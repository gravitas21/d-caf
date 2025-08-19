"""
Utilities to sample star positions and velocities for a (non-binary) cluster
following the turbulent core model used in D-CAF.

This module intentionally avoids any binary initialisation. It only samples
single-star positions and velocities consistent with a power-law density
profile and the associated velocity dispersion scaling.

Depends on: AMUSE (units, Particles), NumPy, SciPy (for hyp2f1), matplotlib (only for tests).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from amuse.lab import units, Particles, new_kroupa_mass_distribution
from amuse.units.quantities import ScalarQuantity
from scipy.special import hyp2f1


@dataclass
class TurbulentCoreParams:
    Rc: ScalarQuantity                 # Core radius [length]
    sigma_surf: ScalarQuantity         # Surface velocity dispersion (1D) [speed]
    sigma_1d: ScalarQuantity           # Volume-weighted 1D velocity dispersion [speed]
    phi_Pmean: float                   # Dimensionless pressure factor
    phi_geom: float                    # Geometric factor for aspect ratio


def _compute_phi_geom(aspect_ratio: float) -> float:
    """Geometric correction for an oblate spheroid with z-stretch = aspect_ratio.
    aspect_ratio = 1 gives phi_geom = 1.
    """
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be > 0")
    if aspect_ratio == 1.0:
        return 1.0
    return float(hyp2f1(0.5, -0.25, 1.5, 1.0 - aspect_ratio ** -2) * aspect_ratio ** 0.5)


def turbulent_core_params(
    Mc: ScalarQuantity = 3000 | units.MSun,
    sfe: float = 0.5,
    kp: float = 1.5,
    alpha_vir: float = 1.0,
    phi_Pc: float = 2.0,
    phi_B: float = 2.8,
    surface_density: ScalarQuantity = 0.1 | units.g * units.cm ** -2,
    fg: float = 1.0,
    aspect_ratio: float = 1.0,
    ar_velocity_scale: bool = True,
    ) -> TurbulentCoreParams:
    """
    Compute key turbulent-core scalings.

    Returns Rc, sigma_surf, and volume-weighted sigma_1d.
    """
    phi_geom = _compute_phi_geom(aspect_ratio)
    kP = 2.0 * (kp - 1.0)
    A = (3.0 - kp) * (kp - 1.0) * fg
    phi_Pmean = (3.0 * np.pi / 20.0) * fg * phi_B * alpha_vir * phi_geom

    # Surface velocity dispersion 
    b = (phi_Pc * phi_Pmean / A / (kP ** 2) / (phi_B ** 4)) ** (1.0 / 8.0)
    c = (Mc / (60 | units.MSun)) ** 0.25
    d = (surface_density / (1 | units.g * units.cm ** -2)) ** 0.25
    sigma_surf = (1.91 * b * c * d) | units.kms

    # Core radius
    aR = (A / (kP ** 2) / phi_Pc / phi_Pmean) ** 0.25
    bR = (Mc / (60 | units.MSun)) ** 0.5
    cR = (surface_density / (1 | units.g * units.cm ** -2)) ** (-0.5)
    Rc = (0.071 * aR * bR * cR) | units.parsec

    # Volume-weighted (global) 1D sigma from surface scaling
    sigma_1d = (2.0 * (3.0 - kp) / (8.0 - 3.0 * kp) * sigma_surf).in_(units.kms)

    if aspect_ratio != 1.0 and ar_velocity_scale:
        vscale = np.sqrt(
            np.arcsinh(aspect_ratio ** 2 - 1.0) / (aspect_ratio ** 2 - 1.0) * phi_geom
        )
        sigma_surf *= np.sqrt(vscale)
        sigma_1d *= np.sqrt(vscale)

    return TurbulentCoreParams(
        Rc=Rc, sigma_surf=sigma_surf, sigma_1d=sigma_1d, phi_Pmean=float(phi_Pmean), phi_geom=float(phi_geom)
    )


def sample_turbulent_core_cluster(
    *,
    Mc: ScalarQuantity = 3000 | units.MSun,
    sfe: float = 0.5,
    kp: float = 1.5,
    alpha_vir: float = 1.0,
    phi_Pc: float = 2.0,
    phi_B: float = 2.8,
    surface_density: ScalarQuantity = 0.1 | units.g * units.cm ** -2,
    fg: float = 1.0,
    aspect_ratio: float = 1.0,
    keps: float = -1.0,
    masses: Optional[ScalarQuantity] = None,
    nstars: Optional[int] = None,
    m_equal: Optional[ScalarQuantity] = None,
    ar_velocity_scale: bool = True,
    seed: Optional[int] = 432,
) -> tuple[Particles, TurbulentCoreParams, float]: 
    """
    Create a single-stars cluster realisation with positions and velocities consistent with the
    turbulent-core model.

    You can provide either:
      - `masses`: a Quantity array of stellar masses [MSun]; or
      - `nstars` AND `m_equal`: use equal-mass stars of mass `m_equal`.

    Returns (stars, params), where `stars` is an AMUSE Particles set containing .mass, .x/.y/.z,
    .vx/.vy/.vz, `params` holds Rc and dispersions, and `sfe_eff` is the realised stellar mass
    fraction.
    """
    if masses is None:
        if nstars is None or m_equal is None:
            raise ValueError("Provide either `masses` or both `nstars` and `m_equal`.")
        masses = (np.ones(nstars) | units.none) * m_equal
    else:
        # ensure AMUSE quantity
        masses = masses.in_(units.MSun)

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    params = turbulent_core_params(
        Mc=Mc,
        sfe=sfe,
        kp=kp,
        alpha_vir=alpha_vir,
        phi_Pc=phi_Pc,
        phi_B=phi_B,
        surface_density=surface_density,
        fg=fg,
        aspect_ratio=aspect_ratio,
        ar_velocity_scale=ar_velocity_scale,
    )

    Mstars_target = (sfe * Mc).in_(units.MSun)
    Mstars_eff = masses.sum().in_(units.MSun)
    sfe_eff = (Mstars_eff / Mc)
    # Note: the masses may not add to Mstars_target, then the resulting sfe is not exact

    n = len(masses)

    # Mass-order sampling: cumulative mass defines radius quantiles
    # Generalised exponent from the user's implementation (keps = -1 -> 1/(3-kp))
    expo = 2.0 / (6.0 - kp * (keps + 3.0))

    # Randomise order to avoid spatial correlations with mass unless wanted
    order = np.arange(n)
    rng.shuffle(order)
    m_sorted = masses[order]

    cmass = np.cumsum(m_sorted.value_in(units.MSun)) | units.MSun
    r = (cmass / Mstars_eff) ** expo * params.Rc

    # Isotropic angles
    u = rng.uniform(size=n)
    costheta = 2.0 * u - 1.0
    sintheta = np.sqrt(1.0 - costheta ** 2)
    phi = 2.0 * np.pi * rng.uniform(size=n)

    x = r * sintheta * np.cos(phi)
    y = r * sintheta * np.sin(phi)
    z = r * costheta
    if aspect_ratio != 1.0:
        z *= aspect_ratio

    # Velocity dispersion scaling: sigma(r) = sigma_surf * (r/Rc)^{(2-kp)/2}
    s1d = (params.sigma_surf * (r / params.Rc) ** ((2.0 - kp) / 2.0)).in_(units.kms)

    vx = (rng.normal(scale=1.0, size=n) | units.none) * s1d
    vy = (rng.normal(scale=1.0, size=n) | units.none) * s1d
    vz = (rng.normal(scale=1.0, size=n) | units.none) * s1d

    stars = Particles(n)
    # Place back in original indexing order to keep `masses` alignment predictable
    inv = np.argsort(order)
    stars.mass = m_sorted[inv]
    stars.x = x[inv]
    stars.y = y[inv]
    stars.z = z[inv]
    stars.vx = vx[inv]
    stars.vy = vy[inv]
    stars.vz = vz[inv]
    stars.radius = (0.01 | units.parsec)  # placeholder, will be overwritten when using stellar evolution

    return stars, params, float(sfe_eff)


# -----------------------
# Simple test helper (optional)
# -----------------------

def make_kroupa_masses(Mstars: ScalarQuantity, mmax: ScalarQuantity = 100 | units.MSun) -> ScalarQuantity:
    """Draw Kroupa IMF masses until the sum reaches ~Mstars, then trim the last draw."""
    mtot = 0 | units.MSun
    out = []
    while mtot < Mstars:
        m = new_kroupa_mass_distribution(1, mass_max=mmax)[0]
        out.append(m.value_in(units.MSun))
        mtot += m
    masses = (np.array(out) | units.MSun)
    # Remove overshoot if any
    if masses.sum() > Mstars and len(masses) > 0:
        masses = masses[:-1]
    return masses


# test_tc_core.py
"""
Quick executable test: create a cluster and plot XY positions coloured by speed,
plus a radial sigma profile vs. the expected scaling.

Usage:
    python test_tc_core.py
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Model setup
    Mc = 3000 | units.MSun
    sfe = 0.5
    kp = 1.5
    aspect_ratio = 1.0  # try 2.0 for a stretched z-axis

    Mstars = sfe * Mc
    masses = make_kroupa_masses(Mstars, mmax=100 | units.MSun)

    stars, params, sfe_eff = sample_turbulent_core_cluster(
        Mc=Mc,
        sfe=sfe,
        kp=kp,
        surface_density=0.1 | units.g * units.cm ** -2,
        alpha_vir=1.0,
        phi_Pc=2.0,
        phi_B=2.8,
        fg=1.0,
        aspect_ratio=aspect_ratio,
        keps=-1.0,
        masses=masses,
        seed=432,
    )

    # XY scatter coloured by speed
    print(f"effective SFE realised: {sfe_eff:.3f}")
    r = (stars.position.lengths() / params.Rc)
    speed = stars.velocity.lengths().value_in(units.kms)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sc = plt.scatter(
        stars.x.value_in(units.parsec),
        stars.y.value_in(units.parsec),
        c=speed,
        s=6,
        alpha=0.8,
    )
    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title("Turbulent-core XY positions (colour: |v| km/s)")
    cbar = plt.colorbar(sc)
    cbar.set_label("speed [km/s]")
    plt.axis("equal")

    # Radial sigma profile vs expected
    plt.subplot(1, 2, 2)
    # Bin by r/Rc
    bins = np.linspace(0, 1.2, 20)
    which = np.digitize(r, bins)
    sig_obs = []
    rmid = []
    for b in range(1, len(bins)):
        sel = which == b
        if sel.any():
            vx = stars[sel].vx.value_in(units.kms)
            vy = stars[sel].vy.value_in(units.kms)
            vz = stars[sel].vz.value_in(units.kms)
            sig_obs.append( np.sqrt(np.var(vx) + np.var(vy) + np.var(vz)) / np.sqrt(3.0) )
            #sig_obs.append(stars[sel].velocity.lengths().std().value_in(units.kms))
            rmid.append(0.5 * (bins[b] + bins[b - 1]))

    rmid = np.array(rmid)
    sig_obs = np.array(sig_obs)

    rplot = np.linspace(1e-3, 1.2, 200)
    sig_exp = (params.sigma_surf * (rplot ) ** ((2.0 - kp) / 2.0)).value_in(units.kms)

    plt.plot(rplot, sig_exp, lw=2, label="expected σ(r)")
    if len(rmid) > 0:
        plt.scatter(rmid, sig_obs, s=18, label="measured (binned)")
    plt.xlabel("r / Rc")
    plt.ylabel("σ [km/s]")
    plt.title("Velocity dispersion profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig('turbulent_clump_model_test.pdf')
    #plt.show()

# -----------------------
# Pytest numerical check
# -----------------------
def test_sigma_slope_matches():
    Mc = 1000 | units.MSun
    sfe = 0.5
    kp = 1.5
    masses = make_kroupa_masses(sfe * Mc, mmax=50 | units.MSun)
    stars, params, _ = sample_turbulent_core_cluster(Mc=Mc, sfe=sfe, kp=kp, masses=masses, seed=42)
    r = (stars.position.lengths() / params.Rc)
    vx = stars.vx.value_in(units.kms)
    vy = stars.vy.value_in(units.kms)
    vz = stars.vz.value_in(units.kms)
    sig_local = np.sqrt(np.var(vx) + np.var(vy) + np.var(vz)) / np.sqrt(3.0)
    # expected slope (log σ vs log r)
    slope_theory = (2.0 - kp) / 2.0
    # measure slope with linear fit in log space
    mask = r > 1e-2
    coeffs = np.polyfit(np.log10(r[mask]), np.log10(stars.velocity.lengths().value_in(units.kms)[mask]), 1)
    slope_measured = coeffs[0]
    assert np.isclose(slope_measured, slope_theory, atol=0.2)

