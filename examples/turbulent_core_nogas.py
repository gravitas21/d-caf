"""
Example: stars-only D-CAF run using the turbulent-core model for initial conditions.

- Builds stellar ICs from models.tc (no gas code).
- Sets up PeTar via dcaf_runner and evolves for a short duration.

Usage (typical):
    python example_stars_only_tc.py \
        --n-stars 2000 \
        --m-cluster 1500  # in Msun (optional; if omitted, IMF draws sum to random total) \
        --r-scale 0.5     # in parsec \
        --t-end 0.1       # in Myr \
        --dt 0.005        # in Myr \
        --seed 42

Notes:
- It assumes you have `models/turbulentcoremodel.py` providing:
    * turbulent_core_params(...)
    * make_kroupa_masses(...)
    * sample_turbulent_core_cluster(...)
- No typing module; explicit, readable Python.
"""

import argparse
from amuse.lab import units

# Import the model-independent runner utilities
from dcaf_runner import PetarConfig, assemble_system, run_dcaf

# Import your IC generator from the models package
from models import turbulentclumpmodel as tcm


def build_stars_from_turbulent_core(n_stars, m_cluster, r_scale, seed, **tc_kwargs):
    """Helper to produce a Particles set using the turbulent-core model.

    Parameters
    ----------
    n_stars : int
    m_cluster : float or None
        Cluster mass in Msun if provided; otherwise draw n_stars by IMF.
    r_scale : float
        Scale radius in parsec for the model.
    seed : int or None
    tc_kwargs : dict
        Extra keyword arguments forwarded to turbulent_core_params.
    """
    if m_cluster is not None:
        masses = tc.make_kroupa_masses(n_stars=n_stars, m_cluster=m_cluster | units.MSun, seed=seed)
    else:
        masses = tc.make_kroupa_masses(n_stars=n_stars, seed=seed)

    params = tc.turbulent_core_params(r_scale=r_scale | units.parsec, **tc_kwargs)
    stars = tc.sample_turbulent_core_cluster(masses=masses, params=params, seed=seed)
    return stars


def main():
    parser = argparse.ArgumentParser(description="D-CAF stars-only example with turbulent-core ICs (PeTar).")
    parser.add_argument("--n-stars", type=int, default=1000, help="Number of stars to sample.")
    parser.add_argument("--m-cluster", type=float, default=None, help="Total cluster mass in Msun (optional).")
    parser.add_argument("--r-scale", type=float, default=0.5, help="Scale radius in parsec for the core model.")
    parser.add_argument("--t-end", type=float, default=0.05, help="Final time in Myr.")
    parser.add_argument("--dt", type=float, default=0.01, help="Step size in Myr.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--theta", type=float, default=0.5, help="PeTar tree opening angle.")
    parser.add_argument("--eta", type=float, default=0.1, help="PeTar timestep parameter.")
    parser.add_argument("--dt-soft", type=float, default=None, help="PeTar soft step in Myr (optional).")
    parser.add_argument("--no-diagnostics", action="store_true", help="Suppress step diagnostics printing.")

    args = parser.parse_args()

    # Build stellar ICs from your turbulent-core module
    stars = build_stars_from_turbulent_core(
        n_stars=args.n_stars,
        m_cluster=args.m_cluster,
        r_scale=args.r_scale,
        seed=args.seed,
    )

    # Configure PeTar
    petar_cfg = PetarConfig(theta=args.theta, eta=args.eta)
    if args.dt_soft is not None:
        petar_cfg.dt_soft = args.dt_soft | units.Myr

    # Assemble a stars-only system (no gas)
    system, comps = assemble_system(stars, petar_cfg=petar_cfg, gas_code=None)

    # Evolve
    t_end = args.t_end | units.Myr
    dt = args.dt | units.Myr
    try:
        run_dcaf(t_end=t_end, dt=dt, system=system, stars=stars, diagnostics=not args.no_diagnostics)
    finally:
        # Clean shutdown
        try:
            system.stop()
        except Exception:
            try:
                comps.get("petar", None) and comps["petar"].stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
