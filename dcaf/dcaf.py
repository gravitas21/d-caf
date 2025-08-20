"""
D-CAF: model-independent runner utilities to set up PeTar for stars and a
background gas provided by the user.

- This module sets up **PeTar** explicitly and (optionally) couples star–gas via
  **Bridge**.
"""

from dataclasses import dataclass, field

# --- AMUSE imports
from amuse.lab import units, Particles
from amuse.couple.bridge import Bridge
from amuse.community.petar.interface import Petar


# =====================
# Configuration objects
# =====================
@dataclass
class PetarConfig:
    """Configuration for the PeTar stellar dynamics code.

    Attributes
    ----------
    theta : float
        Tree opening angle (smaller = more accurate, slower).
    eta : float
        Timestep parameter (smaller = more accurate, slower).
    dt_soft : object | None
        Optional soft step (AMUSE units), e.g. `0.001 | units.Myr`.
    redirection : str
        'none', 'file', or 'stdout'.
    extra_options : dict
        Additional parameters forwarded to `petar.parameters` if they exist.
    """
    theta: float = 0.5
    eta: float = 0.1
    dt_soft: object = None
    redirection: str = "none"
    extra_options: dict = field(default_factory=dict)


@dataclass
class BridgeConfig:
    """Bridge coupling configuration."""
    timestep: object = field(default_factory = lambda: 0.01 | units.Myr)
    use_threading: bool = False
    verbose = True


# =====================
# Validators
# =====================

def validate_stars(stars):
    """Check that the provided Particles set has the necessary attributes with
    units.

    Required attributes: mass, x, y, z, vx, vy, vz.
    """
    if stars is None or not isinstance(stars, Particles):
        raise ValueError("`stars` must be an AMUSE Particles set.")

    required = ["mass", "x", "y", "z", "vx", "vy", "vz"]
    for attr in required:
        if not hasattr(stars, attr):
            raise ValueError(f"Particles are missing required attribute: {attr}")
        val = getattr(stars, attr)
        try:
            _ = val.unit  # ensure it has units
        except Exception:
            raise ValueError(f"Attribute `{attr}` must have AMUSE units.")
    return True


# =====================
# PeTar setup helper
# =====================

def setup_petar(stars, cfg = None):
    """Instantiate and initialize PeTar with provided stellar particles."""
    validate_stars(stars)

    if cfg is None:
        cfg = PetarConfig()

    petar = Petar(redirection=cfg.redirection)

    # Basic tuning
    petar.parameters.theta = cfg.theta
    petar.parameters.eta = cfg.eta
    if cfg.dt_soft is not None:
        petar.parameters.dt_soft = cfg.dt_soft

    # Apply any extra options provided by the user
    for k, v in (cfg.extra_options or {}).items():
        if hasattr(petar.parameters, k):
            setattr(petar.parameters, k, v)

    # Send initial conditions to the code
    petar.particles.add_particles(stars)

    return petar


# =====================
# Bridge assembly
# =====================

def assemble_system(stars , petar_cfg = None, gas_code = None, 
                    bridge_cfg = None):
    """
    Create PeTar for stars and optionally couple to a provided gas code via
    Bridge.

    Inputs:

    stars : amuse.datamodel.particles.Particles set. Depending on the model to
        use, should contain the masses, coordinates and velocities of the stars.
        Depending on the star cluster assembly mechanism, positions and
        velocities could be override at run time.

    petar_cfg : dcaf.PetarConfig dataclass containing the Petar configuration
        parameters. 

    bridge_cfg  : dcaf.BridgeConfig dataclass containing bridfge configuration
        parameters 

    gas_code    : User defined background gas class. Should contain methods such
        as .get_potential_at_point and .evolve. Gas will not be influenced by
        the stars as this framework is one directional focusing on the evolution
        of the stars.
    """
    # TODO: figure out what methods exactly will need here for the gas, and
    # update the documentation.

    validate_stars(stars)
    petar = setup_petar(stars, petar_cfg)
    components = {"petar": petar}

    if gas_code is None:
        return petar, components

    if bridge_cfg is None:
        bridge_cfg = BridgeConfig()

    if not hasattr(gas_code, "evolve_model"):
        #TODO: be more flexible with the gas code to use a user defined class
        raise ValueError("`gas_code` must be an AMUSE community code instance.")

    bridge = Bridge(
            timestep = bridge_cfg.timestep,
            use_threading=bridge_cfg.use_threading,
            verbose = bridge_cfg.verbose
            )

    bridge.add_system(petar, (gas_code,))

    # In the usual bridge scheme, we require a star_to_gas code to direction
    # the stars gravitational influenece on the gas
    # In this framework we do not model the gas, then we do not require 
    # this code.
    bridge.add_system(gas_code, None )

    components["gas"] = gas_code
    components["bridge"] = bridge
    return components


# =====================
# Runner (evolve)
# =====================

def run_model(t_end, dt, system, stars, diagnostics=True):
    """Evolve an already-assembled system for a given duration."""
    validate_stars(stars)

    time = 0.0 | units.Myr
    while time < t_end - 0.5 * dt:
        system.evolve_model(time + dt)
        if hasattr(system, "model_time"):
            time = system.model_time
        else:
            time = time + dt

        if diagnostics:
            try:
                ekin = stars.kinetic_energy()
                # TODO: Add the potential of the gas here
                epot = stars.potential_energy(G=units.constants.G)
                print(
                    "t={:.3f} Myr  E_kin={}  E_pot={}  N={}".format(
                        time.value_in(units.Myr), ekin.in_(units.J), epot.in_(units.J), len(stars)
                    )
                )
            except Exception:
                pass

    return system, stars


if __name__ == "__main__":
    pass

