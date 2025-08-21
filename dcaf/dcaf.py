"""
D-CAF: model-independent runner utilities to set up PeTar for stars and a
background gas provided by the user.

- This module sets up **PeTar** explicitly and (optionally) couples star–gas via
  **Bridge**.
"""
# --- AMUSE imports
from amuse.datamodel import Particles
from amuse.units import units, nbody_system
from amuse.couple.bridge import Bridge
from amuse.community.petar.interface import Petar

from dcaf.models.config.parameters import get_default_configuration



# =====================
# Bridge assembly
# =====================

class StarFormationFramework:
    def __init__(self,stars, config = None, converter = None ):
        if config is None:
            self.config = get_default_configuration()
        else:
            self.config = config

        self.converter = converter

        #TODO: need a user defined or better automated way to define the
        #converter
        self.converter = nbody_system.nbody_to_si(stars.total_mass(),
                                                  stars.virial_radius())
        self.current_time = None

    def initialize_sytem(self, stars , #not finished
                        petar_cfg = None, 
                        gas_code = None, 
                        bridge_cfg = None,
                        starformation_cfg = None
                        ):

        # TODO: figure out what methods exactly will need here for the gas, and
        # update the documentation.

        self.setup_petar()

        if gas_code is not None:
            self.setup_bridge()

            self.bridge.add_system(petar, (gas_code,))

            # In the usual bridge scheme, we require a star_to_gas code to direction
            # the stars gravitational influenece on the gas
            # In this framework we do not model the gas, then we do not require 
            # this code.
            self.bridge.add_system(gas_code, None )

            self.system = self.bridge
        else:
            self.bridge = None
            self.code = self.petar

        self.current_time = 0 |units.Myr

    def setup_petar(self):
        """Initialize Petar. Note that no particles are added here. """

        cfg = self.config['petar']
        self.petar_code = Petar(self.converter,redirection=cfg.redirection,
                                number_of_workers = cfg.number_of_workers)

        self.petar_code.parameters.theta = cfg.theta
        if cfg.dt_soft is not None:
            self.petar.parameters.dt_soft = cfg.dt_soft

    def setup_bridge(self):
        cfg = self.config['bridge']
        self.bridge = Bridge(
                timestep = cfg.timestep,
                use_threading=cfg.use_threading,
                verbose = cfg.verbose
                )
    
    def setup_gas(self):
        #TODO: add setup gas routine to StarFormationFramework
        pass


    def evolve_model(self,t_end, dt,  diagnostics=True):
        time = 0.0 | units.Myr
        while time < t_end - 0.5 * dt:
            self.code.evolve_model(time + dt)
            time = time + dt
            self.current_time = time
            if diagnostics:
                try:
                    ekin = self.stars_in_code.kinetic_energy()
                    epot = self.stars_in_code.potential_energy(G=units.constants.G)
                    print(
                        "t={:.3f} Myr  E_kin={}  E_pot={}  N={}".format(
                            time.value_in(units.Myr), ekin.in_(units.J), epot.in_(units.J), len(stars)
                        )
                    )
                except Exception:
                    pass



if __name__ == "__main__":
    pass

