"""
D-CAF: model-independent runner utilities to set up PeTar for stars and a
background gas provided by the user.

- This module sets up **PeTar** explicitly and (optionally) couples star–gas via
  **Bridge**.
"""
# --- AMUSE imports
#from amuse.datamodel import Particles
from amuse.units import units, nbody_system
from amuse.couple.bridge import Bridge
from amuse.community.petar.interface import Petar

from dcaf.config.parameters import get_default_configuration
#from dcaf.framework import StarFormationFramework


class DcafSystem:
    def __init__(self,framework, config = None, converter = None, gas_code = None ):
        if config is None:
            self.config = get_default_configuration()
        else:
            self.config = config

        self.gas_code = gas_code
        self.converter = converter
        self.framework = framework
        self.dt_out = 0.5 | units.Myr #TODO: this should be on a config file

        #TODO: need a user defined or better automated way to define the
        #converter
        stars = framework.target_stars
        self.converter = nbody_system.nbody_to_si(stars.total_mass(),
                                                  stars.virial_radius())
        self.current_time = None

    def initialize_sytem(self,
                        ):
        # TODO: figure out what methods exactly will need here for the gas, and
        # update the documentation.
        self.setup_petar()

        if self.gas_code is not None:
            self.setup_bridge()

            self.bridge_code.add_system(self.petar_code, (self.gas_code,))

            # In the usual bridge scheme, we require a star_to_gas code to direction
            # the stars gravitational influenece on the gas
            # In this framework we do not model the gas, then we do not require 
            # this code.
            self.bridge_code.add_system(self.gas_code, None )

            self.code = self.bridge_code
        else:
            self.bridge_code = None
            self.code = self.petar_code

        self.current_time = 0 |units.Myr

    def setup_petar(self):
        """Initialize Petar. Note that no particles are added here. """

        cfg = self.config['petar']
        self.petar_code = Petar(self.converter,redirection=cfg.redirection,
                                number_of_workers = cfg.number_of_workers)

        self.petar_code.parameters.theta = cfg.theta
        if cfg.dt_soft is not None:
            self.petar_code.parameters.dt_soft = cfg.dt_soft

    def setup_bridge(self):
        cfg = self.config['bridge']
        self.bridge_code = Bridge(
                timestep = cfg.timestep,
                use_threading=cfg.use_threading,
                verbose = cfg.verbose
                )
    
    def setup_gas(self):
        #TODO: add setup gas routine to StarFormationFramework
        pass


    def evolve_model(self, t_end):

        time = self.current_time
        t_output = time + self.dt_out
        while time < t_end :

            # ask framework for the next formation event
            tnext, starsnext = self.framework.get_next_stars()
            if tnext is None:
                tnext = t_end

            # Select the next event stored, so that
            # i_event = 0 : advance to the end
            # i_event = 1 : call output routine
            # i_event = 2 : form stars and continue
            print(t_end,t_output,tnext)
            event_times = (t_end, t_output, tnext) 
            i_event, t_stop = min(enumerate(event_times), key=lambda x: x[1])
            t_stop = min(t_end, t_output, tnext)

            ### Evolve the model to the stop condition
            ## This is particularly important at start, so petar do not evolve
            #without particles.
            # TODO: make sure petar evolves with 2 or more particles
            # May need to make a small code to handle one single particle
            # in the prescence of the potential
            if tnext > t_stop:
                self.code.evolve_model(t_stop)


            # Do work and prepare to restart the loop
            time = t_stop
            self.current_time = time

            # 1) Output event
            if i_event == 1 :
                self._output()  # placeholder for your output routine
                t_output += self.dt_out

            # 2) Star-formation event
            if i_event == 2 :
                self._add_new_stars(starsnext)

        # If you want a final output right at t_end when the loop stops early:
        #if self.current_time < t_end:
        #    self.code.evolve_model(t_end)
        #    self.current_time = t_end
        #    print(f"evolved to: {t_end}")
        #    if abs(self.current_time - t_output) <= eps:
        #        self._output()

    def _output(self):
        print('evolved to %s'%self.current_time)

    def _add_new_stars(self,stars):
        self.petar_code.particles.add_particles(stars)

if __name__ == "__main__":
    pass

