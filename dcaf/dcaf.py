"""
D-CAF: model-independent runner utilities to set up PeTar for stars and a
background gas provided by the user.

- This module sets up **PeTar** explicitly and (optionally) couples star–gas via
  **Bridge**.
"""
import sys
# --- AMUSE imports
from amuse.datamodel import Particles
from amuse.units import units, nbody_system
from amuse.couple.bridge import Bridge
from amuse.community.petar.interface import Petar
from amuse.io import write_set_to_file

from dcaf.utilities.parameters import get_default_configuration
#from dcaf.framework import StarFormationFramework


class DcafSystem:
    def __init__(self,framework, config = None, converter = None, gas_code =
                 None,
                 output_folder = './',
                 star_factory = None):
        if config is None:
            self.config = get_default_configuration()
        else:
            self.config = config

        self.star_factory = star_factory
        self.gas_code = gas_code
        self.converter = converter
        self.framework = framework
        self.dt_out = 0.5 | units.Myr #TODO: this should be on a config file
        self.output_folder = output_folder #TODO: this should be on a config file
        self.__current_snapshot = 0
        self.snapshot_basename = 'stars_'

        #TODO: need a user defined or better automated way to define the
        #converter
        stars = framework.target_stars
        self.converter = nbody_system.nbody_to_si(stars.total_mass(),
                                                  stars.virial_radius())
        self.target_stars = stars
        self.formed_stars = Particles()
        self.current_time = None

    def initialize_sytem(self,
                        ):
        # TODO: figure out what methods exactly will need here for the gas, and
        # update the documentation.
        self.setup_petar()
        #self.channel_from_code_to_memmory = \
        #    self.petar_code.particles.new_channel_to(self.formed_stars)

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
        print('Evolving to %s'%t_end)

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
            event_times = (t_end, t_output, tnext) 
            i_event, t_stop = min(enumerate(event_times), key=lambda x: x[1])
            t_stop = min(t_end, t_output, tnext)
            print('evolving from (event: %i) %s -> %s'%(i_event,self.current_time,t_stop))

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
                print('\
        ####################### WRITING OUTPUT #######################\
                ')
                self.write_output()  # placeholder for your output routine
                t_output += self.dt_out

            # 2) Star-formation event
            if i_event == 2 :
                print('\
        ####################### ADDING STARS #######################\
                ')
                self._add_new_stars(starsnext)

        # If you want a final output right at t_end when the loop stops early:
        #if self.current_time < t_end:
        #    self.code.evolve_model(t_end)
        #    self.current_time = t_end
        #    print(f"evolved to: {t_end}")
        #    if abs(self.current_time - t_output) <= eps:
        #        self._output()

    def write_output(self):
        self.channel_from_code_to_memmory.copy()

        filename = '%s/%s%03i'%(self.output_folder,self.snapshot_basename,
                                         self.__current_snapshot )
        print('Writing output to %s.hdf5'%filename)
        write_set_to_file(self.formed_stars,filename+'.hdf5')
        self.__current_snapshot += 1

    def _add_new_stars(self,stars):
        if self.star_factory is not None:
            stars = self.star_factory(stars,self.formed_stars)

        self.petar_code.particles.add_particles(stars)
        self.formed_stars = self.petar_code.particles.copy()
        self.channel_from_code_to_memmory = \
            self.petar_code.particles.new_channel_to(self.formed_stars)



if __name__ == "__main__":
    pass

