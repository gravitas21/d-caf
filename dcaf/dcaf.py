"""
D-CAF: model-independent runner utilities to set up PeTar for stars and a
background gas provided by the user.

- This module sets up **PeTar** explicitly and (optionally) couples star–gas via
  **Bridge**.
"""
from __future__ import annotations
import os

from amuse.datamodel import Particles
from amuse.units import units, nbody_system
from amuse.couple.bridge import Bridge
from amuse.community.petar.interface import Petar
from amuse.io import write_set_to_file

from dcaf.utilities.parameters import get_default_configuration
# from dcaf.framework import StarFormationFramework  


class DcafSystem:
    def __init__(
        self,
        framework,# requires: target_stars, next_formation_time(), form_stars()
        config = None,
        converter = None,
        gas_code = None,
        output_folder = "./dcaf_output/",
    ):
        self.config = config or get_default_configuration()

        self.gas_code = gas_code
        self.framework = framework

        self.dt_out = 0.5 | units.Myr   # TODO: move to config
        self.output_folder = output_folder
        self.snapshot_basename = "stars_"
        self.__current_snapshot = 0

        # Converter: use provided one or derive from framework target stars
        stars0 = framework.target_stars
        if len(stars0) < 2:
            raise ValueError('Framework target stars must have at least two \
            particles')
        if converter is None:
            converter = nbody_system.nbody_to_si(stars0.total_mass(),\
                    stars0.virial_radius())
        self.converter = converter

        # Runtime state
        self.target_stars = stars0
        self.formed_stars = Particles()
        self.current_time = None

        # Codes 
        self.petar_code = None
        self.bridge_code = None
        self.code = None

        # Particle channel
        self._channel_code_to_mem = None

    def initialize_system(self):
        """Instantiate PeTar and, if present, Bridge. Also, add initial stars."""

        #lets have the first set of stars from the star formation framework,
        # to set the initial time and initial stars:
        tnext = self.framework.get_next_formation_time()
        if tnext is None:
            raise Exception('No formation events in framework')
        newstars = self.framework.form_stars(Particles())

        # Initialize time
        self.current_time = tnext
        #TODO: For gas implementation, make sure here that the gas evolve up to
        # this point

        self._setup_petar(initial_time=tnext)

        if self.gas_code is not None:
            self._setup_bridge()
            # In this framework we only need gas to star interaction
            self.bridge_code.add_system(self.petar_code, (self.gas_code,))
            if self.gas_code is not None:
                self.bridge_code.add_system(self.gas_code, None)
            self.code = self.bridge_code
        else:
            self.bridge_code = None
            self.code = self.petar_code

        #add the initial stars
        self._add_new_stars(newstars)

        # Make sure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

    # --- setup helpers -----------------------------------------------------
    def _setup_petar(self, initial_time = 0 |units.Myr ):
        """Initialize PeTar. No particles are added here."""
        cfg = self.config["petar"]
        self.petar_code = Petar(self.converter,mode = 'cpu',
                                redirection = cfg.redirection,
                                number_of_workers = cfg.number_of_workers)

        self.petar_code.parameters.theta = cfg.theta
        self.petar_code.parameters.r_bin = cfg.r_bin
        self.petar_code.parameters.r_out = cfg.r_out 
        self.petar_code.parameters.dt_soft = cfg.dt_soft
        self.petar_code.parameters.begin_time = initial_time

    def _setup_bridge(self):
        cfg = self.config["bridge"]
        self.bridge_code = Bridge(timestep=cfg.timestep,
                                  use_threading=cfg.use_threading,
                                  verbose=cfg.verbose)

    def setup_gas(self):
        # TODO: add setup gas routine to StarFormationFramework
        pass

    # --- main loop ---------------------------------------------------------

    def evolve_model(self, t_end):
        """Advance the coupled system to t_end, interleaving outputs and formation events."""
        if self.current_time is None:
            raise RuntimeError("Call initialize_system() before evolve_model().")

        print(f"Evolving to {t_end}")

        time = self.current_time
        t_output = time + self.dt_out

        while time < t_end:
            # Next event times
            tnext = self.framework.get_next_formation_time()
            if tnext is None:
                tnext = t_end

            # 0: finish; 1: output; 2: form stars
            event_times = (t_end, t_output, tnext)
            i_event, t_stop = min(enumerate(event_times), key=lambda x: x[1])
            t_stop = min(t_end, t_output, tnext)
            print(f"evolving from (event: {i_event}) {self.current_time} -> {t_stop}")

            # Evolve dynamics up to t_stop if we actually need to advance time
            # Optional safety: avoid evolving with <2 particles if you suspect PeTar dislikes that
            n_now = len(self.petar_code.particles)
            if (time < t_stop) and (n_now >= 2 or n_now == 0):
                # Evolve only if there is either a reasonable N or no stars (some codes allow)
                self.code.evolve_model(t_stop)

            # Update clock
            time = t_stop
            self.current_time = time

            # 1) Output event
            if i_event == 1:
                print("####################### WRITING OUTPUT #######################")
                self.write_output()
                t_output += self.dt_out

            # 2) Star-formation event
            if i_event == 2:
                print("####################### ADDING STARS #######################")
                new_stars = self.framework.form_stars(self.formed_stars)
                self._add_new_stars(new_stars)

            # --- I/O ---------------------------------------------------------------

    def write_output(self):
        # If we have a live channel, refresh the shadow copy; otherwise, fall back to a direct copy
        if self._channel_code_to_mem is not None:
            self._channel_code_to_mem.copy()
        else:
            self.formed_stars = self.petar_code.particles.copy()

        filename = os.path.join(self.output_folder, f"{self.snapshot_basename}{self.__current_snapshot:03d}")
        print(f"Writing output to {filename}.hdf5")
        output_stars = self.formed_stars.copy()
        output_stars.collection_attributes.timestamp = self.current_time.in_(
                units.Myr)
        write_set_to_file(self.formed_stars, filename + ".hdf5")
        self.__current_snapshot += 1

    # --- internals ---------------------------------------------------------

    def _add_new_stars(self, stars: Particles):
        # Add to PeTar and refresh shadow copy + channel
        self.petar_code.particles.add_particles(stars)
        self.formed_stars = self.petar_code.particles.copy()
        self._channel_code_to_mem = self.petar_code.particles.new_channel_to(self.formed_stars)


if __name__ == "__main__":
    pass
