"""
D-CAF: Dynamic Cluster Assembly Framework
"""
from __future__ import annotations
import os
import numpy as np

from amuse.datamodel import Particles
from amuse.units.constants import G
from amuse.units import units, nbody_system
from amuse.couple.bridge import Bridge
try: 
    from amuse.community.petar.interface import Petar
    PETAR_INSTALLED = True
except:
    print('Warning PeTar not installed, can not run simulations' )
    PETAR_INSTALLED = False

from amuse.io import write_set_to_file

from dcaf.utilities.parameters import get_default_configuration
from dcaf.utilities.logger import setup_logger
from dcaf.utilities.config import load_config

#cfg = load_config("config.yaml")
# from dcaf.framework import StarFormationFramework  

class DcafSystem:
    def __init__(
        self,
        framework,# requires: target_stars, next_formation_time(), form_stars()
        config = None,
        converter = None,
        gas_code = None,
        output_folder = "./dcaf_output/",
        log_level = 'debug'
    ):
        self.config = config or get_default_configuration()


        self.framework = framework

        self.dt_out = 0.5 | units.Myr   # TODO: move to config
        self.output_folder = output_folder
        self.snapshot_basename = "stars_"
        self.__current_snapshot = 0
        self.logger = setup_logger(self.output_folder,log_level)

        if self.framework.background_gas:
            self.gas_code = self.framework.background_gas

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
        self._formed_stars = Particles()
        self._formed_stars_mod = True # this tells that the formed_stars set may
                            #   have changed or not yet ready
        self.model_time = None

        # Codes 
        self.petar_code = None
        self.bridge_code = None
        self.code = None

        # Particle channel
        self._channel_code_to_mem = None

        self.dt_soft_eff = None

        # Energy checks
        self._total_energy = [0 | units.J, 0 | units.J]        # [previous, current]
        # Energy components
        # [Tstars, Ustars_self, U_stars_gas, W_gas]
        self._energy_components = [0 | units.J]*4
        # Cumulatie individual energy budgets
        # [ E_new_stars, E_SE, E_gas_evol, E_binaries ]
        self._energy_budgets = [0 | units.J]*4
        self._energy_header_written = False
        # last step, cumulative
        self._energy_errors = [0, 0]

    @property
    def formed_stars(self):
        """
        Return already formed stars updated. Only update the particles if model
        time has drifted or more stars has been added. Otherwise, the stored copy is provided.
        """
        if self.petar_code is None:
            return self._formed_stars

        self.logger.debug( f'[FORMED STARS] Requested at time: {self.model_time.in_(units.Myr)} '
                          )
        if ( getattr(self._formed_stars.collection_attributes, "code_time", -1|units.Myr) != self.petar_code.model_time 
            or len(self._formed_stars) != len(self.petar_code.particles) ):
            self.update_formed_stars()
        return self._formed_stars

    def update_formed_stars(self):
        with self.logger.timing('[UPDATING STARS] ********************'):
            self._formed_stars = self.petar_code.particles.copy()
            self._formed_stars.collection_attributes.code_time = self.petar_code.model_time

    def initialize_system(self):
        """Instantiate PeTar and, if present, Bridge. Also, add initial stars."""
        if not PETAR_INSTALLED:
            print('PeTar not installed. System not initialized')
            return

        with self.logger.timing('[DCAF] Initializing system *********************'):
            # lets validate the framework before anything:
            dt_soft = self.config["petar"].dt_soft  #should be nbody
            if dt_soft is None:
                raise ValueError('dt_soft must be provided in this '

                            'implemenation')
            if not (dt_soft.unit == nbody_system.time and 
                    np.isclose(
                        np.log2(dt_soft.value_in(nbody_system.time)), 
                        round(np.log2(dt_soft.value_in(nbody_system.time)
                    )), atol=1e-12)):
                raise ValueError(
                    "[DCAF] dt_soft must be in nbody_system.time units and an "
                    " exact power of 2.")

            self.dt_soft_eff = self.converter.to_si(dt_soft)
            self._validate_formation_schedule(dt_soft)

            #lets have the first set of stars from the star formation framework,
            # to set the initial time and initial stars:
            tnext = self.framework.get_next_formation_time()
            if tnext is None:
                raise Exception('No formation events in framework')
            tnext = self._ceil_to_block(tnext)
            newstars = self.framework.form_stars(Particles())

            #TODO: For gas implementation, make sure here that the gas evolve up to
            # this point
            with self.logger.timing('Initializing PeTar'):
                self._setup_petar()

            #add the initial stars
            self.model_time = tnext #for logging
            self._add_new_stars(newstars)

            # Make sure output directory exists
            os.makedirs(self.output_folder, exist_ok=True)

            # Initialize time

            self.dt_soft_eff = self.petar_code.parameters.dt_soft
            self.logger.info('[DCAF] effective dt_soft changed from'
                    f' {self.config["petar"].dt_soft}'
                    f' {self.converter.to_nbody(self.dt_soft_eff)}'
                    f' ({self.dt_soft_eff.in_(units.Myr)})'
                    )
            if abs(self.converter.to_nbody(self.dt_soft_eff)
                - self.config["petar"].dt_soft) > 1e-15 |nbody_system.time:
                raise Exception (
                    '[DCAF] effective dt_soft changed from'
                    f' {self.config["petar"].dt_soft}'
                    #f'{self.dt_soft_eff.in_(units.Myr)}'
                    f' {self.converter.to_nbody(self.dt_soft_eff)}'
                    ' This may happened if dt_soft was not provided in nbody'
                    ' units'
                    )

            self.model_time = tnext
            self.petar_code.parameters.begin_time = self.model_time

            #also advance background gas
            # this may be taken care by bridge. TODO: check after methods are ready
            if self.gas_code:
                self.logger.info('[DCAF] [BGAS] evolved to' 
                                 f'{self.model_time.in_(units.Myr)} ' )
                self.gas_code.evolve_model(self.model_time)

            #initialize bridge
            if self.gas_code is not None:
                n_timestep = 1 # every how many blocktimesteps should we do a kick?
                self._setup_bridge()

                self.logger.info('[DCAF] [BRIDGE] setup with effective time-step: '
                                 f'{(self.dt_soft_eff * n_timestep).in_(units.Myr)}')
                # In this framework we only need gas to star interaction
                self.bridge_code.add_system(self.petar_code,(self.gas_code,),False)
                # is this line needed?
                self.bridge_code.add_system(self.gas_code,)
                self.code = self.bridge_code
            else:
                self.logger.info('[DCAF][BRIDGE] no BGAS found. Evolving only with '
                    'PeTar')
                self.bridge_code = None
                self.code = self.petar_code
        #write initial output
        self.write_output()

    # --- setup helpers -----------------------------------------------------
    def _setup_petar(self):
        """Initialize PeTar. No particles are added here."""
        cfg = self.config["petar"]
        self.petar_code = Petar(self.converter,mode = 'cpu',
                                redirection = cfg.redirection,
                                number_of_workers = cfg.number_of_workers)

        self.petar_code.parameters.theta = cfg.theta
        self.petar_code.parameters.r_bin = cfg.r_bin
        self.petar_code.parameters.r_out = cfg.r_out 
        self.petar_code.parameters.dt_soft = cfg.dt_soft

    def _setup_bridge(self):
        cfg = self.config["bridge"]
        if cfg.timestep is None:
            timestep = self.dt_soft_eff
        else:
            timestep = cfg.timestep


        self.bridge_code = Bridge(timestep = timestep,
                                  use_threading=cfg.use_threading,
                                  verbose=cfg.verbose)

    def setup_gas(self):
        # TODO: add setup gas routine to StarFormationFramework
        #   I think we dont need this with the current implementation
        pass

    # --- main loop ---------------------------------------------------------

    def evolve_model(self, t_end):
        """Advance the coupled system to t_end, interleaving outputs and
        formation events."""
        if self.model_time is None:
            raise RuntimeError("Call initialize_system() before evolve_model().")

        self.logger.info(f"[DCAF] Evolving to {t_end.in_(units.Myr)}")

        time = self.model_time
        t_output = self._ceil_to_block( time + self.dt_out )

        while time < t_end:
            # Next event times
            tnext = self.framework.get_next_formation_time()
            if tnext is None:
                tnext = 10*t_end
            tnext = self._ceil_to_block( tnext )

            # 0: finish; 1: output; 2: form stars
            event_times = (t_end, t_output, tnext)
            i_event, t_stop = min(enumerate(event_times), key=lambda x: x[1])
            self.logger.info(
                    f"[DCAF] (next event id: {i_event}) evolving from  "
                    f"{self.model_time.in_(units.Myr)} to "
                    f"{t_stop.in_(units.Myr)}"
                    )

            # Evolve dynamics up to t_stop if we actually need to advance time
            n_now = len(self.petar_code.particles)
            if (time < t_stop) and (n_now >= 2 or n_now == 0):
                # Evolve only if there is either a reasonable N or no stars (some codes allow)
                with self.logger.timing('[DCAF] Evolving *********************'):
                    self.code.evolve_model(t_stop)
                    # make formed stars is updated next time is accessed
                    self._formed_stars_mod = True


            # Update clock
            time = t_stop
            self.model_time = time

            # 1) Output event
            if i_event == 1:
                self.write_output()
                t_output += self.dt_out

            # 2) Star-formation event
            if i_event == 2:
                with self.logger.timing('[GENERATE STARS]*********'):
                    new_stars = self.framework.form_stars(self.formed_stars)
                self._add_new_stars(new_stars)

            # --- I/O ---------------------------------------------------------------

    def write_output(self):
        with self.logger.timing('[WRITING OUTPUT]', False):
            self.logger.info(
                f"[DCAF] [WRITING OUTPUT] Snap: {self.__current_snapshot}, "
                f"Time {self.model_time.in_(units.Myr)} *************"
            )
            filename = os.path.join(self.output_folder, f"{self.snapshot_basename}{self.__current_snapshot:03d}")

            self.formed_stars.collection_attributes.model_time = self.model_time
            write_set_to_file(self.formed_stars, filename + ".amuse", format='amuse')
            self.__current_snapshot += 1

            if self.framework.background_gas and hasattr(self.framework.background_gas, 'write_output'):
                self.framework.background_gas.write_output()

            self._energy_check()
            self._write_energy_row()

    # --- internals ---------------------------------------------------------

    def _add_new_stars(self, stars):
        with self.logger.timing('[ADDING STARS]', False):
            nactive = len(self.petar_code.particles)
            self.logger.info(f'[ADDING STARS]  Time: {self.model_time.value_in(units.Myr)}  '
                             f'{len(stars)} to {nactive}/{len(self.target_stars)} **************')

            # Add to PeTar
            #if len(self.petar_code.particles) == 0 :
            #    #no stars yet, lets add stars first and use the full code
            #    #potential energy
            #    self.petar_code.particles.add_particles(stars)
            #    self.__inject_new_stars_energy(stars,first_call = True)
            #else:
            #    # regular method, use petar potential method for new stars, 
            #    # and calculate 
            #    self.__inject_new_stars_energy(stars)
            #   


            if len(self.petar_code.particles) == 0 :
                U0 = 0 | units.J
            else:
                U0 = self.petar_code.potential_energy
            self.petar_code.particles.add_particles(stars)
            self.__inject_new_stars_energy(stars,U0=U0)


            for s in stars:
                self.logger.info(
                    f"[NEW_STAR] "
                    f"{self.model_time.value_in(units.Myr):.6f} "
                    f"{s.key:d} "
                    f"{s.mass.value_in(units.MSun):.6f} "
                    f"{s.x.value_in(units.parsec):.8f} "
                    f"{s.y.value_in(units.parsec):.8f} "
                    f"{s.z.value_in(units.parsec):.8f} "
                    f"{s.vx.value_in(units.kms):.6f} "
                    f"{s.vy.value_in(units.kms):.6f} "
                    f"{s.vz.value_in(units.kms):.6f}"
                )

    def __inject_new_stars_energy(self, new_stars, U0 = 0 | units.J):
        """
        Add the injected energy of `new_stars` to the budget
        """
        # --- kinetic of new stars
        v2 = new_stars.vx**2 + new_stars.vy**2 + new_stars.vz**2
        dE_kin = 0.5 * (new_stars.mass * v2).sum()

        dE_star = self.petar_code.potential_energy - U0

        # --- gas potential at new-star positions (assumed constant background here)
        if getattr(self, "gas_code", None) is not None:
            phi_gas = self.gas_code.get_potential_at_point(
                0 | units.m, new_stars.x, new_stars.y, new_stars.z
            )
            dE_gas = (new_stars.mass * phi_gas).sum()
        else:
            dE_gas = 0 | units.J

        dE = dE_kin + dE_star + dE_gas

        # add energy to the budget. Should be added to current and last, because
        # they should be equal
        self._total_energy[0] += dE
        self._total_energy[1] += dE
        # add to the individual energy budget for added stars
        self._energy_budgets[0] += dE

        if hasattr(self, "logger"):
            self.logger.debug(
                f"[ADD_STARS] dE_injected={dE.in_(units.J)}; "
                f"E_budget={self._total_energy[1].in_(units.J)}"
            )
        return dE


    def _ceil_to_block(self,t_si):
        """
        Get the closest time to a dt_soft multiple.
        We will perform operations only on those times for better performance
        """
        if self.dt_soft_eff is None:
            raise Exception( 'dt_soft_eff must be updated' )
        dtnb = self.converter.to_nbody(self.dt_soft_eff).number
        knb  = self.converter.to_nbody(t_si).number / dtnb
        k    = np.ceil(knb)                                 # nearest integer index
        return self.converter.to_si( ( k * dtnb)  | nbody_system.time )


    def _energy_check(self):
        stars = self.formed_stars
        # Ekin
        v2 = stars.vx**2 + stars.vy**2 + stars.vz**2
        Tstars = 0.5 * (stars.mass * v2).sum()

        # Estars_self (prefer solver-native)
        Ustars_self = self.petar_code.potential_energy

        # E_stars_gas (constant gas)
        if getattr(self, "gas_code", None) is not None:
            phi = self.gas_code.get_potential_at_point(0 | units.m, stars.x, stars.y, stars.z)
            U_stars_gas = (stars.mass * phi).sum()
        else:
            U_stars_gas = 0 | units.J

        W_gas = 0 | units.J  # constant background TODO
        # to the total energy I should add here the W_gas difference between
        # last check and now. This W_gas should be tracked internally by the 
        # gas_code every time bridge sends a kick.

        self._total_energy[0] = self._total_energy[1]
        self._total_energy[1] = Tstars + Ustars_self + U_stars_gas 

        # numerical errors 
        eps          = np.abs(self._total_energy[1] - self._total_energy[0])
        Eref  = self._total_energy[1]
        relative_error = abs(eps / Eref) if Eref != (0 | units.J) else 0.0

        self._energy_components = [Tstars, Ustars_self, U_stars_gas, W_gas]
        self._energy_errors[0] = relative_error # step error
        self._energy_errors[1] += relative_error # cumulative error

    def _write_energy_row(self):
        """
        Append to energy file
        """
        path = os.path.join(self.output_folder, "energy.dat")
        file_is_new = not os.path.exists(path)

        # extract scalars in SI (J) and time in Myr
        t_myr = float(self.model_time.value_in(units.Myr))
        t_nb = float( self.converter.to_nbody(self.model_time).number )
        vals = [
            float(self._total_energy[1].value_in(units.J)),  # Total energy
            float(self._energy_errors[0] ), # step error
            float(self._energy_errors[1] ), # cumulative error
            float(self._energy_components[0].value_in(units.J)),  # Ekin
            float(self._energy_components[1].value_in(units.J)),  # U_self
            float(self._energy_components[2].value_in(units.J)),  # U_bg
            float(self._energy_components[3].value_in(units.J)),  # W_gas
            float(self._energy_budgets[0].value_in(units.J)),  # E_new_stars
            float(self._energy_budgets[1].value_in(units.J)),  # E_SE
            float(self._energy_budgets[2].value_in(units.J)),  # E_gas_evol
            float(self._energy_budgets[3].value_in(units.J)),  # E_bin
        ]

        with open(path, "a") as f:
            if file_is_new:
                f.write(
                    "# units in Joules by default \n"
                    "# t_Myr t_nb E |dE/E|  sum(|dE/E|) T_*  U_*,*  U_*,gas"
                    "  W_gas  E_new_stars  E_SE   E_gas_evol  E_bin\n"
                )

            line = f"{t_myr:.6e}  {t_nb:.6e}  "
            for v in vals:
                line += f'{v:.6e}  '
            line += '\n'
            f.write(line)

    def _validate_formation_schedule(self,dt_soft_nb):
        """
        Validate that the intended formation schedule do not violate a set of
        rules designed to avoid adding stars twice during the same dt_soft
        block. Dcaf will be delay the formation events to the end of the next
        dt_soft block for better performance, but this means it can not add
        twice on the same block since may cause unintended results.
        Instead of handeling those events in dcaf, we leave the user to handle
        the decission on how to proceed. In case those rules are brokent, we
        raise an informative exception with suggested modifications to either
        the formation schedule or the chosen dt_soft.
        As a rule of thumb, formation events should be separated at least twice
        dt_soft in order to be consistent.

        The scheduled times should follow these rules:
          - First formation time t0 may be < dt_soft (we will place the first add at >= dt_soft).
          - From the first forming block onward, no two formation times may fall in the same dt_soft block.
        """
        dt_soft = self.converter.to_si(dt_soft_nb)
        ftimes = getattr(self.framework, "formation_times", None)
        if not ftimes or len(ftimes) == 0:
            raise ValueError("[DCAF][VALIDATOR] formation_times is empty.")

        ftimes = [ t.value_in(units.Myr) for t in ftimes ] | units.Myr

        t0 = ftimes[0]

        t_eff0 = dt_soft if t0 < dt_soft else self._ceil_to_block(t0)

        blocks = np.floor(((ftimes - t_eff0) / dt_soft)).astype(int)

        # Check duplicates after the first
        if len(np.unique(blocks[1:])) != len(blocks[1:]):
            # Gaps between consecutive events
            gaps = np.diff(np.array(ftimes))
            min_gap = np.min(gaps[1:]) if len(gaps) > 1 else None

            raise ValueError("\n"
                "[DCAF][VALIDATOR] Two or more formation events fall in the "
                "same dt_soft block. Counting from the first forming block at "
                f"{t_eff0.in_(units.Myr)} "
                f"(minimum width is dt_soft= {dt_soft.in_(units.Myr)})\n"
                f"Currently the minimum space between formation events provided"
                f" is {min_gap.in_(units.Myr) if min_gap else 'N/A'}. "
                " The solution is either decrease dt_soft below the minimum"
                " space between events or increase the space between events "
                " to a value greater than dt_soft. "
                " If using a class inherited from "
                "dcaf.framework.starformation.StarFormationFramework this means "
                "setting the dt_tolerance > dt_soft "
            )
if __name__ == "__main__":
    pass
