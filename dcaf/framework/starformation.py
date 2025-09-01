from math import isfinite
from amuse.units import units
from amuse.datamodel.particles import Particles
#pop would do, but this is more efficient
from collections import deque

class StarFormationFramework :
    def __init__( self, target_stars, star_formation_rate = 'infty',
                 nstart = 2):
        self.target_stars = target_stars
        self.star_formation_rate = star_formation_rate
        self.nstart = nstart

        self.initialize_framework()

    def initialize_framework(self):
        """
        This method is called at __init__(). Leave it here in case the user
        needs to perform extra initialization steps.
        """
        self.schedule_formation()

    def get_next_formation_time(self):
        return self.__next_formation_time

    def extract_next_event(self):
        """
        Retrieve next scheduled stars and setup the next formation event.
        This function should be called by form_stars to obtain new stars to
        form.
        """
        #retrieve the new stars and forward framework time to current time
        new_stars = self.__next_stars
        self.current_time = self.__next_formation_time
        self.__setup_next_event()

        return new_stars

    def schedule_formation(self, t0=0 | units.Myr, 
                           dt_tolerance = 1e-12 |units.Myr):
        """
        Build a schedule of star-formation in batches. The sequence of formation
        follows the order of stars. If another sequence is needed, resort the
        positions of target_stars


        Returns
        -------
        formation_sequence : list[Particles]
            Each element is a Particles batch to be added at the corresponding
            time.
        formation_times : list (AMUSE time quantities)
            Times to add those batches.

        dt_tolerance : stars formed closer in time than this will be added
            together.

        Behaviour
        ---------
        - Instantaneous: if `star_formation_rate` is None, 'infty'/'inf', or non-finite (∞), add
          all stars at `t0`.
        - Constant: otherwise, assume a constant star formation rate (SFR). Each
          star i forms at `t0 + (cumulative_mass_i / sfr)`, but the first event
          batches the first two stars together at the *second* star's time
          (`per_star_times[1]`) so the second star is not pulled earlier.
        - Binaries should be added together (TODO)

        """
        sfr = self.star_formation_rate
        stars = self.target_stars

        if len(stars) == 0:
            self.formation_sequence,self.formation_times = deque([]), deque([])
            return

        # Instantaneous modes
        if (sfr is None or (isinstance(sfr, str) and 
                            sfr.lower() in {"infty", "inf"}) ):
            self.formation_sequence = deque([stars.copy()])
            self.formation_times = deque([t0])
            return

        # Constant sfr from amuse quantity
        if not hasattr(sfr, "unit"):
            raise ValueError(
                "SFR must be an AMUSE quantity with units of MSun/Myr"
            )

        # --- Treat non-finite or non-positive SFR as instantaneous
        sfr_val = sfr.value_in(units.MSun / units.Myr)
        if not isfinite(sfr_val) or sfr_val <= 0.0:
            raise ValueError(
                    "Invalid SFR [%S], must be positive and finite"%sfr_val)
        # Set up the formation time of the stars
        # The formation order is set by the position on the Particle set.
        masses = stars.mass
        cum_mass = masses.cumsum()
        per_star_times = t0 + cum_mass / sfr

        if len(stars)< self.nstart:
            raise Exception('Framework must contain at least enough stars for \
            the first batch of nstart = [%i] stars'%self.nstart)

        # First batch: 
        first_time = per_star_times[self.nstart - 1]
        first_batch = Particles()
        first_batch.add_particles(stars[0:self.nstart])

        formation_sequence = [first_batch]
        formation_times = [first_time]

        # --- Remaining stars (indices >= self.nstart)
        for i in range(self.nstart, len(stars)):
            this_time = per_star_times[i]
            # If last batch has same time, append to it
            if ( formation_times 
                    and abs(this_time - formation_times[-1]) < dt_tolerance
                ):
                formation_sequence[-1].add_particles(stars[i:i+1])
            else:
                formation_sequence.append(stars[i:i+1].copy())
                formation_times.append(this_time)

        self.formation_sequence = deque(formation_sequence)
        self.formation_times =  deque(formation_times)

        self.__setup_next_event()

    def __setup_next_event(self):

        if len(self.formation_sequence) > 0:
            newstars = self.formation_sequence.popleft()
            formation_time = self.formation_times.popleft()
            self.__next_formation_time = formation_time
            self.__next_stars = newstars

    def form_stars(self,active_stars=Particles()):
        """
        Retrieve the new stars applying the formation rules and schedule
        the next formation event.

        By default the new stars are passed directly from the scheduled stars,
        i.e. with the positions and velocities from the original particle list.

        If a more complex formation scenario is needed, for instance using the
        formation rules from dcaf.factory.distance_based, then this function
        should be overwritten.

        Note that the first call of this function will be done with an empty set
        of active_stars, then it should handle such case.

        Example:

        Here is a basic example using the function generate_stars from
        dcaf.factory.distance_based (see doc).
        It generate new positions [and velocities?] for the number of requested
        stars based on the position of the existing set and a predefined PDF of
        closest neighbours (see REFERENCE).

        Note that the function MUST handle EMPTY active_stars and the
        generate_stars function MUST handle n_new == 0

        In this example, if active_stars is empty will just return the schedule
        stars with their original coordinates on a gradual formation simulation
        with NO GAS background.
        If n_new== 0 new_stars is an empty set of Particles

        from dcaf.factory import distance_based 
        from dcaf.dcaf import DcafSystem

        class MyFormationFramework(StarFormationFramework):

            def form_stars(self,active_stars):
                # Get the next scheduled stars to form. This method also prepare
                # the next event for the next extract_next_event call.

                next_stars = self.extract_next_event()
                n_new = len(next_stars)
                
                # Obtain new positions based on the existing stars
                # Note that generate_stars should handle n_new == 0
                 The first call will be done with an empty
                # active_stars and such case must be handled here.

                if len(active_stars) ==  0:
                    new_stars = next_stars # first time, keep original
                    coordinates
                else:
                    new_stars = distance_based.generate_stars( active_stars, n_new )
                
                if len(new_stars) > 0 :
                    new_stars.mass = next_stars.mass

                return new_stars

        # Setup the final stars
        ntot = 1000
        Rpl = 10 |units.pc
        masses = new_kroupa_mass_distribution(ntot)
        target_stars =  new_plummer_model(ntot)
        target_stars.mass = masses

        # Setup the final time and the star formation rate as constant
        tend = 10 | units.Myr
        star_formation_rate = masses.sum() / tend

        framework = MyFormationFramework(target_stars,star_formation_rate = star_formation_rate)
        
        # run with default code configuration 
        system = DcafSystem( framework )

        """

        new_stars = self.extract_next_event()

        return new_stars
