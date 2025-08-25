from math import isfinite
from amuse.units import units
from amuse.datamodel.particles import Particles
#pop would do, but this is more efficient
from collections import deque

class StarFormationFramework :
    def __init__( self, target_stars, star_formation_rate = 'infty' ):
        self.target_stars = target_stars
        self.star_formation_rate = star_formation_rate
        self.schedule_formation()

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
        stars = self.target_stars  # adjust if your pending-stars live elsewhere

        if len(stars) == 0:
            self.formation_sequence,self.formation_times = deque([]), deque([])
            return

        # --- Instantaneous modes
        if sfr is None or (isinstance(sfr, str) and sfr.lower() in {"infty", "inf"}):
            self.formation_sequence,self.formation_times = deque([stars.copy()]), deque([t0])
            return

        # --- Coerce plain numeric SFR to AMUSE quantity (assume MSun/Myr)
        if isinstance(sfr, (int, float)):
            sfr = sfr * (units.MSun / units.Myr)

        # --- Validate SFR units
        if not hasattr(sfr, "unit"):
            raise ValueError(
                "sfr must be an AMUSE quantity with units of MSun/Myr or a number"
            )

        # --- Treat non-finite or non-positive SFR as instantaneous
        try:
            sfr_val = sfr.value_in(units.MSun / units.Myr)
            if not isfinite(sfr_val) or sfr_val <= 0.0:
               self.formation_sequence,self.formation_times = deque([stars.copy()]), deque([t0])
               return
        except Exception:
            # If unit conversion fails, be conservative
            self.formation_sequence,self.formation_times = deque([stars.copy()]), deque([t0])

        # --- Per-star appearance times from cumulative mass (respect current order)
        masses = stars.mass
        cum_mass = masses.cumsum()
        per_star_times = t0 + cum_mass / sfr

        # --- If there's only one star total, we cannot make a 2-star first batch.
        if len(stars) == 1:
            self.formation_sequence,self.formation_times = deque([stars[0:1].copy()]), deque([per_star_times[0]])
            return

        # --- First batch: indices 0 and 1 together at the *second* star's time
        first_time = per_star_times[1]
        first_batch = Particles()
        first_batch.add_particles(stars[0:1])
        first_batch.add_particles(stars[1:2])

        formation_sequence = [first_batch]
        formation_times = [first_time]

        # --- Remaining stars (indices >= 2)
        for i in range(2, len(stars)):
            this_time = per_star_times[i]
            # If last batch has same time, append to it
            if ( formation_times 
                    and abs(this_time - formation_times[-1]) < dt_tolerance
                ):
                formation_sequence[-1].add_particles(stars[i:i+1])
            else:
                formation_sequence.append(stars[i:i+1].copy())
                formation_times.append(this_time)

        self.formation_sequence,self.formation_times = deque(formation_sequence), deque(formation_times)


    def get_next_stars(self):
        """ Get the next scheduled stars for formation. 
        returns:
            formation_time : ScalarQuantity 
            newstars : Particles 
        """
        if len(self.formation_sequence) > 0:
            newstars = self.formation_sequence.popleft()
            formation_time = self.formation_times.popleft()
            return formation_time,newstars
        else:
            return None, None


