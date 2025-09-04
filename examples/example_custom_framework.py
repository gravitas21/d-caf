"""
Custom StarFormationFramework example
-------------------------------------

This is a basic example of using the function generate_stars from
dcaf.factory.distance_based (see doc). It generate new positions [and
velocities? (TODO)] for the number of stars requested based on the position of
the existing set and a predefined PDF of closest neighbours,
which by default is a lognormal distribution with mu = 1e-2 pc and a sigma_log
= 0.9 (see REFERENCE here).

Note that the framework.form_stars method MUST handle an initial call with
EMPTY active_stars and the generate_stars function MUST handle n_new == 0

In this example, if active_stars is empty will just return the schedule stars
with their original coordinates on a gradual formation simulation with NO GAS
background.

If n_new== 0 new_stars is an empty set of Particles
"""

from amuse.lab import nbody_system, new_kroupa_mass_distribution, units, new_plummer_model

from dcaf.dcaf import DcafSystem
from dcaf.framework import StarFormationFramework
from dcaf.factory import distance_based
from dcaf.utilities.parameters import get_default_configuration

class MyFormationFramework(StarFormationFramework):

    def form_stars(self,active_stars):
        # Get the next scheduled stars to form. 
        # the extract_next_event method retrieves the next scheduled forming
        # stars and delete them from the scedule list.

        next_stars = self.extract_next_event()
        n_new = len(next_stars)

        # Obtain new positions based on the existing stars Note that
        # generate_stars should handle n_new == 0 The first call will be done
        # with an empty active_stars and such case must be handled here.

        if len(active_stars) ==  0:
            new_stars = next_stars # first time, keep original coordinates
        else:
            new_stars = distance_based.generate_stars( active_stars, n_new )

        if len(new_stars) > 0 :
            new_stars.mass = next_stars.mass

        return new_stars


if __name__ == '__main__':

    # Create the final set of stars
    # Note that only the first batch of star will keep their original positions

    ntot = 2000
    Rpl = 0.5 |units.pc
    masses = new_kroupa_mass_distribution(ntot)
    converter = nbody_system.nbody_to_si(masses.sum(),Rpl)
    nworkers = 50


    target_stars =  new_plummer_model(ntot,convert_nbody = converter)
    target_stars.mass = masses

    # Setup the final time and the star formation rate as constant
    t_end = 10 | units.Myr
    # let's form stars continously until 5 Myr mark
    star_formation_rate = masses.sum() / t_end*0.25

    # star formation rate could be:
    #   - float : constant str
    #   - a function of time (TODO)
    #   - a table with time and sfr columns, where inbetween values will be
    #   interpolated linearly (TODO)
    #   - 'infty' on which case all stars will be added on the first call. 
    framework = MyFormationFramework(target_stars,
                                     star_formation_rate = star_formation_rate,
                                     nstart = 10 #initial batch of stars
                                     )
    
    cfg = get_default_configuration()
    cfg['petar'].number_of_workers = nworkers
    #print(cfg['petar'].number_of_workers)
    #exit()
    
    # run with default code configuration 
    System = DcafSystem( framework, converter = converter, config= cfg )
    System.dt_out = 0.01 |units.Myr

    # Initialize the codes and add initial batch of stars
    System.initialize_system()

    System.evolve_model(t_end)
