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

import numpy as np

from amuse.lab import nbody_system, new_kroupa_mass_distribution, units, new_plummer_model
from amuse.units.quantities import zero
from amuse.units.constants import G

from dcaf.backgroundgas.plummer import PlummerSphere
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

class MyEvolvingPlummer(PlummerSphere):
    """
    A Plummer Sphere model for the background potential with an exponential
    decay mass with decay timescale tau.
    """
    def __init__(self, mtot, rscale, alpha_vir = 1.0,
                 tau_mass = None,
                 tau_radius = None,
                 tdelay = 0 |units.Myr
                 ):
        #The original PlummersSphere input (note mdot = 0, we are not using it)
        super().__init__(mtot, rscale, alpha_vir, mdot = zero)

        # needed for the evolution equations
        self.M0 = mtot
        self.R0 = rscale

        self.tau_radius = tau_radius
        self.tau_mass = tau_mass

        #fallback to None if given zero tau timescales
        if tau_radius :
            self.tau_radius = tau_radius if tau_radius > zero else None
        if tau_mass :
            self.tau_mass = tau_mass if tau_mass > zero else None

        self.tdelay = tdelay if tdelay >= zero else zero

    def evolve_model(self, tend):
        """
        Using Kroupa, Aarseth and Hurley (2001) prescription.
        """
        self.model_time = tend

        if self.model_time > self.tdelay:
            if self.tau_mass:
                self.mtot = self.M0 * \
                        np.exp( -(tend - self.tdelay)/self.tau_mass   )

            if self.tau_radius:
                self.rscale = self.R0 * (
                    1 + ( (tend - self.tdelay)/self.tau_radius )**0.5 
                )


if __name__ == '__main__':

    ################# General parameters:
    ## for Petar
    nworkers = 90

    ################ Parent cloud properties
    Mcloud = 1e4 | units.MSun
    alpha_vir = 1
    Rpl = 5 | units.pc
    tff = 2.341 * ( (Rpl**3) / G/Mcloud ).sqrt()

    #lets evolve for 3 times the initial free fall time:
    t_end = 3*tff

    # this will be the converter used through the simulation
    converter = nbody_system.nbody_to_si(Mcloud, 1.697*Rpl)

    ################ star formation properties
    sfe = 0.2
    Mstars = sfe*Mcloud
    # number of stars in the first batch (must be greater than 1)
    nstart = 10

    # lets form all stars in half a half-mass radius free fall time
    sfr = 2 * Mstars / tff

    ################ Target set of stars
    #create the masses list
    nguess = int(Mstars / units.MSun(2))
    masses = new_kroupa_mass_distribution(nguess,
                                          mass_min=0.08|units.MSun,
                                          mass_max=100|units.MSun )

    masses = masses[ masses.cumsum() <=  Mstars  ]
    nstars = len(masses)

    # Lets give initial coordinates from a similar Plummer model as the
    # background. Note that only the coordinates of the initial stars will be
    # used, the rest will be overridden by the star formation framework

    target_stars = new_plummer_model(nstars,convert_nbody = converter)
    target_stars.mass = masses


    ############ Background potential definition
    # lets define the background potential instance
    # We will only evolve the mass of the background gas.
    # It will decay on the same timescale than the formation of the stars
    # Note that in this prescription the SFE is not conserved during the
    # simulation, neither in realistic simulations.
    # This mimics the formation of stars at the same time than feedback clears
    # out the gas within the stellar cluster, but still star formation happen
    # in locally over-dense areas of the cloud, until we reach the final SFE.
    #
    # Then, the background potential is:
    # Lets begin the mass decay after the first batch of stars have formed:
    # 
    tdelay = target_stars.mass[:nstart].sum() / sfr
    plummer_background = MyEvolvingPlummer(Mcloud, Rpl, alpha_vir = alpha_vir,
                 tau_mass = tff/2,
                 tdelay = tdelay
                 )

    # Construct the framework with the background potential
    framework = MyFormationFramework(target_stars, 
                                     star_formation_rate = sfr,
                                     nstart = nstart, #initial batch of stars
                                     background_gas= plummer_background
                                     )

    cfg = get_default_configuration()
    cfg['petar'].number_of_workers = nworkers

    # run with default code configuration 
    System = DcafSystem( framework, converter = converter, config= cfg )
    System.dt_out = 0.01 |units.Myr

    # Initialize the codes and add initial batch of stars
    System.initialize_system()

    System.evolve_model(t_end)
