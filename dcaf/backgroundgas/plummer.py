from __future__ import annotations
from amuse.units import units
from amuse.units.quantities import zero
from amuse.units.constants import G
from .base import BackgroundPotential
import numpy as np

class PlummerSphere(BackgroundPotential):
    """
    Plummer background potential with a mass decay and radial expansion.
    Used in Farias+2025 (in prep)

    Parameters
    ----------
    mtot : quantity
        Total mass (e.g., 1e5 | units.MSun)
    rscale : quantity
        Plummer scale length 'a' (e.g., 1 | units.pc)
    alpha_vir : float
        Virial parameter alpha_vir = 2T/|W|; 
        velocity dispersion scales as sqrt(alpha_vir)

    Gas evolution parameters:

    mdot : quantity. Scale mass depletion rate  (implementing automatic)
        
    t0  : quantity. Initial time of the model (important since temb = tge - t0)

    t_ge : quantity. Gas expulsion time
    t_col : quantity. Cloud collapse timescale (if zero then will stay constant)
    t_exp : quantity. Cloud expansion timescale 
    """
    def __init__(self, mtot, rscale, alpha_vir: float = 1.0,
                 mdot = 0.0 | units.MSun/units.Myr,
                 t0  = 0 | units.Myr,
                 t_ge = 3.0 | units.Myr,
                 t_col = 5.0 | units.Myr,
                 t_exp = 2 | units.Myr,
                 ):
        super().__init__(mtot, rscale)
        self.alpha_vir = float(alpha_vir)
        self.mdot = mdot
        self.model_time = 0.0 | units.Myr
        self.t0 = t0
        self.t_ge = t_ge
        self.t_col = t_col
        self.t_exp = t_exp
        self.rscale_0 = self.rscale
        self.mtot_0 = self.mtot

    def get_potential_at_point(self, eps, x, y, z):
        r2 = x**2 + y**2 + z**2
        a2 = self.rscale**2
        #print('time',self.model_time)
        #print('r2,a2,mtot',r2,a2,self.mtot)
        #print
        return - G * self.mtot / (r2+a2).sqrt()

    def get_gravity_at_point(self, eps, x, y, z):
        r2 = x**2 + y**2 + z**2
        a2 = self.rscale**2
        denom32 = (r2 + a2)**1.5
        coef = - G * self.mtot / denom32
        ax = coef * x
        ay = coef * y
        az = coef * z
        return ax, ay, az

    def get_mass_inside_radius(self, r):
        return self.mtot * r**3 / (r**2 + self.rscale**2)**1.5

    def get_1d_velocity_dispersion_at_point(self, x, y, z):
        """
        From Heggie & Hut (2003):

        In virial equilibrium:
            sigma_1D^2(r) = (G M)/(6 a) * (1 + r^2/a^2)^(-1/2) 

        We can scale the velocities to an arbitrary virial ratio by:
            sigma_1D^2(r,alpha_vir) = sigma_1D^2(r) * sqrt(alpha_vir)
        """
        a = self.rscale
        a2 = a**2
        r2 = x**2 + y**2 + z**2
        sigma2 = (G * self.mtot) / (6.0 * a) * (1.0 + (r2/a2) )**(-0.5) * self.alpha_vir
        return sigma2.sqrt()

    def evolve_model(self, tend):
        """
        Advance to time `tend` using the two-phase background model:

            Embedded phase  (t <= t_ge):
                a(t) = a0 * sqrt(1 - (t - t0)/t_col)
                M(t) = M0 + mdot * (t - t0)

            Expansion phase (t > t_ge):
                a(t) = a(t_ge) * exp((t - t_ge)/tau_exp)
                M(t) = M_break  (constant)

        where mdot < 0 gives mass loss.
        """
        # current to new time interval
        t = tend
        t0 = self.t0
        tge = self.t_ge
        tcol = self.t_col
        texp = self.t_exp

        # --- radius evolution ---
        if t <= tge:
            # collapse phase
            if tcol > zero:
                fac =  1.0 - (t - t0)/tcol 
                self.rscale = self.rscale_0 * fac**0.5
        else:
            # expansion phase
            # compute a at the break to ensure continuity
            if tcol == zero:
                a_break = self.rscale_0
            else:
                facb =  1.0 - (tge - t0)/tcol
                a_break = self.rscale_0 * facb**0.5
            self.rscale = a_break * np.exp((t - tge)/texp )

        # --- mass evolution ---
        if t <= tge:
            self.mtot = self.mtot_0 + self.mdot * (t - t0)
        else:
            # constant after break
            self.mtot = self.mtot_0 + self.mdot * (tge - t0)

        if self.mtot < (0.0 | self.mtot.unit):
            self.mtot = 0.0 | self.mtot.unit

        # update model clock
        self.model_time = t
        return

    def get_potential_derivative_at_point(self, x, y, z):
        r2 = x**2 + y**2 + z**2
        R  = (r2 + self.rscale**2).sqrt()

        Mdot = self.mdot if self.model_time <= self.t_ge else 0 | self.mdot.unit
        if self.model_time <= self.t_ge:
            if self.t_col > zero:
                adot = -0.5 * ( self.rscale_0**2 )  / self.rscale / self.t_col
            else:
                adot = 0 | units.pc/ units.Myr
        else:
            adot = self.rscale / self.t_exp


        term1 = -G*Mdot/R

        term2 =   G * self.mtot * self.rscale * adot / (R**3)

        return term1 + term2

def test_plummer_evolution(
    mtot = 1e5 | units.MSun,
    rscale = 1 | units.pc,
    mdot = -1e4 | (units.MSun / units.Myr),
    t0 = 0 | units.Myr,
    t_ge = 3 | units.Myr,
    t_col = 5 | units.Myr,
    t_exp = 2 | units.Myr,
    model = None ,
    times = [0, 1, 2, 3, 4, 6, 8] | units.Myr,
    test_position = 'auto'
    ):
    """
    Test the evolution of the plummer model.
    model is an instance of the cloud, if given, then the parameters are ignored
    the the model instance is used instead.
    """

    if model is None:
        model = PlummerSphere(
            mtot=mtot,
            rscale=rscale,
            mdot=mdot,
            t0=t0,
            t_ge=t_ge,
            t_col=t_col,
            t_exp=t_exp,
            alpha_vir=1.0
        )

    # --- test times and point ---
    if test_position == 'auto':
        xtest = 0|units.pc
        ytest = 0|units.pc
        ztest = model.rscale
    else:
        xtest,ytest,ztest = test_position

    print(f"{'t [Myr]':>8} {'a [pc]':>10} {'M [Msun]':>12} "
          f"{'Phi(r)':>9} {'dPhi/dt':>15}")
    print("-" * 60)
    for t in times:
        model.evolve_model(t)
        phi = model.get_potential_at_point(0|units.pc,xtest,ytest,ztest)
        phidot = model.get_potential_derivative_at_point(xtest,ytest,ztest)

        print(f"{t.value_in(units.Myr):8.2f} "
              f"{model.rscale.value_in(units.pc):10.3f} "
              f"{model.mtot.value_in(units.MSun):12.1f} "
              f"{phi.value_in(units.kms**2):12.3e}"
              f"{phidot.value_in(units.kms**2/units.Myr):12.3e}")

if __name__ == "__main__":
    test_plummer_evolution()
