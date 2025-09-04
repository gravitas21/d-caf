from __future__ import annotations
from amuse.units import units
from amuse.units.constants import G
from .base import BackgroundPotential

class PlummerSphere(BackgroundPotential):
    """
    Plummer background potential with a constant mass decay defined by mdot.

    Parameters
    ----------
    mtot : quantity
        Total mass (e.g., 1e5 | units.MSun)
    rscale : quantity
        Plummer scale length 'a' (e.g., 1 | units.pc)
    alpha_vir : float
        Virial parameter alpha_vir = 2T/|W|; 
        velocity dispersion scales as sqrt(alpha_vir)
    mdot : quantity
        Constant mass loss/accretion rate
    """
    def __init__(self, mtot, rscale, alpha_vir: float = 1.0,
                 mdot = 0.0 | units.MSun/units.Myr):
        super().__init__(mtot, rscale)
        self.alpha_vir = float(alpha_vir)
        self.mdot = mdot
        self.model_time = 0.0 | units.Myr

    def get_potential_at_point(self, eps, x, y, z):
        r2 = x**2 + y**2 + z**2
        return - G * self.mtot / (r2).sqrt()

    def get_gravity_at_point(self, eps, x, y, z):
        r2 = x**2 + y**2 + z**2
        denom32 = (r2)**1.5
        coef = - G * self.mtot / denom32
        static_coef = coef * x, coef * y, coef * z


        return static_coef

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
        r = np.sqrt(x**2 + y**2 + z**2)
        sigma2 = (G * self.mtot) / (6.0 * a) * (1.0 + (r/a)**2)**(-0.5) * self.alpha_vir
        return sigma2.sqrt()

    def evolve_model(self, tend):
        """
        Advance to time `tend`, applying constant mass loss:
            M(t) = M0 + mdot * (t - t0)

        Negative mdot means the mass decays.
        """
        dt = tend - self.model_time
        self.mtot += self.mdot * dt
        if self.mtot < (0.0 | self.mtot.unit):
            self.mtot = 0.0 | self.mtot.unit
        self.model_time = tend
