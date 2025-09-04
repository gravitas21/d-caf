from amuse.units import units
from amuse.units.quantities import zero
import numpy as np

class BackgroundPotential:
    """
    Base class for Background Gas potentials used in D-CAF

    Parameters
    ----------
    mtot : quantity
        Total mass of the potential (e.g., 1e5 | units.MSun).
    rscale : quantity
        Scale radius (interpretation depends on subclass).
    """

    def __init__(self, mtot, rscale):
        self.mtot = mtot
        self.rscale = rscale

        # What attributes should be saved into a file?
        self.logfile = open('background_gas.dat','a')
        self.output_attributes = ['mtot','rscale' ] 
        self.output_units = [ units.MSun, units.parsec  ]
        self.__first_output = True

    # --- Bridge-facing API ---
    def get_potential_at_point(self, eps, x, y, z):
        """
        Return the gravitational potential at (x, y, z).
        """
        raise NotImplementedError

    def get_gravity_at_point(self, eps, x, y, z):
        """
        Return the gravitational acceleration vector at (x, y, z).
        Should return (ax, ay, az).
        """
        raise NotImplementedError

    def get_mass_inside_radius(self, r):
        """
        Return the mass enclosed within radius r.
        """
        raise NotImplementedError

    def initialize_code(self):
        self.model_time = zero
        pass

    def commit_parameters(self):
        pass

    def commit_particles(self):
        pass

    def cleanup_code(self):
        self.logfile.close()
        pass

    def evolve_model(self,tend):
        self.model_time = tend
        pass

    def write_output(self):
        if self.__first_output:
            header = ['# Time [Myr]']
            header += [f'{k} [{u}]' if u is not None else k
                       for k,u in zip(self.output_attributes, self.output_units)]
            self.logfile.write(' '.join(header) + '\n')
            self.__first_output = False

        vals = [self.model_time.value_in(units.Myr)]
        for k,u in zip(self.output_attributes,self.output_units):
            v = getattr(self, k)
            try :
                vals.append(v.value_in(u) if u is not None else float(v))
            except:
                vals.append(np.Nan)
        self.logfile.write(' '.join(f'{x:.6g}' for x in vals) + '\n')


    def stop(self):
        self.cleanup_code()
