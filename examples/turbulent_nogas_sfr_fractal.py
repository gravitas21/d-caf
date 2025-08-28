
from amuse.lab import units
from dcaf.dcaf import DcafSystem
from dcaf.models import turbulentclumpmodel as tcm
from dcaf.framework import StarFormationFramework

from dcaf.factory.fractal import generate_fractal_cascade

# --- User parameters ---
Mc = 3000 | units.MSun          # Mass of the parent cloud [Msun]
sfe = 0.5                 # star formation efficiency
surface_density = 0.1 | units.g * units.cm ** -2   # TCM Sigma
seed = 42                 # RNG seed
t_end = 5 | units.Myr
dt_out = 0.01  | units.Myr
## a toy star formation rate for testing purposes
sfr =  Mc*sfe / t_end 
fractal_dimension = 1.6

# --- Build stellar ICs ---
Mstars = sfe * Mc
masses = tcm.make_kroupa_masses(Mstars, mmax=100 | units.MSun)
stars, params, sfe_eff = tcm.make_turbulent_core_cluster(
    Mc=Mc,
    sfe=sfe,
    k_rho=1.5,
    surface_density= surface_density,
    alpha_vir=1.0,
    phi_Pc=2.0,
    phi_B=2.8,
    fg=1.0,
    aspect_ratio=1,
    keps=-1.0,
    masses=masses,
    seed=seed,
)

def star_factory(new_stars,current_stars):
    # must handle empty current_stars
    print('calling star factory')
    if len(current_stars) <=1:
        return new_stars
    new_posvel = generate_fractal_cascade(current_stars,D_target=fractal_dimension,n_new=len(new_stars),return_as_particles=True,
        min_box = 22**params.Rc.value_in(units.pc) )
    new_posvel.mass = new_stars.mass
    print('done')
    return new_posvel

framework = StarFormationFramework(stars,star_formation_rate=sfr) #default create all stars at begining
TCM = DcafSystem(framework,star_factory=star_factory)
TCM.initialize_sytem()
TCM.dt_out = dt_out

print('initialized fine')

TCM.evolve_model(t_end)
