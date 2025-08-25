from amuse.lab import units
from dcaf.dcaf import DcafSystem
from dcaf.models import turbulentclumpmodel as tcm
from dcaf.framework import StarFormationFramework

# --- User parameters ---
Mc = 3000 | units.MSun          # Mass of the parent cloud [Msun]
sfe = 0.5                 # star formation efficiency
surface_density = 0.1 | units.g * units.cm ** -2   # TCM Sigma
seed = 42                 # RNG seed
t_end = 5 | units.Myr
dt_out = 0.01  | units.Myr
## a toy star formation rate for testing purposes
sfr =  Mc*sfe / t_end 

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

framework = StarFormationFramework(stars,star_formation_rate=sfr,
                                   dt_tolerance = 0.001 | u.Myr) #default create all stars at begining
TCM = DcafSystem(framework)
TCM.initialize_sytem()
TCM.dt_out = dt_out

print('initialized fine')

TCM.evolve_model(t_end)

