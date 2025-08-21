from amuse.lab import units
from dcaf import StarFormationFramework
from dcaf.models import turbulentclumpmodel as tcm

# --- User parameters ---
Mc = 3000 | units.MSun          # Mass of the parent cloud [Msun]
sfe = 0.5                 # star formation efficiency
surface_density = 0.1 | units.g * units.cm ** -2   # TCM Sigma
seed = 42                 # RNG seed
t_end = 5 | units.Myr
dt = 0.1  | units.Myr

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


TCM = StarFormationFramework(stars)

TCM.evolve_model(t_end,dt)

