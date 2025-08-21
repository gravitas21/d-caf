# TODO: I am not sure how to handle the configuration yet.. lets decide after
# the script gets more complex
from amuse.units import units

class PetarConfig:
    """Configuration for the PeTar stellar dynamics code.

    Attributes
    ----------
    theta : float
        Tree opening angle (smaller = more accurate, slower).
    dt_soft : object | None
        Optional soft step (AMUSE units), e.g. `0.001 | units.Myr`.
    redirection : str
        'none', 'file', or 'stdout'.
    extra_options : dict
        Additional parameters forwarded to `petar.parameters` if they exist.
    """
    def __init__(self,**kw):
        self.theta = 0.5
        self.dt_soft = None
        self.redirection: str = "none"
        self.number_of_workers = 10


class BridgeConfig:
    """Bridge coupling configuration."""
    def __init__(self,**kw):
        self.timestep =  0.01 | units.Myr
        self.use_threading = False
        self.verbose = True

class GasConfig:
    """Bridge coupling configuration."""
    def __init__(self,**kw):
        pass


def get_default_configuration():
    return dict( petar = PetarConfig() , bridge = BridgeConfig, 
                gas = GasConfig()  )
