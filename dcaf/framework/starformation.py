from amuse.units import units
#pop would do, but this is more efficient
from collections import deque


class StarFormationFramework :
    def __init__( self, target_stars ):
        self.target_stars = target_stars
        self.star_formation_rate = 'infty'
        self.schedule_formation()

    def schedule_formation(self):

        if self.star_formation_rate == 'infty':
            self.formation_sequence = deque([self.target_stars])
            self.formation_times = deque([ 0 |units.Myr ])
        else:
            #TODO: add finite star formation rate times
            pass


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


