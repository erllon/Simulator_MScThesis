from beacons.beacon import Beacon
from beacons.SCS.path_tree import PathTree
import numpy as np
from helpers import polar_to_vec as p2v
from copy import deepcopy

class SCS(Beacon):

    def __init__(self, range, xi_max=5, d_perf=1, d_none=3,  k=1, a=1, v=np.array([1, 0]), pos=None):
        super().__init__(range, xi_max, d_perf, d_none, k, a, v, pos)
        self.path_tree = PathTree(self)
    
    def generate_target_pos(self, beacons, ENV, next_min):
        angle = np.pi/15
        next_min.target_pos = p2v(self.range, angle)
        next_min.target_angle = angle
        next_min.first_target_pos = deepcopy(next_min.target_pos)
    
    """
    PLOTTING STUFF
    """
    def plot(self, axis):
        self.point = axis.plot(*self.pos, color="black", marker="o", markersize=8)[0]
        self.annotation = axis.annotate(self.ID, xy=(self.pos[0], self.pos[1]), fontsize=14)
        return self.point, self.annotation
    
    def toJson(self):
        jsonDict = {
        'Type': 'SCS',
        'ID': self.ID#,
        # 'Neighbors': [neigh.toJson() for neigh in self.neighbors],
        # 'Pathtree': self.,
        # 'pos_traj': self._pos_traj.tolist(),#,np.array([3,4]).reshape(2,1)])).to_json(orient='values')#,
        # 'force_traj': self._v_traj.tolist(),#np.array([np.array([7,8]).reshape(2,1),np.array([9,10]).reshape(2,1)]).tolist(),
        # 'heading_traj': self._heading_traj.tolist(),#np.array([np.array([5,5]).reshape(2,1),np.array([6,6]).reshape(2,1)]).tolist(),
        # 'xi_traj': self._xi_traj.tolist()##np.array([1,2,3,4,5,6]).tolist()
        }
        return jsonDict