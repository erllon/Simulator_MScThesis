from beacons.beacon import Beacon
from beacons.SCS.path_tree import PathTree
import numpy as np
from helpers import polar_to_vec as p2v
from copy import deepcopy

class SCS(Beacon):

    def __init__(self, scs_id, range, xi_max, d_perf, d_none, d_tau, k=1, a=1, v=np.array([1, 0]), pos=None):
        super().__init__(scs_id, range, xi_max, d_perf, d_none, d_tau, k, a, v, pos)
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
        self.point = axis.plot(*self.pos, color="black", marker="o", markersize=6)[0]
        """test_redoing_plot_large_cR_2.json"""
        self.annotation = axis.annotate(self.ID, xy=(self.pos[0]-0.2, self.pos[1]+0.17), fontsize=14)
        """redoing_zig_zag_test_30_54.json"""
        # self.annotation = axis.annotate(self.ID, xy=(self.pos[0]+0.07, self.pos[1]+0.17), fontsize=14)
        """deployment_comp_uniform_small_cR_15_drones_3.json"""
        # self.annotation = axis.annotate(self.ID, xy=(-0.45, self.pos[1]-0.25), fontsize=14)
        """redoing_stripa_test_80_drones_9.json"""
        # self.annotation = axis.annotate(self.ID, xy=(-0.45, self.pos[1]-0.75), fontsize=14)

        return self.point, self.annotation
    
    def toJson(self):
        jsonDict = {
        'Type': 'SCS',
        'ID': self.ID
        }
        return jsonDict