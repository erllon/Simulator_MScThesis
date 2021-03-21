from beacons.beacon import Beacon
from beacons.SCS.path_tree import PathTree
import numpy as np
from helpers import polar_to_vec as p2v

class SCS(Beacon):

    def __init__(self, range, xi_max=5, d_perf=1, d_none=3,  k=1, a=1, v=np.array([1, 0]), pos=None):
        super().__init__(range, xi_max, d_perf, d_none, k, a, v, pos)
        self.path_tree = PathTree(self)
    
    def generate_target_pos(self, beacons, ENV, next_min):
        next_min.target_pos = p2v(self.range, np.pi/15)