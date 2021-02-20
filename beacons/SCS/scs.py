from beacons.beacon import Beacon
from beacons.SCS.path_tree import PathTree

class SCS(Beacon):

    def __init__(self, range, xi_max=5, d_perf=1, d_none=3, pos=None):
        super().__init__(range, xi_max, d_perf, d_none, pos)
        self.path_tree = PathTree(self)