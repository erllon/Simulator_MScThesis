from deployment.deployment_helpers import (
    get_obstacle_forces as gof,
    get_neighbor_forces as gnf
)
from helpers import (
    rot_z_mat as R_z,
    normalize
)
from beacons.MIN.min import MinState 
from abc import ABC, abstractmethod
import numpy as np

class AtTargetException(Exception):
    """
    Raised when there are no more beacons, i.e. the 
    MIN has arrived sufficiently close to its target.
    """

class FollowingStrategy(ABC):

    MAX_FOLLOWING_SPEED = 2
    MIN_RSSI_SWITCH_BEACON = np.exp(-0.05)#0.95#np.exp(-0.1)#np.exp(-0.28) #HERE
    DEADZONE_RSSI_STRENGTH = np.exp(-0.05)

    def __init__(self, same_num_neighs_differentiator, rand_lim = np.pi/4):
        self.__snnd = same_num_neighs_differentiator
        self.__prev_RSSI, self.__curr_RSSI = 0, 0
        self.beacons_to_follow = None
        self.__deadzone_v = None
        self.__moved_past_target = False
        self.__rand_lim = rand_lim

    def compute_next_beacon_to_follow(self):
        self.btf = self.beacons_to_follow.pop(0)

    def is_following_final_target(self):
        return self.btf == self.target

    @abstractmethod
    def prepare_following(self, MIN, beacons, SCS):
        self.__compute_target(beacons, SCS)
        print(f"{MIN.ID} targeting {self.target.ID}")

    @abstractmethod
    def get_following_velocity(self, MIN, beacons, ENV):
        pass
    
    def __compute_target(self, beacons, SCS):
        self.target = beacons[0]
        if len(beacons) > 1:
            tmp = beacons[1:]
            num_neighs = np.array([len(b.neighbors) for b in tmp])
            min_neigh_indices, = np.where(num_neighs == num_neighs.min())
            if len(min_neigh_indices) > 1:
                self.target =  self.__snnd(tmp[min_neigh_indices], lambda beacon: np.linalg.norm(SCS.get_vec_to_other(beacon)))
            else:
                self.target = tmp[min_neigh_indices[0]]

    @staticmethod
    def follow_velocity_wrapper(func):
        def func_wrapper(self, MIN, beacons, ENV):
            self.__prev_RSSI = self.__curr_RSSI
            self.__curr_RSSI = MIN.get_xi_to_other_from_model(self.btf)#MIN.get_RSSI(self.btf)#
            # print(f"self.__curr_RSSI: {self.__curr_RSSI}")
            if self.is_following_final_target():
                #HERE
                if self.__curr_RSSI >= FollowingStrategy.MIN_RSSI_SWITCH_BEACON:
                    # print(f"np.linalg.norm(MIN.pos - self.target.pos): {np.linalg.norm(MIN.pos - self.target.pos)}")
                    if np.linalg.norm(MIN.pos - self.target.pos) < 0.15:#(MIN.pos == self.target.pos).all(): #HERE
                        print(f"MIN.pos: {MIN.pos}")
                        print(f"self.target.pos: {self.target.pos}")
                        print("--------------------------------")

                        raise AtTargetException
                    if self.__deadzone_v is None:
                        """
                        Deadzone reached. Move directly towards target until target is passed
                        """
                        self.__deadzone_v = self.MAX_FOLLOWING_SPEED*normalize(MIN.get_vec_to_other(self.target))
                    if self.__curr_RSSI < self.__prev_RSSI and not self.__moved_past_target:
                        """
                        Have now moved beyond target. Move away from target until deadzone is exited
                        """
                        vec = self.MAX_FOLLOWING_SPEED*np.append(normalize(-MIN.get_vec_to_other(self.target)), [0]).reshape(3, 1)
                        self.__deadzone_v = (R_z(np.random.uniform(-self.__rand_lim, self.__rand_lim))@vec)[:2].reshape(2, )
                        self.__moved_past_target = True
                    return self.__deadzone_v
                elif not self.__deadzone_v is None:
                        raise AtTargetException

            if self.__curr_RSSI >= FollowingStrategy.MIN_RSSI_SWITCH_BEACON and not self.is_following_final_target():
                print(f"self.__curr_RSSI: {self.__curr_RSSI}")
                print(f"MIN.pos: {MIN.pos}")
                print(f"self.btf.pos: {self.btf.pos}")
                
                self.compute_next_beacon_to_follow()
            return func(self, MIN, beacons, ENV)
        return func_wrapper