from deployment.following_strategies.following_strategy import FollowingStrategy, AtTargetException
from deployment.deployment_helpers import get_obstacle_forces as gof
from helpers import normalize
import numpy as np


class NewAttractiveFollow(FollowingStrategy):
    
    def __init__(self, K_o, same_num_neighs_differentiator=lambda MINs, k: min(MINs, key=k)):
        super().__init__(same_num_neighs_differentiator)
        self.__K_o = K_o
    
    def __compute_target(self, beacons, SCS):
        self.target = beacons[-1]
        # if len(beacons) > 1:
        #     tmp = beacons[1:]
        #     num_neighs = np.array([len(b.neighbors) for b in tmp])
        #     min_neigh_indices, = np.where(num_neighs == num_neighs.min())
        #     if len(min_neigh_indices) > 1:
        #         self.target =  self.__snnd(tmp[min_neigh_indices], lambda beacon: np.linalg.norm(SCS.get_vec_to_other(beacon)))
        #     else:
        #         self.target = tmp[min_neigh_indices[0]]

    def prepare_following(self, MIN, beacons, SCS):
        # super().prepare_following(MIN, beacons, SCS)
        self.target = beacons[-1]
        print(f"{MIN.ID} targeting {self.target.ID}")
        self.beacons_to_follow = SCS.path_tree.get_beacon_path_to_target(self.target.ID)
        SCS.path_tree.add_node(MIN, self.target.ID)
        self.compute_next_beacon_to_follow()
        # TODO: Add a shortest path algorithm here so that drone 5 does not necessarily have to follow 1-2-3-4,
        #       but maybe 1-3-4 and then start exploring

    @FollowingStrategy.follow_velocity_wrapper
    def get_following_velocity(self, MIN, beacons, ENV):
        
        xi_is = np.array([MIN.get_xi_to_other_from_model(b) for b in beacons])
        
        MIN._xi_traj = np.column_stack((MIN._xi_traj, xi_is)) #np.max(xi_is) #axis=0, but should be 1D...
        MIN.compute_neighbors(beacons)

        # MIN.neighbors = list(filter(lambda other: MIN.get_xi_to_other_from_model(other) > self.__RSSI_threshold and MIN != other, beacons))

        F_o = gof(self.__K_o, MIN, ENV)
        F_btf = MIN.get_vec_to_other(self.btf)
        F_att = -MIN.K_target * (MIN.pos - self.btf.pos)#-MIN.K_target * (MIN.pos - MIN.target_pos)
        F_btf_aug = self.MAX_FOLLOWING_SPEED*normalize(F_btf) if np.linalg.norm(F_btf) > self.MAX_FOLLOWING_SPEED else F_btf
        # F = F_o + F_btf_aug
        F = F_o + F_att
        """
        TODO: return a non-zero net force when the MIN the deployed MIN is following is the target
        (to ensure than we travel further into the environment)
        """
        return self.MAX_FOLLOWING_SPEED*normalize(F)