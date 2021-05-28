from deployment.following_strategies.following_strategy import FollowingStrategy, AtTargetException
from deployment.deployment_helpers import get_obstacle_forces as gof
from helpers import normalize
import numpy as np


class AttractiveFollow(FollowingStrategy):
    
    def __init__(self, K_o, same_num_neighs_differentiator=lambda MINs, k: min(MINs, key=k)):
        super().__init__(same_num_neighs_differentiator)
        self.__K_o = K_o  
    
    def __compute_target(self, beacons, SCS):
        self.target = beacons[-1]

    def prepare_following(self, MIN, beacons, SCS):
        # super().prepare_following(MIN, beacons, SCS)
        self.target = beacons[-1]
        print(f"{MIN.ID} targeting {self.target.ID}")
        self.beacons_to_follow = SCS.path_tree.get_beacon_path_to_target(self.target.ID)
        SCS.path_tree.add_node(MIN, self.target.ID)
        self.compute_next_beacon_to_follow()

    @FollowingStrategy.follow_velocity_wrapper
    def get_following_velocity(self, MIN, beacons, ENV):
        
        xi_is = np.array([MIN.get_xi_to_other_from_model(b) for b in beacons])
        
        MIN._xi_traj = np.column_stack((MIN._xi_traj, xi_is))
        MIN.compute_neighbors(beacons)

        F_o = gof(self.__K_o, MIN, ENV)
        F_btf = MIN.get_vec_to_other(self.btf)
        F_att = -MIN.K_target * (MIN.pos - self.btf.pos)
        F_btf_aug = self.MAX_FOLLOWING_SPEED*normalize(F_btf) if np.linalg.norm(F_btf) > self.MAX_FOLLOWING_SPEED else F_btf
        F = F_o + F_att
        """
        TODO: return a non-zero net force when the MIN the deployed MIN is following is the target
        (to ensure than we travel further into the environment)
        """
        return self.MAX_FOLLOWING_SPEED*normalize(F)