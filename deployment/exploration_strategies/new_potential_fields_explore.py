from deployment.exploration_strategies.exploration_strategy import (
    ExplorationStrategy,
    AtLandingConditionException
)
from deployment.deployment_helpers import (
    get_neighbor_forces as gnf,
    get_obstacle_forces as gof,
    get_generic_force_vector
)
from helpers import normalize, rot_z_mat as R_z, get_vector_angle as gva, polar_to_vec as p2v

import numpy as np
from enum import Enum

class NewPotentialFieldsExplore(ExplorationStrategy):
    class Target(Enum):
        POINT = 0,
        LINE  = 1
    
    def __init__(self, K_n=1, K_o=1, min_force_threshold=0.5, rssi_threshold=0.2, target_point_or_line=Target.LINE):
        self.__K_n = K_n
        self.__K_o = K_o
        self.__min_force_threshold = min_force_threshold
        self.__point_or_line = target_point_or_line   
        self.__RSSI_threshold = rssi_threshold

    def get_exploration_velocity(self, MIN, beacons, ENV):
        xi_is = np.array([])
        xi_is = np.array([MIN.get_xi_to_other_from_model(b) for b in beacons])

        neigh_indices, =  np.where(xi_is > self.__RSSI_threshold*MIN.xi_max)
        xi_is_neigh = xi_is[neigh_indices]
        
        MIN._xi_traj = np.column_stack((MIN._xi_traj, xi_is))
        MIN.compute_neighbors(beacons)

        if np.any(MIN.prev_pos != None):
            MIN.delta_pos = MIN.pos - MIN.prev_pos

        F_att = NewPotentialFieldsExplore.get_attractive_force(self.__K_n, MIN)
        F_rep = NewPotentialFieldsExplore.get_repulsive_force(self.__K_n, self.__K_o, MIN, ENV)
        

        F_sum = F_att + F_rep
        if np.linalg.norm(F_sum) > self.__min_force_threshold and np.any(np.array([MIN.get_xi_to_other_from_model(n) for n in MIN.neighbors]) >= self.MIN_RSSI_STRENGTH_BEFORE_LAND*MIN.xi_max):

            if self.__point_or_line == NewPotentialFieldsExplore.Target.LINE and np.any(MIN.delta_pos != None):
                MIN.generate_virtual_target()
                
            MIN.prev_pos = MIN.pos
            return F_sum if np.linalg.norm(F_sum) < self.MAX_EXPLORATION_SPEED else self.MAX_EXPLORATION_SPEED*normalize(F_sum)
        else:
            if np.linalg.norm(F_sum) <= self.__min_force_threshold:
                print("Too small force")
            if np.any(np.array([MIN.get_xi_to_other_from_model(n) for n in MIN.neighbors]) < self.MIN_RSSI_STRENGTH_BEFORE_LAND*MIN.xi_max):
                print("Too low RSSI")
            raise AtLandingConditionException

    
    @staticmethod
    def get_repulsive_force(K_n, K_o, MIN, ENV):   
        vecs_to_neighs = [
            MIN.get_vec_to_other(n).reshape(2, 1) for n in MIN.neighbors if not (MIN.get_vec_to_other(n) == 0).all()
        ]

        for s in MIN.sensors:
            s.sense(ENV)
        vecs_to_obs = [
            (R_z(MIN.heading)@R_z(s.host_relative_angle))[:2,:2]@p2v(s.measurement.get_val(), s.measurement.get_angle()).reshape(2,)
            for s in MIN.sensors if s.measurement.is_valid()
        ]
        # vecs_to_obs = [
            # p2v(s.measurement.get_val(), s.measurement.get_angle()).reshape(2,)
            # for s in MIN.sensors if s.measurement.is_valid()
        # ]
        
        return get_generic_force_vector(vecs_to_obs, K_o, d_o=MIN.range)
    
    @staticmethod
    def get_attractive_force(K_target, MIN):
        return -K_target * (MIN.pos - MIN.target_pos)
