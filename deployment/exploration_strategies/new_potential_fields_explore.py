from deployment.exploration_strategies.exploration_strategy import (
    ExplorationStrategy,
    AtLandingConditionException
)
from deployment.deployment_helpers import (
    get_neighbor_forces as gnf,
    get_obstacle_forces as gof,
    get_generic_force_vector
)
from helpers import normalize, rot_z_mat as R_z, get_vector_angle as gva

import numpy as np
from enum import Enum

class NewPotentialFieldsExplore(ExplorationStrategy):
    class Target(Enum):
        POINT = 0,
        LINE  = 1
    
    def __init__(self, K_n=1, K_o=1, min_force_threshold=0.5, target_point_or_line=Target.LINE):
        self.__K_n = K_n
        self.__K_o = K_o
        self.__min_force_threshold = min_force_threshold
        self.__point_or_line = target_point_or_line   

    def get_exploration_velocity(self, MIN, beacons, ENV):
        MIN.compute_neighbors(beacons)
        RSSIs = [MIN.get_RSSI(n) for n in MIN.neighbors]
        F_att = NewPotentialFieldsExplore.get_attractive_force(self.__K_n, MIN)
        F_rep = NewPotentialFieldsExplore.get_repulsive_force(self.__K_n, self.__K_o, MIN, ENV)
        # if np.linalg.norm(F_rep) != 0 and MIN.ID ==8:
            # print(f"MIN.pos: {MIN.pos}")
        # if MIN.ID == 8 or MIN.ID ==1:
        #     print(f"np.linalg.norm(F_att): {np.linalg.norm(F_att)}")
        #     print(f"np.linalg.norm(F_rep): {np.linalg.norm(F_rep)}")

        F_sum = F_att + 5*F_rep
        # print(f"np.linalg.norm(F_sum): {np.linalg.norm(F_sum)}")
        a = np.any([MIN.get_RSSI(n) for n in MIN.neighbors] >= self.MIN_RSSI_STRENGTH_BEFORE_LAND)
        b = [MIN.get_RSSI(n) for n in MIN.neighbors] >= self.MIN_RSSI_STRENGTH_BEFORE_LAND
        if np.linalg.norm(F_sum) > self.__min_force_threshold and np.any([MIN.get_RSSI(n) for n in MIN.neighbors] >= self.MIN_RSSI_STRENGTH_BEFORE_LAND):#MIN.get_RSSI(MIN.target_pos) >= self.MIN_RSSI_STRENGTH_BEFORE_LAND:
            # MIN.generate_virtual_target(gva(MIN.target_pos.reshape(2, ))) 
            if self.__point_or_line == NewPotentialFieldsExplore.Target.LINE:
                if np.linalg.norm(F_sum) < self.MAX_EXPLORATION_SPEED:
                    MIN.generate_virtual_target(F_sum, 0.01)#MIN.target_pos += F_sum
                else:
                    MIN.generate_virtual_target(self.MAX_EXPLORATION_SPEED*normalize(F_sum), 0.01)
                     #MIN.target_pos += self.MAX_EXPLORATION_SPEED*normalize(F_sum)
                
                #MIN.generate_virtual_target(gva(MIN.target_pos))
            return F_sum if np.linalg.norm(F_sum) < self.MAX_EXPLORATION_SPEED else self.MAX_EXPLORATION_SPEED*normalize(F_sum)
        else:
            print("Landing due to too small force and satisfactory low RSSI")
            raise AtLandingConditionException

    
    @staticmethod
    def get_repulsive_force(K_n, K_o, MIN, ENV):   
        vecs_to_neighs = [
            MIN.get_vec_to_other(n).reshape(2, 1) for n in MIN.neighbors if not (MIN.get_vec_to_other(n) == 0).all()
        ]

        for s in MIN.sensors:
            s.sense(ENV)
        vecs_to_obs = [
            (R_z(MIN.heading)@R_z(s.host_relative_angle)@s.measurement.get_val())[:2]
            for s in MIN.sensors if s.measurement.is_valid()
        ]

        return get_generic_force_vector(vecs_to_obs, K_o, d_o=MIN.range) #+ get_generic_force_vector(vecs_to_neighs, K_n, d_o=MIN.range)

    # return get_generic_force_vector(vecs_to_obs, K_n)
    
    @staticmethod
    def get_attractive_force(K_target, MIN):
        return -K_target * (MIN.pos - MIN.target_pos)
