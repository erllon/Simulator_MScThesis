from deployment.exploration_strategies.exploration_strategy import (
    ExplorationStrategy,
    AtLandingConditionException
)
from deployment.deployment_helpers import get_obstacle_forces as gof
from beacons.MIN.min import Min, MinState

import numpy as np
from enum import IntEnum

class LineExploreKind(IntEnum):
      ONE_DIM_GLOBAL = 1,
      TWO_DIM_GLOBAL = 2,
      ONE_DIM_LOCAL  = 3,
      TWO_DIM_LOCAL  = 4,

class LineExplore(ExplorationStrategy):

  RSSI_THRESHOLD = 0.5

  def __init__(self, K_o=1, force_threshold=0.01, kind=LineExploreKind.ONE_DIM_GLOBAL): #RSSI_threshold=0.6
    self.K_o = K_o
    self.K_o = K_o
    self.kind = kind
    self.force_threshold = force_threshold
    # self.RSSI_threshold = RSSI_threshold
    # self.ndims = ndims
    self.prev_xi_rand = None
    self.prev_neigh_indices = None

  def prepare_exploration(self, target):
      return super().prepare_exploration(target)

  def get_exploration_velocity(self, MIN, beacons, ENV):
    land_due_to_no_neighs = False

    F_n = np.zeros((2, ))
    F_o = gof(self.K_o, MIN, ENV)

    x_is = np.concatenate([b.pos.reshape(2, 1) for b in beacons], axis=1)
    xi_is = np.array([MIN.get_xi_to_other_from_model(b) for b in beacons])

    if int(self.kind) <= LineExploreKind.TWO_DIM_GLOBAL:
      """GLOBAL METHODS"""

      if self.kind == LineExploreKind.ONE_DIM_GLOBAL:
        x_is = x_is[0, :]

        """ Default values """
        k_is = np.ones(x_is.shape)
        a_is = np.ones(x_is.shape)
        a_is[-1] = 1 + (1/k_is[-1])*np.sum(k_is[:-1])

        """ Leads to equally spaced drones """
        k_is = np.zeros(x_is.shape)
        k_is[-1] = 2*1
        a_is[-1] = 1

        """ Test for 'move back gains' (a_i*k_i = 0 for all 0 < i < n) """
        
        """ test 1 """

        k_is = np.zeros(x_is.shape)
        a_is = np.ones(x_is.shape)
        a_is[-1] = 2
        k_is[-1] = 1

        """ test 2 """

        k_is = np.ones(x_is.shape)
        a_is = np.zeros(x_is.shape)

        a_is[-1] = 2
        k_is[-1] = np.sum(k_is) - 1

        assert k_is[-1]*a_is[-1] > np.sum(k_is) or np.isclose(k_is[-1]*a_is[-1], np.sum(k_is)) and a_is[-1] >= 0,\
           "Conditions on constants a_i and k_i do not hold. Cannot guarantee x_{n+1} > x_{n}"
        
        F_n = np.array([-np.sum(k_is*(MIN.pos[0] - a_is*(x_is + xi_is))), 0])
        F_o = 0*F_o[0]
      
      elif self.kind == LineExploreKind.TWO_DIM_GLOBAL:
        k_is = np.zeros(len(beacons))
        k_is[-1] = 1
        x_is = np.concatenate([b.pos.reshape(2, 1) for b in beacons], axis=1)
        xi_is = np.array([MIN.get_RSSI(b) for b in beacons])*np.ones((2, 1))
        F_n = -np.sum(k_is*(MIN.pos.reshape(2, 1) - (x_is + xi_is)), axis=1).reshape(2, )
   
    else:
      """ LOCAL METHODS """
      neigh_indices, = np.where(xi_is > self.RSSI_THRESHOLD)
      land_due_to_no_neighs = len(neigh_indices) == 0
      if land_due_to_no_neighs:
          print(f"{MIN.ID} STOPPED due to no neighs")
      else:

        x_is = x_is[:, neigh_indices]
        xi_is = xi_is[neigh_indices]

        if self.kind == LineExploreKind.ONE_DIM_LOCAL:
          x_is = x_is[0, :]

          m = np.argmax(x_is)

          k_is = np.ones(x_is.shape)
          a_is = 1.1*np.ones(x_is.shape)

          a_is[m] = (1/k_is[m])*np.sum(k_is) + 1


          """ Leads to equally spaced drones """
          #k_is = np.zeros(x_is.shape)
          #k_is[j] = 2*1
          #a_is[j] = 1

          """ Using qualitative info. about xi function vol. 1"""
          a_is = np.ones(x_is.shape)
          a_is[m] = 1.1
          k_is = np.ones(x_is.shape)

          delta_is = np.array([b.get_xi_max_decrease() for b in beacons[neigh_indices]])

          k_is[m] = (1/(a_is[m]-1))*np.sum(np.delete(k_is*(1+a_is*delta_is), m)) + 0.1
          
          """ Using qualitative info. about xi function vol. 2"""
          k_is = np.ones(x_is.shape)
          a_is = np.ones(x_is.shape)

          a_is[m] = (1/k_is[m])*np.sum(np.delete(k_is*(1+a_is*delta_is), m)) + 1

          assert (k_is[m]*a_is[m] > np.sum(k_is) or np.isclose(k_is[m]*a_is[m], np.sum(k_is))) and a_is[m] >= 0,\
            f"""
            Conditions on constants a_i and k_i do not hold. Cannot guarantee x_n_plus_one > max(x_i) for i in neighbors of nu_n_plus_one.
            {k_is[m]*(a_is[m] - 1)} >=? {np.sum(np.delete(k_is, m))} and {a_is[m]} >= 0.
            """
            
          F_n = np.array([-np.sum(k_is*(MIN.pos[0] - a_is*(x_is + xi_is))), 0])
          F_o = 0*F_o

        else:
          """ Behdads gain approach """

          k_is = np.array([b.k for b in beacons[neigh_indices]])
          a_is = np.array([b.a for b in beacons[neigh_indices]])
          v_is = np.concatenate([
            b.v.reshape(2, 1) for b in beacons[neigh_indices]
          ], axis=1)

          F_n = -np.sum(k_is*(MIN.pos.reshape(2, 1) - a_is*(x_is + xi_is*v_is)), axis=1).reshape(2, )
          F_o = 0*F_o

          MIN.a = np.min(a_is) + 1
          MIN.k = 1

    F = F_n + F_o
    at_landing_condition = land_due_to_no_neighs or np.linalg.norm(F) < self.force_threshold
    
    if at_landing_condition:
      rot_mat_2D = lambda theta: np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
      ])
      v_i_base = np.array([1, 0]).reshape(2, 1)
      MIN.v = rot_mat_2D(MIN.heading + (np.pi/2)*np.random.uniform(-1, 1))@v_i_base
  
      print(np.rad2deg(np.arctan2(MIN.v[1], MIN.v[0])), MIN.ID)
      raise AtLandingConditionException
    return F
#     neigh_indices, = np.where(xi_is > self.RSSI_threshold) #np.where(RSSIs_all <= MIN.range) 
#     xi_is_neigh = xi_is[neigh_indices]
    
#     MIN._xi_traj = np.column_stack((MIN._xi_traj, xi_is)) #np.max(xi_is) #axis=0, but should be 1D...

#     F = None
#     if self.ndims == 1:
#       """ 1D """
#       x_is = np.array([b.pos[0] for b in beacons])
#       k_is, a_is = None, None
      
# # %% Global knowledge
#       """ Default values """
#       # k_is = np.ones(x_is.shape)
#       # a_is = np.ones(x_is.shape)
#       # a_is[-1] = 1 + (1/k_is[-1])*np.sum(k_is[:-1])

#       """ Leads to equally spaced drones """
#       k_is = np.zeros(x_is.shape)
#       a_is = np.zeros(x_is.shape)
#       k_is[-1] = 2*1
#       a_is[-1] = 1

#       """ Test for 'move back gains' (a_i*k_i = 0 for all 0 < i < n) """
      
#       """ test 1 """

#       # k_is = np.zeros(x_is.shape)
#       # a_is = np.ones(x_is.shape)
#       # a_is[-1] = 2
#       # k_is[-1] = 1

#       """ test 2 """

#       # k_is = np.ones(x_is.shape)
#       # a_is = np.zeros(x_is.shape)

#       # a_is[-1] = 2
#       # k_is[-1] = np.sum(k_is) - 1

# # %% Local knowledge
#       # x_is = np.array([beacons[i].pos[0] for i in neigh_indices])
#       # k_is = np.ones(x_is.shape)
#       # a_is = np.zeros(x_is.shape)
#       # k_is[-1] = 2*1
#       # a_is[-1] = 1
# #REMOVE COMMENTS FROM HERE
#       # neigh_indices, = np.where(xi_is > self.RSSI_threshold)
#       # if len(neigh_indices) == 0:
#       #     print(xi_is, self.RSSI_threshold)
#       #     print(f"{MIN.ID} STOPPED due to no neighs")
#       #     raise AtLandingConditionException
      
#       # x_is = x_is[neigh_indices]
#       # xi_is = xi_is[neigh_indices]

#       # m = np.argmax(x_is)

#       # k_is = np.ones(x_is.shape)
#       # a_is = 1.1*np.ones(x_is.shape)

#       # a_is[m] = (1/k_is[m])*np.sum(k_is) + 1



#       # """ Leads to equally spaced drones """
#       # #k_is = np.zeros(x_is.shape)
#       # #k_is[j] = 2*1
#       # #a_is[j] = 1

#       # """ Using qualitative info. about xi function vol. 1"""
#       # a_is = np.ones(x_is.shape)
#       # a_is[m] = 1.1
#       # k_is = np.ones(x_is.shape)

#       # delta_is = np.array([b.get_xi_max_decrease() for b in beacons[neigh_indices]])

#       # k_is[m] = (1/(a_is[m]-1))*np.sum(np.delete(k_is*(1+a_is*delta_is), m)) + 0.1
      
#       # """ Using qualitative info. about xi function vol. 2"""
#       # k_is = np.ones(x_is.shape)
#       # a_is = np.ones(x_is.shape)

#       # a_is[m] = (1/k_is[m])*np.sum(np.delete(k_is*(1+a_is*delta_is), m)) + 1

#       # assert (k_is[m]*a_is[m] > np.sum(k_is) or np.isclose(k_is[m]*a_is[m], np.sum(k_is))) and a_is[m] >= 0,\
#       #   f"""
#       #   Conditions on constants a_i and k_i do not hold. Cannot guarantee x_n_plus_one > max(x_i) for i in neighbors of nu_n_plus_one.
#       #   {k_is[m]*(a_is[m] - 1)} >=? {np.sum(np.delete(k_is, m))} and {a_is[m]} >= 0.
#       #     """
# #TO HERE

#       F_n = -np.sum(k_is*(MIN.pos[0] - a_is*(x_is + xi_is)))
#       F_o = 0*gof(self.K_o, MIN, ENV)[0]
#       F = self.__clamp(np.array([F_n + F_o, 0]), 15) #10, 100
#     else:
#       print("No neighbors")
#       raise AtLandingConditionException
#     if np.linalg.norm(F) < self.force_threshold:
#       print(f"Force lower than threshold: {np.linalg.norm(F)} < {self.force_threshold}")
#       raise AtLandingConditionException
#     return F
    
# %%Gammel kode
    # # SE OVER HVORDAN TING NEDENFOR BLIR REGNET UT, SAMT SE HVA SOM ER KOMMENTERT BORT
    # # RSSIs_all = np.array([np.linalg.norm(MIN.get_vec_to_other(b)) for b in beacons])
    # # neigh_indices, = np.where(RSSIs_all < MIN.range)
    # # Change state of all neighbors to MinState.NEIGHBOR???
    # xi_RSSI = np.array([MIN.get_RSSI(b) for b in beacons]) #np.array([np.linalg.norm(MIN.get_vec_to_other(b)) for b in beacons])#
    # neigh_indices, = np.where(xi_RSSI > self.RSSI_threshold) #np.where(RSSIs_all <= MIN.range) 
    # xi_neigh = xi_RSSI[neigh_indices]
    
    # F = None
    # if len(neigh_indices)!=0:
    #   if self.ndims == 1:
    #     """ 1D """
    #     x_is = np.array([beacons[i].pos[0] for i in neigh_indices])#np.array([beacons[i].pos[0] for i in neigh_indices]) #np.array([b.pos[0] for b in beacons])
    #     k_is = np.ones(x_is.shape)#np.ones(x_is.shape)# * (np.ones(len(x_is)) + np.array(range(len(x_is)))*0.1)#np.zeros(x_is.shape)#np.ones(x_is.shape)
    #     k_is[-1] = 1

    #     epsilon = 3*0.10 #0 because we only want the drones to move to the right?

    #     if np.array(neigh_indices != self.prev_neigh_indices).all():
    #       print(f"prev_neigh: {self.prev_neigh_indices} \t current_neigh: {neigh_indices}")
    #       xi_random = np.random.uniform(epsilon - self.RSSI_threshold, epsilon)
    #       if self.prev_xi_rand != None:
    #         if (xi_random - self.prev_xi_rand) > 0.4:
    #           xi_random = epsilon - 0.05
    #         print(f"prev_random: {self.prev_xi_rand} \t current RAND: {xi_random}")
    #         print("-------------------------------------------------")
    #     else:
    #       xi_random = self.prev_xi_rand

    #     F_n = -1*np.sum(k_is*(x_is- MIN.pos[0]  - (xi_neigh - epsilon + xi_random)))
    #     F_o = 0*gof(self.K_o, MIN, ENV)[0]
    #     F = np.array([F_n + F_o, 0])
    #     self.prev_xi_rand = xi_random
    #     self.prev_neigh_indices = neigh_indices
    #   elif self.ndims == 2:
    #     """ 2D """
    #     x_is = np.array([beacons[i].pos for i in neigh_indices])
    #     k_is = 5*np.ones(len(x_is)) #np.ones(len(x_is)) + np.array(range(len(x_is)))#np.array(range(len(x_is)))%2np.ones(len(x_is)) + np.array(range(len(x_is)))#np.ones(len(x_is)) #np.zeros(len(x_is))
    #     #k_is[-1] = 1
    #     epsilon_x = 4*0.10
    #     epsilon_y = 1*0.10
    #     epsilon = np.hstack((epsilon_x, epsilon_y)).reshape(2, )

    #     # Update the xi_rands when the neighbor set changes
    #     # print(f"(self.prev_neigh_indices == None): {(self.prev_neigh_indices == None)}")
    #     # print(f"(neigh_indices == self.prev_neigh_indices).any(): {(neigh_indices == self.prev_neigh_indices).any()}")
    #     # print(f"prev_neigh: {self.prev_neigh_indices} \t current_neigh: {neigh_indices}")
    #     # print(f"prev_rand: {self.prev_RSSIs_rand} \t current_rand: {RSSIs_random}")

    #     if np.array(neigh_indices != self.prev_neigh_indices).all(): #np.array(self.prev_neigh_indices == None).any() or 
    #       #print(f"prev_neigh: {self.prev_neigh_indices} \t current_neigh: {neigh_indices}")
    #       xi_random_x = np.random.uniform(epsilon_x-self.RSSI_threshold, epsilon_x) #self.RSSI_threshold
    #       xi_random_y = np.random.uniform(epsilon_y-self.RSSI_threshold, epsilon_y)
    #       if np.array(self.prev_xi_rand != None).any():
    #         if (xi_random_x - self.prev_xi_rand[0]) > 0.4:
    #           xi_random_x = self.prev_xi_rand[0] - 0.1#epsilon_x - 0.1            
    #         if (xi_random_y - self.prev_xi_rand[1]) > 0.4:
    #           xi_random_y = self.prev_xi_rand[1] - 0.1 #epsilon_y - 0.1
    #       xi_random = np.hstack((xi_random_x, xi_random_y))         
    #       #print(f"prev_random: {self.prev_xi_rand} \t current RAND: {xi_random}")
    #     else:
    #       xi_random = self.prev_xi_rand
        
    #     F_n = np.zeros((2, ))
    #     for i in range(len(x_is)): #neigh_indices:
    #       #F_n += (k_is[i]*(x_is[i].reshape(2, ) - MIN.pos.reshape(2, ) + RSSIs_all[i] - epsilon.reshape(2, ) + RSSIs_random)).reshape(2, ) #funker
    #       #F_n -= (k_is[i]*(x_is[i] + xi_RSSI[i] - (MIN.pos.reshape(2, ) - epsilon + xi_random))).reshape(2, ) #works
    #       #print((-xi_RSSI[i] - epsilon + xi_random))
    #       #F_n -= (k_is[i]*(x_is[i]-MIN.pos.reshape(2, )  - (-xi_RSSI[i] - epsilon + xi_random))).reshape(2, )
    #       xi_tot = xi_neigh[i] - epsilon + xi_random
    #       #F_n -= (k_is[i]*(MIN.pos.reshape(2, ) - x_is[i] -xi_tot)).reshape(2, ) #Original
    #       F_n -= (k_is[i]*(x_is[i] - MIN.pos.reshape(2, ) - xi_tot)).reshape(2, ) #Chaning order of positions (pos_drone_i - pos_drone_{n+1})
    #       #F_n += (k_is[i]*(MIN.pos.reshape(2, ) - (x_is[i] + xi_RSSI[i] - epsilon + xi_random))).reshape(2, ) #+= WORKS "PERFECTLY"?!?!

    #     F_o = gof(self.K_o, MIN, ENV).reshape(2, )
    #     # print(f"F_n: {F_n}")
    #     # print(f"F_o: {F_o}")
    #     # print("-------------")
    #     F = F_n + F_o
    #     self.prev_xi_rand = xi_random
    #     self.prev_neigh_indices = neigh_indices
    #     # print(f"np.linalg.norm(F): {np.linalg.norm(F)}")
    # else:
    #   print("No neighbors")
    #   raise AtLandingConditionException
    # if np.linalg.norm(F) < self.force_threshold:
    #   print(f"Force lower than threshold: {np.linalg.norm(F)} < {self.force_threshold}")
    #   raise AtLandingConditionException
    # return F

# %%Clamp
  @staticmethod
  def __clamp(F, limit):
    norm_F = np.linalg.norm(F)
    if norm_F > limit:
      return limit*F/norm_F
    return F