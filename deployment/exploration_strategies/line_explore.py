from deployment.exploration_strategies.exploration_strategy import (
    ExplorationStrategy,
    AtLandingConditionException
)
from deployment.deployment_helpers import get_obstacle_forces as gof
from beacons.MIN.min import Min, MinState

import numpy as np
from enum import IntEnum
from helpers import get_vector_angle as gva

class LineExploreKind(IntEnum):
      ONE_DIM_GLOBAL = 1,
      TWO_DIM_GLOBAL = 2,
      ONE_DIM_LOCAL  = 3,
      TWO_DIM_LOCAL  = 4,

class LineExplore(ExplorationStrategy):

  MIN_RSSI_STRENGTH_BEFORE_LAND = 0.22

  def __init__(self, K_o=1, force_threshold=0.1, kind=LineExploreKind.ONE_DIM_GLOBAL): #RSSI_threshold=0.6
    self.K_o = K_o
    self.kind = kind
    self.force_threshold = force_threshold
    # self.RSSI_threshold = RSSI_threshold
    # self.ndims = ndims
    self.prev_xi_rand = None
    self.prev_neigh_indices = None
  
  def __str__(self):
    return "LineExplore"

  def prepare_exploration(self, target):
      return super().prepare_exploration(target)

  def get_exploration_velocity(self, MIN, beacons, ENV):
    land_due_to_no_neighs = False

    F_n = np.zeros((2, ))

    x_is = np.concatenate([b.pos.reshape(2, 1) for b in beacons], axis=1)
    xi_is = np.array([MIN.get_xi_to_other_from_model(b) for b in beacons])
    MIN._xi_traj = np.column_stack((MIN._xi_traj, xi_is))

    """ LOCAL METHODS """
    neigh_indices, = np.where(xi_is > self.MIN_RSSI_STRENGTH_BEFORE_LAND)#np.where(xi_is > self.RSSI_THRESHOLD)
    land_due_to_no_neighs = len(neigh_indices) == 0
    if land_due_to_no_neighs:
        print(f"{MIN.ID} STOPPED due to no neighs")
    else:

      x_is = x_is[:, neigh_indices]
      xi_is = xi_is[neigh_indices]

      if self.kind == LineExploreKind.ONE_DIM_LOCAL:
        x_is = x_is[0, :]

        m = np.argmax(x_is)

        delta_is = np.array([b.get_xi_max_decrease() for b in beacons[neigh_indices]])

        
        """ Test """
        # k_is = np.ones(x_is.shape)
        # a_is = 1.1*np.ones(x_is.shape)

        # a_is[m] = (1/k_is[m])*np.sum(k_is) + 1


        """ Leads to equally spaced drones """
        k_is = np.zeros(x_is.shape)
        a_is = np.zeros(x_is.shape)
        k_is[m] = 2*1
        a_is[m] = 1

        """ Using qualitative info. about xi function vol. 1"""
        # a_is = np.ones(x_is.shape)
        # a_is[m] = 1.5
        # k_is = np.zeros(x_is.shape)
        # k_is[m] = 2

        assert (k_is >= 0).all() and \
               (a_is >= 0).all() and \
               len((k_is[np.nonzero(a_is)])) > 0 and \
               (a_is[m] >= 1).all() and \
               (k_is[m]*(a_is[m]-1) >= np.sum(k_is[:m]*(1 + a_is[:m]*delta_is[:m])) or np.isclose(k_is[m]*(a_is[m]-1), np.sum(k_is[:m]*(1 + a_is[:m]*delta_is[:m])))),\
          f"""
          Conditions on constants a_i and k_i do not hold. Cannot guarantee x_n_plus_one > max(x_i) for i in neighbors of nu_n_plus_one.
          len(k_is[np.nonzero(a_is)]) = {len(k_is[np.nonzero(a_is)])} >? 0
          {k_is} >=? 0 and
          {a_is} >=? 0 and
          {a_is[m]} >=? 1 and
          {k_is[m]*(a_is[m]-1)} >=? {np.sum(k_is[:m] + a_is[:m]*delta_is[:m])} and {a_is[m]} >= 1.
          """
          
        F_n = np.array([-np.sum(k_is*(MIN.pos[0] - a_is*(x_is + xi_is))), 0])
    F_n = self.__clamp(F_n,17)
    F = F_n
    at_landing_condition = land_due_to_no_neighs or np.linalg.norm(F) <= self.force_threshold
    if at_landing_condition:
      # print(f"xi_is[m] = {xi_is[m]}")
      # print(f"F[0] = {F[0]}")
      # print(f"xi_is[m]-F[0] = {xi_is[m]-F[0]}")
      raise AtLandingConditionException
    return F
    

# %%Clamp
  @staticmethod
  def __clamp(F, limit):
    norm_F = np.linalg.norm(F)
    if norm_F > limit:
      return limit*F/norm_F
    return F