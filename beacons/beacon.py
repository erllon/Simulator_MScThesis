import numpy as np
from helpers import (
  get_smallest_signed_angle as ssa,
  get_vector_angle as gva,
  polar_to_vec as p2v,
  normalize,
  euler_int,
  plot_vec,
  rot_z_mat as R_z
)
from abc import ABC, abstractmethod

class Beacon():

  ID_counter = 0

  @staticmethod
  def get_ID():
    ret = Beacon.ID_counter
    Beacon.ID_counter += 1
    return ret

  def __init__(self, b_id, range, xi_max, d_perf, d_none, k=0, a=0, v=np.zeros((2, )), pos=None):
    self.range = range
    self.target_r = range #TODO: Scale this down, so that the target is not generated at border?
    self.pos = pos
    self.prev_pos = None
    self.delta_pos = None
    self.ID = b_id
    self.neighbors = []
    """ STUFF FOR XI MODEL """
    self.xi_max = xi_max
    self.d_perf = d_perf
    self.d_none = d_none
    self._omega = np.pi/(self.d_none - self.d_perf)
    self._phi = -(np.pi*self.d_perf) / (d_none - d_perf)
    self._xi_max_decrease = (self.xi_max/2)*self._omega
    """ Gains """
    self.k = k
    self.a = a
    """ xi direction vector for 2D exploration """ 
    self.v = v
  
  def insert_into_environment(self, env):
    self.pos = env.entrance_point
  
  def is_within_range(self, other):
    dist = np.linalg.norm(self.pos - other.pos)
    return dist < self.range and dist < other.range

  def compute_neighbors(self, others):
    self.neighbors = list(filter(lambda other: self.get_xi_to_other_from_model(other) > 0.2*self.xi_max and self != other, others))
  
  @abstractmethod
  def generate_target_pos(self, beacons, ENV, next_min):
    pass

  def get_xi_to_other_from_model(self, other):
    d = np.linalg.norm(self.pos - other.pos)
    if d <= self.d_perf:
      return self.xi_max
    elif self.d_perf < d and d < self.d_none:
      return (self.xi_max/2) * (1 + np.cos(self._omega*d + self._phi))
    elif self.d_none <= d:
      return 0

  def get_xi_max_decrease(self):
    return self._xi_max_decrease
    
  def get_vec_to_other(self, other):
    return other.pos - self.pos
  
  def __eq__(self, other):
        return self.ID == other.ID
  
  def __str__(self):
        return f"[\n\ttype: {self.__class__.__name__},\n\tID:{self.ID},\n\tneighbors: {len(self.neighbors)},\n\tpos: {self.pos}\n]"


  """""
  PLOTTING STUFF
  """""
  @abstractmethod
  def plot(self, axis, clr="green"):
    pass
  
    
  def calc_uniformity(self):
    """
    A smaller "uniformity"-value means that nodes are
    more uniformly distributed
    """
    if len(self.neighbors) != 0:
      D_ij = [np.linalg.norm(self.get_vec_to_other(neigh)) for neigh in self.neighbors]
      M_ij = np.sum(D_ij)/len(D_ij)
      K_i = len(self.neighbors)

      test = [(d - M_ij)**2 for d in D_ij]
      sum_test = np.sum(test)
      within_parenthesis = 1/K_i * sum_test

      U_i = np.sqrt(within_parenthesis)
      return U_i
    else:
      return 0
      


  