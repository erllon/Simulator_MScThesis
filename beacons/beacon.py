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

  def __init__(self, range, xi_max, d_perf, d_none, k=0, a=0, v=np.zeros((2, )), pos=None):
    self.range = range
    self.target_r = range #TODO: Scale this down, so that the target is not generated at border?
    self.pos = pos
    self.ID = self.get_ID()
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
    # self.neighbors = list(filter(lambda other: self.is_within_range(other) and self != other, others))
    # self.neighbors = []
    # for other in others:
      # if self.get_xi_to_other_from_model(other) > 0.2 and self != other:
        # self.neighbors.append(other)
    self.neighbors = list(filter(lambda other: self.get_xi_to_other_from_model(other) > 0.2 and self != other, others))#self.RSSI_threshold and self != other, others))
  
  @abstractmethod
  def generate_target_pos(self, beacons, ENV, next_min):
    pass
    #Get vectors to neighbors
    #Get vectors to obstacles
    #Sum vectors to form "red" vector
    #Generate target on cirle within interval (angle)
    #Assign generated target to next_min.target_pos    

  # def get_RSSI(self, other):
  #   return np.exp(-np.linalg.norm(self.pos - other.pos))

  def get_xi_to_other_from_model(self, other):
    d = np.linalg.norm(self.pos - other.pos)
    if d <= self.d_perf:
      return self.xi_max
    elif self.d_perf < d and d < self.d_none:
      return (self.xi_max/2) * (1 + np.cos(self._omega*d + self._phi))
    elif self.d_none <= d:#else: #if d_none < d
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
  def plot(self, axis, clr="green"):
    self.point = axis.plot(*self.pos, color=clr, marker="o", markersize=8)[0]
    self.annotation = axis.annotate(self.ID, xy=(self.pos[0], self.pos[1]), fontsize=14)
    theta = np.linspace(0, 2*np.pi)
    # self.radius = axis.plot(
    #   self.pos[0] + self.range*np.cos(theta), self.pos[1] + self.range*np.sin(theta),
    #   linestyle="dashed",
    #   color="black",
    #   alpha=0.3
    # )[0]
    # self.radius2 = axis.plot(
    #   self.pos[0] + self.d_perf*np.cos(theta), self.pos[1] + self.d_perf*np.sin(theta),
    #   linestyle="dashed",
    #   color="black",
    #   alpha=0.3
    # )[0]

    return self.point, self.annotation#, self.radius#, self.radius2


