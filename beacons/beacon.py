import numpy as np

class Beacon():

  ID_counter = 0

  @staticmethod
  def get_ID():
    ret = Beacon.ID_counter
    Beacon.ID_counter += 1
    return ret

  def __init__(self, range, xi_max, d_perf, d_none, pos=None):
    self.range = range
    self.pos = pos
    self.ID = self.get_ID()
    self.neighbors = []

    self._xi_max = xi_max
    self._d_perf = d_perf
    self._d_none = d_none
    self._omega = np.pi/(self._d_none - self._d_perf)
    self._phi = -(np.pi*self._d_perf) / (d_none - d_perf)
    self._xi_max_decrease = (self._xi_max/2)*self._omega
  
  def insert_into_environment(self, env):
    self.pos = env.entrance_point
  
  def is_within_range(self, other):
    dist = np.linalg.norm(self.pos - other.pos)
    return dist < self.range and dist < other.range

  def compute_neighbors(self, others):
    self.neighbors = list(filter(lambda other: self.is_within_range(other) and self != other, others))
  
  def get_RSSI(self, other):
    return np.exp(-np.linalg.norm(self.pos - other.pos))

  def get_xi_to_other_from_model(self, other):
    d = np.linalg.norm(self.pos - other.pos)
    if d < self._d_perf:
      return self._xi_max
    elif self._d_perf <= d and d <= self._d_none:
      return (self._xi_max/2) * (1 + np.cos(self._omega*d + self._phi))
    else: #if d_none < d
      return 0

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
    self.radius = axis.plot(
      self.pos[0] + self.range*np.cos(theta), self.pos[1] + self.range*np.sin(theta),
      linestyle="dashed",
      color="black",
      alpha=0.3
    )[0]
    # if type(axis) == np.ndarray:
    #   self.point = axis[0].plot(*self.pos, color=clr, marker="o", markersize=8)[0]
    #   self.annotation = axis[0].annotate(self.ID, xy=(self.pos[0], self.pos[1]), fontsize=14)
    #   theta = np.linspace(0, 2*np.pi)
    #   self.radius = axis[0].plot(
    #     self.pos[0] + self.range*np.cos(theta), self.pos[1] + self.range*np.sin(theta),
    #     linestyle="dashed",
    #     color="black",
    #     alpha=0.3
    #   )[0]
    # else: #type(axis)==matplotlib.axes._subplots.AxesSubplot
    #   self.point = axis.plot(*self.pos, color=clr, marker="o", markersize=8)[0]
    #   self.annotation = axis.annotate(self.ID, xy=(self.pos[0], self.pos[1]), fontsize=14)
    #   theta = np.linspace(0, 2*np.pi)
    #   self.radius = axis.plot(
    #     self.pos[0] + self.range*np.cos(theta), self.pos[1] + self.range*np.sin(theta),
    #     linestyle="dashed",
    #     color="black",
    #     alpha=0.3
    #   )[0]

    return self.point, self.annotation, self.radius


