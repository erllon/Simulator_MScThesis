from beacons.beacon import Beacon
from beacons.MIN.range_sensor import RangeSensor
from helpers import (
  get_smallest_signed_angle as ssa,
  get_vector_angle as gva,
  polar_to_vec as p2v,
  normalize,
  euler_int,
  plot_vec
)

import numpy as np
from enum import Enum

class MinState(Enum):
  SPAWNED   = 0,
  FOLLOWING = 1,
  EXPLORING = 2,
  LANDED    = 3
  NEIGHBOR  = 4

class Min(Beacon):
      
  clr = {
    MinState.SPAWNED:   "yellow",
    MinState.FOLLOWING: "red",
    MinState.EXPLORING: "green",
    MinState.LANDED:    "black",
    MinState.NEIGHBOR:  "blue",
  }

  def __init__(self, max_range, deployment_strategy, xi_max=5, d_perf=1, d_none=3):
    super().__init__(max_range,xi_max, d_perf, d_none, pos=None)
    self.deployment_strategy = deployment_strategy
    self.sensors = []
    for ang in np.arange(0, 360, 90):
      r = RangeSensor(max_range)
      r.mount(self, ang)

  def insert_into_environment(self, env):
    super().insert_into_environment(env)
    self.state = MinState.SPAWNED
    self.heading =  0#np.pi
    self._pos_traj = self.pos.reshape(2, 1)
    self._heading_traj = np.array([self.heading])
    self.state_traj = np.array([self.state], dtype=object)

  def do_step(self, beacons, SCS, ENV, dt):
    v = self.deployment_strategy.get_velocity_vector(self, beacons, SCS, ENV)
    self.pos = euler_int(self.pos.reshape(2, ), v, dt).reshape(2, )
    #print(self.pos)
    psi_ref = gva(v)
    tau = 0.1
    self.heading = euler_int(self.heading, (1/tau)*(ssa(psi_ref - self.heading)), dt)
    self._pos_traj = np.hstack((self._pos_traj, self.pos.reshape(2, 1)))
    # print(f"self._heading_traj: {self._heading_traj}")
    # print(f"[self.heading]: {[self.heading]}")
    self._heading_traj = np.concatenate((self._heading_traj, [self.heading]))#[self.heading])) #Finn ut hvorfor denne ikke funker
    self.state_traj = np.concatenate((self.state_traj, [self.state]))
  
  def get_pos_traj_length(self):
        return self._pos_traj.shape[1]

  """""
  PLOTTING STUFF
  """""
  def plot(self, axis):
    self.heading_arrow = plot_vec(axis, p2v(1, self.heading), self.pos) #HERE
    return super().plot(axis, clr=self.clr[self.state]) + (self.heading_arrow, )

  def plot_traj_line(self, axis):
    self.traj_line, = axis.plot(*self._pos_traj, alpha=0.4)
    return self.traj_line

  def plot_pos_from_pos_traj_index(self, index):
    new_pos = self._pos_traj[:, index]
    self.point.set_data(new_pos)
    self.point.set_color(self.clr[self.state_traj[index]])
    self.annotation.set_x(new_pos[0])
    self.annotation.set_y(new_pos[1])
    theta = np.linspace(0, 2*np.pi)
    self.radius.set_data(new_pos.reshape(2, 1) + p2v(self.range, theta))
    self.traj_line.set_data(self._pos_traj[:, :index])

    self.heading_arrow.set_data(*np.hstack((new_pos.reshape(2, 1), new_pos.reshape(2, 1) + p2v(1, self._heading_traj[index]).reshape(2, 1))))
    return self.point, self.annotation, self.radius, self.traj_line, self.heading_arrow