from beacons.beacon import Beacon
from beacons.MIN.range_sensor import RangeSensor
from helpers import (
  get_smallest_signed_angle as ssa,
  get_vector_angle as gva,
  polar_to_vec as p2v,
  normalize,
  euler_int,
  plot_vec,
  rot_z_mat as R_z
)

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class MinState(Enum):
  SPAWNED   = 0,
  FOLLOWING = 1,
  EXPLORING = 2,
  LANDED    = 3
  NEIGHBOR  = 4

class VectorTypes(Enum):
  OBSTACLE = 0,
  PREV_MIN = 1,
  TOTAL    = 2,
  INTERVAL = 3

class Min(Beacon):
      
  clr = {
    MinState.SPAWNED:   "yellow",
    MinState.FOLLOWING: "red",
    MinState.EXPLORING: "green",
    MinState.LANDED:    "black",
    MinState.NEIGHBOR:  "blue",
  }

  vec_clr = {
    VectorTypes.OBSTACLE: "blue",
    VectorTypes.PREV_MIN: "green",
    VectorTypes.TOTAL:    "red",
    VectorTypes.INTERVAL: "black"
  }

  def __init__(self, max_range, deployment_strategy, xi_max=5, d_perf=1, d_none=3, k=0, a=0, v=np.zeros((2, ),), K_target=1, target_threshold=0.15):
    super().__init__(max_range,xi_max, d_perf, d_none, pos=None)
    self.K_target = K_target
    self.deployment_strategy = deployment_strategy
    self.target_threshold = target_threshold
    self.sensors = []
    self.target_pos = np.array([None, None]).reshape(2, )
    self.prev = None
    self.test = np.zeros(2)
    self.test2 = np.zeros(2)
    for ang in np.arange(0, 360, 90):
      r = RangeSensor(max_range)
      r.mount(self, ang)

  def insert_into_environment(self, env):
    super().insert_into_environment(env)
    self.state = MinState.SPAWNED
    self.heading =  0
    self._pos_traj = self.pos.reshape(2, 1)
    self._heading_traj = np.array([self.heading])
    self.state_traj = np.array([self.state], dtype=object)
    self._v_traj = np.array([0])
    self._xi_traj = np.zeros((self.ID,1)) #xi for all prev drones

  def do_step(self, beacons, SCS, ENV, dt):
    v = self.deployment_strategy.get_velocity_vector(self, beacons, SCS, ENV)
    self.pos = euler_int(self.pos.reshape(2, ), v, dt).reshape(2, )
    psi_ref = gva(v)
    tau = 0.1
    self.heading = euler_int(self.heading, (1/tau)*(ssa(psi_ref - self.heading)), dt)
    self._pos_traj = np.hstack((self._pos_traj, self.pos.reshape(2, 1)))
    self._heading_traj = np.concatenate((self._heading_traj, [self.heading]))
    self.state_traj = np.concatenate((self.state_traj, [self.state]))

    #As of 21.01 .get_velocity_vector() returns the calculated force, self._force_hist considers the norm of the force
    self._v_traj = np.hstack((self._v_traj, np.linalg.norm(v)))
    #self._xi_traj blir satt i self.deployment_strategy.get_velocity_vector()  

  def generate_target_pos(self, beacons, ENV, prev_min, next_min):
    self.compute_neighbors(beacons)
    # vecs_to_neighs = [
    #     # normalize(self.get_vec_to_other(n).reshape(2, 1)) for n in self.neighbors if not (self.get_vec_to_other(n) == 0).all()
    #     self.get_vec_to_other(n).reshape(2, 1) for n in self.neighbors if not (self.get_vec_to_other(n) == 0).all()
    # ]
    vec_to_prev = self.get_vec_to_other(prev_min).reshape(2, 1)
    for s in self.sensors:
        s.sense(ENV)
    # vecs_to_obs = [
    #     # normalize((R_z(self.heading)@R_z(s.host_relative_angle)@s.measurement.get_val())[:2])
    #     # for s in self.sensors if s.measurement.is_valid()
    #     (R_z(self.heading)@R_z(s.host_relative_angle)@s.measurement.get_val())[:2]
    #     for s in self.sensors if s.measurement.is_valid()
    # ]
    vecs_to_obs = []
    for s in self.sensors:
      if s.measurement.is_valid():
        #vector to obstacle in world frame
        vector = (R_z(self.heading)@R_z(s.host_relative_angle)@s.measurement.get_val())[:2]
        #scaling the vector that points away from obstalce
        #so that obstacles that are close to the drone produce larger vectors
        meas_length = np.linalg.norm(vector)
        vec_to_obs = (self.range - meas_length)*normalize(vector)
        vecs_to_obs.append(vec_to_obs)

    if len(vecs_to_obs) != 0: #if obstacles present
      # tot_vec = self.pos - vec_to_prev.reshape(2, ) - np.sum(vecs_to_obs,axis=0).reshape(2, )
      tot_vec =  - (self.range - np.linalg.norm(vec_to_prev))*normalize(vec_to_prev.reshape(2, )) - np.sum(vecs_to_obs,axis=0).reshape(2, )
    else: #if no obstacles present, only consider vector pointing away from prev min
      tot_vec =  - vec_to_prev.reshape(2, ) #- np.sum(vecs_to_neighs,axis=0).reshape(2, )

    mid_angle = gva(tot_vec) #angle that is "mean" of angle-interval
    target_angle = mid_angle + np.random.uniform(-1,1)*np.pi/4

    Rot_mat = R_z(gva(tot_vec))
    origin_transl = np.hstack((self.pos,0)).reshape((3,1))
    rest = np.array([0,0,0,1])
    a = np.hstack((Rot_mat,origin_transl))
    h_trans_mat = np.vstack((a,rest))#np.vstack((np.vstack((Rot_mat,origin_transl)),rest))

    target_pos = self.pos + R_z(gva(tot_vec))[:2,:2]@p2v(self.target_r,target_angle) #p2v(self.target_r, target_angle)
    self.test = target_pos#gva(target_pos)
    self.test2 = p2v(self.target_r,target_angle)
    next_min.target_pos = target_pos
    next_min.prev_drone = prev_min
    return target_pos
    #Get vectors to obstacles
    #Sum vectors to form "red" vector
    #Generate target on cirle within interval (angle)
    #Assign generated target to next_min.target_pos    
  
  def get_v_traj_length(self):
    return len(self._v_traj)

  def get_pos_traj_length(self):
    return self._pos_traj.shape[1]

  """""
  PLOTTING STUFF
  """""
  def plot(self, axis):
    self.heading_arrow = plot_vec(axis, p2v(1, self.heading), self.pos)
    return super().plot(axis, clr=self.clr[self.state]) + (self.heading_arrow, )
  
  def plot_vectors(self, prev_drone, ENV, axis):
    obstacles = []
    for s in self.sensors:
      s.sense(ENV)
    for s in self.sensors:
      if s.measurement.is_valid():
        #vector to obstacle in world frame
        vector = (R_z(self.heading)@R_z(s.host_relative_angle)@s.measurement.get_val())[:2]
        #scaling the vector that points away from obstalce
        #so that obstacles that are close to the drone produce larger vectors
        meas_length = np.linalg.norm(vector)
        vec_to_obs = (self.range - meas_length)*normalize(vector)
        obstacles.append(vec_to_obs)
    vec_to_prev_drone = self.get_vec_to_other(self.prev)
    self.a = plot_vec(axis, -vec_to_prev_drone, self.pos, clr=self.vec_clr[VectorTypes.PREV_MIN])
    for obs_vec in obstacles:
      self.b = plot_vec(axis, -obs_vec, self.pos, clr=self.vec_clr[VectorTypes.OBSTACLE])

    if len(obstacles) != 0:
      tot_vec = -vec_to_prev_drone- np.sum(obstacles,axis=0).reshape(2, )
    else:
      tot_vec = -vec_to_prev_drone

    self.c = plot_vec(axis, tot_vec, self.pos, clr=self.vec_clr[VectorTypes.TOTAL] )
    interval_vec_1 = R_z(np.pi/4)[:2,:2]@tot_vec
    interval_vec_2 = R_z(-np.pi/4)[:2,:2]@tot_vec
    self.d1 = plot_vec(axis, interval_vec_1, self.pos, clr=self.vec_clr[VectorTypes.INTERVAL])
    self.d2 = plot_vec(axis, interval_vec_2, self.pos, clr=self.vec_clr[VectorTypes.INTERVAL])
    self.test1_1 = axis.plot(*self.test, color="red",marker="o",markersize=8)
    self.test2_2 = axis.plot(*self.test2, color="green",marker="o",markersize=8)


  def plot_traj_line(self, axis):
    self.traj_line, = axis.plot(*self._pos_traj, alpha=0.4)
    return self.traj_line
  
  def plot_force_traj_line(self, axis):
    self.force_traj_line, = axis.plot(np.linspace(start=0, stop=len(self._v_traj),num=len(self._v_traj)), self._v_traj, label=f"Drone {self.ID}")
    return self.force_traj_line

  def plot_xi_traj_line(self, axis):    
    self.xi_traj_line = np.array([])
    for i in range(self._xi_traj.shape[0]):
#     if np.array(neigh_indices != self.prev_neigh_indices).all():      
      if np.array(self._xi_traj[i,:] != 0).any():
        # self.xi_traj_line, = axis.plot(np.linspace(start=0, stop=len(self._xi_traj[i,:]),num=len(self._xi_traj[i,:])), self._xi_traj[i], label=f"Drone {i}")
        tmp, = axis.plot(np.linspace(start=0, stop=len(self._xi_traj[i,:]),num=len(self._xi_traj[i,:])), self._xi_traj[i])#, label=f"Drone {i}")
        self.xi_traj_line = np.append(self.xi_traj_line, tmp)
    return self.xi_traj_line

  def plot_pos_from_pos_traj_index(self, index):
    new_pos = self._pos_traj[:, index]
    self.point.set_data(new_pos)
    self.point.set_color(self.clr[self.state_traj[index]])
    self.annotation.set_x(new_pos[0])
    self.annotation.set_y(new_pos[1])
    theta = np.linspace(0, 2*np.pi)
    self.radius.set_data(new_pos.reshape(2, 1) + p2v(self.range, theta))
    # self.radius2.set_data(new_pos.reshape(2, 1) + p2v(self.d_perf, theta))
    self.traj_line.set_data(self._pos_traj[:, :index])
    self.heading_arrow.set_data(*np.hstack((new_pos.reshape(2, 1), new_pos.reshape(2, 1) + p2v(1, self._heading_traj[index]).reshape(2, 1))))
    return self.point, self.annotation, self.radius, self.traj_line, self.heading_arrow #,self.radius2 

  def plot_force_from_traj_index(self, index):
    new_force = self._v_traj[:index]
    self.force_traj_line.set_data(np.linspace(0,index,num=len(new_force)),new_force)
    return self.force_traj_line
  
  def plot_xi_from_traj_index(self, index):
    for i in range(len(self._xi_traj)):      
      new_xi = self._xi_traj[:,:index+1]
      for j in range(len(new_xi)):      
        self.xi_traj_line[j].set_data(np.linspace(0,index,num=new_xi.shape[1]), new_xi[j])
      return self.xi_traj_line