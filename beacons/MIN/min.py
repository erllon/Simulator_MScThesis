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
    VectorTypes.INTERVAL: "orange"
  }

  def __init__(self, max_range, deployment_strategy, xi_max=5, d_perf=1, d_none=3, k=0, a=0, v=np.zeros((2, ),), K_target=1, target_threshold=0.15, delta_expl_angle=np.pi/4):
    super().__init__(max_range,xi_max, d_perf, d_none, pos=None)
    self.K_target = K_target
    self.deployment_strategy = deployment_strategy
    self.target_threshold = target_threshold
    self.sensors = []
    self.target_pos = np.array([None, None]).reshape(2, )
    self.prev = None
    self.delta_expl_angle = delta_expl_angle
    self.test = np.zeros(2)
    self.test2 = np.zeros(2)
    for ang in np.arange(0, 360, 90):
      r = RangeSensor(max_range)
      r.mount(self, ang)
    self.tot_vec = np.zeros(2)
    self.vecs_to_obs = []
    self.vec_to_prev = np.zeros(2)
    self.obs_vec = np.zeros(2)
    self.neigh_vec = np.zeros(2)


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
    """Generates a target point for next_min
       self calculates the vectors pointing away from obstacles and other drones.
       The vectors pointing away from obstacles forms a total_obstacle_vector, and the 
       target is generated at an angle that is the mean of total_obstacle_vector and the 
       mean of the angles of all the vectors pointing away from the neighboring drones"""

    self.compute_neighbors(beacons)

    """Computing vectors FROM neighbors TO drone"""
    vecs_from_neighs = []
    ang_from_neighs = []
    for n in self.neighbors:
      if not (self.get_vec_to_other(n) == 0).all():
        vec_from_neigh = -self.get_vec_to_other(n).reshape(2, 1)
        dist = np.linalg.norm(vec_from_neigh)
        vecs_from_neighs.append((self.range - dist)*normalize(vec_from_neigh))
        ang_from_neighs.append(gva(vec_from_neigh.reshape(2, )))
    tot_vec_from_neigh = np.sum(vecs_from_neighs,axis=0)
    avg_ang_from_neigh = np.sum(ang_from_neighs, axis=0)/len(ang_from_neighs)

    """Only calculating vector FROM current TO previous drone"""
    # self.vec_to_prev = self.get_vec_to_other(prev_min).reshape(2, 1)
    # vec_from_prev = -self.get_vec_to_other(prev_min).reshape(2, 1)
    
    """Calculating vectors FROM drone TO obstacles"""
    for s in self.sensors:
      s.sense(ENV)

    vecs_from_obs = []
    ang_from_obs = []

    for s in self.sensors:
      if s.measurement.is_valid():
        """Vector FROM drone TO obstacle in world frame"""
        vec_from_obs = -(R_z(self.heading)@R_z(s.host_relative_angle)@s.measurement.get_val())[:2]
        """Scaling the vector that points towards the obstalce
           so that obstacles that are close to the drone produce larger vectors"""
        meas_length = np.linalg.norm(vec_from_obs)
        """Vector FROM drone TO obstacle"""

        vec_from_obs = (self.range - meas_length)*normalize(vec_from_obs)
        vecs_from_obs.append(vec_from_obs.reshape(2, ))
        ang_from_obs.append(gva(vec_from_obs.reshape(2, )))

    expl_ang = 0
    ang_tot_vec_from_obs = 0

    if len(vecs_from_obs) != 0:
      """If obstacles present"""

      tot_vec_from_obs = np.sum(vecs_from_obs,axis=0)
      ang_tot_vec_from_obs = gva(tot_vec_from_obs)
      self.obs_vec = p2v(1, ang_tot_vec_from_obs)#p2v(1, tot_ang_from_obs)

      tot_vec_comb = tot_vec_from_obs.reshape(2, ) + tot_vec_from_neigh.reshape(2, )
      ang_tot_vec_comb = gva(tot_vec_comb)
      expl_ang = ang_tot_vec_comb#ang_tot_vec_from_obs #0.5*(ang_tot_vec_from_obs - avg_ang_from_neigh) 
    else:
      """If no obstacles present in range of drone"""
      # self.tot_vec = np.sum(vecs_from_neighs, axis=0).reshape(2, )
      # expl_ang = -np.sum(ang_to_neighs,axis=0)/len(ang_to_neighs)
      expl_ang = gva(tot_vec_from_neigh.reshape(2, )) #avg_ang_from_neigh
    

    rand = np.random.uniform(-1,1)*np.pi/4#self.delta_expl_angle #np.pi/4
    print(f"rand: {rand*180/np.pi}")
    target_angle = expl_ang + rand #mid_angle + rand#np.random.uniform(-1,1)*np.pi/4
    self.tot_vec = p2v(1, target_angle)
    
    print(f"ang_tot_vec_from_obs: {ang_tot_vec_from_obs}")
    self.neigh_vec = p2v(1, avg_ang_from_neigh)#p2v(1, np.sum(ang_from_neighs, axis=0)/len(ang_from_neighs))
    
    # Rot_mat = R_z(gva(self.tot_vec))
    # origin_transl = np.hstack((self.pos,0)).reshape((3,1))
    # rest = np.array([0,0,0,1])
    # a = np.hstack((Rot_mat,origin_transl))
    # h_trans_mat = np.vstack((a,rest))#np.vstack((np.vstack((Rot_mat,origin_transl)),rest))

    target_pos = self.pos + p2v(self.range, target_angle)#p2v(10, target_angle)#.reshape((2,)) #R_z(gva(tot_vec))[:2,:2]@p2v(self.target_r,target_angle)
    # target_pos_tilde = np.hstack((target_pos,0,1)).reshape((4,1))
    # target_pos_world = h_trans_mat @ target_pos_tilde
    next_min.target_pos = target_pos
    next_min.prev_drone = prev_min
    return target_pos  
  
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
    # for obs_vec in self.vecs_to_obs:
      # self.b = plot_vec(axis, -obs_vec, self.pos, clr=self.vec_clr[VectorTypes.OBSTACLE])
    self.c1 = plot_vec(axis, normalize(self.tot_vec), self.pos, clr=self.vec_clr[VectorTypes.TOTAL] )
    # self.c2 = plot_vec(axis, self.tot_vec, np.zeros(2), clr=self.vec_clr[VectorTypes.TOTAL] )
    interval_vec_1 = normalize(R_z(self.delta_expl_angle)[:2,:2]@self.tot_vec)
    interval_vec_2 = normalize(R_z(-self.delta_expl_angle)[:2,:2]@self.tot_vec)
    self.d1 = plot_vec(axis, interval_vec_1, self.pos, clr=self.vec_clr[VectorTypes.INTERVAL])
    self.d2 = plot_vec(axis, interval_vec_2, self.pos, clr=self.vec_clr[VectorTypes.INTERVAL])

    if np.linalg.norm(self.obs_vec) != 0: 
      self.e = plot_vec(axis, self.obs_vec, self.pos, clr="blue")
    self.f = plot_vec(axis, self.neigh_vec, self.pos, clr="green")

    #self.e = plot_vec(axis, self.vec_to_prev, self.pos, clr=self.vec_clr[VectorTypes.PREV_MIN])

    # self.e1 = plot_vec(axis, interval_vec_1, np.zeros(2), clr=self.vec_clr[VectorTypes.INTERVAL])
    # self.e2 = plot_vec(axis, interval_vec_2, np.zeros(2), clr=self.vec_clr[VectorTypes.INTERVAL])
    # self.test1_1 = axis.plot(*self.test, color="red",marker="o",markersize=8)
    # axis.annotate(f"{self.ID+1}", self.test)
    # self.test2_2 = axis.plot(*self.test2, color="green",marker="o",markersize=8)
    # axis.annotate(f"{self.ID+1}", self.test2)


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