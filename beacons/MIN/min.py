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
from copy import deepcopy

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class MinState(Enum):
  SPAWNED   = 0
  FOLLOWING = 1
  EXPLORING = 2
  LANDED    = 3
  NEIGHBOR  = 4

class VectorTypes(Enum):
  OBSTACLE = 0
  PREV_MIN = 1
  TOTAL    = 2
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

  def __init__(self, min_id, max_range, deployment_strategy, xi_max, d_perf, d_none, d_tau, k=0, a=0, v=np.zeros((2, ),), K_target=1, target_threshold=0.15, delta_expl_angle=np.pi/4):
    super().__init__(min_id, max_range, xi_max, d_perf, d_none, d_tau, pos=None)
    self.K_target = K_target
    self.deployment_strategy = deployment_strategy
    self.target_threshold = target_threshold
    self.sensors = []
    self.target_pos = np.array([None, None]).reshape(2, )
    self.target_angle = None
    self.prev = None
    self.next = None
    self.delta_expl_angle = delta_expl_angle
    for ang in np.arange(0, 360, 90):
      r = RangeSensor(max_range)
      r.mount(self, ang)
    self.tot_vec = np.zeros(2)
    self.vecs_to_obs = []
    self.vec_to_prev = np.zeros(2)
    self.obs_vec = np.zeros(2)
    self.closest_obs_vec = np.zeros(2)
    self.neigh_vec = np.zeros(2)

    self.first_target_pos = None
    self.final_target_pos = None
    self.vecs_from_obs = []


  def insert_into_environment(self, env):
    super().insert_into_environment(env)
    self.state = MinState.SPAWNED
    self.heading =  0
    self._pos_traj = self.pos.reshape(2, 1)
    self._heading_traj = np.array([self.heading])
    self.state_traj = np.array([self.state], dtype=object)
    self._v_traj = np.array([0])
    self._xi_traj = np.zeros((self.ID,1)) #xi to/for all prev drones


  def do_step(self, beacons, SCS, ENV, dt):
    v = self.deployment_strategy.get_velocity_vector(self, beacons, SCS, ENV)
    self.prev_pos = self.pos
    self.pos = euler_int(self.pos.reshape(2, ), v, dt).reshape(2, )
    if (self.prev_pos != None).any():
      self.delta_pos = self.pos - self.prev_pos
    psi_ref = gva(v)
    tau = 0.1
    self.heading = euler_int(self.heading, (1/tau)*(ssa(psi_ref - self.heading)), dt)
    self._pos_traj = np.hstack((self._pos_traj, self.pos.reshape(2, 1)))
    self._heading_traj = np.concatenate((self._heading_traj, [self.heading]))
    self.state_traj = np.concatenate((self.state_traj, [self.state]))

    self._v_traj = np.hstack((self._v_traj, np.linalg.norm(v)))
    
  def generate_target_pos(self, beacons, ENV, prev_min, next_min):
    """Generates a target point for next_min
       self calculates the vectors pointing away from obstacles and other drones.
       The vectors pointing away from obstacles forms a total_obstacle_vector, and the 
       target is generated at an angle that is the mean of total_obstacle_vector and the 
       mean of the angles of all the vectors pointing away from the neighboring drones"""

    self.compute_neighbors(beacons)

    """Computing vectors FROM neighbors TO drone"""
    vecs_from_neighs, ang_from_neighs = Min.get_neigh_vecs_and_angles(self)
    vecs_from_neighs_arr = np.array(vecs_from_neighs)
    if vecs_from_neighs_arr.shape[0] == 1:
      tot_vec_from_neigh = vecs_from_neighs_arr#np.array(vecs_from_neighs) # test#vecs_from_neighs
    else:
      sortidxs = np.argsort(np.linalg.norm(vecs_from_neighs_arr, axis=-1))#np.argsort(np.linalg.norm(vecs_from_neighs[:], axis=-1))
      sorted_vecs_from_neighs = vecs_from_neighs_arr[sortidxs]#test[sortidxs]#vecs_from_neighs[sortidxs]
      # sorted_ang_from_neighs = ang_from_neighs[sortidxs]

    # tot_vec_from_neigh = np.sum(vecs_from_neighs,axis=0)
    # avg_ang_from_neigh = np.sum(ang_from_neighs, axis=0)/len(ang_from_neighs)
    # THE VECTOR CORRESPONDING TO THE CLOSEST NEIGHBOR WILL BE THE **LAST** VECTOR IN sorted_vecs_from_neighs DUE TO SCALING
      tot_vec_from_neigh = sorted_vecs_from_neighs[-1]#np.sum(sorted_vecs_from_neighs[0],axis=0)#np.sum(sorted_vecs_from_neighs[:4],axis=0)#
    
    """Calculating vectors FROM drone TO obstacles"""
    # vecs_from_obs, ang_from_obs = Min.get_obs_vecs_and_angles(self, ENV)
    vecs_from_obs, _ = Min.get_obs_vecs_and_angles(self, ENV)

    expl_ang = 0
    ang_tot_vec_from_obs = 0
    ang_closest_obs = 0

    self.vecs_from_obs = vecs_from_obs
    if len(vecs_from_obs) != 0: # If obstacles present
      tot_vec_from_obs = np.sum(vecs_from_obs,axis=0)
      ang_tot_vec_from_obs = gva(tot_vec_from_obs)
      self.obs_vec = p2v(1, ang_tot_vec_from_obs)

      tot_vec_comb = tot_vec_from_obs.reshape(2, ) + tot_vec_from_neigh.reshape(2, )
      ang_tot_vec_comb = gva(tot_vec_comb)
      expl_ang = ang_tot_vec_comb 
    else:
      """If no obstacles present in range of drone"""
      expl_ang = gva(tot_vec_from_neigh.reshape(2, ))
    

    rand = np.random.uniform(-1,1)*self.delta_expl_angle
    print(f"Drone {self.ID} generating target for drone {next_min.ID}")
    print(f"expl_ang from drone {self.ID}: {np.rad2deg(expl_ang)}")
    print(f"rand from drone {self.ID}: {np.rad2deg(rand)}")
    next_min.target_angle = expl_ang + rand

    self.tot_vec = p2v(1, expl_ang)
    
    self.neigh_vec = tot_vec_from_neigh

    # Could increase the distance the target point is generated at, to increase the force applied to the min
    # It is the fact that the distance is 1 that the force is not saturated when entering the exploration phase
    target_pos = self.pos + p2v(1, next_min.target_angle) #1.5
    
    if next_min.first_target_pos == None:
      next_min.first_target_pos = deepcopy(target_pos.reshape(2, ))
      print(f"first_target: {next_min.first_target_pos}")
    
    next_min.target_pos = target_pos
    next_min.prev_drone = prev_min
    self.next = next_min
    return target_pos  
  
  def generate_virtual_target(self):
    self.target_pos += p2v(np.linalg.norm(self.delta_pos), self.target_angle)
    self.final_target_pos = deepcopy(self.target_pos)

  @staticmethod
  def get_neigh_vecs_and_angles(MIN):
    vecs_from_neighs, ang_from_neighs = [], []#np.array([]), np.array([]) #[], []#
    for n in MIN.neighbors:
      if not (MIN.get_vec_to_other(n) == 0).all():
        vec_from_neigh = -MIN.get_vec_to_other(n)#.reshape(2,))
        dist = np.linalg.norm(vec_from_neigh) #when using xi for RSSI, dist will be in the interval (0, 1.7916)
        scaling = MIN.d_tau#4#2#4#2#4#MIN.d_none#1.7916#

        vecs_from_neighs.append(((scaling-dist)/scaling)*MIN.range*normalize(vec_from_neigh))
        # vecs_from_neighs.append((MIN.range*(scaling-dist)/scaling)*normalize(vec_from_neigh))
        ang_from_neighs.append(gva(vec_from_neigh.reshape(2, )))
        # vecs_from_neighs = np.append(vecs_from_neighs, (scaling-dist)*normalize(vec_from_neigh),axis=-1)
        # ang_from_neighs = np.append(ang_from_neighs, gva(vec_from_neigh.reshape(2, )))
    return vecs_from_neighs, ang_from_neighs
    
  @staticmethod
  def get_obs_vecs_and_angles(MIN, ENV):
    vecs_from_obs, ang_from_obs = [], []
    for s in MIN.sensors:
      s.sense(ENV)  #Each sensor will now have/know the smallest distance to an obstacle and at what angle the obstacle is
      if s.measurement.is_valid():
        """Vector FROM drone TO obstacle in world frame"""
        
        length = s.measurement.get_val()
        angle_sensor_frame = s.measurement.get_angle()
        meas_vec_sensor_frame = p2v(length, angle_sensor_frame)
        meas_vec_world_frame = R_z(s.host.heading)[:2,:2]@R_z(s.host_relative_angle)[:2,:2]@meas_vec_sensor_frame
        vec_away_from_obs = -meas_vec_world_frame
        angle_world_frame = angle_sensor_frame + s.host.heading + s.host_relative_angle
        vec_from_obs = -p2v(length, angle_world_frame)
        """Scaling the vector that points towards the obstalce
          so that obstacles that are close to the drone produce larger vectors"""
        meas_length = np.linalg.norm(vec_from_obs)
        """Vector FROM drone TO obstacle"""

        vec_from_obs = (MIN.range - meas_length)*normalize(vec_from_obs)
        vec_from_obs_scaled = (MIN.range - meas_length)*MIN.range*normalize(vec_away_from_obs)
        # vec_from_obs_scaled = ((MIN.range - meas_length)/1)*normalize(vec_away_from_obs)


        # vecs_from_obs.append(vec_from_obs.reshape(2, ))
        vecs_from_obs.append(vec_from_obs_scaled.reshape(2, ))

        ang_from_obs.append(gva(vec_from_obs.reshape(2, )))

    return vecs_from_obs, ang_from_obs     

  def get_v_traj_length(self):
    return len(self._v_traj)

  def get_pos_traj_length(self):
    return self._pos_traj.shape[1]

  """""
  PLOTTING STUFF
  """""
  def plot(self, axis):
    self.heading_arrow = plot_vec(axis, p2v(1, self.heading), self.pos)
    self.point = axis.plot(*self.pos, color=self.clr[self.state], marker="o", markersize=4)[0]
    self.annotation = axis.annotate(self.ID, xy=(self.pos[0], self.pos[1]+0.003), fontsize=14)

    return self.point, self.annotation
  
  def plot_vectors(self, ENV, axis):

    plot_vec(axis, normalize(self.tot_vec), self.pos, clr=self.vec_clr[VectorTypes.TOTAL] )
    interval_vec_1 = normalize(R_z(self.delta_expl_angle)[:2,:2]@self.tot_vec)
    interval_vec_2 = normalize(R_z(-self.delta_expl_angle)[:2,:2]@self.tot_vec)
    plot_vec(axis, interval_vec_1, self.pos, clr=self.vec_clr[VectorTypes.INTERVAL])
    plot_vec(axis, interval_vec_2, self.pos, clr=self.vec_clr[VectorTypes.INTERVAL])

    # plot_vec(axis, self.neigh_vec, self.pos, clr='green')

  def plot_traj_line(self, axis):
    self.traj_line, = axis.plot(*self._pos_traj, alpha=0.4)
    return self.traj_line
  
  def plot_force_traj_line(self, axis):
    self.force_traj_line, = axis.plot(np.linspace(start=0, stop=len(self._v_traj),num=len(self._v_traj)), self._v_traj, label=rf"$\nu_{{{self.ID}}}$")
    return self.force_traj_line

  def plot_xi_traj_line(self, axis):  
    xi_clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  
    self.xi_traj_line = np.array([])
    for i in range(self._xi_traj.shape[0]):
      if np.array(self._xi_traj[i,:] != 0).any():
        tmp, = axis.plot(np.linspace(start=0, stop=len(self._xi_traj[i,:]),num=len(self._xi_traj[i,:])), self._xi_traj[i], label=rf"$\nu_{{{i}}}$")#), color=xi_clr_cycle[i-1])
        # if i == 0:
        #   tmp, = axis.plot(np.linspace(start=0, stop=len(self._xi_traj[i,:]),num=len(self._xi_traj[i,:])), self._xi_traj[i], label=rf"$\nu_{{{i}}}$", color=xi_clr_cycle[self._xi_traj.shape[0]])
        # else:
        #   tmp, = axis.plot(np.linspace(start=0, stop=len(self._xi_traj[i,:]),num=len(self._xi_traj[i,:])), self._xi_traj[i], label=rf"$\nu_{{{i}}}$", color=xi_clr_cycle[i-1])
        self.xi_traj_line = np.append(self.xi_traj_line, tmp)
    return self.xi_traj_line

  def plot_pos_from_pos_traj_index(self, index):
    new_pos = self._pos_traj[:, index]
    self.point.set_data(new_pos)
    self.point.set_color(self.clr[self.state_traj[index]])
    self.annotation.set_x(new_pos[0])
    self.annotation.set_y(new_pos[1])
    
    self.traj_line.set_data(self._pos_traj[:, :index])
    self.heading_arrow.set_data(*np.hstack((new_pos.reshape(2, 1), new_pos.reshape(2, 1) + p2v(1, self._heading_traj[index]).reshape(2, 1))))
    return self.point, self.annotation, self.traj_line, self.heading_arrow

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

  def toJson(self): 
    jsonDict = {
        'Type': 'MIN',
        'ID': self.ID,
        'Neighbor IDs': [neigh.ID for neigh in self.neighbors],
        'pos_traj': self._pos_traj.tolist(),
        'force_traj': self._v_traj.tolist(),
        'heading_traj': self._heading_traj.tolist(),
        'xi_traj': self._xi_traj.tolist(),
        'state_traj': [state.value for state in self.state_traj],
        'vectors': {}
    }
    return jsonDict
