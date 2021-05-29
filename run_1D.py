from environment import Env

from beacons.beacon import Beacon
from beacons.SCS.scs import SCS
from beacons.MIN.min import Min, MinState

from deployment.following_strategies.attractive_follow import AttractiveFollow
from deployment.exploration_strategies.potential_fields_explore import PotentialFieldsExplore
from deployment.following_strategies.no_follow import NoFollow
from deployment.exploration_strategies.line_explore import LineExplore, LineExploreKind
from deployment.deployment_fsm import DeploymentFSM

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter 

from helpers import polar_to_vec as p2v

import timeit
import cProfile, pstats, io
from pstats import SortKey
import json, codecs

data = {}
data['parameters'] = []
data['environment'] = []
data['beacons'] = []
data['uniformity'] = []

uniformity_list = []

def simulate(dt, mins, scs, env):
  pr = cProfile.Profile()
  pr.enable()

  scs.insert_into_environment(env)
  beacons = np.array([scs], dtype=object)
  data['beacons'].append(scs.toJson())
  
  mins[0].prev = scs
  scs.generate_target_pos(beacons, env, mins[0])
  uniformity_list.append(np.sum([beacon.calc_uniformity() for beacon in beacons])/len(beacons))

  # If the deployment should stop when some condition on the uniformity is fulfilled
  delta_uniformity = 0
  delta_limit = 0.5
  limit = 0.3
  i = 0

  tic = timeit.default_timer()

  while i < N_mins:#uniformity_list[-1] < limit and i < N_mins:#delta_uniformity <= limit:
    mins[i].insert_into_environment(env)
    while not mins[i].state == MinState.LANDED:
      mins[i].do_step(beacons, scs, env, dt)
    if i < len(mins)-1: #If i is not the index of the last min
      mins[i+1].prev = mins[i]
    beacons = np.append(beacons, mins[i])
    data['beacons'].append(mins[i].toJson())
    for b in beacons:
      b.compute_neighbors(beacons)
    if isinstance(beacons[i], Min): 
      data['beacons'][i]['vectors'] = {
          'tot_vec': beacons[i].tot_vec.tolist(),
          'obs_vec': beacons[i].obs_vec.tolist(),
          'neigh_vec': beacons[i].neigh_vec.tolist()
        }
    
      
    uniformity_list.append(np.sum([beacon.calc_uniformity() for beacon in beacons])/len(beacons))
    delta_uniformity = uniformity_list[-1] - uniformity_list[-2]
    
    
    print(f"min {mins[i].ID} landed at pos\t\t\t {mins[i].pos}")
    print(f"min {mins[i].ID} neighbors: {[n.ID for n in mins[i].neighbors]}")
    print(f"distance between n+1 and n = {np.linalg.norm(beacons[-1].pos-beacons[-2].pos)}")
    print("------------------")
    i += 1
  pr.disable()
  toc = timeit.default_timer()
  tot = toc - tic
  print(f"minimum number of neighbors: {min(beacons, key=lambda b: len(b.neighbors))}") 
  print(f"Total elapsed time for simulation: {tot}")
  print(f"minimum number of neighbors: {min(beacons, key=lambda b: len(b.neighbors))}")
  print("positions of the beacons:")
  pos_diff = []
  for i in range(len(beacons)):
    print(beacons[i].pos[0])
    if i > 0:
      pos_diff.append(beacons[i].pos[0] - beacons[i-1].pos[0])  
  return beacons

def write_to_file(file_path, data_to_write):
  with open(file_path, 'w') as outfile:
    json.dump(data_to_write, outfile, separators=(',', ':'), sort_keys=True, indent=2)


if __name__ == "__main__":
  # If _animate=True it is recommended that plot_properties=False due to slow animation
  # If it is desirable to animate the deployment AND the properties it is recommended to save the animation and watch the saved animation
  # Decreasing _save_count decreases the time it takes to save the animation
  _animate, save_animation, plot_properties = False, False, True
  start_animation_from_min_ID = 0

# %% Plotting styles
  # set styles
  if not _animate:
  # Animation runs way faster when using default styles
  # The below styles gives nicer looking plots
    try:
        # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
        # gives quite nice plots
        plt_styles = ["science", "grid", "bright", "no-latex"]
        plt.style.use(plt_styles)
        print(f"pyplot using style set {plt_styles}")
    except Exception as e:
      print(e)
      print("setting grid and only grid and legend manually")
      plt.rcParams.update(
          {
              # setgrid
              "axes.grid": True,
              "grid.linestyle": ":",
              "grid.color": "k",
              "grid.alpha": 0.5,
              "grid.linewidth": 0.5,
              # Legend
              "legend.frameon": True,
              "legend.framealpha": 1.0,
              "legend.fancybox": True,
              "legend.numpoints": 1,
          }
      )
  else:
    plt.rcParams.update(
      {
          # setgrid
          "axes.grid": True,
          "grid.linestyle": ":",
          "grid.color": "k",
          "grid.alpha": 0.5,
          "grid.linewidth": 0.5,
          # Legend
          "legend.frameon": True,
          "legend.framealpha": 1.0,
          "legend.fancybox": True,
          "legend.numpoints": 1,
      }
    )
    

# %% Environment initialization

  env = Env(
    np.array([
      0, 0
    ]),
    obstacle_corners = []
  )
  data['environment'].append(env.toJson())

# %%Parameter initialization
  max_range = 3
  _xi_max = 1
  _d_perf = 0.1
  _d_none = 2.5
  _K_o = 0.9

  N_mins = 6
  file_path = r'json_files\test_1D.json'
  dt = 0.01

  scs = SCS(Beacon.get_ID(), max_range,d_tau=None, xi_max=_xi_max, d_perf=_d_perf, d_none=_d_none)

  """ Line exploration """
  mins = [
    Min(
      Beacon.get_ID(),
      max_range,
      DeploymentFSM(
        NoFollow(),
        LineExplore(
          kind=LineExploreKind.ONE_DIM_LOCAL,
        )
      ),
      xi_max=_xi_max,
      d_perf=_d_perf,
      d_none=_d_none,
      d_tau=None
    ) for i in range(N_mins)
  ]

  beacons = simulate(dt, mins, scs, env)
  data['uniformity'] = [float(number) for number in uniformity_list]

  data['parameters'] = {
    'N_mins': len(beacons)-1,#N_mins, len(beacons)-1 so that it works when deploying unknown number of mins
    'Max_range' : max_range,
    'K_o': _K_o,
    'xi_max': _xi_max,
    'd_perf': _d_perf,
    'd_none': _d_none,
    'delta_expl_angle': None #_delta_expl_angle
  }

  write_to_file(file_path, data)
  
  fig = plt.figure(figsize=(5,5), tight_layout=True)
  fig.canvas.set_window_title(f"Deployment {file_path[:-5]}")
  
  if plot_properties:
    if _animate:
      ax1_1 = fig.add_subplot(2,1,1)
      ax1_2 = fig.add_subplot(2,1,2)
      ax1_1.title.set_text("Deployment")
      ax1_2.title.set_text(r"$\left\|\| F_{applied} \right\|\|$")      
    else:
      fig2 = plt.figure(figsize=(5,5))
      fig2.canvas.set_window_title(f"Properties {file_path[:-5]}")

      ax1_1 = fig.add_subplot(1,1,1)
      ax2_1 = fig2.add_subplot(2,1,1)
      ax2_2 = fig2.add_subplot(2,1,2)

      ax1_1.title.set_text("Deployment")
      ax2_1.title.set_text(r"$\left\|\| F_{applied} \right\|\|$")
      ax2_2.title.set_text(r"$\xi$ from neighbors")
      time_label = ax2_2.set_xlabel(r"$time$",labelpad=-4,loc="right")
      xi_label = ax2_2.set_ylabel(r"$\xi$",labelpad=-15, loc='top')
      xi_label.set_rotation(0)
  else:
    ax = fig.add_subplot(1,1,1)
    ax.title.set_text("Deployment")


  if _animate:
    for mn in beacons[1:start_animation_from_min_ID]: #SCS is already plotted
      if plot_properties:
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        mn.plot_force_traj_line(ax1_2)
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        mn.plot_force_traj_line(ax1_2)
      else:
        mn.plot(ax)
        mn.plot_vectors(env, ax)

    offset, min_counter = [0], [start_animation_from_min_ID]

    def init():
      if plot_properties:
        scs.plot(ax1_1)
        env.plot(ax1_1)
        artists = []
        for mn in mins:
          artists += mn.plot(ax1_1)
          artists += (mn.plot_traj_line(ax1_1), )
          artists += (mn.plot_force_traj_line(ax1_2), )
          mn.plot_pos_from_pos_traj_index(0)
          mn.plot_force_from_traj_index(0)
        if start_animation_from_min_ID == 0:
          ax1_2.legend(ncol=1, prop={'size': 9}, bbox_to_anchor=(1,1), loc='upper left')
      else:
        scs.plot(ax)
        env.plot(ax)
        artists = []
        for mn in mins:
          artists += mn.plot(ax)
          artists += (mn.plot_traj_line(ax), )
          mn.plot_pos_from_pos_traj_index(0)
      return artists

    def animate(i):
      try:
        if i - offset[0] >= mins[min_counter[0]].get_pos_traj_length():
          offset[0] += mins[min_counter[0]].get_pos_traj_length()
          min_counter[0] += 1
        if plot_properties:
          plt_pos_traj = mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
          plt_force_traj = mins[min_counter[0]].plot_force_from_traj_index(i-offset[0])
          return  plt_force_traj, plt_pos_traj, # plt_xi_traj 
        else:
          plt_pos_traj = mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
          return plt_pos_traj
      except:
        print("Animation finished")
  
    anim = FuncAnimation(fig, animate, init_func=init, interval=2, blit=False)
    
    _save_count = 1750
    anim = FuncAnimation(fig, animate, init_func=init, interval=2, blit=False, save_count=_save_count)
    
    if save_animation:
        writergif = PillowWriter(fps=60) 

        animation_name_gif = "animation_test1D.gif"
        print("Saving animation. Depending on the choise of 'save_count' this might take some time...")
        print(f"Chosen 'save_count' = {_save_count}")
        anim.save(animation_name_gif,writer=writergif)   
        print(f"Animation saved to {animation_name_gif}")
  else:
    if plot_properties:
      env.plot(ax1_1)
      scs.plot(ax1_1)
      for mn in beacons[1:]:#SCS is already plotted, using beacons instead of mins so that only landed mins are taken into account
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        mn.plot_force_traj_line(ax2_1)
      beacons[-1].plot_xi_traj_line(ax2_2)
      ax2_1.legend(ncol=1, prop={'size': 9}, bbox_to_anchor=(1.05,1), loc='upper left')

    else:
      env.plot(ax)
      scs.plot(ax)
      for j in range(len(beacons)-1):
      # len(beacons)-1 so that it works when deploying unknown number of mins
        mins[j].plot(ax)
        mins[j].plot_traj_line(ax)
        if j == 0:
          mins[j].plot_vectors(env,ax)
        else:
          mins[j].plot_vectors(env,ax)
      ax.legend(ncol=2, prop={'size': 9})
      ax.axis('equal')

  if not save_animation:
    plt.show()

# %%
