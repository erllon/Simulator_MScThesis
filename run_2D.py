from environment import Env

from beacons.beacon import Beacon
from beacons.SCS.scs import SCS
from beacons.MIN.min import Min, MinState

from deployment.following_strategies.attractive_follow import AttractiveFollow
from deployment.following_strategies.straight_line_follow import StraightLineFollow
from deployment.following_strategies.new_attractive_follow import NewAttractiveFollow
from deployment.exploration_strategies.potential_fields_explore import PotentialFieldsExplore
from deployment.exploration_strategies.new_potential_fields_explore import NewPotentialFieldsExplore
from deployment.exploration_strategies.heuristic_explore import HeuristicExplore
from deployment.following_strategies.no_follow import NoFollow
from deployment.exploration_strategies.line_explore import LineExplore, LineExploreKind
from deployment.deployment_fsm import DeploymentFSM

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

  # If we deploy drones until some condition on the uniformity is fulfilled
  delta_uniformity = 0
  delta_limit = 0.5
  limit = 0.3#2#1.5#2#0.05 #Try 1.5?
  i = 0

  tic = timeit.default_timer()

  while i < N_mins:#uniformity_list[-1] < limit and i < N_mins:#delta_uniformity <= limit:
  # for i in range(len(mins)):
    print(f"file: {file_path}")
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
    print(f"min {mins[i].ID} tot_vec = {mins[i].tot_vec}")
    print(f"distance between n+1 and n = {np.linalg.norm(beacons[-1].pos-beacons[-2].pos)}")
    print(f"min {mins[i].ID} target\t\t\t\t {mins[i].target_pos}")
    print(f"min {mins[i].ID} neighbors: {[n.ID for n in mins[i].neighbors]}")
    if not mins[i].deployment_strategy.get_target() is None:
          print(f"Its target now has {len(mins[i].deployment_strategy.get_target().neighbors)} neighs", )
    print(f"uniformity after min {mins[i].ID} landed: {uniformity_list[-1]}")
    print("------------------")
    i += 1
  pr.disable()
  toc = timeit.default_timer()
  tot = toc - tic
  print(f"minimum number of neighbors: {min(beacons, key=lambda b: len(b.neighbors))}") 
  print(f"Total elapsed time for simulation: {tot}")
  
  # s = io.StringIO()
  # sortby = SortKey.CUMULATIVE
  # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
  # ps.print_stats()
  # print(s.getvalue())
  return beacons

def write_to_file(file_path, data_to_write):
  with open(file_path, 'w') as outfile:
    json.dump(data_to_write, outfile, separators=(',', ':'), sort_keys=True, indent=2)


if __name__ == "__main__":
  _animate, save_animation, plot_propterties = False, False, True
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
  obstacle_corners_1D = [
      np.array([
        [-10, -10],
        [ -6, -10],
      ]),
    ]

  obs_zig_zag = [
      np.array([
        [-1, -1],
        [-1,   10],
        [15,     10],
        [15,     -1],
      ]),
      np.array([
        [4.0,  5],
        [4.0, -1],
        [5, -1],
        [5,  5]
      ]),
      np.array([
        [9.0,  10],
        [9.0, 4.9],
        [10.0, 4.9],
        [10.0,  10]
      ])
    ]
  open_small = [
      np.array([
        [-1, -1],
        [-1,   5],
        [5,      5],#[7,    7],
        [5,     -1],#[7,    -1],#,
      ]),
    ]
  
  open_large = [
      np.array([
        [-1, -1],
        [-1,   10],
        [10,      10],
        [10,     -1],
      ]),
    ]

  open_w_sq_obs = [
      np.array([
        [-1, -1],
        [-1,   12],
        [12,     12],
        [12,     -1],
      ]),
      np.array([
        [2, 2],
        [2, 9],
        [9, 9],
        [9, 2]
      ])      
    ]

  stripa = [
    np.array( 
      [
        [-1, -1],
        [(-1 + 20),-1],
        [(-1 + 20), (-1 + 3)],
        [(-1 + 20 + 2),   (-1 + 3)],
        [(-1 + 20 + 2),  -1],
        [((-1 + 20 + 2) + 20), -1],
        [((-1 + 20 + 2) + 20), (-1 + 3)],
        [((-1 + 20 + 2) + 20 + 2),  (-1 + 3)],
        [((-1 + 20 + 2) + 20 + 2),  -1],
        [(((-1 + 20 + 2) + 20 + 2) + 20), -1],
        [(((-1 + 20 + 2) + 20 + 2) + 20), 9],
        [(((-1 + 20 + 2) + 20 + 2)), 9],
        [((-1 + 20 + 2) + 20 + 2), (9 - 3)],
        [((-1 + 20 + 2) + 20) , (9 - 3)],
        [((-1 + 20 + 2) + 20) ,9],
        [((-1 + 20 + 2)), 9],
        [(-1 + 20 + 2) ,(9 - 3)],
        [(-1 + 20) ,(9 - 3)],
        [(-1 + 20) ,9],
        [-1,  9],
        [-1,  9]
      ],
    )
  ]

  open_uniformity_comp = [
      np.array([
        [-1, -1],
        [-1,  9],
        [ 9,  9],
        [ 9, -1],
      ]),
    ]

  env = Env(
    np.array([
      0, 0
    ]),
    obstacle_corners = open_uniformity_comp#open_large #stripa#open_small#[]#open_w_sq_obs #open_large#obs_zig_zag#[]#
  )
  data['environment'].append(env.toJson())

# %%Parameter initialization
  max_range = 2#3
  _xi_max = 1
  _d_perf = 0.18#0.1
  _d_none = 5.6#2.5
  _delta_expl_angle = np.pi/4 #0#np.pi/6
  _K_o = 0.9

  N_mins = 15
  file_path = r'json_files\ds_test_123.json'
  dt = 0.01

  scs = SCS(Beacon.get_ID(), max_range,xi_max=_xi_max, d_perf=_d_perf, d_none=_d_none)
  """ Potential fields exploration """
  mins = [
    Min(
      Beacon.get_ID(),
      max_range,
      DeploymentFSM(
        NewAttractiveFollow(K_o=_K_o),
        NewPotentialFieldsExplore(K_o=_K_o, target_point_or_line=NewPotentialFieldsExplore.Target.LINE)
      ),
      xi_max=_xi_max,
      d_perf=_d_perf,
      d_none=_d_none,
      delta_expl_angle=_delta_expl_angle
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
    'delta_expl_angle': _delta_expl_angle
  }

  write_to_file(file_path, data)
  
  fig = plt.figure(figsize=(5,5))#plt.figure(figsize=(5.2,3))
  fig.canvas.set_window_title(f"Deployment {file_path[:-5]}")
  
  if plot_propterties:
    if _animate:
      ax1_1 = fig.add_subplot(3,1,1)
      ax1_2 = fig.add_subplot(3,1,2)
      ax1_3 = fig.add_subplot(3,1,3, sharex=ax1_2)
      ax1_1.title.set_text("Deployment")
      ax1_2.title.set_text(r"$\left\|\| F_{applied} \right\|\|$")
      ax1_3.title.set_text(r"$\xi$ from neighbors")
    else:
      fig2 = plt.figure(figsize=(5,5))#plt.figure(figsize=(5,5), tight_layout=True)
      fig2.canvas.set_window_title(f"Properties {file_path[:-5]}")

      ax1_1 = fig.add_subplot(1,1,1)
      ax2_1 = fig2.add_subplot(2,1,1)
      ax2_2 = fig2.add_subplot(2,1,2)

      ax1_1.title.set_text("Deployment")
      ax2_1.title.set_text(r"$\left\|\| F_{applied} \right\|\|$")
      ax2_2.title.set_text(r"$\xi$ from neighbors")
  else:
    ax = fig.add_subplot(1,1,1)
    ax.title.set_text("Deployment")


  if _animate:
    for mn in beacons[1:start_animation_from_min_ID]: #SCS is already plotted
      if plot_propterties:
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        mn.plot_force_traj_line(ax1_2)
        mn.plot_xi_traj_line(ax1_3)
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        mn.plot_force_traj_line(ax1_2)
        mn.plot_xi_traj_line(ax1_3)
      else:
        mn.plot(ax)
        mn.plot_vectors(env, ax)

    offset, min_counter = [0], [start_animation_from_min_ID]

    def init():
      if plot_propterties:
        scs.plot(ax1_1)
        env.plot(ax1_1)
        artists = []
        for mn in mins:
          artists += mn.plot(ax1_1)
          artists += (mn.plot_traj_line(ax1_1), )
          artists += (mn.plot_force_traj_line(ax1_2), )
          artists += (mn.plot_xi_traj_line(ax1_3), )
          mn.plot_pos_from_pos_traj_index(0)
          mn.plot_force_from_traj_index(0)
          mn.plot_xi_from_traj_index(0)
        if start_animation_from_min_ID == 0:
          ax1_2.legend(ncol=2, prop={'size': 9})  
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
      if i - offset[0] >= mins[min_counter[0]].get_pos_traj_length():
        offset[0] += mins[min_counter[0]].get_pos_traj_length()
        min_counter[0] += 1
      if plot_propterties:
        plt_pos_traj = mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
        plt_force_traj = mins[min_counter[0]].plot_force_from_traj_index(i-offset[0])
        plt_xi_traj = mins[min_counter[0]].plot_xi_from_traj_index(i-offset[0])
        return  plt_force_traj, plt_xi_traj, plt_pos_traj
      else:
        plt_pos_traj = mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
        return plt_pos_traj
  
    anim = FuncAnimation(fig, animate, init_func=init, interval=2, blit=False)
    
    if save_animation:
      animation_name = "animation.gif"
      print("Saving animation")
      anim.save(animation_name)
      print(f"Animation saved to {animation_name}")
  else:
    if plot_propterties:
      env.plot(ax1_1)
      scs.plot(ax1_1)
      for mn in beacons[1:]:#SCS is already plotted, using beacons instead of mins so that only landed mins are taken into account
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        mn.plot_vectors(env, ax1_1)
        mn.plot_force_traj_line(ax2_1)
        mn.plot_xi_traj_line(ax2_2)
      ax2_1.legend(ncol=2, prop={'size': 9})

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
  fig_uniformity = plt.figure(figsize=(5,5))#plt.figure(figsize=(5.2,3))
  fig_uniformity.canvas.set_window_title(f"Uniformity {file_path[:-5]}")

  ax_uniformity = fig_uniformity.add_subplot(1,1,1)
  ax_uniformity.set(
    xlabel = 'Beacons',
    ylabel = 'Uniformity',
    title = 'Uniformity'
  )

  plt.xticks(range(len(uniformity_list)+1)) #ints on x-axis
  ax_uniformity.plot(uniformity_list)
  ax_uniformity.plot(uniformity_list, "or")

  plt.show()

# %%
