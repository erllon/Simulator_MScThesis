from environment import Env

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

from plot_fields import FieldPlotter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from helpers import polar_to_vec as p2v

import timeit
import cProfile, pstats, io
from pstats import SortKey


def simulate(dt, mins, scs, env):
  pr = cProfile.Profile()
  pr.enable()
  scs.insert_into_environment(env)
  beacons = np.array([scs], dtype=object)

  #scs.generate_target_pos(beacons,env, mins[0])
  # mins[0].target_pos = p2v(mins[0].target_r,np.random.uniform(0, np.pi/2))
  # mins[0].prev = scs
  mins[0].prev = scs
  scs.generate_target_pos(beacons, env, mins[0])  
  tic = timeit.default_timer()
  for i in range(len(mins)):
    mins[i].insert_into_environment(env)
    while not mins[i].state == MinState.LANDED:
      mins[i].do_step(beacons, scs, env, dt)
    if i < len(mins)-1: #as long as i is not the index of the last min
      # if i-1 > 0: #"prev_drone" for drone 1 will be the scs
        # mins[i].generate_target_pos(beacons,env,mins[i-1], mins[i+1])
      # else:
        # mins[i].generate_target_pos(beacons,env,scs, mins[i+1])
      mins[i+1].prev = mins[i]
    beacons = np.append(beacons, mins[i])    
    for b in beacons:
      b.compute_neighbors(beacons)
    print(f"min {mins[i].ID} landed at pos\t\t\t {mins[i].pos}")
    print(f"min {mins[i].ID} target\t\t\t\t {mins[i].target_pos}")
    print(f"min {mins[i].ID} neighbors: {[n.ID for n in mins[i].neighbors]}")
    if not mins[i].deployment_strategy.get_target() is None:
          print(f"Its target now has {len(mins[i].deployment_strategy.get_target().neighbors)} neighs\n------------------", )
  pr.disable()
  toc = timeit.default_timer()
  tot = toc - tic
  print(f"minimum number of neighbors: {min(beacons, key=lambda b: len(b.neighbors))}") 
  print(f"Total elapsed time for simulation: {tot}")

  s = io.StringIO()
  sortby = SortKey.CUMULATIVE
  ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
  ps.print_stats()
  print(s.getvalue())
  return beacons   

if __name__ == "__main__":
# %% Plotting styles
  # set styles
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
              "axes.grid": False,#True,
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
        [-0.5, -0.5],
        [-0.5,   10],
        [15,     10],
        [15,     -0.5],
      ]),
      np.array([
        [4.0,  5],
        [4.0, -0.5],
        [5, -0.5],
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
        [-0.5, -0.5],
        [-0.5,   5],
        [5,      5],
        [5,     -0.5],
      ]),
    ]
  
  open_large = [
      np.array([
        [-0.5, -0.5],
        [-0.5,   7],
        [7,      7],
        [7,     -0.5],
      ]),
    ]

  open_w_sq_obs = [
      np.array([
        [-0.5, -0.5],
        [-0.5,   12],
        [12,     12],
        [12,     -0.5],
      ]),
      np.array([
        [2.5, 2.5],
        [2.5, 9.5],
        [9.5, 9.5],
        [9.5, 2.5]
      ])      
    ]

  env = Env(
    # np.array([
    #   -9.8, -5.2
    # ]),
    np.array([
      0, 0
    ]),
    obstacle_corners = open_large#open_w_sq_obs#obs_zig_zag#open_small#obs_zig_zag#open_w_sq_obs#open_large##open_small#[]#obs_zig_zag #[]
  )

# %%Parameter initialization
  _animate, save_animation, plot_propterties = False, False, False
  start_animation_from_min_ID = 0

  max_range = 3 #0.51083#float(-np.log(-0.6))#3 #0.75    0.51083

  N_mins = 5#18#7#2*5#3
  dt = 0.01#0.01

  scs = SCS(max_range)
  """ Potential fields exploration
  mins = [
    Min(
      max_range,
      DeploymentFSM(
        AttractiveFollow(
          K_o = 0.001,
          same_num_neighs_differentiator=lambda MINs, k: min(MINs, key=k)
        ),
        PotentialFieldsExplore(
          K_n=1,
          K_o=1,
          min_force_threshold=0.1
        )
      )
    ) for _ in range(N_mins)
  ]
  """
  """ Line exploration """

  mins = [
    Min(
      max_range,
      DeploymentFSM(
        # NoFollow(),
        # LineExplore(
        #   # RSSI_threshold=0.5,
        #   K_o= 5*1*(i+1),#30,# 12 0.1,#0.01, #12 works somewhat with TWO_DIM_LOCAL, else much lower (0.4-ish)
        #   kind=LineExploreKind.TWO_DIM_LOCAL,
        # )
        NewAttractiveFollow(K_o=.2),
        NewPotentialFieldsExplore(K_o=.2, target_point_or_line=NewPotentialFieldsExplore.Target.LINE)
      ),
      xi_max=1,
      d_perf=0.1,
      d_none=2.5,#2.1,
      delta_expl_angle=np.pi/6#0#np.pi/4#np.pi/6#0#np.pi/6#np.pi/4 #0
    ) for i in range(N_mins)
  ]

  beacons = simulate(dt, mins, scs, env)
  
  fig = plt.figure(figsize=(5,5))
  
  if plot_propterties:
    if _animate:
      # fig, ax = plt.subplots(nrows=3,ncols=1)
      ax1_1 = fig.add_subplot(3,1,1)
      ax1_2 = fig.add_subplot(3,1,2)
      ax1_3 = fig.add_subplot(3,1,3, sharex=ax1_2)
      ax1_1.title.set_text("Deployment")
      ax1_2.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
      ax1_3.title.set_text(r"$\xi$ from neighbors")               #Set title
    else:
      fig2 = plt.figure(figsize=(5,5))
      ax1_1 = fig.add_subplot(1,1,1)
      ax2_1 = fig2.add_subplot(2,1,1)
      ax2_2 = fig2.add_subplot(2,1,2)

      ax1_1.title.set_text("Deployment")
      ax2_1.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
      ax2_2.title.set_text(r"$\xi$ from neighbors")               #Set title2
  else:
    ax = fig.add_subplot(1,1,1)
    # fig, ax = plt.subplots(1,1)
    ax.title.set_text("Deployment")


  if _animate:
    for mn in mins[:start_animation_from_min_ID]:
      if plot_propterties:
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        # mn.plot_vectors(mn.prev, env, ax[0])
        mn.plot_force_traj_line(ax1_2)
        mn.plot_xi_traj_line(ax1_3)
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        # mn.plot_vectors(mn.prev, env, ax[0])
        mn.plot_force_traj_line(ax1_2)
        mn.plot_xi_traj_line(ax1_3)
      else:
        mn.plot(ax)
        mn.plot_vectors(mn.prev, env, ax)

    offset, min_counter = [0], [start_animation_from_min_ID]

    def init():
      if plot_propterties:
        scs.plot(ax1_1)
        env.plot(ax1_1)
        artists = []
        for mn in mins:
          artists += mn.plot(ax1_1)
          artists += (mn.plot_traj_line(ax1_1), ) #Type: Line2D(_line6)
          artists += (mn.plot_force_traj_line(ax1_2), )
          artists += (mn.plot_xi_traj_line(ax1_3), )
          mn.plot_pos_from_pos_traj_index(0)
          mn.plot_force_from_traj_index(0)
          mn.plot_xi_from_traj_index(0)
        if start_animation_from_min_ID == 0:
          ax1_2.legend()  
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
        return  plt_force_traj, plt_xi_traj, plt_pos_traj #,mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0]), mins[min_counter[0]].plot_force_from_traj_index(i-offset[0]) #2
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
      for mn in mins:
        mn.plot(ax1_1)
        mn.plot_traj_line(ax1_1)
        mn.plot_vectors(mn.prev, env, ax1_1)
        mn.plot_force_traj_line(ax2_1)
        mn.plot_xi_traj_line(ax2_2)
      ax2_1.legend()

    else:
      env.plot(ax)
      scs.plot(ax)
      for j in range(len(mins)):#mn in mins:
        mins[j].plot(ax)
        mins[j].plot_traj_line(ax)
        if j == 0:
          mins[j].plot_vectors(scs,env,ax)
        else:
          mins[j].plot_vectors(mins[j-1],env,ax)
      ax.legend()
      ax.axis('equal')

  plt.show()