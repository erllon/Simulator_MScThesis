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

from plot_fields import FieldPlotter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from helpers import polar_to_vec as p2v

import timeit
import cProfile, pstats, io
from pstats import SortKey
import json, codecs
import jsons

data = {}
data['parameters'] = []
data['environment'] = []
data['beacons'] = []

def simulate(dt, mins, scs, env):
  pr = cProfile.Profile()
  pr.enable()



  scs.insert_into_environment(env)
  beacons = np.array([scs], dtype=object)
  data['beacons'].append(scs.toJson())
  # data['beacons'].append({
  #       'Type': 'SCS',
  #       'ID': '0',
  #       # 'Pathtree': [0],
  #       'pos_traj': np.array([0,0]).reshape(2,1).tolist(),
  #       'force_traj': np.array([0,0]).reshape(2,1).tolist(),
  #       'heading_traj': np.array([0,0]).reshape(2,1).tolist(),
  #       'xi_traj': np.array([0]).tolist(),
  #       #Vectors and stuff
  #   })

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
    data['beacons'].append(mins[i].toJson())
    # jsons_test = jsons.dump(mins[i])
  

    # data['beacons'].append({
    #     'Type': 'MIN',
    #     'ID': mins[i].ID,
    #     # 'Pathtree': mins[i].pa,
    #     'pos_traj': mins[i]._pos_traj.tolist(),#np.array([np.array([1,2]).reshape(2,1),np.array([3,4]).reshape(2,1)]).tolist(),
    #     'force_traj': mins[i]._v_traj.tolist(),#np.array([np.array([7,8]).reshape(2,1),np.array([9,10]).reshape(2,1)]).tolist(),
    #     'heading_traj': mins[i]._heading_traj.tolist(),#np.array([np.array([5,5]).reshape(2,1),np.array([6,6]).reshape(2,1)]).tolist(),
    #     'xi_traj': mins[i]._xi_traj.tolist(),#np.array([1,2,3,4,5,6]).tolist()
    #     #Vectors and stuff
    # })


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
  file_path = 'data_1.json'
  # data = {}
  # data['beacons'] = []
  # for i in range(len(beacons)):
  #   data['beacons'].append(beacons[i].toJson())
  # with open('data_1.json', 'w') as outfile:
    # json.dump(scs.toJson(), outfile, separators=(',', ':'), sort_keys=True, indent=2)
  for i in range(len(beacons)):
    json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=2)
    # jsons.dump(beacons,outfile)
  obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
  b_new = json.loads(obj_text)
  # s = io.StringIO()
  # sortby = SortKey.CUMULATIVE
  # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
  # ps.print_stats()
  # print(s.getvalue())
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
# # %% Environment initialization
#   obstacle_corners_1D = [
#       np.array([
#         [-10, -10],
#         [ -6, -10],
#       ]),
#     ]

#   obs_zig_zag = [
#       np.array([
#         [-1, -1],
#         [-1,   10],
#         [15,     10],
#         [15,     -1],
#       ]),
#       np.array([
#         [4.0,  5],
#         [4.0, -1],
#         [5, -1],
#         [5,  5]
#       ]),
#       np.array([
#         [9.0,  10],
#         [9.0, 4.9],
#         [10.0, 4.9],
#         [10.0,  10]
#       ])
#     ]
#   open_small = [
#       np.array([
#         [-1, -1],
#         [-1,   5],
#         [5,      5],
#         [5,     -1],
#       ]),
#     ]
  
#   open_large = [
#       np.array([
#         [-1, -1],
#         [-1,   10],
#         [10,      10],
#         [10,     -1],
#       ]),
#     ]

#   open_w_sq_obs = [
#       np.array([
#         [-1, -1],
#         [-1,   12],
#         [12,     12],
#         [12,     -1],
#       ]),
#       np.array([
#         [2, 2],
#         [2, 9],
#         [9, 9],
#         [9, 2]
#       ])      
#     ]

#   env = Env(
#     # np.array([
#     #   -9.8, -5.2
#     # ]),
#     np.array([
#       0, 0
#     ]),
#     obstacle_corners = open_large#open_w_sq_obs #open_large#open_small#obs_zig_zag#open_w_sq_obs#open_large##open_small#[]#obs_zig_zag #[]
#   )
#   data['environment'].append(env.toJson())

# # %%Parameter initialization
#   _animate, save_animation, plot_propterties, replay = False, False, False, True
#   start_animation_from_min_ID = 0

#   max_range = 3 #0.51083#float(-np.log(-0.6))#3 #0.75    0.51083

#   N_mins = 6#18#7#2*5#3
#   dt = 0.01#0.01

#   scs = SCS(Beacon.get_ID(), max_range)
#   """ Potential fields exploration
#   mins = [
#     Min(
#       max_range,
#       DeploymentFSM(
#         AttractiveFollow(
#           K_o = 0.001,
#           same_num_neighs_differentiator=lambda MINs, k: min(MINs, key=k)
#         ),
#         PotentialFieldsExplore(
#           K_n=1,
#           K_o=1,
#           min_force_threshold=0.1
#         )
#       )
#     ) for _ in range(N_mins)
#   ]
#   """
#   """ Line exploration """

#   mins = [
#     Min(
#       Beacon.get_ID(),
#       max_range,
#       DeploymentFSM(
#         # NoFollow(),
#         # LineExplore(
#         #   # RSSI_threshold=0.5,
#         #   K_o= 5*1*(i+1),#30,# 12 0.1,#0.01, #12 works somewhat with TWO_DIM_LOCAL, else much lower (0.4-ish)
#         #   kind=LineExploreKind.TWO_DIM_LOCAL,
#         # )
#         NewAttractiveFollow(K_o=.3),
#         NewPotentialFieldsExplore(K_o=.3, target_point_or_line=NewPotentialFieldsExplore.Target.LINE)
#       ),
#       xi_max=1,
#       d_perf=0.1,
#       d_none=2.5,#2.1,
#       delta_expl_angle=np.pi/4#0#np.pi/4#np.pi/6#0#np.pi/6#np.pi/4 #0
#     ) for i in range(N_mins)
#   ]

#   data['parameters'] = {
#     'N_mins': N_mins,
#     'Max_range' : max_range,
#     'K_o': 0.3,
#     'xi_max': 1,
#     'd_perf': 0.1,
#     'd_none': 2.5,
#     'delta_expl_angle': np.pi/4
#   }

#   beacons = simulate(dt, mins, scs, env)

  
#   fig = plt.figure(figsize=(5,5))
  
#   if plot_propterties:
#     if _animate:
#       # fig, ax = plt.subplots(nrows=3,ncols=1)
#       ax1_1 = fig.add_subplot(3,1,1)
#       ax1_2 = fig.add_subplot(3,1,2)
#       ax1_3 = fig.add_subplot(3,1,3, sharex=ax1_2)
#       ax1_1.title.set_text("Deployment")
#       ax1_2.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
#       ax1_3.title.set_text(r"$\xi$ from neighbors")               #Set title
#     else:
#       fig2 = plt.figure(figsize=(5,5))
#       ax1_1 = fig.add_subplot(1,1,1)
#       ax2_1 = fig2.add_subplot(2,1,1)
#       ax2_2 = fig2.add_subplot(2,1,2)

#       ax1_1.title.set_text("Deployment")
#       ax2_1.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
#       ax2_2.title.set_text(r"$\xi$ from neighbors")               #Set title2
#   else:
#     ax = fig.add_subplot(1,1,1)
#     # fig, ax = plt.subplots(1,1)
#     ax.title.set_text("Deployment")


#   if _animate:
#     for mn in mins[:start_animation_from_min_ID]:
#       if plot_propterties:
#         mn.plot(ax1_1)
#         mn.plot_traj_line(ax1_1)
#         # mn.plot_vectors(mn.prev, env, ax[0])
#         mn.plot_force_traj_line(ax1_2)
#         mn.plot_xi_traj_line(ax1_3)
#         mn.plot(ax1_1)
#         mn.plot_traj_line(ax1_1)
#         # mn.plot_vectors(mn.prev, env, ax[0])
#         mn.plot_force_traj_line(ax1_2)
#         mn.plot_xi_traj_line(ax1_3)
#       else:
#         mn.plot(ax)
#         mn.plot_vectors(mn.prev, env, ax)

#     offset, min_counter = [0], [start_animation_from_min_ID]

#     def init():
#       if plot_propterties:
#         scs.plot(ax1_1)
#         env.plot(ax1_1)
#         artists = []
#         for mn in mins:
#           artists += mn.plot(ax1_1)
#           artists += (mn.plot_traj_line(ax1_1), ) #Type: Line2D(_line6)
#           artists += (mn.plot_force_traj_line(ax1_2), )
#           artists += (mn.plot_xi_traj_line(ax1_3), )
#           mn.plot_pos_from_pos_traj_index(0)
#           mn.plot_force_from_traj_index(0)
#           mn.plot_xi_from_traj_index(0)
#         if start_animation_from_min_ID == 0:
#           ax1_2.legend()  
#       else:
#         scs.plot(ax)
#         env.plot(ax)
#         artists = []
#         for mn in mins:
#           artists += mn.plot(ax)
#           artists += (mn.plot_traj_line(ax), )
#           mn.plot_pos_from_pos_traj_index(0)
#       return artists

#     def animate(i):
#       if i - offset[0] >= mins[min_counter[0]].get_pos_traj_length():
#         offset[0] += mins[min_counter[0]].get_pos_traj_length()
#         min_counter[0] += 1
#       if plot_propterties:
#         plt_pos_traj = mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
#         plt_force_traj = mins[min_counter[0]].plot_force_from_traj_index(i-offset[0])
#         plt_xi_traj = mins[min_counter[0]].plot_xi_from_traj_index(i-offset[0])
#         return  plt_force_traj, plt_xi_traj, plt_pos_traj #,mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0]), mins[min_counter[0]].plot_force_from_traj_index(i-offset[0]) #2
#       else:
#         plt_pos_traj = mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
#         return plt_pos_traj
  
#     anim = FuncAnimation(fig, animate, init_func=init, interval=2, blit=False)
    
#     if save_animation:
#       animation_name = "animation.gif"
#       print("Saving animation")
#       anim.save(animation_name)
#       print(f"Animation saved to {animation_name}")
#   else:
#     if plot_propterties:
#       env.plot(ax1_1)
#       scs.plot(ax1_1)
#       for mn in mins:
#         mn.plot(ax1_1)
#         mn.plot_traj_line(ax1_1)
#         mn.plot_vectors(mn.prev, env, ax1_1)
#         mn.plot_force_traj_line(ax2_1)
#         mn.plot_xi_traj_line(ax2_2)
#       ax2_1.legend()

#     else:
#       env.plot(ax)
#       scs.plot(ax)
#       for j in range(len(mins)):#mn in mins:
#         mins[j].plot(ax)
#         mins[j].plot_traj_line(ax)
#         if j == 0:
#           mins[j].plot_vectors(scs,env,ax)
#         else:
#           mins[j].plot_vectors(mins[j-1],env,ax)
#       ax.legend()
#       ax.axis('equal')

#   # plt.show()
  _animate, save_animation, replay = True, False, True
  start_animation_from_min_ID = 0
  if replay:
    obj_text = codecs.open('data_1.json', 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)

    obstacle_corners2 = [np.array(corner) for corner in b_new['environment'][0]['Obstacle_corners']]
    entrance_point2 = np.array(b_new['environment'][0]['Entrance_point'])
    
    env2 = Env(
      entrance_point2,
      obstacle_corners = obstacle_corners2
    )

    K_o2 = b_new['parameters']['K_o']
    max_range2 = b_new['parameters']['Max_range']
    N_mins2 = b_new['parameters']['N_mins']
    d_none2 = b_new['parameters']['d_none']
    d_perf2 = b_new['parameters']['d_perf']
    delta_expl_angle2 = b_new['parameters']['delta_expl_angle']
    xi_max2 = b_new['parameters']['xi_max']


    scs2 = SCS(b_new['beacons'][0]['ID'],max_range2)
    scs2.insert_into_environment(env2)

    mins2 = [
      Min(b_new['beacons'][i+1]['ID'],
        max_range2,
        None,
        # DeploymentFSM(
        #   # NoFollow(),
        #   # LineExplore(
        #   #   # RSSI_threshold=0.5,
        #   #   K_o= 5*1*(i+1),#30,# 12 0.1,#0.01, #12 works somewhat with TWO_DIM_LOCAL, else much lower (0.4-ish)
        #   #   kind=LineExploreKind.TWO_DIM_LOCAL,
        #   # )
        #   NewAttractiveFollow(K_o=.3),
        #   NewPotentialFieldsExplore(K_o=.3, target_point_or_line=NewPotentialFieldsExplore.Target.LINE)
        # ),
        xi_max=xi_max2,
        d_perf=d_perf2,
        d_none=d_none2,#2.1,
        delta_expl_angle=delta_expl_angle2#np.pi/4#0#np.pi/4#np.pi/6#0#np.pi/6#np.pi/4 #0
      ) for i in range(N_mins2)
    ]

    for e in range(len(mins2)):
      mins2[e]._pos_traj = np.array(b_new['beacons'][e+1]['pos_traj'])
      mins2[e]._v_traj = np.array(b_new['beacons'][e+1]['force_traj'])
      mins2[e]._heading_traj = np.array(b_new['beacons'][e+1]['heading_traj'])
      mins2[e]._xi_traj = np.array(b_new['beacons'][e+1]['xi_traj'])
      mins2[e].state_traj = [MinState(state[0]) if type(state)==list else MinState(state) for state in b_new['beacons'][e+1]['state_traj']]#np.array(b_new['beacons'][e+1]['state_traj'])#[MinState(mstate_int[0]) for mstate_int in np.array(b_new['beacons'][e+1]['state_traj'])]#np.array(b_new['beacons'][e+1]['state_traj'])
      
      mins2[e].heading = mins2[e]._heading_traj[-1]
      mins2[e].pos = np.array([mins2[e]._pos_traj[0][-1], mins2[e]._pos_traj[1][-1]])
      mins2[e].state = MinState(mins2[e].state_traj[-1])
      t = 2

    fig3 = plt.figure(figsize=(5,5))
    ax3 = fig3.add_subplot(1,1,1)
    

    if _animate:
      for mn in mins2[:start_animation_from_min_ID]:
        # if plot_propterties:
        #   mn.plot(ax1_1)
        #   mn.plot_traj_line(ax1_1)
        #   # mn.plot_vectors(mn.prev, env, ax[0])
        #   mn.plot_force_traj_line(ax1_2)
        #   mn.plot_xi_traj_line(ax1_3)
        #   mn.plot(ax1_1)
        #   mn.plot_traj_line(ax1_1)
        #   # mn.plot_vectors(mn.prev, env, ax[0])
        #   mn.plot_force_traj_line(ax1_2)
        #   mn.plot_xi_traj_line(ax1_3)
        # else:
        mn.plot(ax3)
        mn.plot_vectors(mn.prev, env2, ax3)

      offset, min_counter = [0], [start_animation_from_min_ID]

      def init():
        # if plot_propterties:
          # scs.plot(ax1_1)
          # env.plot(ax1_1)
          # artists = []
          # for mn in mins:
          #   artists += mn.plot(ax1_1)
          #   artists += (mn.plot_traj_line(ax1_1), ) #Type: Line2D(_line6)
          #   artists += (mn.plot_force_traj_line(ax1_2), )
          #   artists += (mn.plot_xi_traj_line(ax1_3), )
          #   mn.plot_pos_from_pos_traj_index(0)
          #   mn.plot_force_from_traj_index(0)
          #   mn.plot_xi_from_traj_index(0)
          # if start_animation_from_min_ID == 0:
          #   ax1_2.legend()  
        # else:
        scs2.plot(ax3)
        env2.plot(ax3)
        artists = []
        for mn in mins2:
          artists += mn.plot(ax3)
          artists += (mn.plot_traj_line(ax3), )
          mn.plot_pos_from_pos_traj_index(0)
        return artists

      def animate(i):
        if i - offset[0] >= mins2[min_counter[0]].get_pos_traj_length():
          offset[0] += mins2[min_counter[0]].get_pos_traj_length()
          min_counter[0] += 1
        # if plot_propterties:
        #   plt_pos_traj = mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
        #   plt_force_traj = mins[min_counter[0]].plot_force_from_traj_index(i-offset[0])
        #   plt_xi_traj = mins[min_counter[0]].plot_xi_from_traj_index(i-offset[0])
        #   return  plt_force_traj, plt_xi_traj, plt_pos_traj #,mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0]), mins[min_counter[0]].plot_force_from_traj_index(i-offset[0]) #2
        # else:
        plt_pos_traj = mins2[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
        return plt_pos_traj
    
      anim = FuncAnimation(fig3, animate, init_func=init, interval=2, blit=False)
      
      if save_animation:
        animation_name = "animation.gif"
        print("Saving animation")
        anim.save(animation_name)
        print(f"Animation saved to {animation_name}")
    else:
      scs2.plot(ax3)
      env2.plot(ax3)
      for j in range(len(mins2)):#mn in mins:
        
        mins2[j].plot(ax3)
        mins2[j].plot_traj_line(ax3)
        if j == 0:
          mins2[j].plot_vectors(scs2,env2,ax3)
        else:
          mins2[j].plot_vectors(mins2[j-1],env2,ax3)
      ax3.legend()
      ax3.axis('equal')

    # else:
    #   env.plot(ax)
    #   scs.plot(ax)
    #   for j in range(len(mins)):#mn in mins:
    #     mins[j].plot(ax)
    #     mins[j].plot_traj_line(ax)
    #     if j == 0:
    #       mins[j].plot_vectors(scs,env,ax)
    #     else:
    #       mins[j].plot_vectors(mins[j-1],env,ax)
    #   ax.legend()
    #   ax.axis('equal')



  test = 2
  plt.show()