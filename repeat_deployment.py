
from environment import Env

from beacons.beacon import Beacon
from beacons.SCS.scs import SCS
from beacons.MIN.min import Min, MinState

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import timeit
import cProfile, pstats, io
from pstats import SortKey
import json, codecs
import jsons


_animate, save_animation, plot_propterties = True, False, False
start_animation_from_min_ID = 0
if not _animate:
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
file_path = 'data_from_deployment_1.json'
obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
json_data = json.loads(obj_text)

obstacle_corners2 = [np.array(corner) for corner in json_data['environment'][0]['Obstacle_corners']]
entrance_point2 = np.array(json_data['environment'][0]['Entrance_point'])

env2 = Env(
    entrance_point2,
    obstacle_corners = obstacle_corners2
)

K_o2 = json_data['parameters']['K_o']
max_range2 = json_data['parameters']['Max_range']
N_mins2 = json_data['parameters']['N_mins']
d_none2 = json_data['parameters']['d_none']
d_perf2 = json_data['parameters']['d_perf']
delta_expl_angle2 = json_data['parameters']['delta_expl_angle']
xi_max2 = json_data['parameters']['xi_max']


scs2 = SCS(json_data['beacons'][0]['ID'],max_range2)
scs2.insert_into_environment(env2)

mins2 = [
    Min(json_data['beacons'][i+1]['ID'],
    max_range2,
    None,
    xi_max=xi_max2,
    d_perf=d_perf2,
    d_none=d_none2,
    delta_expl_angle=delta_expl_angle2
    ) for i in range(N_mins2)
]

for e in range(len(mins2)):
    mins2[e]._pos_traj = np.array(json_data['beacons'][e+1]['pos_traj'])
    mins2[e]._v_traj = np.array(json_data['beacons'][e+1]['force_traj'])
    mins2[e]._heading_traj = np.array(json_data['beacons'][e+1]['heading_traj'])
    mins2[e]._xi_traj = np.array(json_data['beacons'][e+1]['xi_traj'])
    mins2[e].state_traj = [MinState(state[0]) if type(state)==list else MinState(state) for state in json_data['beacons'][e+1]['state_traj']]#np.array(json_data['beacons'][e+1]['state_traj'])#[MinState(mstate_int[0]) for mstate_int in np.array(json_data['beacons'][e+1]['state_traj'])]#np.array(json_data['beacons'][e+1]['state_traj'])

    mins2[e].heading = mins2[e]._heading_traj[-1]
    mins2[e].pos = np.array([mins2[e]._pos_traj[0][-1], mins2[e]._pos_traj[1][-1]])
    mins2[e].state = MinState(mins2[e].state_traj[-1])
    t = 2

fig = plt.figure(figsize=(5,5))

if plot_propterties:
    if _animate:
        ax1_1 = fig.add_subplot(3,1,1)
        ax1_2 = fig.add_subplot(3,1,2)
        ax1_3 = fig.add_subplot(3,1,3, sharex=ax1_2)
        ax1_1.title.set_text("Deployment")
        ax1_2.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
        ax1_3.title.set_text(r"$\xi$ from neighbors")
    else:
        fig2 = plt.figure(figsize=(5,5))
        ax1_1 = fig.add_subplot(1,1,1)
        ax2_1 = fig2.add_subplot(2,1,1)
        ax2_2 = fig2.add_subplot(2,1,2)

        ax1_1.title.set_text("Deployment")
        ax2_1.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
        ax2_2.title.set_text(r"$\xi$ from neighbors") 
else:
    ax = fig.add_subplot(1,1,1)
    # fig, ax = plt.subplots(1,1)
    ax.title.set_text("Deployment")

# fig3 = plt.figure(figsize=(5,5))
# ax3 = fig3.add_subplot(1,1,1)


if _animate:
    for mn in mins2[:start_animation_from_min_ID]:
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
            mn.plot_vectors(mn.prev, env2, ax)

    offset, min_counter = [0], [start_animation_from_min_ID]

    def init():
        if plot_propterties:
            scs2.plot(ax1_1)
            env2.plot(ax1_1)
            artists = []
            for mn in mins2:
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
            scs2.plot(ax)
            env2.plot(ax)
            artists = []
            for mn in mins2:
                artists += mn.plot(ax)
                artists += (mn.plot_traj_line(ax), )
                mn.plot_pos_from_pos_traj_index(0)
            return artists

    def animate(i):
        if i - offset[0] >= mins2[min_counter[0]].get_pos_traj_length():
            offset[0] += mins2[min_counter[0]].get_pos_traj_length()
            min_counter[0] += 1
        if plot_propterties:
            plt_pos_traj = mins2[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
            plt_force_traj = mins2[min_counter[0]].plot_force_from_traj_index(i-offset[0])
            plt_xi_traj = mins2[min_counter[0]].plot_xi_from_traj_index(i-offset[0])
            return  plt_force_traj, plt_xi_traj, plt_pos_traj #,mins[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0]), mins[min_counter[0]].plot_force_from_traj_index(i-offset[0]) #2
        else:
            plt_pos_traj = mins2[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
            return plt_pos_traj

    anim = FuncAnimation(fig, animate, init_func=init, interval=2, blit=False)
    
    if save_animation:
        animation_name = "animation.gif"
        print("Saving animation")
        anim.save(animation_name)   
        print(f"Animation saved to {animation_name}")
else:
    if plot_propterties:
        env2.plot(ax1_1)
        scs2.plot(ax1_1)
        for mn in mins2:
            mn.plot(ax1_1)
            mn.plot_traj_line(ax1_1)
            mn.plot_vectors(mn.prev, env2, ax1_1)
            mn.plot_force_traj_line(ax2_1)
            mn.plot_xi_traj_line(ax2_2)
        ax2_1.legend()
    else:
        env2.plot(ax)
        scs2.plot(ax)
        
        for j in range(len(mins2)):#mn in mins:
            mins2[j].plot(ax)
            mins2[j].plot_traj_line(ax)
        if j == 0:
            mins2[j].plot_vectors(scs2,env2,ax)
        else:
            mins2[j].plot_vectors(mins2[j-1],env2,ax)
        ax.legend()
        ax.axis('equal')

plt.show()