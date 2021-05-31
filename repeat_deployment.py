
from environment import Env

from beacons.beacon import Beacon
from beacons.SCS.scs import SCS
from beacons.MIN.min import Min, MinState

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter 

import timeit
import cProfile, pstats, io
from pstats import SortKey
import json, codecs
from copy import deepcopy

from tqdm import tqdm


_animate, save_animation, plot_propterties = False,False,True#True, True, False

if not _animate or (_animate and save_animation):
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
            "axes.grid": False,
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

folder_path = r'json_files'
file_name = r'\test_1D.json'

file_path = folder_path + file_name
print(f"Reading json-file: '{file_path}'")
obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
print("Finished reading json-file...")
print("Making json_data...")
json_data = json.loads(obj_text)
print("Finishing making json_data...")
obstacle_corners_from_json = [np.array(corner) for corner in json_data['environment'][0]['Obstacle_corners']]
entrance_point_from_json = np.array(json_data['environment'][0]['Entrance_point'])

K_o_from_json = json_data['parameters']['K_o']
max_range_from_json = json_data['parameters']['Max_range']
N_mins_from_json = json_data['parameters']['N_mins']
d_none_from_json = json_data['parameters']['d_none']
d_perf_from_json = json_data['parameters']['d_perf']
try:
    d_tau_from_json = json_data['parameters']['d_tau']
except:
    d_tau_from_json = None
delta_expl_angle_from_json = json_data['parameters']['delta_expl_angle']
xi_max_from_json = json_data['parameters']['xi_max']

scs_from_json = SCS(json_data['beacons'][0]['ID'], max_range_from_json, xi_max=xi_max_from_json, d_perf=d_perf_from_json, d_none=d_none_from_json,d_tau=d_tau_from_json)

env_from_json = Env(
    entrance_point_from_json,
    obstacle_corners = obstacle_corners_from_json
)

start_animation_from_min_ID = 0
stop_min_ID = N_mins_from_json


scs_from_json.insert_into_environment(env_from_json)

mins2 = [
    Min(json_data['beacons'][i+1]['ID'], #i+1 because [0] is the SCS
    max_range_from_json,
    None,
    xi_max=xi_max_from_json,
    d_perf=d_perf_from_json,
    d_none=d_none_from_json,
    d_tau=d_tau_from_json,
    delta_expl_angle=delta_expl_angle_from_json
    ) for i in range(N_mins_from_json)
]


for e in tqdm(range(len(mins2))):
    mins2[e]._pos_traj = np.array(json_data['beacons'][e+1]['pos_traj'])
    mins2[e]._v_traj = np.array(json_data['beacons'][e+1]['force_traj'])
    mins2[e]._heading_traj = np.array(json_data['beacons'][e+1]['heading_traj'])
    mins2[e]._xi_traj = np.array(json_data['beacons'][e+1]['xi_traj'])
    mins2[e].state_traj = [MinState(state[0]) if type(state)==list else MinState(state) for state in json_data['beacons'][e+1]['state_traj']]
    if e != len(mins2)-1:
        mins2[e].tot_vec = np.array(json_data['beacons'][e+1]['vectors']['tot_vec'])
        mins2[e].obs_vec = np.array(json_data['beacons'][e+1]['vectors']['obs_vec'])

    mins2[e].heading = mins2[e]._heading_traj[-1]
    mins2[e].pos = np.array([mins2[e]._pos_traj[0][-1], mins2[e]._pos_traj[1][-1]])
    mins2[e].state = MinState(mins2[e].state_traj[-1])
mins2[-1]._xi_traj = np.array(json_data['beacons'][e+1]['xi_traj'])
mins_to_plot = deepcopy(mins2[:stop_min_ID])

uniformity_list = json_data['uniformity']

fig = plt.figure(figsize=(5,4))
#zig_zag: figsize=(5.3, 3.7)
#open: figsize=(5,5)
#stripa: figsize=(5,4)
fig.canvas.set_window_title('Replay')

if plot_propterties:
    if _animate:
        ax1_1 = fig.add_subplot(3,1,1)
        ax1_2 = fig.add_subplot(3,1,2)
        ax1_3 = fig.add_subplot(3,1,3, sharex=ax1_2)
        ax1_1.title.set_text("Deployment")
        ax1_2.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
        ax1_3.title.set_text(r"$\xi$ from neighbors")
    else:
        fig2 = plt.figure(figsize=(5.2,3))
        fig2.canvas.set_window_title('Replay force')
        fig3 = plt.figure(figsize=(5.2,3))
        fig3.canvas.set_window_title('Replay xi')
        ax1_1 = fig.add_subplot(1,1,1)
        ax2_1 = fig2.add_subplot(1,1,1)
        ax3_1 = fig3.add_subplot(1,1,1)


        ax1_1.grid(False)
        ax2_1.title.set_text(r"$\left\|\| F_{applied} \right\|\|$") #Set title
        ax3_1.title.set_text(r"$\xi$ from neighbors") 
else:
    ax = fig.add_subplot(1,1,1)


if _animate:
    for mn in mins_to_plot[:start_animation_from_min_ID]:
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

    offset, min_counter = [0], [start_animation_from_min_ID]

    def init():
        if plot_propterties:
            scs_from_json.plot(ax1_1)
            env_from_json.plot(ax1_1)
            artists = []
            for mn in mins_to_plot:
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
            scs_from_json.plot(ax)
            env_from_json.plot(ax)
            artists = []
            for mn in mins_to_plot:
                artists += mn.plot(ax)
                artists += (mn.plot_traj_line(ax), )
                mn.plot_pos_from_pos_traj_index(0)
            return artists

    def animate(i):
        try:
            if i - offset[0] >= mins_to_plot[min_counter[0]].get_pos_traj_length():
                offset[0] += mins_to_plot[min_counter[0]].get_pos_traj_length()
                min_counter[0] += 1
            if plot_propterties:
                plt_pos_traj = mins_to_plot[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
                plt_force_traj = mins_to_plot[min_counter[0]].plot_force_from_traj_index(i-offset[0])
                plt_xi_traj = mins_to_plot[min_counter[0]].plot_xi_from_traj_index(i-offset[0])
                return  plt_force_traj, plt_xi_traj, plt_pos_traj
            else:
                plt_pos_traj =  mins_to_plot[min_counter[0]].plot_pos_from_pos_traj_index(i - offset[0])
                return plt_pos_traj
        except:
            print("Animation finished")
    
    _save_count = 2000
    anim = FuncAnimation(fig, animate, init_func=init, interval=2, blit=False, save_count=_save_count)
    
    if save_animation:
        writergif = PillowWriter(fps=60)

        animation_name_gif = "animation_test.gif"
        print("Saving animation. Depending on the choise of 'save_count' this might take some time")
        print(f"Chosen 'save_count' = {_save_count}")
        anim.save(animation_name_gif,writer=writergif)   
        print(f"Animation saved to {animation_name_gif}")
else:
    if plot_propterties:
        env_from_json.plot(ax1_1)
        scs_from_json.plot(ax1_1)
        for mn in mins_to_plot:
            mn.plot(ax1_1)
            mn.plot_traj_line(ax1_1)
            mn.plot_force_traj_line(ax2_1)
        mins_to_plot[-1].plot_xi_traj_line(ax3_1)
        
        ax2_1.legend(ncol=1, prop={'size': 9}, handlelength=1, bbox_to_anchor=(1.13,1), borderaxespad=0)
        ax3_1.legend(ncol=1, prop={'size': 9}, handlelength=1, bbox_to_anchor=(1.13,1), borderaxespad=0)


    else:
        env_from_json.plot(ax)
        scs_from_json.plot(ax)
        
        for j in range(len(mins_to_plot)):
            mins_to_plot[j].plot(ax)
            mins_to_plot[j].plot_traj_line(ax)
        
        ax.grid(False)

fig_uniformity = plt.figure(figsize=(5.2,3))
fig_uniformity.canvas.set_window_title('Replay uniformity')

ax_uniformity = fig_uniformity.add_subplot(1,1,1)
ax_uniformity.set(
    xlabel = '# of deployed agents',
    title = 'Uniformity'
)
ax_uniformity.set_xticks(range(0,len(uniformity_list[:stop_min_ID])+1,10))
ax_uniformity.plot(uniformity_list[:stop_min_ID+1])
ax_uniformity.plot(uniformity_list[:stop_min_ID+1], "or", markersize=2)

if not save_animation:
    plt.show()