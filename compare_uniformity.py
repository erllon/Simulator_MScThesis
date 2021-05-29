
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
num_of_runs = 100
global_list = []
for i in tqdm(range(num_of_runs)):
    folder_path = r'..\large-json-files\Uniformity_comp_small_sR\Correct'
    file_name = r'\correct_avg_unif_comp_small_rs_15_drones_' + str(i+1) + r'.json'

    file_path = folder_path + file_name #r'json_files\unif_comp_26.json' #
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    json_data = json.loads(obj_text)

    uniformity_list = json_data['uniformity']
    global_list.append(uniformity_list)

t = 2
average_uniformity = np.sum(global_list,axis=0)/num_of_runs

fig = plt.figure(figsize=(5.2,3)) #plt.figure(figsize=(5,5)) #
fig.canvas.set_window_title(rf'Avg Uniformity, $num\_runs = {num_of_runs}$')

ax = fig.add_subplot(1,1,1)
ax.set(
    xlabel = '# of deployed agents',#'Beacons',
    # ylabel = 'Uniformity',
    title = 'Uniformity'
)

plt.xticks(range(len(average_uniformity)+1)) #ints on x-axis
ax.plot(average_uniformity)
ax.plot(average_uniformity, "or", markersize=2)
# e = [i for i in range(len(average_uniformity))]
# ee = average_uniformity
# ax.bar([i for i in range(len(average_uniformity))], [u for u in average_uniformity], 0.1)
# ax.plot(average_uniformity, "or", markersize=4)
print(f"Final avg uniformity: {average_uniformity[-1]}")
plt.show()