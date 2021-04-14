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

equal_dist = 1.2267#1.7915
# 1.7915, hypothenus equal to the distance representing RSSI_THRESHOLD_NEIGH
# 12267, three mins form a isosceles(?) triangle (likebeint trekant)

open_ref = [
    np.array([
    [-1.7915, -1.7915],
    [-1.7915,  5.3745],
    [ 5.3745,  5.3745],#[7,    7],
    [ 5.3745,  -1.7915],#[7,    -1],#,
    ]),
]

open_ref2 = [
    np.array([
    [-1*equal_dist, -1*equal_dist],
    [-1*equal_dist,  3*equal_dist],
    [ 3*equal_dist,  3*equal_dist],#[7,    7],
    [ 3*equal_dist, -1*equal_dist],#[7,    -1],#,
    ]),
]

open_ref3 = [
    np.array([
    [-1*equal_dist, -1*equal_dist],
    [-1*equal_dist,            10],
    [ 10,                      10],#[7,    7],
    [ 10,           -1*equal_dist],#[7,    -1],#,
    ]),
]

env = Env(
    np.array([
    0, 0
    ]),
    obstacle_corners = open_ref2#open_ref2#open_large #open_w_sq_obs #open_large#obs_zig_zag#[]#
)

# %%Parameter initialization
max_range = 3
_xi_max = 1
_d_perf = 0.1
_d_none = 2.5
_delta_expl_angle = np.pi/4 #np.pi/6
_K_o = 0.6

N_mins = 8
dt = 0.01

uniformity_list = []
delta_uniformity_list = []

scs = SCS(Beacon.get_ID(), max_range)
scs.insert_into_environment(env)

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

num_rows = num_cols = int(np.sqrt(N_mins+1))

row_counter = 0
col_counter = 1 #SCS is already placed at column 0

for i in range(len(mins)):
    mins[i].pos = np.array([col_counter*equal_dist,row_counter*equal_dist]).reshape(2, )
    print(f"min {mins[i].ID} is positioned at {mins[i].pos}")
    if (col_counter+1)%num_cols == 0:
        row_counter += 1
    if col_counter < (num_cols-1):
        col_counter += 1
    else:
        col_counter = 0
    


# mins[0].pos = np.array([1*equal_dist,0]).reshape(2,)
# mins[1].pos = np.array([2*equal_dist,0]).reshape(2,)
# mins[2].pos = np.array([0*equal_dist,1*equal_dist]).reshape(2,)
# mins[3].pos = np.array([1*equal_dist,1*equal_dist]).reshape(2,)
# mins[4].pos = np.array([2*equal_dist,1*equal_dist]).reshape(2,)#np.array([1.8,0.96]).reshape(2,)#
# mins[5].pos = np.array([0*equal_dist,2*equal_dist]).reshape(2,)#np.array([3.2,0.9]).reshape(2,)#
# mins[6].pos = np.array([1*equal_dist,2*equal_dist]).reshape(2,)
# mins[7].pos = np.array([2*equal_dist,2*equal_dist]).reshape(2,)


# for i in range(len(mins)):
#     mins[i].pos = np.array([i*equal_dist,]).reshape(2, )

fig = plt.figure(figsize=(5,5))
ax_ref = fig.add_subplot(1,1,1)

env.plot(ax_ref)
scs.plot(ax_ref)
uniformity_list.append(scs.calc_uniformity())
beacons = np.array([scs], dtype=object)


for mn in mins:
    mn.heading = 0
    mn.state = MinState.LANDED
    mn.compute_neighbors(beacons)
    beacons = np.append(mn, beacons)
    uniformity_list.append(np.sum([beacon.calc_uniformity() for beacon in beacons]))
    delta_uniformity_list.append(uniformity_list[-1]-uniformity_list[-2])
    # print(f"min {mn.ID} has {len(mn.neighbors)} neighs")

for mn2 in mins:
    # uniformity_list.append(np.sum([beacon.calc_uniformity() for beacon in beacons]))
    mn2.plot(ax_ref)

test = 0
for beacon in beacons:
    test += beacon.calc_uniformity()
    print(f"beacon_id: {beacon.ID}")
    print(f"calc_uniform: {beacon.calc_uniformity()}")

fig_uniformity = plt.figure(figsize=(5,5))
ax_uniformity = fig_uniformity.add_subplot(1,1,1)
ax_uniformity.set(
xlabel = 'Beacons',
ylabel = 'Uniformity',
title = 'Uniformity'
)

plt.xticks(range(len(uniformity_list)+1)) #ints on x-axis
ax_uniformity.plot(uniformity_list)
ax_uniformity.plot(uniformity_list, "or")

fig_delta_uniformity = plt.figure(figsize=(5,5))
ax_delta_uniformity = fig_delta_uniformity.add_subplot(1,1,1)
ax_delta_uniformity.set(
xlabel = 'Beacons',
ylabel = 'Delta uniformity',
title = 'Delta uniformity'
)

plt.xticks(range(len(delta_uniformity_list)+1)) #ints on x-axis
ax_delta_uniformity.plot(delta_uniformity_list)
ax_delta_uniformity.plot(delta_uniformity_list, "or")


print(f"delta_uniformity_list: {delta_uniformity_list}")
plt.show()
print("FINISHED")

