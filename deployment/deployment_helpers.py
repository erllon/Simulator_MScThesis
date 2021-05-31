import numpy as np
from helpers import rot_z_mat as R_z, polar_to_vec as p2v
import sys

def get_neighbor_forces(K_n, MIN):
    vecs_to_neighs = [
        MIN.get_vec_to_other(n).reshape(2, 1) for n in MIN.neighbors if not (MIN.get_vec_to_other(n) == 0).all()
    ]
    return get_generic_force_vector(vecs_to_neighs, K_n, d_o=MIN.range)

def get_obstacle_forces(K_o, MIN, ENV):
    for s in MIN.sensors:
        s.sense(ENV)
    vecs_to_obs = [
        (R_z(MIN.heading)@R_z(s.host_relative_angle))[:2,:2]@p2v(s.measurement.get_val(),s.measurement.get_angle())
        for s in MIN.sensors if s.measurement.is_valid()
    ]
    return get_generic_force_vector(vecs_to_obs, K_o, d_o=MIN.range)

def get_generic_force_vector(vecs, gain, sigma_x=1, sigma_y=1, d_o=1):
    try:
        mat = np.vstack(vecs)
        """Reciprocal force within range"""
        within_range_mat = [vec for vec in mat if np.linalg.norm(vec) < d_o]
        individual_recip_force_within_range = [-gain*vec/np.linalg.norm(vec,axis=0)**3*(1/np.linalg.norm(vec,axis=0)-1/d_o) for vec in within_range_mat]        
        reciprocal_force_within_range = np.sum(individual_recip_force_within_range,axis=0)
        
        return reciprocal_force_within_range
    except:
        # e = sys.exc_info()[0]
        # print(f"Error: {e}")
        return np.zeros((2, ))