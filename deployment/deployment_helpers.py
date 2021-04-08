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

def get_generic_force_vector(vecs, gain, sigma_x=1, sigma_y=1, d_o=1): #TODO: Add force threshold,
    # try:
    #     mat = np.concatenate(vecs, axis=1)
    #     F= -gain*np.sum(mat/np.linalg.norm(mat, axis=0)**3, axis=1)
    #     return -gain*np.sum(mat/np.linalg.norm(mat, axis=0)**3, axis=1)
    # except ValueError:
    #     return np.zeros((2, )) 
    try:
        mat = np.vstack(vecs)#np.concatenate(vecs, axis=1) #inneholder 2 arrays, x-coords in [0] and y-coords in [1]

        """Exponential force"""
        # first_x = -2*(gain)/sigma_x**2 * mat[0,:] #vec
        # first_y = -2*(gain)/sigma_y**2 * mat[1,:] #vec
        # first = np.vstack((first_x, first_y))

        # exponent_x = mat[0,:]**2/sigma_x**2
        # exponent_y = mat[1,:]**2/sigma_y**2
        # exponent = -(exponent_x + exponent_y)

        # tot = first*np.e**exponent
        # exponential_force = np.sum(tot,axis=1)
        first_x = -2*(gain)/sigma_x**2 * mat[:,0] #vec
        first_y = -2*(gain)/sigma_y**2 * mat[:,1] #vec
        first = np.vstack((first_x, first_y))

        exponent_x = mat[:,0]**2/sigma_x**2
        exponent_y = mat[:,1]**2/sigma_y**2
        exponent = -(exponent_x + exponent_y)

        tot = first*np.e**exponent
        exponential_force = np.sum(tot,axis=1)
        
        """Reciprocal force"""
        within_range_mat = []
        for i in range(len(mat)):
            vec = mat[i]
            if np.linalg.norm(vec) < d_o:
                within_range_mat.append(vec)
        #within_range_mat = [m for m in mat if np.linalg.norm(m,axis=0) < d_o]
        reciprocal_force = -gain*np.sum(mat/np.linalg.norm(mat, axis=0)**3, axis=1)
        # print(f"reciprocal_force: {reciprocal_force}")
        #reciprocal_force = -gain*np.sum(mat/np.linalg.norm(mat, axis=0)**3*(1/np.linalg.norm(mat, axis=0) - 1/d_o), axis=1)
        #the below line is correctly calculated
        test = [-gain*vec/np.linalg.norm(vec,axis=0)**3*(1/np.linalg.norm(vec,axis=0)-1/d_o) for vec in within_range_mat]
        
        test_sum = np.sum(test,axis=0)
        reciprocal_force_within_range = test_sum#-gain*np.sum(within_range_mat/np.linalg.norm(within_range_mat, axis=1)**3*(1/np.linalg.norm(mat, axis=0) - 1/d_o), axis=1)
        """Quadratic force"""
        quadratic_force = -1*gain/5*np.sum(mat,axis=1)
        # print(f"quad_force: {quadratic_force}")
        # print(f"np.linalg.norm(reciprocal_force): {np.linalg.norm(reciprocal_force)}")
        # print(f"np.linalg.norm(reciprocal_force_within_range): {np.linalg.norm(reciprocal_force_within_range)}")
        # print("---------------------------------------------------------------------")
        
        # print(f"reciprocal_force: {reciprocal_force}")
        return reciprocal_force_within_range#reciprocal_force ##reciprocal_force_within_range#quadratic_force#reciprocal_force#exponential_force#exponential_force
    except:
        # e = sys.exc_info()[0]
        # print(f"Error: {e}")
        return np.zeros((2, ))