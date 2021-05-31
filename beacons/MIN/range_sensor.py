import numpy as np
from helpers import (
    rot_z_mat as R_z,
    polar_to_vec as p2v
    )

class RangeReading():
    def __init__(self, measured_range, measured_angle=None):
        self.__measured_range = measured_range
        self.__measured_angle = measured_angle
        self.__is_valid = measured_range != np.inf
    
    def get_val(self):
        return self.__measured_range
    
    def get_angle(self):
        return self.__measured_angle

    def is_valid(self):
        return self.__is_valid

    def __str__(self):
        return f"Range reading: {self.__measured_range} [m], angle: {self.__measured_angle}"

class RangeSensor():

    def __init__(self, max_range):
        self.max_range = max_range
        self.measurement_dist = None
        self.measurement_angle = None

    def mount(self, host, angle_deg):
        self.host = host
        self.host_relative_angle = np.deg2rad(angle_deg)
        host.sensors.append(self)
        self.rot_mat = R_z(angle_deg)

    def sense(self, environment):
        if environment.obstacle_corners == []:
            self.measurement = RangeReading(np.inf)
        else:
            measurement_from_sensors = [np.array(self.__sense_aux(corners)) for corners in environment.obstacle_corners]
            
            min_measurement = min(measurement_from_sensors, key=lambda x:x[0])
            self.measurement = RangeReading(min_measurement[0], min_measurement[1])


    def __sense_aux(self, corners):
        """Computes the distance to an obstacle

        Args:
            corners ndarray: n-by-2 array representing the n corners of an obstacle as (x,  y)-coords. It is assumed that
            there is a line between the first and last corner defined in the array.

        Returns:
            float: distance along sensor-frame x-axis to nearest obstacle (inf if no obstacle is within range)
        """
        
        valid_crossings_dict = {
            "lengths" : np.array([]),
            "angles"  : np.array([])
        }
        valid_crossings = np.array([np.inf])
        closed_corners = np.vstack((corners, corners[0, :])) #Contains all the line segments that defines an obstacle

        num_of_rays = 5 # 5 rays total, 2 pairs + "0-angle"
        fov_angle = np.deg2rad(27) #Total field-of-view
        start_ang = -fov_angle/2.0
        if num_of_rays == 1:
            delta_ang = 0
        else:
            delta_ang = fov_angle/(num_of_rays-1)
        
        for i in range(num_of_rays):
            current_ray_angle = start_ang + i*delta_ang

            A_1 = p2v(1, self.host_relative_angle + self.host.heading + current_ray_angle).reshape(2, 1)
            max_t = np.array([self.max_range, 1])

            for j in np.arange(corners.shape[0]):
                x1, x2 = closed_corners[j, :], closed_corners[j+1, :]

                A = np.hstack((A_1, (x1-x2).reshape(2, 1)))

                b = x1 - self.host.pos
                try:
                    t = np.linalg.solve(A, b)
                    if (t >= 0).all() and (t <= max_t).all():
                        valid_crossings = np.hstack((valid_crossings, np.linalg.norm(t)))
                        valid_crossings_dict["lengths"] = np.hstack((valid_crossings_dict["lengths"], np.linalg.norm(t)))
                        valid_crossings_dict["angles"] = np.hstack((valid_crossings_dict["angles"], current_ray_angle))
                except np.linalg.LinAlgError:
                    pass
        
        if len(valid_crossings_dict['lengths']) > 0:
            length = min(valid_crossings_dict['lengths'])
            index = np.argmin((valid_crossings_dict['lengths']))
            angle = valid_crossings_dict['angles'][index]
        else:
            length = np.inf
            angle = None
        return length, angle
