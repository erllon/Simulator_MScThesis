from matplotlib.patches import Polygon

class Env():
      
  """Class representing an environment.

  Args:
      entrance_point (ndarray): 2-by-0 numpy array containing (x, y)-coords of 
      environment entrance point.

      obstacle_corners (list, optional): List of n-by-2 numpy arrays. Each numpy array contains 
      the corners of a polygon. It is assumed that there is a line between the 
      first corner and the last corner. Defaults to [].
  """

  def __init__(self, entrance_point, obstacle_corners = []):
    self.obstacle_corners = obstacle_corners
    self.entrance_point = entrance_point

  def plot(self, axis):
    for i in range(len(self.obstacle_corners)):
      axis.add_patch(Polygon(self.obstacle_corners[i], fill=False))
    
  
  def toJson(self):
    jsonDict = {
      'Obstacle_corners': [obstacle_corner.tolist() for obstacle_corner in self.obstacle_corners],
      'Entrance_point': self.entrance_point.tolist()
    }
    return jsonDict