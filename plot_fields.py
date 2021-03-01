import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrow

class FieldPlotter():

    def __init__(self, **kwargs):
        #kwargs can either consist of "beacons" and "RSSI_threshold" or be a dictionary "dict" of both
        assert ("beacons" in kwargs and "RSSI_threshold" in kwargs or "dict" in kwargs) 
        if "beacons" in kwargs:
            self.neigh_RSSI_threshold = kwargs["RSSI_threshold"]
            self.config_dict = FieldPlotter.__build_config_dict(kwargs["beacons"])
        else:
            self.neigh_RSSI_threshold = kwargs["dict"]["RSSI_threshold"]
            del kwargs["dict"]["RSSI_threshold"]
            self.config_dict = kwargs["dict"]
    
    @staticmethod
    def __build_config_dict(beacons):
        config_dict = {
            b.ID: {
                "x": b.pos,
                "k": b.k,
                "a": b.a,
                "v": b.v,
                "d_perf": b.d_perf,
                "d_none": b.d_none,
                "xi_max": b.xi_max
            }for b in beacons}
        return config_dict

    def __init_X_Y_Z(self, resolution=0.05):
        x_is = np.concatenate([drone_config["x"].reshape(2,1) for drone_config in self.config_dict.values()], axis=1)

        min_x, max_x = np.min(x_is[0,:]), np.max(x_is[0,:])
        min_y, max_y = np.min(x_is[1,:]), np.max(x_is[1,:])
        
        X, Y = np.meshgrid(
            np.arange(min_x - 5, max_x + 5, resolution),
            np.arange(min_y - 5, max_y + 5, resolution)
        )
        Z = np.zeros(X.shape)

        return X, Y, Z

    def plot_potential_field(self): #TODO: Add potential field from obstacels
        X, Y, Z = self.__init_X_Y_Z()

        fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("U")

        for beacon_ID, drone_config in self.config_dict.items():
            # drone_config.keys(): "x", "k", "a", "v", "d_perf", "d_none", "xi_max"
            x_i, _, _, _, d_perf_i, d_none_i, xi_max_i = drone_config.values()
            XI_i = FieldPlotter.__xi(x_i, d_perf_i, d_none_i, xi_max_i, X, Y)
            Z += FieldPlotter.get_U_i(
                *drone_config.values(),
                X,
                Y,
            )*(XI_i > self.neigh_RSSI_threshold)
            ax.scatter(*drone_config["x"], color="blue" if not beacon_ID == 0 else "green", zorder=100)
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            alpha=0.5
        )

        fig.colorbar(surf, shrink=0.5, aspect=5)
        

    def plot_force_field(self):
        X, Y, _ = self.__init_X_Y_Z(0.5)
        U, V  = np.zeros(X.shape), np.zeros(Y.shape) # x- and y-component of the vectors

        _, ax = plt.subplots()
        ax.title.set_text("F")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        for beacon_ID, drone_config in self.config_dict.items():
            # drone_config has the following keys: "x", "k", "a", "v", "d_perf", "d_none", "xi_max"
            x_i, _, _, v_i, d_perf_i, d_none_i, xi_max_i = drone_config.values()            
            temp_U, temp_V = FieldPlotter.get_F_i(*drone_config.values(), X, Y)
            
            XI_i = FieldPlotter.__xi(x_i, d_perf_i, d_none_i, xi_max_i, X, Y)
            U += temp_U*(XI_i > self.neigh_RSSI_threshold)
            V += temp_V*(XI_i > self.neigh_RSSI_threshold)

            v_i = v_i.reshape(2,)
            ax.add_patch(FancyArrow(x_i[0], x_i[1], v_i[0], v_i[1], color="green"))
            ax.scatter(*drone_config["x"], color="blue" if not beacon_ID == 0 else "green", zorder=100)

        ax.quiver(X, Y, U, V, alpha=0.5)


    @staticmethod
    def __xi(x_i, d_perf, d_none, xi_max, X, Y):
        assert d_none > d_perf

        omega = np.pi*(1/(d_none - d_perf))
        phi = -d_perf*omega

        X_i = np.ones(X.shape)*x_i[0]
        Y_i = np.ones(Y.shape)*x_i[1]

        d = np.sqrt((X-X_i)**2 + (Y - Y_i)**2)

        xi_is = (xi_max/2)*(1 + np.cos(omega*d + phi))
        xi_is[d > d_none] = 0
        xi_is[d < d_perf] = xi_max

        return xi_is

    @staticmethod
    def get_U_i(x_i, k_i, a_i, v_i, d_perf_i, d_none_i, xi_max_i, X, Y):
        x_component = X - a_i*(np.ones(X.shape)*x_i[0] + v_i[0]*FieldPlotter.__xi(x_i, d_perf_i, d_none_i, xi_max_i, X, Y))
        y_component = Y - a_i*(np.ones(Y.shape)*x_i[1] + v_i[1]*FieldPlotter.__xi(x_i, d_perf_i, d_none_i, xi_max_i, X, Y))
        # x_component = X
        # y_component = Y
        return (1/2)*k_i*(x_component**2 + y_component**2)

    @staticmethod
    def get_F_i(x_i, k_i, a_i, v_i, d_perf_i, d_none_i, xi_max_i, X, Y):
        F_x = -k_i*(X - a_i*(np.ones(X.shape)*x_i[0] + v_i[0]*FieldPlotter.__xi(x_i, d_perf_i, d_none_i, xi_max_i, X, Y)))
        F_y = -k_i*(Y - a_i*(np.ones(Y.shape)*x_i[1] + v_i[1]*FieldPlotter.__xi(x_i, d_perf_i, d_none_i, xi_max_i, X, Y)))
        return F_x, F_y

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

def f(X, Y, r_th):
    try:
        # mat = np.concatenate(vecs, axis=1)
        # return -gain*np.sum(mat/np.linalg.norm(mat, axis=0)**3, axis=1)
        x_part2 = ((X-0*np.ones(X.shape))**2)
        y_part2 = ((Y-0*np.ones(Y.shape))**2)
        r_i = x_part2 + y_part2
        return 0.5*(1/r_i-1/r_th)**2
    except ValueError:
        print("Except")
        return np.zeros((2, ))
    
    # if (r_i < r_th).any() and (r_i != 0).all():
    #     return 0.5*(1/r_i-1/r_th)**2
    # else:
    #     return np.zeros(X.shape)

if __name__=="__main__":
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

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)

    x_part1 = ((X+6*np.ones(X.shape))**2)/2
    y_part1 = ((Y+6*np.ones(Y.shape))**2)/2
    exponent1 = -(x_part1+y_part1)
    Z1 = np.exp(exponent1)

    x_part2 = ((X-6*np.ones(X.shape))**2)/1
    y_part2 = ((Y-6*np.ones(Y.shape))**2)/1
    exponent2 = -(x_part2+y_part2)
    Z2 = np.exp(exponent2)

    Z = Z1 + Z2
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.title.set_text(r"$U_o = k_{obs}exp\left\{-\left(\frac{(x-x_{obs,i})^2}{\sigma_x^2} + \frac{(y-y_{obs,i})^2}{\sigma_y^2}\right)\right\}$")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    