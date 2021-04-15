import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def xi(d): 
    if d <= d_perf:
        return xi_max
    elif d_perf < d and d < d_none:
        return (xi_max/2) * (1 + np.cos(omega*d + phi))
    elif d_none <= d:
        return 0
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

d_perf = 0.1
d_none = 2.5
xi_max = 1

omega = np.pi/(d_none - d_perf)
phi = -(np.pi*d_perf) / (d_none - d_perf)

D = np.linspace(0, d_none + 0.5)

xi_fun = np.vectorize(xi)
XI = [xi(d) for d in D]

plt.plot(D, XI, color="blue")
plt.axvline(x=d_perf, ymin=0, ymax=xi_max+0.5, color='green', label='d_perf')
plt.axvline(x=d_none, ymin=0, ymax=xi_max+0.5, color="red", label='d_none')
# plt.axhline(y=xi_max, xmin=0, xmax=D[-1], color="black", label='xi_max')
plt.axvline(x=0, color="black")
plt.axhline(y=0, color="black")
plt.ylim([-0.1, xi_max + 0.5])
plt.legend()    
plt.show()


