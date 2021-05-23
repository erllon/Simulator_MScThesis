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
d_none = 2.8#2.5
xi_max = 1

omega = np.pi/(d_none - d_perf)
phi = -(np.pi*d_perf) / (d_none - d_perf)

D = np.linspace(0, d_none + 0.5)

xi_fun = np.vectorize(xi)
XI = [xi(d) for d in D]

fig = plt.figure()#plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.grid(False)
ax.plot(D, XI, color="blue")

# plt.title(r"$\xi(\|\|\mathbf{x}_i - \mathbf{x}_j\|\|,...)$")
# plt.axvline(x=d_perf, ymin=0, ymax=xi_max+0.5, color='green', label=r'$d_{perf}$', linestyle=':')
# plt.axvline(x=d_none, ymin=0, ymax=xi_max+0.5, color="red", label=r'$d_{none}$', linestyle='-.')
# plt.axhline(y=xi_max, xmin=0, xmax=D[-1], color="black", label='xi_max')
ax.axvline(x=0, color="black")
ax.axhline(y=0, color="black")
ax.set_xlim([0,d_none + 0.5])
ax.set_ylim([-0.1, xi_max + 0.2])
d_label = ax.set_xlabel(r"$d_{i,j}$",labelpad=-4,loc="right")
xi_label = ax.set_ylabel(r"$\xi_{i,j}$",labelpad=-15, loc='top')
xi_label.set_rotation(0)
# plt.legend()    
plt.show()
