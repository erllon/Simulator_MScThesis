import numpy as np
import matplotlib.pyplot as plt

x_0 = np.array([-4,-4])
x_1 = np.array([-3,-3])
x_2 = np.array([-2,-3])
x_3 = np.array([-1.9,-1.6])


x = np.arange(-4.0, 4.0, 0.1)
y = np.arange(-4.0, 4.0, 0.1)
X, Y = np.meshgrid(x, y)

Z = 0.5*(np.sqrt((x_0[0]-X)**2 + (x_0[1]-Y)**2)**2) + np.sqrt((x_1[0]-X)**2 + (x_1[1]-Y)**2)**2# + np.sqrt((x_2[0]-X)**2 + (x_2[1]-Y)**2)**2+ np.sqrt((x_3[0]-X)**2 + (x_3[1]-Y)**2)**2)   #X*Y #np.sin(X)*np.cos(Y)
print(f"Z: {Z}")
print(f"min_Z = {np.min(Z)}")
print(f"indices = {np.argmin(Z)}")
# TODO:  Find the equilibrium point (overleaf) and plot it in the figure, approach below/here does not work
xmin, ymin = np.unravel_index(np.argmin(Z), Z.shape)
print(f"(xmin,ymin) = ({xmin,ymin})")
X_min = X[xmin,ymin]
Y_min = Y[xmin,ymin]
Z_min = 0.5*(np.sqrt((x_0[0]-X_min)**2 + (x_0[1]-Y_min)**2)**2 + np.sqrt((x_1[0]-X_min)**2 + (x_1[1]-Y_min)**2)**2 + np.sqrt((x_2[0]-X_min)**2 + (x_2[1]-Y_min)**2)**2+ np.sqrt((x_3[0]-X_min)**2 + (x_3[1]-Y_min)**2)**2)   #X*Y #np.sin(X)*np.cos(Y)
print(f"(X_min,Y_min) = ({X_min,Y_min})")
fig, ax = plt.subplots()
ax.plot(X_min, Y_min, 'or')
# ax.plot(x_0[0], x_0[1], 'ob')
# ax.plot(x_1[0], x_1[1], 'og')
# ax.plot(x_2[0], x_2[1], 'oy')
# ax.plot(x_3[0], x_3[1], 'oc')
#ax.contour()returns QuadContourSet (Base: ContourSet)
lower = np.min(Z)
levels_arr = np.array([lower + i for i in range(4)])
cs  = ax.contour(X,Y,Z, levels=levels_arr)#[2,3,4,5])#, levels=[-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
# ax.clabel(cs, inline=True, 
#           fontsize=10)
plt.show()