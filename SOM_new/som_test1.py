import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from SOM_d1 import SOM

#load data
# data = np.loadtxt("data1.txt")

resolution = 2
D = 2
data = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(1, -1, resolution))
data = np.dstack(data)
data = data.reshape(resolution ** 2, D)

def update(t, ax_latent, ax_observable, obss, lats):
    ax_latent.cla()
    ax_observable.cla()

    z_y = np.zeros((np.size(lats, axis=0), np.size(som.z, axis=0)))

    ax_latent.scatter(som.z[:, 0], z_y[0])
    ax_observable.scatter(data[:, 0], data[:, 1])

    ax_latent.scatter(lats[t], z_y[t])
    ax_observable.plot(obss[t][:, 0], obss[t][:, 1], color='orange')

    plt.title("{}time".format(t))

fig = plt.figure(figsize=(12, 6))
ax_latent = fig.add_subplot(121)
ax_observable = fig.add_subplot(122)
lats = []
obss = []

#learn
som = SOM()
som.initialize(data)
for t in range(60):
    som.fit(data, t)

    obss.append(som.y)
    lats.append(som.z)

#plot
fargs = [ax_latent, ax_observable, obss, lats]
ani = animation.FuncAnimation(fig, update, fargs=fargs, frames=60, interval=100)
ani.save("test.gif", writer="pillow")

plt.show()