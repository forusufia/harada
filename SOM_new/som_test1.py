import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from SOM_d1 import SOM

#load data
# data = np.loadtxt("dataoriginal.txt")

# resolution = 10
# resolution2 = 5
# D = 2
# data = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(1, -1, resolution2))
# data = np.dstack(data)
# data = data.reshape(resolution * resolution2, D)

# x = np.arange(0, 1.05, 0.05)
# x = np.reshape(x, (len(x), 1))
# y = (-0.2*x)+0.2
# y = np.reshape(y, (len(y), 1))
# dat = np.concatenate([x,y], axis=1)
# a = np.arange(0, 1, 0.05)
# a = np.reshape(a, (len(a), 1))
# b = np.array([0]*20)
# b = np.reshape(b, (len(b), 1))
# c = np.concatenate([a,b], axis=1)
# data = np.concatenate([dat, c], axis = 0)

x = np.arange(-3.14, 3.14, 0.1)
y = np.sin(x)
data = np.stack([x, y], 1)

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
# ax_latent = fig.add_subplot(122)
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