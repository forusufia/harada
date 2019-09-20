import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from SOM_T import TSOM

#load data
data = np.loadtxt("datatsom.txt")
data = np.reshape(data, (10, 10, 3))

# x = np.arange(-3, 3, 0.1)
# y = np.sin(x)
# data = np.stack([x, y], 1)


# def update(t, ax_latent, ax_observable, obss, lats):
#     ax_latent.cla()
#     ax_observable.cla()
#
#     ax_latent.scatter(som.z[:, 0], som.z[:, 1])
#     ax_observable.scatter(data[:, 0], data[:, 1])
#
#     ax_latent.scatter(lats[t][:, 0], lats[t][:, 1])
#     ax_observable.scatter(obss[t][:, 0], obss[t][:, 1])
#
#     plt.title("{}time".format(t))
#
# fig = plt.figure(figsize=(10, 6))
# ax_latent = fig.add_subplot(121)
# ax_observable = fig.add_subplot(122)
# lats = []
# obss = []

#learn
som = TSOM()
som.initialize(data)
for t in range(50):
    som.fit(data, t)

    # obss.append(som.y)
    # lats.append(som.z)

#plot
# fargs = [ax_latent, ax_observable, obss, lats]
# ani = animation.FuncAnimation(fig, update, fargs=fargs, frames=50, interval=100)
# ani.save("test.gif", writer="pillow")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[:, :, 0], data[:, :, 1], data[:, :, 2])
ax.scatter(som.y[:, :, 0], som.y[:, :, 1], som.y[:, :, 2])

# plt.plot(data[:, 0], data[:, 1])
# plt.plot(som.y[:, 0], som.y[:, 1], color='red')

plt.show()