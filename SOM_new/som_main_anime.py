import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from SOM import SOM

#load data
# data = np.loadtxt("dataoriginal.txt")
data = np.loadtxt("data1.txt")

# resolution = 5
# resolution2 = 5
# D = 2
# data = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(1, -1, resolution2))
# data = np.dstack(data)
# data = data.reshape(resolution * resolution2, D)
# a = np.array([[-0.4, -2], [0.21, -4], [0.62, -3]])
# data = np.concatenate([data, a])

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

# x = np.arange(-3.14, 3.14, 0.1)
# y = np.sin(x)
# data = np.stack([x, y], 1)

def update(t, ax_latent, ax_observable, obss, lats):
    ax_latent.cla()
    ax_observable.cla()

    ax_latent.scatter(som.z[:, 0], som.z[:, 1])
    ax_observable.scatter(data[:, 0], data[:, 1])

    ax_latent.scatter(lats[t][:, 0], lats[t][:, 1])
    ax_observable.scatter(obss[t][:, 0], obss[t][:, 1])

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

    #lat, = ax_latent.scatter(som.z[:, 0], som.z[:, 1], s=5)
    #obs, = ax_observable.scatter(som.y[:, 0], som.y[:, 1])
    #lats.append([lat])
    #obss.append([obs])
    obss.append(som.y)
    lats.append(som.z)

#plot
#ax_latent.scatter(som.zeta[:, 0], som.zeta[:, 1], s = 50, alpha = 0.5)
fargs = [ax_latent, ax_observable, obss, lats]
# ani = animation.FuncAnimation(fig, update, fargs=fargs, interval=100, repeat="True")
ani = animation.FuncAnimation(fig, update, fargs=fargs, frames=60, interval=100)
ani.save("test2.gif", writer="pillow")
# ani2 = animation.FuncAnimation(fig, obss, interval=100)
#ax_observable.scatter(data[:, 0], data[:, 1], color='red', alpha=0.5)

plt.show()