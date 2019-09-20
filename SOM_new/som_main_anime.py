import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from SOM import SOM

#load data
data = np.loadtxt("data1.txt")

def update(t, ax_latent, ax_observable, obss, lats):
    ax_latent.cla()
    ax_observable.cla()

    ax_latent.scatter(som.z[:, 0], som.z[:, 1])
    ax_observable.scatter(data[:, 0], data[:, 1])

    ax_latent.scatter(lats[t][:, 0], lats[t][:, 1])
    ax_observable.scatter(obss[t][:, 0], obss[t][:, 1])

    plt.title("{}time".format(t))

fig = plt.figure(figsize=(10, 6))
ax_latent = fig.add_subplot(121)
ax_observable = fig.add_subplot(122)
lats = []
obss = []

#learn
som = SOM()
som.initialize(data)
for t in range(50):
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
ani = animation.FuncAnimation(fig, update, fargs=fargs, frames=50, interval=100)
ani.save("test.gif", writer="pillow")
# ani2 = animation.FuncAnimation(fig, obss, interval=100)
#ax_observable.scatter(data[:, 0], data[:, 1], color='red', alpha=0.5)

plt.show()