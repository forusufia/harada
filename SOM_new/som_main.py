import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from SOM import SOM
from Grad_norm import Grad_Norm

#load data
data = np.loadtxt("data1.txt")

#learn
som = SOM()
som.initialize(data)
for t in range(50):
    som.fit(data, t)

zero = np.zeros((np.size(som.y, axis=0), 1))

grad = Grad_Norm(data, som.z)
grad.draw_umatrix()
#plot
# fig = plt.figure(figsize=(10, 5))
# ax_latent = fig.add_subplot(121)
# ax_latent.scatter(som.zeta[:, 0], som.zeta[:, 1], s = 50, alpha = 0.5)
# ax_latent.scatter(som.z[:, 0], som.z[:, 1], s = 5)
#
# y = np.concatenate((som.y, zero), axis=1)
# wf = y.reshape(som.resolution, som.resolution, 3)
# ax_observable = fig.add_subplot(122, projection='3d')
# ax_observable.view_init(elev=90, azim=0)
# ax_observable.plot_wireframe(wf[:, :, 0], wf[:, :, 1], wf[:, :, 2])
# ax_observable.scatter(data[:, 0], data[:, 1], np.array([0]*400), color='red', alpha=0.5)
#
# plt.show()