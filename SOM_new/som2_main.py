import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from SOM import SOM
from SOM_2 import SOM2

#load data
data = np.loadtxt("data1.txt")

#初期化
file = 9
D = 2
som = []
X = []

#子SOM
for file_num in range(file):
    file_name = "data" + str(file_num + 1) + ".txt"
    print(file_name)
    X.append(np.loadtxt(file_name))
    som.append(SOM())
    som[file_num].initialize(X[file_num])

#V
v = []
for file_num in range(file):
    som[file_num].fit(X[file_num], 1)
    ch_som = som[file_num].y
    v.append(ch_som.flatten())
    print("CH_SOM"+str(file_num+1))
    # print("y", som[file_num].y)
# print(v)

#親SOM
som_p = SOM2()
som_p.initialize(v)
som_p.fit(v, 1)

#コピーバック
w_li = som_p.y[som_p.k_star]
print("w_li", np.size(w_li, axis=0), np.size(w_li, axis=1))
for file_num in range(file):
    w_li_re = np.reshape(w_li[file_num], (100, 2))
    som[file_num].y = w_li_re

#loop
T = 50
for t in range(1, T):
    # V
    v = []
    for file_num in range(file):
        som[file_num].fit(X[file_num], t)
        ch_som = som[file_num].y
        v.append(ch_som.flatten())
        # print("CH_SOM" + str(file_num + 1))

    som_p.fit(v, t)

    # コピーバック
    w_li = som_p.y[som_p.k_star]
    for file_num in range(file):
        w_li_re = np.reshape(w_li[file_num], (100, 2))
        som[file_num].y = w_li_re

# print(som_p.y)
# print(som_p.h)
print("h", np.size(som_p.h, axis=0), np.size(som_p.h, axis=1))
print("v", np.size(v, axis=0), np.size(v, axis=1))
print("y", np.size(som[0].y, axis=0), np.size(som[0].y, axis=1))
print("w", np.size(som_p.y, axis=0), np.size(som_p.y, axis=1))
print("l", np.size(som_p.k_star, axis=0))
# print(w_li)

#plot
fig = plt.figure(figsize=(10, 5))
ax_latent = fig.add_subplot(121)
ax_latent.scatter(som_p.zeta[:, 0], som_p.zeta[:, 1], s = 50, alpha = 0.5)
ax_latent.scatter(som_p.z[:, 0], som_p.z[:, 1], s = 50, color='black')

s_w = np.reshape(som_p.y[50], (100, 2))
ax2 = fig.add_subplot(122)
for i in range(file):
    ax2.scatter(som[i].y[:, 0], som[i].y[:, 1], label = str(i+1))
ax2.scatter(s_w[:, 0], s_w[:, 1], label='P')

plt.legend()
plt.show()
