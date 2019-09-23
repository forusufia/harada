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

SOM = SOM()
SOM.initialize(data)
for t in range(50):
    SOM.fit(data, t)

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

#plot
# plt.scatter(SOM.y)
for i in range(file):
    plt.scatter(som[i].y[:, 0], som[i].y[:, 1], label = str(i))

plt.show()