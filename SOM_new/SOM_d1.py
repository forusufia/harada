import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# SOM class
class SOM:

    def __init__(self,
                 D = 1,
                 resolution = 25,
                 sigma_max = 1.0,
                 sigma_min = 0.1,
                 tau = 40.0
                 ):
        self.D = D
        self.resolution = resolution
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau

        self.k_star = None
        self.zeta = None
        self.z = None
        self.h = None
        self.sigma = None
        self.y = None

    def fit(self, data, t):
        #self.initialize(data)
        # self.__e_step(data)
        self.__m_step(data, t)
        self.__e_step(data)

    def initialize(self, data):
        self.zeta = create_zeta(self.resolution, self.D)
        np.random.seed(0)
        self.y = np.random.rand(self.resolution, np.size(data, axis=1))
        # da = np.zeros((41, 2))
        # for i in range(41):
        #     da[i] = np.array([5.5, i / 41])
        # self.y = da
        self.z = np.random.rand(np.size(data, axis=0), self.D)
        # pca = PCA(np.size(data, axis=1))
        # pca.fit(data)
        # self.y = pca.inverse_transform(np.sqrt(pca.explained_variance_)[None, :] * self.zeta)

    def __e_step(self, data):
        # print("data", np.size(data, axis=0), np.size(data, axis=1))
        # print("y", np.size(self.y, axis=0), np.size(self.y, axis=1))
        self.k_star = np.argmin(cdist(data, self.y), axis=1)
        # self.k_star = np.argmin(np.sum(np.power(data[:, None] - self.y[None, :], 2), axis=1), axis=0)
        # print("k_star", np.size(self.k_star, axis=0))
        # print("zeta", np.size(self.zeta, axis=0), np.size(self.zeta, axis=1))
        self.z = self.zeta[self.k_star, :]

    def __m_step(self, data, t):
        self.h = np.exp(-0.5 * cdist(self.zeta, self.z) / (self.__sigma(t) ** 2))
        # print("k_star", np.size(self.k_star, axis=0))
        # print("z", np.size(self.z, axis=0), np.size(self.z, axis=1))
        # print("zeta", np.size(self.zeta, axis=0), np.size(self.zeta, axis=1))
        # print("h", np.size(self.h, axis=0), np.size(self.h, axis=1))
        # print("data", np.size(data, axis=0), np.size(data, axis=1))
        self.y = np.dot(self.h, data) / np.sum(self.h, axis=1)[:, None]

    def __sigma(self, t):
        self.sigma = self.sigma_max + (self.sigma_min - self.sigma_max) * (t / self.tau)
        if self.sigma < self.sigma_min:
            self.sigma = self.sigma_min
        return self.sigma

def create_zeta(resolution, D):
    zeta = np.linspace(-1, 1, resolution)
    zeta = np.reshape(zeta, (resolution, D))
    return zeta
