import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# SOM class
class TSOM:

    def __init__(self,
                 D = 1,
                 n = 3,
                 resolution = 49,
                 sigma_max = 1.0,
                 sigma_min = 0.05,
                 tau = 50.0
                 ):
        self.D = D
        self.n = n
        self.resolution = resolution
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau

        self.k_star_1 = None
        self.k_star_2 = None
        self.zeta_1 = None
        self.zeta_2 = None
        self.z_1 = None
        self.z_2 = None
        self.r_1 = None
        self.r_2 = None
        self.sigma = None
        self.u_1 = None
        self.u_2 = None
        self.y = None

    def fit(self, data, t):
        #self.initialize(data)
        self.__e_step(data)
        self.__m_step(data, t)

    def initialize(self, data):
        self.zeta_1 = create_zeta(self.resolution, self.D)
        self.zeta_2 = create_zeta(self.resolution, self.D)
        self.u_1 = np.random.rand(10, 50, 3)
        self.u_2 = np.random.rand(50, 10, 3)
        self.y = np.random.rand(50, 50, 3)

    def __e_step(self, data):
        n = 0
        self.k_star_1 = np.argmin(np.sum(np.power(self.u_1[None, :, :, :] - self.y[:, None, :, :], 2), axis=(2,3)), axis=0)
        # print("k1", self.k_star_1.shape)
        self.k_star_2 = np.argmin(np.sum(np.power(self.u_2[:, None, :, :] - self.y[:, :, None, :], 2), axis=(0,3)), axis=0)
        # print("k2", self.k_star_2.shape)
        self.z_1 = self.zeta_1[self.k_star_1]
        self.z_2 = self.zeta_2[self.k_star_2]

    def __m_step(self, data, t):
        self.r_1 = np.exp(-0.5 * np.power(self.zeta_1[:,None]-self.z_1[None,:], 2) / (self.__sigma(t)**2))
        self.r_2 = np.exp(-0.5 * np.power(self.zeta_2[:,None]-self.z_2[None,:], 2) / (self.__sigma(t)**2))
        # print("k_star", np.size(self.k_star, axis=0))
        # print("z", np.size(self.z, axis=0), np.size(self.z, axis=1))
        # print("zeta", np.size(self.zeta, axis=0), np.size(self.zeta, axis=1))
        # print("h", np.size(self.h, axis=0), np.size(self.h, axis=1))
        # print("r_2", np.size(self.r_2, axis=0), np.size(self.r_2, axis=1))
        # print("z_1", np.size(self.z_1, axis=0))
        # print("z_2", np.size(self.z_2, axis=0))
        # self.u_1 = np.sum(self.r_2[None, :, :, None] * data[:, None, :, :], axis=2) / np.sum(self.r_2, axis=0)[None, :, None]
        self.u_1 = np.einsum("jk,ikl -> ijl", self.r_2, data) / np.sum(self.r_2, axis=1)[None, :, None]
        # print("u1", self.u_1.shape)
        # self.u_2 = np.sum(self.r_1[:, :, None, None] * data[None, :, :, :], axis=0) / np.sum(self.r_1, axis=0)[:, None, None]
        self.u_2 = np.einsum("ij,jkl -> ikl", self.r_1, data) / np.sum(self.r_1, axis=1)[:, None, None]
        # print("u2", self.u_2.shape)
        # self.y = np.sum(self.r_1[:, None, :, None, None] * self.r_2[None, :, None, :, None] * data[None, None, :, :, :], axis=(2,3)) / (np.sum(self.r_1, axis=0)[:, None, None, None, None] * np.sum(self.r_2, axis=0)[None, :, None, None, None])
        self.y = np.einsum("ij,kl,jlm -> ikm", self.r_1, self.r_2, data) / (np.sum(self.r_1, axis=1)[:, None, None] * np.sum(self.r_2, axis=1)[None, :, None])
        # print("y", self.y.shape)

    def __sigma(self, t):
        self.sigma = self.sigma_max + (self.sigma_min - self.sigma_max) * (t / self.tau)
        if self.sigma < self.sigma_min:
            self.sigma = self.sigma_min
        return self.sigma

def create_zeta(resolution, D):
    zeta = np.array(np.linspace(-1, 1, resolution))
    return zeta
