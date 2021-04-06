import numpy as np
from pykdtree.kdtree import KDTree

class KullbackLiebler:
    """Approximate Kullback-Liebler divergence.
    
    From Eq 14 in F. Perez-Cruz, Kullback-Leibler divergence estimation of continuous distributions
    doi: 10.1109/ISIT.2008.4595271. http://www.tsc.uc3m.es/~fernando/bare_conf3.pdf
    """

    def __init__(self, obs_data):
        obs_data = obs_data[:, None]
        n, d = obs_data.shape
        rho_d, _ = KDTree(obs_data).query(obs_data, 2)
        self.rho_d = rho_d[:, 1]
        self.d_n = d / n
        self.log_r = np.log(n / (n - 1))
        self.obs_data = obs_data

    def __call__(self, sim_data):
        sim_data = sim_data[:, None]
        nu_d, _ = KDTree(sim_data).query(self.obs_data, 1)
        log_diff = np.log(nu_d / self.rho_d)
        finite = np.isfinite(log_diff)
        return self.d_n * np.sum(log_diff[finite]) + self.log_r
