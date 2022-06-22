import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import multivariate_normal

jax.config.update("jax_platform_name", "cpu")

import blackjax
import blackjax.smc.resampling as resampling

n_samples = 10_000


initial_smc_state = jax.random.multivariate_normal(
    jax.random.PRNGKey(0), jnp.zeros([1]), jnp.eye(1), (n_samples,)
)

print(initial_smc_state)

import jax.random as random
rng_key = random.PRNGKey(314)
num_points = 50
from sklearn.datasets import make_biclusters
X, rows, cols = make_biclusters(
    (num_points, 2), 2, noise=0.6, random_state=314, minval=-3, maxval=3
)
Phi = jnp.c_[jnp.ones(num_points)[:, None], X]
N, M = Phi.shape

multivariate_initial_point = random.multivariate_normal(rng_key, 0.1 + jnp.zeros(M), jnp.eye(M))

print(multivariate_initial_point)
print(np.random.normal(0, 1, size=(N, 1)))

