import pymc as pm
import aesara.tensor as at
from pymc.sampling_jax import get_jaxified_logp
import blackjax
import jax
import logging
import arviz as az
import numpy as np
import xarray as xr

def inference_data(chains, draws, dims, sampling_as_arrays):

    dataset = xr.Dataset(
        {
        "a": (["chain", "draw", "X_dim_0"], sampling_as_arrays),
        },
    coords = {
         "chain": (["chain"], np.arange(chains)),
          "draw": (["draw"], np.arange(draws)),
          "a_dim": (["a_dim"], np.arange(dims))
        }
    )
    return az.InferenceData(posterior=dataset)


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
n = 20

mu1 = np.ones(n) * (1. / 2)
mu2 = -mu1

stdev = 0.1
sigma = np.power(stdev, 2) * np.eye(n)
isigma = np.linalg.inv(sigma)
logdsigma = np.linalg.slogdet(sigma)[1]

w1 = 0.1
w2 = (1 - w1)


def results(idata, path):
    ax = az.plot_trace(idata, compact=True, kind="rank_vlines")
    ax[0, 0].axvline(-0.5, 0, .9, color="k")
    ax[0, 0].axvline(0.5, 0, 0.1, color="k")
    ax[0, 0].set_xlim(-1, 1)
    fig = ax.ravel()[0].figure
    fig.savefig(path)

print("definition")
def two_gaussians(x):
    log_like1 = - 0.5 * n * at.log(2 * np.pi) \
                - 0.5 * logdsigma \
                - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
    log_like2 = - 0.5 * n * at.log(2 * np.pi) \
                - 0.5 * logdsigma \
                - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
    return pm.math.logsumexp([at.log(w1) + log_like1, at.log(w2) + log_like2])

print("creating model")
with pm.Model() as model:
    X = pm.Uniform('X',
                   shape=n,
                   lower=-2. * np.ones_like(mu1),
                   upper=2. * np.ones_like(mu1),
                   testval=-1. * np.ones_like(mu1))
    llk = pm.Potential('llk', two_gaussians(X))
