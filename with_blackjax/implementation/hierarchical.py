import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

data = pd.read_csv('../../radon.csv')

county_names = data.county.unique()
county_idx = data['county_code'].values.astype(np.int32)
n_counties = len(data.county.unique())

with pm.Model() as model:
    mu_a = pm.Normal('mu_a', mu=0., sigma=5)
    sigma_a = pm.HalfNormal('sigma_a', 5)
    mu_b = pm.Normal('mu_b', mu=0., sigma=5)
    sigma_b = pm.HalfNormal('sigma_b', 5)
     
    a_offset = pm.Normal('a_offset', mu=0, sigma=2, shape=n_counties)
    a = pm.Deterministic("a", mu_a + a_offset * sigma_a)
    b_offset = pm.Normal('b_offset', mu=0, sigma=2, shape=n_counties)
    b = pm.Deterministic("b", mu_b + b_offset * sigma_b)
 
    eps = pm.HalfNormal('eps', 5)
    
    radon_est = a[county_idx] + b[county_idx] * data.floor.values
    
    radon_like = pm.Normal('radon_like', mu=radon_est, sigma=eps, observed=data.log_radon)



def plot_results(idata, path):
    nuts_posterior = idata.posterior.stack(samples=("chain", "draw"))
    plt.figure(figsize=(5,6))
    plt.plot((nuts_posterior["mu_b"]+nuts_posterior["b_offset"] * nuts_posterior['sigma_b_log__']).isel(b_offset_dim_0=75), nuts_posterior["sigma_b_log__"], '.C0')
    # This is hacky difference to avoid sampling deterministic variables
    plt.xlabel('B')
    plt.ylabel('sigma_b')
    plt.legend()
    plt.savefig(path)


