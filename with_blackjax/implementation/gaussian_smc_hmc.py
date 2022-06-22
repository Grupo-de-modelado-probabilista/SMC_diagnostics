"""
Gaussian model with Sequential Monte Carlo + Hamiltonian Monte Carlo
"""

import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.mcmc.random_walk import normal
from blackjax.smc.tempered import init

from with_blackjax.implementation.gaussian import model, results
from with_blackjax.implementation.utils import get_jaxified_logprior, get_jaxified_loglikelihood, inference_data

jax.config.update("jax_platform_name", "cpu")
n_samples = 10

logprior = get_jaxified_logprior(model)
loglikelihood = get_jaxified_loglikelihood(model)

inv_mass_matrix = jnp.eye(1)

key = jax.random.PRNGKey(42)

hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=50
)


def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.
    """

    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state


tempered = blackjax.adaptive_tempered_smc(
    logprior,
    loglikelihood,
    blackjax.hmc,
    hmc_parameters,
    resampling.systematic,
    target_ess=0.8,
    mcmc_iter=10
)

rvs = [rv.name for rv in model.value_vars]
import numpy as np
initial_particles = [model.compute_initial_point() for sample in range(0, n_samples)] # this one could be expensive
initial_smc_state = [np.array([ip[rv] for ip in initial_particles]) for rv in rvs]
print(initial_smc_state)
print("Finished generating initial points")
initial_smc_state = init(initial_smc_state)
n_iter, smc_samples = smc_inference_loop(key, tempered.step, initial_smc_state)
print("Number of steps in the adaptive algorithm: ", n_iter.item())
print(smc_samples.particles)

id = inference_data(chains=1,
                    samples_per_chain=n_samples,
                    sampling_as_arrays=[smc_samples.particles[0]],
                    rvs=rvs)

results(id, '../results/results_gaussian_smc_imh.png')