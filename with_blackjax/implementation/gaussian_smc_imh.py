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

def proposal(rng, mean, sigma):
    # TODO:
    # It would be great to infer
    # shape not from parameters, but
    # from the model's properties'
    # This is similar to the way
    # it is done in random walk,
    # is this a jax limitation?
    #sigma = jnp.array([1.0])
    ndim = jnp.ndim(sigma)  # type: ignore[arg-type]
    shape = jnp.shape(jnp.atleast_1d(sigma))[:1]

    if ndim == 1:
        dot = jnp.multiply
    elif ndim == 2:
        dot = jnp.dot
    else:
        raise ValueError
    sample = jax.random.normal(rng, shape) + mean
    move_sample = dot(sigma, sample)
    return move_sample

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


irmh_parameters = {'proposal_distribution': lambda x, y: proposal(rng=x,mean=y, sigma=np.ones(20)*10)}
tempered = blackjax.adaptive_tempered_smc(
    logprior,
    loglikelihood,
    blackjax.irmh,
    irmh_parameters,
    resampling.systematic,
    target_ess=0.8,
    mcmc_iter=10
)

rvs = [rv.name for rv in model.value_vars]
import numpy as np
initial_particles = [model.compute_initial_point() for sample in range(0, n_samples)] # this one could be expensive
initial_smc_state = [np.array([ip[rv] for ip in initial_particles]) for rv in rvs]
print(initial_smc_state)
initial_smc_state = init(initial_smc_state)
n_iter, smc_samples = smc_inference_loop(key, tempered.step, initial_smc_state)
print("Number of steps in the adaptive algorithm: ", n_iter.item())
print(smc_samples.particles)

id = inference_data(chains=1,
                    samples_per_chain=n_samples,
                    sampling_as_arrays=[smc_samples.particles[0]],
                    rvs=rvs)

results(id, '../results/results_gaussian_smc_imh.png')