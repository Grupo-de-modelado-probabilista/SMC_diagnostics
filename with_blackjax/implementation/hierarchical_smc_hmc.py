"""
Hierarchical model with Sequential Monte Carlo + Hamiltonian Monte Carlo
"""

import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling

from with_blackjax.implementation.hierarchical import model
from with_blackjax.implementation.utils import get_jaxified_logprior, get_jaxified_loglikelihood

jax.config.update("jax_platform_name", "cpu")


def inference_loop(rng_key, mcmc_kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, k):
        state, _ = mcmc_kernel(k, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


logprior = get_jaxified_logprior(model)
loglikelihood = get_jaxified_loglikelihood(model)

inv_mass_matrix = jnp.eye(1)
n_samples = 10_000
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



hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
)

tempered = blackjax.adaptive_tempered_smc(
    logprior,
    loglikelihood,
    blackjax.hmc,
    hmc_parameters,
    resampling.systematic,
    0.5,
    mcmc_iter=1,
)

import numpy as np
rvs = [rv.name for rv in model.value_vars]
init_position_dict = model.compute_initial_point()
initial_smc_state = np.array([[jax.numpy.array(init_position_dict[rv]) for rv in rvs] for _ in range(0, n_samples)])

initial_smc_state = tempered.init(initial_smc_state)

n_iter, smc_samples = smc_inference_loop(key, tempered.step, initial_smc_state)
print("Number of steps in the adaptive algorithm: ", n_iter.item())


