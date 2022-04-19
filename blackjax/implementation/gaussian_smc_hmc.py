import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import multivariate_normal
import blackjax
import blackjax.smc.resampling as resampling
jax.config.update("jax_platform_name", "cpu")

def V(x):
    return 5 * jnp.square(jnp.sum(x**2) - 1)


def prior_log_prob(x):
    d = x.shape[0]
    return multivariate_normal.logpdf(x, jnp.zeros((d,)), jnp.eye(d))


linspace = jnp.linspace(-2, 2, 5000).reshape(-1, 1)
lambdas = jnp.linspace(0.0, 1.0, 5)
prior_logvals = jnp.vectorize(prior_log_prob, signature="(d)->()")(linspace)
potential_vals = jnp.vectorize(V, signature="(d)->()")(linspace)
log_res = prior_logvals.reshape(1, -1) - jnp.expand_dims(
    lambdas, 1
) * potential_vals.reshape(1, -1)

density = jnp.exp(log_res)
normalizing_factor = jnp.sum(density, axis=1, keepdims=True) * (
    linspace[1] - linspace[0]
)
density /= normalizing_factor

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(linspace.squeeze(), density.T)
ax.legend(list(lambdas))


def inference_loop(rng_key, mcmc_kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, k):
        state, _ = mcmc_kernel(k, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def full_logprob(x):
    return -V(x) + prior_log_prob(x)


inv_mass_matrix = jnp.eye(1)
n_samples = 10_000
key = jax.random.PRNGKey(42)

hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=50
)

hmc = blackjax.hmc(full_logprob, **hmc_parameters)
hmc_state = hmc.init(jnp.ones((1,)))
hmc_samples = inference_loop(key, hmc.step, hmc_state, n_samples)

samples = np.array(hmc_samples.position[:, 0])
_ = plt.hist(samples, bins=100, density=True)
_ = plt.plot(linspace.squeeze(), density[-1])

nuts_parameters = dict(step_size=1e-4, inverse_mass_matrix=inv_mass_matrix)

nuts = blackjax.nuts(full_logprob, **nuts_parameters)
nuts_state = nuts.init(jnp.ones((1,)))
nuts_samples = inference_loop(key, nuts.step, nuts_state, n_samples)

samples = np.array(nuts_samples.position[:, 0])
_ = plt.hist(samples, bins=100, density=True)
_ = plt.plot(linspace.squeeze(), density[-1])


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

loglikelihood = lambda x: -V(x)

hmc_parameters = dict(
    step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
)

tempered = blackjax.adaptive_tempered_smc(
    prior_log_prob,
    loglikelihood,
    blackjax.hmc,
    hmc_parameters,
    resampling.systematic,
    0.5,
    mcmc_iter=1,
)

initial_smc_state = jax.random.multivariate_normal(
    jax.random.PRNGKey(0), jnp.zeros([1]), jnp.eye(1), (n_samples,)
)
initial_smc_state = tempered.init(initial_smc_state)

n_iter, smc_samples = smc_inference_loop(key, tempered.step, initial_smc_state)
print("Number of steps in the adaptive algorithm: ", n_iter.item())

samples = np.array(smc_samples.particles[:, 0])
_ = plt.hist(samples, bins=100, density=True)
_ = plt.plot(linspace.squeeze(), density[-1])

