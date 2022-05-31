import logging

from pymc.sampling_jax import get_jaxified_logp
import blackjax
import jax
# For now copy pasting a little bit from the gaussian case
from utils import inference_data
from blackjax.implementation.hierarchical import plot_results, model

jax.config.update("jax_platform_name", "cpu")

log = logging.getLogger(__name__)
logprob_fn = get_jaxified_logp(model)
seed = jax.random.PRNGKey(50)
samples_per_chain = 5000
CHAINS = 2
rvs = [rv.name for rv in model.value_vars]
log.info(rvs)

def sample_chain(seed, logprob_fn, rvs):
    init_position_dict = model.compute_initial_point()
    init_position = [init_position_dict[rv] for rv in rvs]
    adapt = blackjax.window_adaptation(blackjax.nuts, logprob_fn, 1000)
    last_state, kernel, _ = adapt.run(seed, init_position)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return states, infos

    states, infos = inference_loop(seed, kernel, last_state, samples_per_chain)
    posterior_samples = states.position
    return posterior_samples


sampled = [sample_chain(seed, logprob_fn, rvs) for chain in range(0, CHAINS)]
posterior = inference_data(CHAINS, samples_per_chain, sampled, rvs)

plot_results(posterior, './blackjax/results_hierarchical_hmc.png')

