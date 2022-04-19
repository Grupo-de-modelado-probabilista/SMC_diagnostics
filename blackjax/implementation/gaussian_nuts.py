from pymc.sampling_jax import get_jaxified_logp
import blackjax
import jax
from blackjax.implementation.gaussian import model, results
from utils import inference_data

jax.config.update("jax_platform_name", "cpu")

logprob_fn = get_jaxified_logp(model)
seed = jax.random.PRNGKey(50)
samples_per_chain = 5000
CHAINS = 10
rvs = [rv.name for rv in model.value_vars]

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


posterior = [sample_chain(seed, logprob_fn, rvs) for chain in range(0, CHAINS)]


id = inference_data(chains=CHAINS,
                    samples_per_chain=samples_per_chain,
                    sampling_as_arrays=posterior,
                    rvs=rvs)

results(id, '../results_gaussian_hmc.png')

