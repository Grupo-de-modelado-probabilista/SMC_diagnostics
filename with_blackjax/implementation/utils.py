from typing import Callable

from jax.experimental.host_callback import id_print
from pymc import Model
from pymc.sampling_jax import get_jaxified_graph
import jax


def get_jaxified_logprior(model: Model) -> Callable:
    model_logpt = model.varlogpt
    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[model_logpt])
    from aesara import pp

    def logp_fn_wrap(particles):
        return logp_fn(*particles)[0]

    return logp_fn_wrap


def get_jaxified_loglikelihood(model: Model) -> Callable:
    """
    This is code that we will need to port into PYMC at some point
    """
    model_logpt = model.datalogpt
    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[model_logpt])

    def logp_fn_wrap(particles):
        """
        Assuming an x,y posterior, where x in
         x in R^{10} and y in R^{50}
        Blackjax operates with a structure like:
        [ALL_X,ALL_Y]
        where ALL_X.shape = (n_particles, 10)
        and ALL_Y.shape =   (n_particles, 50).
        This function will be applied after vmap.
        """

        return logp_fn(*particles)[0]
        #acc_logp = 0.
        #for particle_index in range(0, len(particles[0])):

        #    acc_logp += logp_fn(*[particles[var_index][particle_index]
        #                          for var_index in range(0, len(particles))])
        #return acc_logp

    return logp_fn_wrap


import arviz as az
from arviz import dict_to_dataset
import numpy as np


def inference_data(chains, samples_per_chain, sampling_as_arrays, rvs):
    """
    Given a list of traces, one per chain, each
    trace being a list of arrays, each with samples
    of a given variable
    """
    as_dict = {}
    for index, var in enumerate(rvs):
        as_dict[var] = {}
        all_chains_for_var = {}
        for chain in range(0, len(sampling_as_arrays)):
            all_chains_for_var[chain] = []
            for sample in range(0, samples_per_chain):
                all_chains_for_var[chain].append(np.array(sampling_as_arrays[chain][index][sample]))
        as_dict[var] = np.array([all_chains_for_var[c] for c in range(0, len(sampling_as_arrays))])
    return az.InferenceData(posterior=az.dict_to_dataset(
        as_dict,
        coords={'chain': range(0, chains), 'var': rvs, 'draw': range(0, samples_per_chain)}
    ))