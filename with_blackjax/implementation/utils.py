from typing import Callable

from jax.experimental.host_callback import id_print
from pymc import Model
from pymc.sampling_jax import get_jaxified_graph


def get_jaxified_logprior(model: Model) -> Callable:
    model_logpt = model.varlogpt
    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[model_logpt])

    def logp_fn_wrap(x):
        return logp_fn(*x)

    return logp_fn_wrap


def get_jaxified_loglikelihood(model: Model) -> Callable:
    model_logpt = model.datalogpt
    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[model_logpt])

    def logp_fn_wrap(x):
        to_return = logp_fn(*x)
        id_print(to_return)
        return to_return

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