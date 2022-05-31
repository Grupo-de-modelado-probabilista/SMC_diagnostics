from pymc.sampling_jax import get_jaxified_graph
import pymc as pm

from with_blackjax.implementation.hierarchical import model

with pm.Model() as fast_model:
    x = pm.Normal("x", 0, 1)
    y = pm.Normal("y", x, 1, observed=[0 for _ in range(0,100000)])
    """
    Including decent observations so that prior and likelihood differ
    """

model_logpt = fast_model.datalogpt
log_likelihood_internal = get_jaxified_graph(inputs=fast_model.value_vars, outputs=[model_logpt])
log_likelihood = lambda x: log_likelihood_internal(*x)
print(log_likelihood([2.]))
# This returns DeviceArray(-2.91893853, dtype=float64)

model_logpt = fast_model.varlogpt
log_prior_internal = get_jaxified_graph(inputs=fast_model.value_vars, outputs=[model_logpt])
log_prior = lambda x: log_prior_internal(*x)
print(log_prior([2.]))




model_logpt = model.datalogpt
log_likelihood_internal = get_jaxified_graph(inputs=fast_model.value_vars, outputs=[model_logpt])
log_likelihood = lambda x: log_likelihood_internal(*x)
print(log_likelihood([2.]))
# This returns DeviceArray(-2.91893853, dtype=float64)

model_logpt = model.varlogpt
log_prior_internal = get_jaxified_graph(inputs=fast_model.value_vars, outputs=[model_logpt])
log_prior = lambda x: log_prior_internal(*x)
print(log_prior([2.]))
