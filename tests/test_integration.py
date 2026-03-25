import math

import array_api_compat.numpy as xp
import numpy as np
from aspire import Aspire
from aspire.samples import Samples


def test_ptemcee_integration():
    dims = 2
    parameters = [f"x_{i}" for i in range(dims)]
    prior_bounds = {parameter: [-10, 10] for parameter in parameters}
    rng = np.random.default_rng(0)

    def log_likelihood(samples):
        x = xp.asarray(samples.x)
        mean = 2.0
        std = 1.0
        constant = xp.log(xp.asarray(1 / (std * math.sqrt(2 * math.pi))))
        return xp.sum(constant - 0.5 * ((x - mean) / std) ** 2, axis=-1)

    def log_prior(samples):
        x = xp.asarray(samples.x)
        constant = dims * xp.log(xp.asarray(1 / 10))
        val = xp.where((x >= -10) & (x <= 10), constant, -xp.inf)
        return xp.sum(val, axis=-1)

    training_samples = Samples(
        rng.normal(loc=2.0, scale=1.0, size=(200, dims)),
        parameters=parameters,
        xp=xp,
    )

    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=dims,
        parameters=parameters,
        prior_bounds=prior_bounds,
        flow_matching=False,
        bounded_to_unbounded=True,
        flow_backend="zuko",
    )
    aspire.fit(training_samples, n_epochs=5)

    samples = aspire.sample_posterior(
        n_samples=20,
        sampler="ptemcee",
        nwalkers=10,
        nsteps=20,
        ntemps=2,
        thin=1,
        burn_in=0,
    )

    assert samples.chain_shape == (2, 20)
    assert samples.n_temps == 2
    assert list(samples.parameters) == parameters

    posterior_samples = samples.cold_chain()
    assert posterior_samples.x.shape == (20, dims)
