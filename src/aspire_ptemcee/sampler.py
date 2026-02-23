import math

import numpy as np
import tqdm
from aspire.samplers.mcmc import ParallelTemperedMCMCSampler
from aspire.samples import PTMCMCSamples
from aspire.utils import to_numpy, track_calls

from ._vendor import ptemcee


class PTEmceeSampler(ParallelTemperedMCMCSampler):
    def log_likelihood_wrapper(self, z):
        if np.ndim(z) > 2:
            raise ValueError("Input z must be a 1D or 2D array.")
        return super().log_likelihood_wrapper(z)

    def log_prior_wrapper(self, z):
        if np.ndim(z) > 2:
            raise ValueError("Input z must be a 1D or 2D array.")
        return super().log_prior_wrapper(z)

    @track_calls
    def sample(
        self,
        n_samples: int,
        nwalkers: int,
        nsteps: int = 100,
        ntemps: int = 5,
        burnin: int = 0,
        thin: int = 1,
        Tmax: float | None = math.inf,
        rng=None,
        vectorize: bool = True,
        **kwargs,
    ):
        rng = rng or self.rng or np.random.default_rng()

        self.sampler = ptemcee.Sampler(
            dim=self.dims,
            logl=self.log_likelihood_wrapper,
            logp=self.log_prior_wrapper,
            nwalkers=nwalkers,
            ntemps=ntemps,
            Tmax=Tmax,
            random=rng,
            vectorize=vectorize,
            **kwargs,
        )

        p0 = self.draw_initial_samples(int(nwalkers * ntemps)).x
        z0 = to_numpy(self.preconditioning_transform.fit(p0))

        z0 = z0.reshape((ntemps, nwalkers, self.dims))

        # Use tqgmbar for progress tracking if enabled
        for _ in tqdm.tqdm(
            self.sampler.sample(
                p0=z0,
                iterations=nsteps,
                thin=thin,
            ),
            total=nsteps,
            desc="Sampling with ptemcee",
        ):
            pass

        # Chain has shape (ntemps, nwalkers, nsteps, dims)
        chain_z = self.sampler.chain
        # Transform all chains back to original space
        chain_z_flat = chain_z.reshape(-1, self.dims)
        chain_x_flat, _ = self.preconditioning_transform.inverse(chain_z_flat)
        chain_x = chain_x_flat.reshape(chain_z.shape)

        # PTMCMCSamples expects chain of shape (n_temps, steps, n_walkers, dims), so we need to
        # reshape the chain to match this expectation
        chain_x = np.transpose(chain_x, (0, 2, 1, 3))
        logl_chain = np.transpose(self.sampler.loglikelihood, (0, 2, 1))

        # Create PTMCMCSamples with full chain
        samples_pt = PTMCMCSamples.from_chain(
            chain=chain_x,
            betas=self.sampler.betas,
            log_likelihood=logl_chain,
            parameters=self.parameters,
            xp=self.xp,
            dtype=self.dtype,
            thin=thin,
            burn_in=burnin,
        )

        samples_pt.log_prior = samples_pt.array_to_namespace(
            self.log_prior(samples_pt)
        )
        samples_pt.autocorrelation_time = self.sampler.acor

        return samples_pt
