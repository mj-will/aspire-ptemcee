import logging
import math

import numpy as np
import tqdm
from aspire.samplers.mcmc import ParallelTemperedMCMCSampler
from aspire.samples import PTMCMCSamples
from aspire.utils import to_numpy, track_calls

from ._vendor import ptemcee
from .proposal import FlowProposal

logger = logging.getLogger(__name__)


class PTEmceeSampler(ParallelTemperedMCMCSampler):
    """Parallel Tempered MCMC sampler using ptemcee.

    This uses the vendored of ptemcee, which include changes to support:
    - Vectorized log-likelihood and log-prior evaluations
    - Custom flow-based proposals
    """

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
        n_samples: int | None = None,
        nwalkers: int | None = None,
        nsteps: int = 100,
        ntemps: int = 5,
        burn_in: int = 0,
        thin: int = 1,
        Tmax: float | None = math.inf,
        rng=None,
        vectorize: bool = True,
        proposal: str | None = None,
        checkpoint_callback=None,
        checkpoint_every: int | None = None,
        checkpoint_file_path: str | None = None,
        **kwargs,
    ):
        """Run parallel tempered MCMC sampling using ptemcee.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the posterior
            (after burn-in and thinning). If None, returns all samples.
        nwalkers : int
            The number of walkers to use in the MCMC sampler. If None, defaults
            to the number of samples.
        nsteps : int
            The number of MCMC steps to run.
        ntemps : int
            The number of temperatures to use in the parallel tempering.
        burn_in : int
            The number of initial samples to discard as burn-in.
        thin : int
            The thinning factor to apply to the samples after burn-in.
        Tmax : float
            The maximum temperature to use in the parallel tempering. If None,
            defaults to infinity (no maximum temperature).
        rng : np.random.Generator
            A random number generator to use for sampling. If None, a new
            generator will be created.
        vectorize : bool
            Whether to use vectorized log-likelihood and log-prior evaluations.
        proposal : str or None
            The proposal distribution to use for MCMC sampling. If 'flow', uses
            a flow-based proposal. If None, uses the default proposal in ptemcee.
            Currently, only 'flow' is supported as a custom proposal option.
        checkpoint_callback : callable, optional
            Callback used to save checkpoints. If None, no callback checkpointing
            is applied unless checkpoint_file_path is provided.
        checkpoint_every : int, optional
            Checkpoint frequency control. For PTMCMC this currently only gates
            final checkpoint saving; if <= 0, no checkpoint is written.
        checkpoint_file_path : str, optional
            HDF5 file path where the final checkpoint is written.
        kwargs
            Additional keyword arguments to pass to the ptemcee.Sampler
            constructor.
        """
        rng = rng or self.rng or np.random.default_rng()

        if proposal == "flow":
            logger.info(
                "Using flow-based proposal for PTMCMC sampling with ptemcee."
            )
            proposal = FlowProposal(
                prior_flow=self.prior_flow,
                preconditioning_transform=self.preconditioning_transform,
            )
        elif proposal is not None:
            raise ValueError(
                f"Invalid proposal: {proposal}. Must be 'flow' or None."
            )
        else:
            logger.debug(
                "Using default proposal for PTMCMC sampling with ptemcee."
            )

        nwalkers = nwalkers or n_samples

        self.sampler = ptemcee.Sampler(
            dim=self.dims,
            logl=self.log_likelihood_wrapper,
            logp=self.log_prior_wrapper,
            nwalkers=nwalkers,
            ntemps=ntemps,
            Tmax=Tmax,
            proposal=proposal,
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
        )

        samples_pt.log_prior = samples_pt.array_to_namespace(
            self.log_prior(samples_pt)
        )
        samples_pt.autocorrelation_time = self.sampler.acor

        # Save checkpoint before any post-processing (burn-in/thinning/subsampling).
        self.checkpoint_mcmc_chain(
            samples=samples_pt,
            iteration=nsteps,
            checkpoint_callback=checkpoint_callback,
            checkpoint_every=checkpoint_every,
            checkpoint_file_path=checkpoint_file_path,
        )

        if thin is not None and burn_in is not None:
            logger.info(
                f"Post-processing PTMCMC samples with burn-in of {burn_in} and thinning factor of {thin}."
            )
            samples_pt = samples_pt.post_process(burn_in=burn_in, thin=thin)

        if n_samples is not None:
            logger.info(
                f"Subsampling PTMCMC samples to {n_samples} samples per temperature after burn-in and thinning."
            )
            samples_pt = samples_pt[:n_samples]

        return samples_pt
