from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from aspire.utils import to_numpy

if TYPE_CHECKING:
    from aspire.flows import Flow
    from aspire.transforms import Transform


class FlowProposal:
    """Proposal distribution for MCMC sampling based on a flow model."""

    def __init__(self, prior_flow: Flow, preconditioning_transform: Transform):
        self.prior_flow = prior_flow
        self.preconditioning_transform = preconditioning_transform

    def _log_qz(self, z_flat: np.ndarray) -> np.ndarray:
        x, log_abs_det_jacobian = self.preconditioning_transform.inverse(
            z_flat
        )
        log_qx = self.prior_flow.log_prob(x)
        return to_numpy(log_qx + log_abs_det_jacobian).reshape(-1)

    def propose(
        self,
        current: np.ndarray,
        complement: np.ndarray,
        random: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        ntemps, nwalkers_half, dim = current.shape
        n = ntemps * nwalkers_half

        x_prop, _ = self.prior_flow.sample_and_log_prob(n)
        z_prop, _ = self.preconditioning_transform.forward(x_prop)
        z_prop = to_numpy(z_prop).reshape(ntemps, nwalkers_half, dim)

        current_flat = current.reshape(-1, dim)
        prop_flat = z_prop.reshape(-1, dim)

        log_qratio = (
            self._log_qz(current_flat) - self._log_qz(prop_flat)
        ).reshape(ntemps, nwalkers_half)
        return z_prop, log_qratio
