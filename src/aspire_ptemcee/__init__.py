"""
aspire-ptemcee: Parallel-tempered MCMC sampler for aspire

This package vendors the ptemcee library and provides a wrapper to integrate it
with aspire's sampling framework.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())
