# aspire-ptemcee

Wrapper for using `ptemcee` with `aspire`.

This package vendors `ptemcee` with several minor tweaks to ensure compatibility with `aspire`:

- Replace calls to `np.float` with `float`
- Fixing indexing issues in `autocorr_function` and `autocorr_integrated_time`
- Add `vectorize` argument to support vectorized likelihoods in `aspire`

## Installation

Currently, this package can only be installed from source

```
pip install git+https://github.com/mj-will/aspire-ptemcee.git
```
