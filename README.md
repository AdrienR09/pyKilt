# pyKilt — Kernel-based Inverse Laplace Transform

pyKilt is a lightweight Python package for recovering fluorescence lifetime distributions using the Maximum Entropy Method (MEM). It supports 1D MEM (single-decay) and 2D / global MEM (τ–τ maps or stacks) with optional IRF reconvolution and annealed regularization.

## Quick links
- Full docs: ./docs/ (installation, tutorials, 1D/2D guides, plotting)
- Examples: ./docs/examples/ (1d_example.py, 2d_example.py)

## Installation
From PyPI:
```bash
pip install pyKilt
```
From source (dev):
```bash
git clone <repo>
cd pyKilt
pip install -e .[dev]
```

## Minimal usage

1D (single decay)
```python
import numpy as np
import jax.numpy as jnp
from pyKilt import KiltConfig, run_ilt_1d
from pyKilt.plotting import plot_1d_fit, plot_1d_distribution

time = np.linspace(0, 15, 500)
hist = np.loadtxt("my_decay.txt")
tau = jnp.linspace(0.02, 5, 200)
cfg = KiltConfig(use_irf_convolution=True, optimize_eta=True)
res = run_ilt_1d(time, tau, hist, cfg)

# plot best result
fig, ax = plot_1d_fit(time, hist, res["best"]["fit"])
fig, ax = plot_1d_distribution(tau, res["best"]["A"])
```

2D (global / τ–τ maps)
```python
from pyKilt import KiltConfig, run_ilt_2d
# time: (n_times,), corr_maps: user-provided maps array, tau: lifetime grid
cfg = KiltConfig(global_fit=True, use_irf_convolution=True, optimize_eta=True)
res2d = run_ilt_2d(time, tau, corr_maps, cfg, A_init=A_init)
```

## Key config options (KiltConfig)
- tau (lifetime grid)
- optimize_eta, eta_start, eta_end, eta_rounds — annealing schedule for MEM regularization
- use_irf_convolution, irf, optimize_irf_shift — enable reconvolution and IRF shift search
- global_fit / A_init — for 2D/global fitting
- maxiter, rounds_per_eta, conv_pad, sigma2, baseline — solver tuning

## Tips
- Start with a coarse tau grid and short annealing to find stable settings, then refine.
- Provide experimental IRF and enable reconvolution + IRF shift for accurate fits.
- Inspect `res["trials"]` to monitor annealing progression and `res["best"]` for final outputs.
- Use plotting helpers in `pyKilt.plotting` to visualize fits and τ–τ maps.

## Documentation & examples
See the docs folder for detailed step-by-step guides:
- ./docs/usage/1d.md — 1D workflows and examples
- ./docs/usage/2d.md — 2D/global workflows, A_init suggestions
- ./docs/usage/plotting.md — plotting helpers
- ./docs/examples/ — runnable example scripts

## Contributing
Open issues or PRs for bugs, examples, doc improvements or feature requests.

## License
MIT License
See LICENSE in repository.