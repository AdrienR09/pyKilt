# Quick Start

1. Prepare time axis `time` and decay(s) `hist` or `decays`.
2. Define lifetime grid `tau` (jax.numpy recommended).
3. Provide IRF if available and set reconvolution options.
4. Create `KiltConfig`, tune annealing/eta schedule.
5. Run `run_ilt_1d` or `run_ilt_2d`.
6. Inspect `res["trials"]` and `res["best"]`.

Minimal 1D (example)
```python
import numpy as np
import jax.numpy as jnp
from pyKilt import KiltConfig, run_ilt_1d
from pyKilt.plotting import plot_1d_fit, plot_1d_distribution

time = np.linspace(0, 15, 500)
hist = np.loadtxt("my_decay.txt")
tau = jnp.linspace(0.02, 5, 200)
irf = np.loadtxt("my_irf.txt")

cfg = KiltConfig(use_irf_convolution=True, irf=irf, optimize_eta=True)
res = run_ilt_1d(time, tau, hist, cfg)
```