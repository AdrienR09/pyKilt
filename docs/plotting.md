# Plotting & Visualization

pyKilt provides simple helpers (if available) to visualize fits and distributions.

Common helpers
- plot_1d_fit(time, data, fit): overlay fit vs data
- plot_1d_distribution(tau, A): plot lifetime spectrum
- plot_2d_map_stack(maps, tau, title=None): display τ–τ maps

Example usage
```python
from pyKilt.plotting import plot_1d_fit, plot_1d_distribution, plot_2d_map_stack

fig, ax = plot_1d_fit(time, hist, res1d["best"]["fit"])
fig, ax = plot_1d_distribution(tau, res1d["best"]["A"])

# For 2D
fig, axes = plot_2d_map_stack(res2d["best"]["ilt_maps_tau"], tau, title="Best τ–τ maps")
```