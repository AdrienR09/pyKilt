"""Plot helpers for Kilt."""

import matplotlib.pyplot as plt
import numpy as np

def plot_1d_fit(time, ExpData, fit_model, A_fit, tau, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        ax0, ax1 = ax
    else:
        ax0, ax1 = ax

    ax0.semilogy(time, ExpData, label="Data")
    ax0.semilogy(time, fit_model, 'k-', label="Fit")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Counts")
    ax0.legend()

    ax1.fill_between(tau, A_fit, step="mid")
    ax1.set_xlabel("Lifetime [ns]")
    ax1.set_ylabel("Probability")
    ax1.set_yticks([])

    return ax0.get_figure()

def plot_2d_maps(ilt_maps, tau, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, ilt_maps.shape[0], figsize=(4*ilt_maps.shape[0], 4))
    if ilt_maps.shape[0] == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(ilt_maps[i], extent=(tau[0], tau[-1], tau[-1], tau[0]), vmin=vmin, vmax=vmax)
        ax.set_title(f"ΔT index {i}")
        ax.set_xlabel("τ (ns)")
        ax.set_ylabel("τ (ns)")
        plt.colorbar(im, ax=ax)
    return fig
