import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_1d_fit(time, data, fit, title="Fit vs Data"):
    sns.set_style("whitegrid", {'axes.grid': False})
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    ax.semilogy(time, data, label="Data")
    ax.semilogy(time, fit, 'k-', label="Fit")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.set_title(title)
    return fig, ax

def plot_1d_distribution(tau, A, title="Recovered Lifetime Distribution"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.fill_between(tau, A, step="mid", color='#6BB4F9', edgecolor='#6BB4F9')
    ax.set_xlabel("Lifetime [ns]")
    ax.set_ylabel("Probability")
    ax.set_yticks([])
    ax.set_title(title)
    return fig, ax

def plot_2d_map_stack(ilt_maps_tau, tau, vmax=None, title="ILT τ–τ maps"):
    # ilt_maps_tau: (ΔT, τ, τ)
    D = ilt_maps_tau.shape[0]
    cols = min(D, 4)
    rows = int(np.ceil(D / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(D):
        im = axes[i].imshow(ilt_maps_tau[i], origin="lower",
                            extent=(tau[0], tau[-1], tau[0], tau[-1]),
                            aspect='auto', vmax=vmax)
        axes[i].set_title(f"ΔT idx {i}")
        axes[i].set_xlabel("τ (ns)")
        axes[i].set_ylabel("τ (ns)")
        fig.colorbar(im, ax=axes[i], fraction=0.046)
    for j in range(D, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title)
    return fig, axes

def plot_corr_map(corr_maps, time, idx=0, title="Correlation map"):
    # corr_maps: (ΔT, t, t)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(corr_maps[idx], origin="lower",
                   extent=(time[0], time[-1], time[0], time[-1]),
                   aspect='auto')
    ax.set_title(f"{title} (ΔT idx {idx})")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("t (ns)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig, ax
