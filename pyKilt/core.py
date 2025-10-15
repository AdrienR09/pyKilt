"""
Core primitives shared by 1D & 2D ILT:
- Config (KiltConfig)
- IRF utilities
- Kernels (Laplace)
- Convolution helpers
- Entropy / objective
- numba fdc_core
- JAX helpers (pack/unpack/model/loss/L-BFGS)
"""

from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, List, Dict, Any
import numpy as np
from scipy.signal import convolve
from scipy.linalg import toeplitz

# ----------------------------- Config -----------------------------

@dataclass
class KiltConfig:
    # IRF options
    optimize_irf_shift: bool = True
    irf_shift_list: Optional[Iterable[int]] = None   # e.g. range(0, 5)
    irf: Optional[Iterable[float]] = None   # e.g. range(0, 5)
    conv_pad: int = 0
    baseline: Optional[float] = None  # if None, use ExpData[0]

    # Eta options (either grid or annealing schedule)
    optimize_eta: bool = True
    eta_grid: Optional[np.ndarray] = None            # overrides annealing schedule if provided
    eta_start: float = 1e-5
    eta_end: float = 1e-12
    eta_rounds: int = 5  # number of annealing rounds (logspace from start->end)

    # Optimizer options
    maxiter: int = 2000
    # For 2D JAX LBFGS, "rounds_per_eta" lets you re-seed m_prior between LBFGS runs
    rounds_per_eta: int = 1

    # Numerical
    sigma2: float = 1.0  # stabilizer in chi2 denominator

    # JAX IRF mode for 2D (K_irf = K or toeplitz(IRF) @ K )
    use_irf_convolution: bool = False

    # Let A varies with time (2D ILT)
    time_varying_A: bool = True

# ------------------------ Light JAX management --------------------

def ensure_jax():
    try:
        import jax  # noqa
        import jax.numpy as jnp  # noqa
        from jaxopt import LBFGS  # noqa
        return True
    except Exception:
        return False

JAX_AVAILABLE = ensure_jax()

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from jaxopt import LBFGS

# -------------------------- IRF utilities -------------------------

def gaussian_irf(time: np.ndarray, center: float, width: float) -> np.ndarray:
    irf = np.exp(-((time - center) ** 2) / (2 * width ** 2))
    s = irf.sum()
    return irf if s == 0 else irf / s

def shift_irf(irf: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(irf, shift)

# -------------------------- Kernels -------------------------------

def exp_kernel(time: np.ndarray, tau: np.ndarray) -> np.ndarray:
    # K(t, τ) = exp(-t/τ)
    return np.exp(-np.outer(time, 1.0 / tau))

def make_irf_matrix_conv(K: np.ndarray, irf: np.ndarray, conv_pad: int = 0) -> np.ndarray:
    """
    Convolve each column of K with IRF, normalize each col.
    """
    tlen, tau_bins = K.shape
    out = np.zeros_like(K)
    pad = conv_pad
    irf_pad = np.pad(irf, pad, mode="constant", constant_values=0.0)
    Kp = np.pad(K, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    for i in range(tau_bins):
        col = Kp[:, i + pad]
        conv_col = convolve(col, irf_pad, mode="full")[pad : pad + tlen]
        s = conv_col.sum()
        out[:, i] = conv_col / s if s > 0 else conv_col
    return out

def toeplitz_irf_matrix(irf: np.ndarray, tau_bins: int) -> np.ndarray:
    return toeplitz(np.r_[irf[0], np.zeros(tau_bins - 1)], irf)

# --------------------- Entropy / Objective (NumPy) ----------------

def entropy_np(A: np.ndarray, m_prior: np.ndarray) -> float:
    return float(np.sum(np.where(A > 1e-12,
                                 A - m_prior - A * np.log(A / (m_prior + 1e-12)),
                                 0.0)))

def reconvolve_np(A: np.ndarray, K_irf: np.ndarray, y0: float) -> np.ndarray:
    return K_irf @ A + y0

def objective_np(A: np.ndarray, ExpData: np.ndarray, K_irf: np.ndarray,
                 y0: float, m_prior: np.ndarray, eta: float, sigma2: float = 1.0) -> float:
    model = reconvolve_np(A, K_irf, y0)
    chi2 = np.sum((ExpData - model) ** 2 / (model + sigma2))
    ent = entropy_np(A, m_prior)
    return float(chi2 - eta * ent)

# --------------------------- fdc_core (numba) ---------------------

try:
    import numba as nb

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def fdc_core(macrotimes, microtimes, dT, ddT, time, corr_maps):
        for n in nb.prange(dT.size):
            for i in nb.prange(macrotimes.size):
                t_start = macrotimes[i] + dT[n] - ddT
                t_end   = macrotimes[i] + dT[n] + ddT
                j0 = np.searchsorted(macrotimes, t_start, side='left')
                j1 = np.searchsorted(macrotimes, t_end,   side='right')

                if microtimes[i] < time[0] or microtimes[i] > time[-1]:
                    continue
                k = np.searchsorted(time, microtimes[i], side='left') - 1
                if k < 0:
                    continue

                for j in range(j0, j1):
                    if i == j:
                        continue
                    if microtimes[j] < time[0] or microtimes[j] >= time[-1]:
                        continue
                    l = np.searchsorted(time, microtimes[j], side='right') - 1
                    if l >= 0:
                        corr_maps[n, k, l] += 1
except Exception:
    # Provide a slow fallback if numba missing
    def fdc_core(macrotimes, microtimes, dT, ddT, time, corr_maps):
        for n in range(dT.size):
            for i in range(macrotimes.size):
                t_start = macrotimes[i] + dT[n] - ddT
                t_end   = macrotimes[i] + dT[n] + ddT
                j0 = np.searchsorted(macrotimes, t_start, side='left')
                j1 = np.searchsorted(macrotimes, t_end,   side='right')

                if microtimes[i] < time[0] or microtimes[i] > time[-1]:
                    continue
                k = np.searchsorted(time, microtimes[i], side='left') - 1
                if k < 0:
                    continue

                for j in range(j0, j1):
                    if i == j:
                        continue
                    if microtimes[j] < time[0] or microtimes[j] >= time[-1]:
                        continue
                    l = np.searchsorted(time, microtimes[j], side='right') - 1
                    if l >= 0:
                        corr_maps[n, k, l] += 1

# --------------------------- JAX helpers --------------------------

if JAX_AVAILABLE:
    # Packing upper-triangular blocks for U
    def jax_build_packers(tau_bins: int, delay_bins: int):
        tri_idx = jnp.triu_indices(tau_bins)
        tri_len = tri_idx[0].size

        def pack(A, Ulist):
            parts = [A.ravel()]
            parts.extend([U[tri_idx] for U in Ulist])
            return jnp.concatenate(parts)

        @jit
        def unpack(theta):
            A = theta[:tau_bins * tau_bins].reshape(tau_bins, tau_bins)
            Ulist = []
            offset = tau_bins * tau_bins
            for _ in range(delay_bins):
                u_flat = theta[offset: offset + tri_len]
                U = jnp.zeros((tau_bins, tau_bins))
                U = U.at[tri_idx].set(u_flat)
                Ulist.append(U)
                offset += tri_len
            return A, Ulist

        return pack, unpack, tri_idx, tri_len

    @jit
    def jax_entropy(A, m_prior):
        return jnp.sum(jnp.where(A > 1e-12,
                                 A - m_prior - A * jnp.log(A / (m_prior + 1e-12)),
                                 0.0))

    @jit
    def jax_model_maps(A, Ulist, K_irf):
        # (ΔT, τ, τ) stack
        G_stack = jnp.stack([U @ U.T for U in Ulist])
        M_tau = A @ G_stack @ A.T    # (ΔT, τ, τ)
        return K_irf @ M_tau @ K_irf.T  # (ΔT, t, t)

    @jit
    def jax_loss(theta, eta, m_prior, corr_maps, K_irf, unpack_fn, sigma2: float):
        tau_bins = m_prior.shape[0]
        delay_bins = corr_maps.shape[0]
        A, Ulist = unpack_fn(theta, delay_bins)
        sim = jax_model_maps(A, Ulist, K_irf)
        chi2 = jnp.mean((corr_maps - sim) ** 2 / (sim + sigma2))
        ent = jax_entropy(A, m_prior)
        return chi2 - eta * ent

    def jax_lbfgs_run(theta0, loss_fun, maxiter):
        # jaxopt LBFGS expects (value, grad) if value_and_grad=True
        solver = LBFGS(fun=loss_fun, value_and_grad=True, maxiter=maxiter)
        sol = solver.run(theta0)
        return sol.params, sol.state.value
