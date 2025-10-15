"""
2D ILT (MEM) with JAX acceleration and LBFGS from jaxopt.
- Uses numba fdc_core (in core.py) to build corr_maps (via io.py helper)
- IRF shift loop
- η annealing/scan with optional multiple LBFGS rounds per η
"""

from typing import Dict, Any, List
import numpy as np

from .core import (
    KiltConfig, JAX_AVAILABLE, exp_kernel, toeplitz_irf_matrix,
    jax, jnp, jit, LBFGS,
    jax_build_packers, jax_model_maps, jax_loss, jax_lbfgs_run, shift_irf, jax_entropy
)

def _eta_schedule(cfg: KiltConfig) -> np.ndarray:
    if cfg.eta_grid is not None:
        return np.array(cfg.eta_grid, dtype=float)
    return np.logspace(np.log10(cfg.eta_start), np.log10(cfg.eta_end), cfg.eta_rounds)

def run_ilt_2d(
    time: np.ndarray,             # (t,)
    tau: np.ndarray,              # (τ,)
    corr_maps: np.ndarray,        # (ΔT, t, t)
    cfg: KiltConfig,
    A_init: np.ndarray = None,  # optional initial A (τ, τ)
) -> Dict[str, Any]:
    if not JAX_AVAILABLE:
        raise RuntimeError("run_ilt_2d requires jax + jaxopt installed.")

    # Shapes
    t_bins = time.size
    tau_bins = tau.size
    delay_bins = corr_maps.shape[0]

    # Base decay kernel K(t, τ)
    K = jnp.exp(-jnp.outer(jnp.array(time), 1.0 / jnp.array(tau)))  # (t, τ)

    # Prepare JAX pack/unpack
    pack, unpack, tri_idx, tri_len = jax_build_packers(tau_bins, delay_bins)

    def loss(theta, eta, m_seed, K_irf, sigma2):
        A, Ulist = unpack(theta)
        sim = jax_model_maps(A, Ulist, K_irf)
        chi2 = jnp.mean((corr_maps - sim) ** 2 / (sim + sigma2))
        ent = jax_entropy(A, m_seed)
        return chi2 - eta * ent

    if cfg.irf_shift_list is None:
        cfg.irf_shift_list = range(0, 1 if not cfg.optimize_irf_shift else 3)

    # IRF
    irf_np = np.array(cfg.irf, dtype=float)
    best = dict(loss=np.inf)
    trials: List[Dict[str, Any]] = []

    # Prepare corr as jax array
    corr = jnp.array(corr_maps, dtype=jnp.float32)

    for shift in cfg.irf_shift_list:
        # Align IRF
        irf_shifted_np = irf_np
        if cfg.optimize_irf_shift:
            rise_fl = int(np.argmax(np.diag(corr_maps[0])))
            rise_irf = int(np.argmax(irf_np)) + int(shift)
            irf_shifted_np = shift_irf(irf_np, rise_fl - rise_irf)

        # Build K_irf (either plain K or Toeplitz-convolved K)
        if cfg.use_irf_convolution:
            T = toeplitz_irf_matrix(irf_shifted_np, tau_bins)  # (τ, τ)
            K_irf = jnp.array(T.T @ np.array(K))  # simple linear operator
        else:
            K_irf = K

        # Initialize A0 and U0
        if A_init is not None:
            A0 = jnp.array(A_init, dtype=jnp.float32)
        else:
            # Default initialization
            A0 = np.ones((tau_bins, tau_bins), dtype=jnp.float32)
        A0 /= A0.sum()  # Normalize the initial A matrix
        U0 = [jnp.eye(tau_bins) for _ in range(delay_bins)]
        theta0 = pack(A0, U0)
        m_prior = jnp.ones_like(A0)

        for eta in _eta_schedule(cfg):
            # Allow multiple LBFGS rounds per eta (annealing refinement)
            theta_seed = theta0
            m_seed = m_prior
            for _ in range(int(cfg.rounds_per_eta)):
                loss_fun = lambda th: (loss(th, eta, m_seed, K_irf, cfg.sigma2),
                                       jax.grad(lambda th: loss(th, eta, m_seed, K_irf, cfg.sigma2))(th))
                # run LBFGS
                params_opt, Q_val = jax_lbfgs_run(theta_seed, loss_fun, int(cfg.maxiter))
                theta_seed = params_opt
                A_tmp, _Utmp = unpack(theta_seed)
                # update m_prior (column mean heuristic)
                m_seed = jnp.mean(A_tmp, axis=1, keepdims=True) * jnp.ones_like(A_tmp)

            # Final evaluation for this (shift, eta)
            A_fin, U_fin = unpack(theta_seed)
            ilt_maps_tau = A_fin @ jnp.stack([U @ U.T for U in U_fin]) @ A_fin.T # (ΔT, τ, τ)
            model_out = jax_model_maps(A_fin, U_fin, K_irf)             # (ΔT, t, t)

            trial = {
                "shift": int(shift), "eta": float(eta), "loss": float(Q_val),
                "A": np.array(A_fin), "U": [np.array(u) for u in U_fin],
                "ilt_maps_tau": np.array(ilt_maps_tau),
                "model": np.array(model_out),
                "irf_shifted": irf_shifted_np.copy(),
            }
            trials.append(trial)

            if float(Q_val) < best.get("loss", np.inf):
                best = trial

    return {
        "best": best,
        "trials": trials,
        "meta": {
            "opt_irf_shift": bool(cfg.optimize_irf_shift),
            "opt_eta": bool(cfg.optimize_eta),
            "rounds_per_eta": int(cfg.rounds_per_eta),
            "sigma2": float(cfg.sigma2),
            "use_irf_convolution": bool(cfg.use_irf_convolution),
        }
    }
