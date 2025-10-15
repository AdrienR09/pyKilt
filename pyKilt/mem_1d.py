"""
1D ILT (MEM) with optional IRF shift optimization and η annealing/scan.
SciPy L-BFGS-B optimizer (NumPy objective). JAX not required for 1D.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.optimize import minimize

from .core import (
    KiltConfig, exp_kernel, make_irf_matrix_conv, shift_irf,
    objective_np, reconvolve_np
)

def _eta_schedule(cfg: KiltConfig) -> np.ndarray:
    if cfg.eta_grid is not None:
        return np.array(cfg.eta_grid, dtype=float)
    return np.logspace(np.log10(cfg.eta_start), np.log10(cfg.eta_end), cfg.eta_rounds)

def run_ilt_1d(
    time: np.ndarray,
    tau: np.ndarray,
    ExpData: np.ndarray,
    cfg: KiltConfig,
    A_init: np.ndarray = None,
    m_prior: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Returns dict containing best A, fit curve, chosen (shift, eta), and per-trial history.
    """
    tau_bins = tau.size
    K = exp_kernel(time, tau)

    if cfg.irf_shift_list is None:
        cfg.irf_shift_list = range(0, 1 if not cfg.optimize_irf_shift else 5)

    if A_init is None:
        A_init = np.ones(tau_bins) / tau_bins
    if m_prior is None:
        m_prior = A_init.copy()

    y0 = ExpData[0] if cfg.baseline is None else cfg.baseline

    best = dict(loss=np.inf)
    trials: List[Dict[str, Any]] = []

    for shift in cfg.irf_shift_list:
        # Align IRF if requested
        irf_shifted = cfg.irf
        if cfg.optimize_irf_shift:
            rise_fl = int(np.argmax(ExpData))
            rise_irf = int(np.argmax(cfg.irf)) + int(shift)
            irf_shifted = shift_irf(cfg.irf, rise_fl - rise_irf)

        if cfg.use_irf_convolution:
            K_irf = make_irf_matrix_conv(K, irf_shifted, conv_pad=cfg.conv_pad)
        else:
            K_irf = K

        # Anneal/scan eta
        A_seed = A_init.copy()
        m = m_prior.copy()
        for eta in _eta_schedule(cfg):
            res = minimize(
                objective_np, A_seed,
                args=(ExpData, K_irf, y0, m, float(eta), cfg.sigma2),
                method="L-BFGS-B",
                bounds=[(0, None)] * tau_bins,
                options={"maxiter": int(cfg.maxiter)}
            )
            A_fit = res.x
            loss = float(res.fun)
            model = reconvolve_np(A_fit, K_irf, y0)

            trials.append({
                "shift": int(shift), "eta": float(eta),
                "loss": loss, "A": A_fit.copy(), "fit": model.copy()
            })

            # re-seed for annealing
            A_seed = A_fit.copy()
            m = A_fit.copy()

            if loss < best["loss"]:
                best = dict(shift=int(shift), eta=float(eta), loss=loss,
                            A=A_fit.copy(), fit=model.copy(),
                            irf_shifted=irf_shifted.copy())

    return {
        "best": best,
        "trials": trials,
        "meta": {
            "baseline": float(y0),
            "conv_pad": int(cfg.conv_pad),
            "sigma2": float(cfg.sigma2),
            "opt_irf_shift": bool(cfg.optimize_irf_shift),
            "opt_eta": bool(cfg.optimize_eta),
        }
    }
