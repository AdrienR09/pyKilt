"""Shared optimizer helpers (small utilities)."""

from typing import Callable, Any, List, Tuple

def run_irf_shift_loop(scan_fn: Callable, irf_shift_list, eta_grid, *args, **kwargs):
    """
    Generic driver for scanning IRF shifts and eta. `scan_fn` performs a single-fit
    for given (shift, eta) and must return (loss, outputs_dict).
    """
    best_loss = float("inf")
    best_out = None
    loss_history = []
    for shift in irf_shift_list:
        for eta in eta_grid:
            loss, out = scan_fn(shift, eta, *args, **kwargs)
            loss_history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_out = (shift, eta, out)
    return best_loss, best_out, loss_history
