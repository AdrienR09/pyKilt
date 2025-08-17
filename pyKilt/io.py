"""
I/O helpers + corr_map builder that wraps fdc_core.
"""

from typing import Dict, Tuple
import pickle
import numpy as np

from .core import fdc_core

def load_pickles(ut_path: str, mt_path: str) -> Tuple[Dict, Dict]:
    with open(ut_path, "rb") as f:
        ut = pickle.load(f)
    with open(mt_path, "rb") as f:
        mt = pickle.load(f)
    return ut, mt

def load_npz(path: str) -> dict:
    return dict(np.load(path, allow_pickle=True))

def build_corr_maps_from_dicts(
    ut_dict: Dict, mt_dict: Dict,
    delays: np.ndarray, ddT: float, bins: np.ndarray,
    tau_min: float = 0.0,
    Ith: float = 700.0,
    acquisition_time: float = 30.0,
    N_hist: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loop through molecules in ut_dict / mt_dict, filter and build corr_maps via fdc_core.
    Returns (corr_maps, decay_hist).
    """
    tau_bins = len(bins) - 1
    corr_maps = np.zeros((len(delays), tau_bins, tau_bins), dtype=np.float64)
    decay = np.zeros(tau_bins, dtype=np.float64)

    for mol in ut_dict.keys():
        macrotimes, microtimes = mt_dict[mol], ut_dict[mol]

        macrotimes = np.nan_to_num(macrotimes, 0.0)
        macrotimes = macrotimes[macrotimes > 0.0]
        if macrotimes.size == 0:
            continue
        macrotimes -= macrotimes.min()

        microtimes = np.nan_to_num(microtimes, 0.0) * 1e9
        if microtimes.size == 0:
            continue
        microtimes = microtimes % 12.5

        # Intensity threshold
        trace, tr_bins = np.histogram(macrotimes, bins=N_hist)
        trace = trace * N_hist / acquisition_time
        if np.mean(trace) < Ith:
            continue

        # Windowing
        mask = microtimes > tau_min
        macrotimes = macrotimes[mask]
        microtimes = microtimes[mask]
        microtimes -= tau_min
        if microtimes.size == 0:
            continue

        decay += np.histogram(microtimes, bins=tau_bins)[0]

        macrotimes = macrotimes.astype(np.float64)
        microtimes = microtimes.astype(np.float64)

        fdc_core(macrotimes, microtimes, delays, ddT, bins, corr_maps)

    return corr_maps, decay

