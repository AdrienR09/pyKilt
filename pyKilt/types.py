from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Dict, Any

@dataclass
class ILT1DConfig:
    time: np.ndarray              # measured time axis (t)
    tau: np.ndarray               # lifetime axis (τ)
    irf: np.ndarray               # IRF kernel
    conv_pad: int = 10
    eta_grid: Optional[np.ndarray] = None
    irf_shift_list: Optional[np.ndarray] = None
    m_prior: Optional[np.ndarray] = None
    y0: float = 0.0               # baseline
    maxiter: int = 1000
    verbose: bool = True

@dataclass
class ILT2DConfig:
    time: np.ndarray
    tau: np.ndarray
    corr_maps: np.ndarray
    irf: np.ndarray
    delay_bins: int
    eta_grid: Optional[np.ndarray] = None
    irf_shift_list: Optional[np.ndarray] = None
    m_prior: Optional[np.ndarray] = None
    maxiter: int = 2000
    verbose: bool = True

@dataclass
class ILTResult:
    best_params: Dict[str, Any]
    best_loss: float
    loss_history: List[float]
    best_irf_shift: Optional[int]
    best_eta: Optional[float]
    outputs: Dict[str, Any]
