"""Frost probability and risk tier computation."""

from typing import Tuple
import numpy as np
from scipy.special import erf

from backend.app.config import settings


def compute_frost_probability(
    mu: np.ndarray | float,
    sigma: np.ndarray | float,
    threshold_c: float = settings.FROST_THRESHOLD_C,
) -> np.ndarray | float:
    """Compute frost probability P(T < threshold) = Phi((threshold - mu) / sigma).
    
    When sigma -> 0, returns step function (1.0 if mu < threshold else 0.0).
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    # Avoid divide-by-zero
    safe_sigma = np.maximum(sigma_arr, 1e-6)
    z = (threshold_c - mu_arr) / (safe_sigma * np.sqrt(2.0))
    # Phi(z) = 0.5 * (1 + erf(z))
    p_frost = 0.5 * (1.0 + erf(z))

    # If input was scalar, return scalar float
    if np.isscalar(mu) and np.isscalar(sigma):
        return float(p_frost.item())
    return p_frost


def classify_risk_tier(
    p_frost: np.ndarray | float,
    mu: np.ndarray | float,
    sigma: np.ndarray | float,
    threshold_c: float = settings.FROST_THRESHOLD_C,
) -> Tuple[np.ndarray | bool, np.ndarray | bool]:
    """Classify parcels into:
    - critical_action: p_frost > 0.50
    - watch_tier: p_frost > 0.20 or high uncertainty near the freezing threshold (|mu| <= 1.5*sigma).
    """
    p_arr = np.asarray(p_frost, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sig_arr = np.asarray(sigma, dtype=np.float64)

    critical = p_arr > 0.50
    near_threshold = np.abs(mu_arr - threshold_c) <= (1.5 * sig_arr)
    watch = (p_arr > 0.20) | (near_threshold & (sig_arr >= 0.5))

    if np.isscalar(p_frost):
        return bool(critical.item()), bool(watch.item())
    return critical, watch
