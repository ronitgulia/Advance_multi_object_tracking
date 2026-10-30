"""
evaluate.py
===========
Benchmarking and evaluation module.

Generates synthetic ground-truth trajectories, injects noise, runs both
Kalman Filter and One Euro Filter, computes quantitative metrics, and
produces matplotlib plots saved to /results/.

Metrics
───────
• RMSE       : Root Mean Squared Error vs. ground truth
• Jitter     : Mean variance of first-order differences (frame-to-frame jitter)
• Lag        : Cross-correlation peak offset between filter output and GT
• Robustness : RMSE when a fraction of observations are randomly dropped

Author: Portfolio project — Kalman Filter Sensor Fusion
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add src/ to path so we can import our modules
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kalman_filter import KalmanFilterCV
from src.one_euro_filter import OneEuroFilter2D

# Results directory — two levels up from src/
_RESULTS_DIR = _SRC.parent / "results"


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_trajectory(
    n_frames: int = 300,
    velocity: tuple[float, float] = (2.0, 1.5),
    start: tuple[float, float] = (50.0, 50.0),
    measurement_noise_std: float = 8.0,
    drop_fraction: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Generate a constant-velocity ground-truth trajectory with Gaussian
    measurement noise and optional frame drops.

    Parameters
    ----------
    n_frames : int
        Number of frames to simulate.
    velocity : (vx, vy)
        Ground-truth velocity in pixels/frame.
    start : (px0, py0)
        Starting position.
    measurement_noise_std : float
        Std dev of position measurement noise (pixels).
    drop_fraction : float
        Fraction of frames where the detection is 'missed' (0.0–1.0).
    rng : numpy random Generator

    Returns
    -------
    dict with keys:
        'ground_truth'  : ndarray (N, 2) — true positions
        'measurements'  : ndarray (N, 2) — noisy measurements (NaN = missed)
        'observed_mask' : ndarray (N,)  bool — True where measurement exists
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    gt = np.zeros((n_frames, 2), dtype=np.float64)
    gt[0] = start
    for k in range(1, n_frames):
        gt[k] = gt[k - 1] + np.array(velocity)

    # Gaussian noise on measurements
    noise = rng.normal(0, measurement_noise_std, size=(n_frames, 2))
    measurements = gt + noise

    # Random frame drops
    observed_mask = np.ones(n_frames, dtype=bool)
    if drop_fraction > 0.0:
        n_drop = int(n_frames * drop_fraction)
        drop_idx = rng.choice(n_frames, size=n_drop, replace=False)
        observed_mask[drop_idx] = False
        measurements[drop_idx] = np.nan

    return {
        "ground_truth": gt,
        "measurements": measurements,
        "observed_mask": observed_mask,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Running filters on synthetic data
# ──────────────────────────────────────────────────────────────────────────────

def run_kalman_filter(
    measurements: np.ndarray,
    observed_mask: np.ndarray,
    process_noise_std: float = 2.0,
    measurement_noise_std: float = 8.0,
) -> np.ndarray:
    """
    Run the KalmanFilterCV on a sequence of 2-D measurements.

    Parameters
    ----------
    measurements : ndarray (N, 2), NaN where missed
    observed_mask : ndarray (N,) bool
    process_noise_std, measurement_noise_std : filter tuning

    Returns
    -------
    ndarray (N, 2) — KF position estimates for every frame
    """
    kf = KalmanFilterCV(
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
    )

    # Initialise from first valid measurement
    first_valid = np.where(observed_mask)[0][0]
    kf.initialise(measurements[first_valid])

    positions = np.full((len(measurements), 2), np.nan)

    for k in range(first_valid, len(measurements)):
        pred = kf.predict()
        if observed_mask[k]:
            state = kf.update(measurements[k])
        else:
            state = pred  # predict-only on missed frames
        positions[k] = state[:2]

    return positions


def run_one_euro_filter(
    measurements: np.ndarray,
    observed_mask: np.ndarray,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
) -> np.ndarray:
    """
    Run the OneEuroFilter2D on a sequence of 2-D measurements.

    On missed frames the filter repeats its last output (no observation
    -> signal appears stationary -> strong smoothing applied).

    Returns
    -------
    ndarray (N, 2) — OEF position estimates
    """
    oef = OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta)

    first_valid = np.where(observed_mask)[0][0]
    positions = np.full((len(measurements), 2), np.nan)
    last_pos = measurements[first_valid]

    for k in range(first_valid, len(measurements)):
        z = measurements[k] if observed_mask[k] else last_pos
        out = oef.filter(z)
        positions[k] = out
        if observed_mask[k]:
            last_pos = measurements[k]

    return positions


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """RMSE of 2-D position estimates vs. ground truth (pixels)."""
    valid = ~np.isnan(estimates[:, 0])
    diff = estimates[valid] - ground_truth[valid]
    return float(np.sqrt(np.mean(diff**2)))


def compute_jitter(estimates: np.ndarray) -> float:
    """
    Jitter = mean variance of frame-to-frame position differences.

    A smoother trajectory has smaller jitter.
    """
    valid = ~np.isnan(estimates[:, 0])
    pos = estimates[valid]
    diffs = np.diff(pos, axis=0)
    return float(np.var(diffs, axis=0).mean())


def compute_lag(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Estimate the lag (in frames) introduced by the filter.

    Uses cross-correlation of the x-dimension signal.
    Positive lag -> filter trails ground truth.
    """
    valid = ~np.isnan(estimates[:, 0])
