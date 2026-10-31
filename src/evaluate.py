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
    est_x = estimates[valid, 0] - estimates[valid, 0].mean()
    gt_x = ground_truth[valid, 0] - ground_truth[valid, 0].mean()
    corr = np.correlate(est_x, gt_x, mode="full")
    lag = corr.argmax() - (len(gt_x) - 1)
    return float(lag)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory_comparison(
    ground_truth: np.ndarray,
    measurements: np.ndarray,
    kf_estimates: np.ndarray,
    oef_estimates: np.ndarray,
    observed_mask: np.ndarray,
    title: str = "Trajectory Comparison",
    filename: str = "trajectory_comparison.png",
) -> Path:
    """2-D trajectory overlay plot."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    valid_obs = observed_mask

    # Ground truth
    ax.plot(ground_truth[:, 0], ground_truth[:, 1],
            color="#e2e2e2", linewidth=2.5, label="Ground Truth", zorder=3)

    # Raw noisy measurements
    ax.scatter(
        measurements[valid_obs, 0], measurements[valid_obs, 1],
        color="#f7971e", s=12, alpha=0.6, label="Noisy Measurements", zorder=2,
    )

    # Missed frames
    missed = ~observed_mask
    if missed.any():
        # Mark first/last GT positions to show where drops occurred
        ax.scatter(
            ground_truth[missed, 0], ground_truth[missed, 1],
            marker="x", color="#ff4757", s=30, alpha=0.7, label="Dropped Frames", zorder=2,
        )

    # KF estimate
    valid_kf = ~np.isnan(kf_estimates[:, 0])
    ax.plot(kf_estimates[valid_kf, 0], kf_estimates[valid_kf, 1],
            color="#2ed573", linewidth=2, label="Kalman Filter", zorder=4)

    # OEF estimate
    valid_oef = ~np.isnan(oef_estimates[:, 0])
    ax.plot(oef_estimates[valid_oef, 0], oef_estimates[valid_oef, 1],
            color="#ff6b81", linewidth=2, linestyle="--", label="One Euro Filter", zorder=4)

    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (pixels)", color="#aaaaaa")
    ax.set_ylabel("Y (pixels)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white", fontsize=10)

    out_path = _RESULTS_DIR / filename
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved -> {out_path}")
    return out_path


def plot_error_over_time(
    ground_truth: np.ndarray,
    kf_estimates: np.ndarray,
    oef_estimates: np.ndarray,
    title: str = "Position Error Over Time",
    filename: str = "error_over_time.png",
) -> Path:
    """Per-frame L2 position error for KF and OEF."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    valid = ~np.isnan(kf_estimates[:, 0])
    frames = np.where(valid)[0]

    kf_err  = np.linalg.norm(kf_estimates[valid]  - ground_truth[valid], axis=1)
    oef_err = np.linalg.norm(oef_estimates[valid] - ground_truth[valid], axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.plot(frames, kf_err,  color="#2ed573", linewidth=1.5, label="Kalman Filter Error")
    ax.plot(frames, oef_err, color="#ff6b81", linewidth=1.5, linestyle="--", label="One Euro Filter Error")
    ax.fill_between(frames, kf_err, alpha=0.15, color="#2ed573")
    ax.fill_between(frames, oef_err, alpha=0.15, color="#ff6b81")

    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Frame", color="#aaaaaa")
    ax.set_ylabel("L2 Error (pixels)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white", fontsize=10)

    out_path = _RESULTS_DIR / filename
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved -> {out_path}")
    return out_path


def plot_benchmark_table(rows: list[dict], filename: str = "benchmark_table.png") -> Path:
    """
    Render a styled metrics table as an image.

    Parameters
    ----------
    rows : list of dicts with keys:
        'scenario', 'filter', 'rmse', 'jitter', 'lag'
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    col_labels = ["Scenario", "Filter", "RMSE (px)", "Jitter (px²)", "Lag (frames)"]
    cell_data = [
        [r["scenario"], r["filter"], f"{r['rmse']:.2f}", f"{r['jitter']:.3f}", f"{r['lag']:.1f}"]
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(10, 0.45 * (len(rows) + 2) + 1.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#0f3460")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(rows) + 1):
        bg = "#16213e" if i % 2 == 0 else "#1a1a2e"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
            table[i, j].set_text_props(color="white")
            # Highlight winning filter (lower RMSE) in green
            if col_labels[j] == "RMSE (px)":
                pass  # colour applied after all rows known

    ax.set_title("Benchmark Results", color="white", fontsize=14, fontweight="bold", pad=10)

    out_path = _RESULTS_DIR / filename
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Full benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def run_full_benchmark(n_frames: int = 300, verbose: bool = True) -> dict:
    """
    Run KF vs OEF benchmark across three scenarios:
      1. Low noise (easy)
      2. High noise (challenging)
      3. High noise + 30% frame drops (robustness)

    Returns
    -------
    dict mapping scenario name -> metric dict
    """
    scenarios = [
        {"name": "Low Noise",           "noise": 4.0,  "drop": 0.00},
        {"name": "High Noise",           "noise": 15.0, "drop": 0.00},
        {"name": "High Noise + Drops",   "noise": 15.0, "drop": 0.30},
    ]

    all_results = {}
    table_rows: list[dict] = []
    rng = np.random.default_rng(seed=7)

    for sc in scenarios:
        data = generate_synthetic_trajectory(
            n_frames=n_frames,
            measurement_noise_std=sc["noise"],
            drop_fraction=sc["drop"],
            rng=rng,
        )
        gt   = data["ground_truth"]
        meas = data["measurements"]
        mask = data["observed_mask"]

        kf_est  = run_kalman_filter(meas, mask,
                                    process_noise_std=2.0,
                                    measurement_noise_std=sc["noise"])
        oef_est = run_one_euro_filter(meas, mask)

        metrics = {
            "kf":  {"rmse": compute_rmse(kf_est,  gt),
                    "jitter": compute_jitter(kf_est),
                    "lag":    compute_lag(kf_est, gt)},
            "oef": {"rmse": compute_rmse(oef_est, gt),
                    "jitter": compute_jitter(oef_est),
                    "lag":    compute_lag(oef_est, gt)},
        }
        all_results[sc["name"]] = metrics

        # Safe filename
        safe_name = sc["name"].lower().replace(" ", "_").replace("+", "plus")

        # Trajectory plot
        plot_trajectory_comparison(
            gt, meas, kf_est, oef_est, mask,
            title=f"Trajectory Comparison — {sc['name']}",
            filename=f"trajectory_{safe_name}.png",
        )

        # Error-over-time plot
        plot_error_over_time(
            gt, kf_est, oef_est,
            title=f"Error Over Time — {sc['name']}",
            filename=f"error_{safe_name}.png",
        )

        for fname, fmetrics in [("Kalman Filter", metrics["kf"]),
                                  ("One Euro Filter", metrics["oef"])]:
            table_rows.append({
                "scenario": sc["name"],
                "filter": fname,
                "rmse":   fmetrics["rmse"],
                "jitter": fmetrics["jitter"],
                "lag":    fmetrics["lag"],
            })

        if verbose:
            print(f"\n[{sc['name']}]")
            print(f"  KF  -> RMSE={metrics['kf']['rmse']:.2f}  Jitter={metrics['kf']['jitter']:.3f}  Lag={metrics['kf']['lag']:.1f}")
            print(f"  OEF -> RMSE={metrics['oef']['rmse']:.2f}  Jitter={metrics['oef']['jitter']:.3f}  Lag={metrics['oef']['lag']:.1f}")

    plot_benchmark_table(table_rows)

    return all_results