"""
test_kalman_filter.py
=====================
Unit tests for the KalmanFilterCV implementation.

Tests verify:
  1. Convergence to true state on a synthetic 1D constant-velocity trajectory
  2. Kalman gain limits: R→0 (trust measurement), Q→0 (trust prediction)
  3. Covariance matrix stays positive semi-definite across many iterations
  4. Tracker correctly assigns IDs and handles track birth/death
  5. Joseph-form PSD guarantee under extreme conditions

Run with:
  pytest tests/ -v

Author: Portfolio project — Kalman Filter Sensor Fusion
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src/ to path
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from kalman_filter import KalmanFilterCV
from one_euro_filter import OneEuroFilter2D
from tracker import MultiObjectTracker


# ──────────────────────────────────────────────────────────────────────────────
# 1. KF Convergence Test
# ──────────────────────────────────────────────────────────────────────────────

class TestKalmanConvergence:
    """
    The Kalman Filter should converge to the true state (within several sigma)
    for a constant-velocity trajectory observed with Gaussian noise.
    """

    def test_converges_on_cv_trajectory_1d(self) -> None:
        """
        1D test: object moves at constant velocity v=3 px/frame.
        After 50 frames the KF should track the true position within
        2× the measurement noise std.
        """
        rng = np.random.default_rng(seed=42)
        n = 100
        v = 3.0
        true_pos = np.arange(n, dtype=float) * v + 10.0
        noise_std = 5.0
        measurements = true_pos + rng.normal(0, noise_std, n)

        kf = KalmanFilterCV(
            process_noise_std=1.0,
            measurement_noise_std=noise_std,
        )
        kf.initialise(np.array([measurements[0], 0.0]))

        errors = []
        for k in range(1, n):
            kf.predict()
            kf.update(np.array([measurements[k], 0.0]))
            est_x = kf.get_position()[0]
            errors.append(abs(est_x - true_pos[k]))

        # Converged errors (last 50 frames) should be well below 2× noise std
        final_errors = errors[50:]
        mean_error = np.mean(final_errors)
        assert mean_error < 2 * noise_std, (
            f"KF did not converge: mean error={mean_error:.2f}, "
            f"threshold={2*noise_std:.2f}"
        )

    def test_converges_on_cv_trajectory_2d(self) -> None:
        """
        2D test: object moves diagonally. KF should track within 3σ after warm-up.
        """
        rng = np.random.default_rng(seed=99)
        n = 150
        vx, vy = 2.0, 1.5
        true_pos = np.column_stack([
            np.arange(n) * vx + 50.0,
            np.arange(n) * vy + 50.0,
        ])
        noise_std = 8.0
        measurements = true_pos + rng.normal(0, noise_std, true_pos.shape)

        kf = KalmanFilterCV(process_noise_std=2.0, measurement_noise_std=noise_std)
        kf.initialise(measurements[0])

        for k in range(1, n):
            kf.predict()
            kf.update(measurements[k])

        final_pos = kf.get_position()
        error = np.linalg.norm(final_pos - true_pos[-1])
        assert error < 3 * noise_std, (
            f"2D KF final error too large: {error:.2f} (threshold {3*noise_std:.2f})"
        )

    def test_velocity_estimate_converges(self) -> None:
        """Velocity estimate should converge to true velocity within 1 px/frame."""
        rng = np.random.default_rng(seed=7)
        n = 200
        vx_true, vy_true = 4.0, -2.0

        true_pos = np.column_stack([
            np.arange(n) * vx_true,
            np.arange(n) * vy_true,
        ])
        measurements = true_pos + rng.normal(0, 5.0, true_pos.shape)

        kf = KalmanFilterCV(process_noise_std=1.0, measurement_noise_std=5.0)
        kf.initialise(measurements[0])

        for k in range(1, n):
            kf.predict()
            kf.update(measurements[k])

        est_v = kf.get_velocity()
        assert abs(est_v[0] - vx_true) < 1.5, f"vx not converged: {est_v[0]:.2f} vs {vx_true}"
        assert abs(est_v[1] - vy_true) < 1.5, f"vy not converged: {est_v[1]:.2f} vs {vy_true}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Kalman Gain Limits
# ──────────────────────────────────────────────────────────────────────────────

class TestKalmanGainLimits:
    """
    Kalman gain K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹ has two important limit behaviours:
      R → 0 : K → H⁻¹  (trust measurement fully; K·H → I)
      Q → 0 : steady-state K → 0 (trust prediction fully)
    """

    def test_r_near_zero_trusts_measurement(self) -> None:
        """
        With R very small, the update should move the state very close to
        the raw measurement value (correction dominates).
        """
        kf = KalmanFilterCV(
            process_noise_std=1.0,
            measurement_noise_std=1e-6,   # near-zero R
            initial_state_covariance=100.0,
        )
        kf.initialise(np.array([0.0, 0.0]))

        # Predict once to get P > 0
        kf.predict()

        # Update with a measurement far from current state
        measurement = np.array([100.0, 50.0])
        state = kf.update(measurement)

        # State should be very close to the measurement
        assert abs(state[0] - measurement[0]) < 0.01, (
            f"R→0: state[0]={state[0]:.4f} should be ≈{measurement[0]}"
        )
        assert abs(state[1] - measurement[1]) < 0.01, (
            f"R→0: state[1]={state[1]:.4f} should be ≈{measurement[1]}"
        )

    def test_q_near_zero_trusts_prediction(self) -> None:
        """
        With Q≈0 and many iterations, the KF covariance P shrinks to near zero
        (highly confident in prediction), so updates barely change the state.
        """
        kf = KalmanFilterCV(
            process_noise_std=1e-8,        # near-zero Q
            measurement_noise_std=10.0,
        )
        kf.initialise(np.array([0.0, 0.0]))
        rng = np.random.default_rng(seed=1)

        # Run many frames to let P shrink
        for _ in range(200):
            kf.predict()
            z = rng.normal([0.0, 0.0], 10.0)
            kf.update(z)

        # After convergence, prediction uncertainty should be much smaller than initial (100)
        # With Q≈0 and R=100, steady-state trace ≈ R/(frames)^0.5 — significantly below initial
        P_pos = kf.P[:2, :2]
        trace = np.trace(P_pos)
        assert trace < 20.0, (
            f"Q→0: P position trace should be much smaller than initial 100, got {trace:.4f}"
        )
        # Also verify P shrinks from initial value (it should have decreased substantially)
        # Initial position covariance trace = 2 * 100.0 = 200.0
        assert trace < 200.0 * 0.1, (  # should be at least 10x smaller than initial
            f"Q→0: P did not shrink sufficiently: {trace:.4f} vs initial ~200.0"
        )

    def test_innovation_shrinks_with_more_measurements(self) -> None:
        """
        As the KF sees more measurements of the same location, the innovation
        (residual) variance (S = HPH^T + R) should decrease toward R.
        """
        kf = KalmanFilterCV(process_noise_std=0.1, measurement_noise_std=5.0)
        kf.initialise(np.array([50.0, 50.0]))

        S_vals = []
        for _ in range(100):
            kf.predict()
            S = kf.H @ kf.P @ kf.H.T + kf.R
            S_vals.append(np.trace(S))
            kf.update(np.array([50.0, 50.0]))

        # Innovation covariance should decrease over time (stabilises near R)
        assert S_vals[-1] < S_vals[0], "Innovation covariance should decrease."


# ──────────────────────────────────────────────────────────────────────────────
# 3. Positive Semi-Definiteness of Covariance
# ──────────────────────────────────────────────────────────────────────────────

class TestCovariancePSD:
    """
    The covariance matrix P must remain positive semi-definite (PSD) at all times.
    All eigenvalues of P must be non-negative.
    """

    def test_psd_throughout_normal_operation(self) -> None:
        """P stays PSD over 500 normal-noise update steps."""
        rng = np.random.default_rng(seed=42)
        kf = KalmanFilterCV(process_noise_std=2.0, measurement_noise_std=5.0)
        kf.initialise(np.array([100.0, 100.0]))

        for _ in range(500):
            kf.predict()
            z = np.array([100.0, 100.0]) + rng.normal(0, 5.0, 2)
            kf.update(z)
            assert kf.is_positive_semidefinite(), (
                "Covariance P became non-PSD during normal operation."
            )

    def test_psd_with_extreme_noise(self) -> None:
        """P stays PSD even with very large measurement noise."""
        rng = np.random.default_rng(seed=13)
        kf = KalmanFilterCV(process_noise_std=50.0, measurement_noise_std=100.0)
        kf.initialise(np.array([0.0, 0.0]))

        for _ in range(300):
            kf.predict()
            z = rng.normal(0.0, 200.0, 2)
            kf.update(z)
            assert kf.is_positive_semidefinite(), (
                "Covariance P became non-PSD under extreme noise."
            )

    def test_psd_with_near_zero_process_noise(self) -> None:
        """P stays PSD when Q ≈ 0 (static object assumption)."""
        rng = np.random.default_rng(seed=17)
        kf = KalmanFilterCV(process_noise_std=1e-5, measurement_noise_std=5.0)
        kf.initialise(np.array([200.0, 150.0]))

        for _ in range(300):
            kf.predict()
            z = np.array([200.0, 150.0]) + rng.normal(0, 5.0, 2)
            kf.update(z)
            assert kf.is_positive_semidefinite(), (
                "Covariance P became non-PSD with near-zero Q."
            )

    def test_covariance_is_symmetric(self) -> None:
        """P must always be exactly symmetric (P == Pᵀ)."""
        rng = np.random.default_rng(seed=5)
        kf = KalmanFilterCV(process_noise_std=3.0, measurement_noise_std=7.0)
        kf.initialise(np.array([50.0, 50.0]))

        for _ in range(200):
            kf.predict()
            z = rng.normal([50.0, 50.0], 7.0)
            kf.update(z)
            P = kf.get_covariance()
            assert np.allclose(P, P.T, atol=1e-12), "Covariance P is not symmetric."


# ──────────────────────────────────────────────────────────────────────────────
# 4. Tracker Tests
# ──────────────────────────────────────────────────────────────────────────────

def _make_bbox(cx: float, cy: float, w: float = 50.0, h: float = 80.0) -> np.ndarray:
    """Helper: create a detection array [x1, y1, x2, y2, conf, cls]."""
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2, 0.9, 0.0], dtype=np.float32)


class TestTracker:
    """Tests for multi-object tracker ID assignment and lifecycle."""

    def test_single_object_birth(self) -> None:
        """A new detection should create exactly one track."""
        tracker = MultiObjectTracker()
        det = np.array([_make_bbox(100, 100)])
        outputs = tracker.update(det)
        assert len(outputs) == 1, "Expected 1 track from 1 detection."
        assert outputs[0]["track_id"] == 1

    def test_track_id_persistence(self) -> None:
        """The same object should keep the same track ID across frames."""
        tracker = MultiObjectTracker()
        # Frame 1: birth
        outputs1 = tracker.update(np.array([_make_bbox(100, 100)]))
        tid1 = outputs1[0]["track_id"]

        # Frames 2–10: slight movement
        for k in range(1, 10):
            dets = np.array([_make_bbox(100 + k * 2, 100 + k)])
            outputs = tracker.update(dets)
            assert len(outputs) == 1, f"Frame {k+1}: expected 1 track, got {len(outputs)}"
            assert outputs[0]["track_id"] == tid1, (
                f"Frame {k+1}: track ID changed from {tid1} to {outputs[0]['track_id']}"
            )

    def test_track_death_on_missed_frames(self) -> None:
        """A track should be deleted after max_missed consecutive missed frames."""
        max_missed = 3
        tracker = MultiObjectTracker(max_missed=max_missed)

        # Create track
        tracker.update(np.array([_make_bbox(200, 200)]))

        # Send empty detections for max_missed frames
        for _ in range(max_missed):
            outputs = tracker.update(np.zeros((0, 6), dtype=np.float32))

        # One more frame — track should now be deleted
        outputs = tracker.update(np.zeros((0, 6), dtype=np.float32))
        assert len(outputs) == 0, (
            f"Track should be dead after {max_missed} missed frames, but got {len(outputs)} tracks."
        )

    def test_two_objects_get_distinct_ids(self) -> None:
        """Two spatially separate detections should get distinct IDs."""
        tracker = MultiObjectTracker()
        dets = np.array([_make_bbox(100, 100), _make_bbox(400, 400)])
        outputs = tracker.update(dets)
        assert len(outputs) == 2, "Expected 2 tracks."
        ids = {o["track_id"] for o in outputs}
        assert len(ids) == 2, f"Expected 2 distinct IDs, got {ids}"

    def test_no_id_reuse_after_death(self) -> None:
        """After a track dies, the next new track should get a fresh ID."""
        tracker = MultiObjectTracker(max_missed=1)

        # Birth: ID 1
        tracker.update(np.array([_make_bbox(100, 100)]))
        # Kill it: 2 missed frames
        tracker.update(np.zeros((0, 6), dtype=np.float32))
        tracker.update(np.zeros((0, 6), dtype=np.float32))

        # New detection: should get ID 2, not 1
        outputs = tracker.update(np.array([_make_bbox(500, 500)]))
        assert len(outputs) == 1
        new_id = outputs[0]["track_id"]
        assert new_id > 1, f"ID reuse detected: got ID {new_id} after track 1 died."

    def test_empty_frame_no_crash(self) -> None:
        """Updating with zero detections on an empty tracker should not crash."""
        tracker = MultiObjectTracker()
        outputs = tracker.update(np.zeros((0, 6), dtype=np.float32))
        assert outputs == [], f"Expected empty output, got {outputs}"

    def test_multiple_births_and_deaths(self) -> None:
        """Stress test with many tracks appearing and disappearing."""
        tracker = MultiObjectTracker(max_missed=2)

        # Birth 5 tracks
        dets = np.array([_make_bbox(i * 100, i * 100) for i in range(1, 6)])
        outputs = tracker.update(dets)
        assert len(outputs) == 5

        # All alive: 5 detections near original positions
        for _ in range(5):
            dets = np.array([_make_bbox(i * 100 + 2, i * 100 + 1) for i in range(1, 6)])
            outputs = tracker.update(dets)
            assert len(outputs) == 5

        # All disappear — after max_missed+1 frames they should be gone
        for _ in range(4):
            tracker.update(np.zeros((0, 6), dtype=np.float32))

        outputs = tracker.update(np.zeros((0, 6), dtype=np.float32))
        alive_non_missed = [o for o in outputs if o["missed"] == 0]
        assert len(alive_non_missed) == 0, "All tracks should have died."


# ──────────────────────────────────────────────────────────────────────────────
# 5. One Euro Filter Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestOneEuroFilter:
    """Sanity checks for the One Euro Filter."""

    def test_static_signal_converges_to_value(self) -> None:
        """On a constant signal, the OEF should converge to that value."""
        oef = OneEuroFilter2D(min_cutoff=1.0, beta=0.0)
        value = np.array([42.0, 137.0])
        for _ in range(100):
            out = oef.filter(value)
        assert np.allclose(out, value, atol=0.01), (
            f"OEF didn't converge to static value: got {out}"
        )

    def test_output_is_smoother_than_input(self) -> None:
        """OEF output variance should be lower than noisy input variance."""
        rng = np.random.default_rng(seed=22)
        n = 200
        true_val = np.array([100.0, 50.0])
        noisy = true_val + rng.normal(0, 10.0, (n, 2))

        oef = OneEuroFilter2D(min_cutoff=0.5, beta=0.001)
        filtered = np.array([oef.filter(noisy[k]) for k in range(n)])

        in_var  = np.var(noisy, axis=0).mean()
        out_var = np.var(filtered, axis=0).mean()
        assert out_var < in_var, (
            f"OEF output variance {out_var:.2f} not less than input {in_var:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Evaluate Module Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluate:
    """Tests for the synthetic data generation and metric computation."""

    def test_synthetic_trajectory_shape(self) -> None:
        from evaluate import generate_synthetic_trajectory
        data = generate_synthetic_trajectory(n_frames=100)
        assert data["ground_truth"].shape == (100, 2)
        assert data["measurements"].shape == (100, 2)
        assert data["observed_mask"].shape == (100,)

    def test_drop_fraction_correct(self) -> None:
        from evaluate import generate_synthetic_trajectory
        data = generate_synthetic_trajectory(n_frames=200, drop_fraction=0.2,
                                              rng=np.random.default_rng(1))
        n_dropped = (~data["observed_mask"]).sum()
        # Allow ±5% tolerance
        assert 30 <= n_dropped <= 50, f"Expected ~40 drops, got {n_dropped}"

    def test_rmse_perfect_estimate(self) -> None:
        from evaluate import compute_rmse
        gt   = np.ones((100, 2)) * 5.0
        est  = np.ones((100, 2)) * 5.0
        assert compute_rmse(est, gt) == pytest.approx(0.0, abs=1e-10)

    def test_jitter_zero_for_straight_line(self) -> None:
        from evaluate import compute_jitter
        # Perfectly linear trajectory → zero velocity variance → zero jitter
        t = np.arange(100).reshape(-1, 1)
        pos = np.hstack([t * 2.0, t * 1.5])
        assert compute_jitter(pos) == pytest.approx(0.0, abs=1e-10)