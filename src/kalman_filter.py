"""
kalman_filter.py
================
From-scratch Kalman Filter implementation using NumPy only.

Motion model: Constant Velocity (CV)
  State vector x = [px, py, vx, vy]^T
  - (px, py) : 2D position
  - (vx, vy) : 2D velocity (pixels/frame)

No external filter libraries are used — all linear algebra is done
explicitly with numpy so the implementation is fully transparent and
interview-demonstrable.

Author: Portfolio project — Kalman Filter Sensor Fusion
"""

from __future__ import annotations

import numpy as np


class KalmanFilterCV:
    """
    Linear Kalman Filter with a 2-D Constant-Velocity (CV) motion model.

    State vector  x ∈ ℝ⁴ :  [px, py, vx, vy]ᵀ
    Measurement   z ∈ ℝ²  :  [px, py]ᵀ   (position only)

    Discrete-time system equations
    ───────────────────────────────
    Predict  :  x̂⁻ₖ = F · x̂ₖ₋₁
                P⁻ₖ  = F · Pₖ₋₁ · Fᵀ + Q

    Update   :  Kₖ   = P⁻ₖ · Hᵀ · (H · P⁻ₖ · Hᵀ + R)⁻¹
                x̂ₖ  = x̂⁻ₖ + Kₖ · (zₖ − H · x̂⁻ₖ)
                Pₖ   = (I − Kₖ · H) · P⁻ₖ · (I − Kₖ · H)ᵀ + Kₖ · R · Kₖᵀ
                       (Joseph form — numerically stable, always PSD)

    Parameters
    ----------
    dt : float
        Time step in seconds (or frames). Default = 1.0 (frame-based).
    process_noise_std : float
        Standard deviation of process noise (acceleration uncertainty).
        Governs how much we believe the velocity can change each step.
    measurement_noise_std : float
        Standard deviation of position measurement noise (pixels).
    initial_state_covariance : float
        Initial diagonal value of P — reflects uncertainty in the
        initial state estimate.
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 2.0,
        measurement_noise_std: float = 5.0,
        initial_state_covariance: float = 100.0,
    ) -> None:
        self.dt = dt

        # ── State transition matrix F ──────────────────────────────────
        # Encodes constant-velocity kinematics:
        #   pₖ = pₖ₋₁ + vₖ₋₁·Δt
        #   vₖ = vₖ₋₁           (velocity constant between steps)
        #
        #   F = │ 1  0  dt  0 │
        #       │ 0  1   0 dt │
        #       │ 0  0   1  0 │
        #       │ 0  0   0  1 │
        self.F: np.ndarray = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # ── Measurement matrix H ───────────────────────────────────────
        # We only observe position, not velocity:
        #   z = H · x  →  H picks the first two rows
        #
        #   H = │ 1  0  0  0 │
        #       │ 0  1  0  0 │
        self.H: np.ndarray = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # ── Process noise covariance Q ─────────────────────────────────
        # Derived from the discrete-time noise input model where
        # acceleration aₖ ~ N(0, σ²_a) drives the state:
        #
        #   Q = G · Gᵀ · σ²_a
        #
        # where G = [Δt²/2, Δt²/2, Δt, Δt]ᵀ (acceleration-to-state coupling)
        #
        # This is the standard piecewise-constant white-noise acceleration
        # (DWNA) model — see Bar-Shalom "Estimation with Applications" §6.3.
        sigma_a = process_noise_std  # acceleration noise std
        G = np.array(
            [dt**2 / 2, dt**2 / 2, dt, dt], dtype=np.float64
        )  # shape (4,)
        # Q = outer(G, G) * sigma_a^2, but we want separate x/y coupling
        # so we build block-diagonal structure:
        Gx = np.array([dt**2 / 2, 0.0, dt, 0.0], dtype=np.float64)
        Gy = np.array([0.0, dt**2 / 2, 0.0, dt], dtype=np.float64)
        self.Q: np.ndarray = sigma_a**2 * (
            np.outer(Gx, Gx) + np.outer(Gy, Gy)
        )

        # ── Measurement noise covariance R ─────────────────────────────
        # Diagonal: independent noise on x and y position measurements.
        sigma_z = measurement_noise_std
        self.R: np.ndarray = np.diag([sigma_z**2, sigma_z**2]).astype(np.float64)

        # ── State & covariance initialised in initialise() ─────────────
        self.x: np.ndarray = np.zeros((4, 1), dtype=np.float64)   # state mean
        self.P: np.ndarray = np.eye(4, dtype=np.float64) * initial_state_covariance

        # Identity for Joseph-form update (cached)
        self._I4: np.ndarray = np.eye(4, dtype=np.float64)

        # History of posterior state means (for plotting/evaluation)
        self.history: list[np.ndarray] = []

    # ──────────────────────────────────────────────────────────────────
    def initialise(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        """
        Set the initial state from a measured position.

        Parameters
        ----------
        position : ndarray of shape (2,)
            Initial (px, py) from the first detection.
        velocity : ndarray of shape (2,), optional
            Initial velocity estimate. Defaults to zero.
        """
        v = velocity if velocity is not None else np.zeros(2, dtype=np.float64)
        self.x = np.array(
            [position[0], position[1], v[0], v[1]], dtype=np.float64
        ).reshape(4, 1)
        # Reinitialise covariance — high uncertainty at birth of track
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.P[2, 2] = 500.0  # higher initial velocity uncertainty
        self.P[3, 3] = 500.0

    # ──────────────────────────────────────────────────────────────────
    def predict(self) -> np.ndarray:
        """
        Kalman predict step (time update).

        Propagates the state estimate and covariance one step forward
        using the constant-velocity model:

            x̂⁻ₖ = F · x̂ₖ₋₁          (predicted state mean)
            P⁻ₖ  = F · Pₖ₋₁ · Fᵀ + Q  (predicted covariance)

        WHY: This step captures what we *expect* to happen based on the
        motion model alone — the "prior" in Bayesian language.

        Returns
        -------
        ndarray of shape (4,)
            Predicted state vector [px, py, vx, vy].
        """
        # Propagate state mean through linear motion model
        self.x = self.F @ self.x

        # Propagate covariance — F·P·Fᵀ spreads uncertainty through
        # the dynamics; Q adds the motion model's inherent uncertainty
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.flatten()

    # ──────────────────────────────────────────────────────────────────
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Kalman update step (measurement update).

        Fuses the sensor measurement z with the predicted state x̂⁻
        to produce the posterior estimate x̂⁺ (the "corrected" state).

        Steps
        ─────
        1. Innovation (residual):
               y = z − H · x̂⁻
           What the sensor *actually* reported minus what we *predicted*
           it would report.

        2. Innovation covariance:
               S = H · P⁻ · Hᵀ + R

        3. Kalman gain:
               K = P⁻ · Hᵀ · S⁻¹
           Optimal weighting: trust measurement more when R is small
           (sensor is precise) and trust prediction more when R is large.

        4. State correction:
               x̂⁺ = x̂⁻ + K · y

        5. Covariance correction (Joseph form — numerically stable):
               I_KH = I − K · H
               P⁺   = I_KH · P⁻ · I_KH^T + K · R · K^T

           WHY Joseph form over simple (I−KH)·P⁻?
           The simple form is equivalent only when K is *exactly* optimal.
           Floating-point errors make K sub-optimal in practice; Joseph
           form is always symmetric and PSD regardless of K accuracy.

        Parameters
        ----------
        measurement : ndarray of shape (2,)
            Observed [px, py] position from detector.

        Returns
        -------
        ndarray of shape (4,)
            Updated (posterior) state vector [px, py, vx, vy].
        """
        z = measurement.reshape(2, 1).astype(np.float64)

        # 1. Innovation — gap between observation and prediction
        y = z - self.H @ self.x

        # 2. Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # 3. Kalman gain — numerically stable via linear solve instead of explicit inversion
        #    K = P⁻·Hᵀ·S⁻¹
        #    Equivalently: S·Kᵀ = H·P⁻  →  solve S·X = H·P⁻ for X (shape 2×4), then K = Xᵀ
        #    np.linalg.solve(A, B): A=(2,2), B=(2,4) → X=(2,4) → K=X.T is (4,2)
        _HP = self.H @ self.P                             # shape (2, 4)
        K = np.linalg.solve(S, _HP).T                    # shape (4, 2)

        # 4. State correction
        self.x = self.x + K @ y

        # 5. Covariance correction — Joseph form
        I_KH = self._I4 - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Record posterior for analysis
        self.history.append(self.x.flatten().copy())

        return self.x.flatten()

    # ──────────────────────────────────────────────────────────────────
    def get_position(self) -> np.ndarray:
        """Return current position estimate [px, py]."""
        return self.x[:2].flatten()

    def get_velocity(self) -> np.ndarray:
        """Return current velocity estimate [vx, vy]."""
        return self.x[2:].flatten()

    def get_state(self) -> np.ndarray:
        """Return full state [px, py, vx, vy]."""
        return self.x.flatten()

    def get_covariance(self) -> np.ndarray:
        """Return current covariance matrix P (4×4)."""
        return self.P.copy()

    def is_positive_semidefinite(self) -> bool:
        """
        Check whether P is positive semi-definite.
        Uses eigenvalue check: all eigenvalues ≥ −ε (small tolerance for
        floating-point errors).
        """
        eigenvalues = np.linalg.eigvalsh(self.P)
        return bool(np.all(eigenvalues >= -1e-8))
