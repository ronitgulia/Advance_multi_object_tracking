"""
one_euro_filter.py
==================
From-scratch One Euro Filter implementation using NumPy only.

The One Euro Filter (Casiez et al., CHI 2012) is a simple, adaptive
low-pass filter designed to reduce jitter while minimising lag.

Key idea
────────
At any given timestep, two things can be true:
  • If the signal is moving SLOWLY  → jitter dominates → use a LOW cutoff
    (= strong smoothing).
  • If the signal is moving QUICKLY → lag dominates   → use a HIGH cutoff
    (= less smoothing).

The filter achieves this by making the cutoff frequency a function of
the estimated derivative (speed) of the signal.

Algorithm (per dimension)
────────────────────────
1. Estimate derivative:
       dX̂ = LPF(ẋ, f_c_derivative)
   where ẋ = (x − x_prev) / dt  and  f_c_derivative = d_cutoff (fixed, e.g. 1 Hz)

2. Adaptive cutoff:
       f_c = f_min + β · |dX̂|

3. Smooth signal:
       X̂  = LPF(x, f_c)

   where LPF(x, f_c) = α·x + (1−α)·X̂_prev
         α = 2π·f_c·dt / (2π·f_c·dt + 1)   (first-order IIR alpha)

References
──────────
Casiez, G., Roussel, N., Vogel, D. (2012). "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems." In Proceedings of CHI 2012.
DOI: 10.1145/2207676.2208639

Author: Portfolio project — Kalman Filter Sensor Fusion
"""

from __future__ import annotations

import numpy as np


class _LowPassFilter1D:
    """
    First-order infinite impulse response (IIR) low-pass filter for a
    single scalar signal.

    Transfer function (z-domain):
        H(z) = α / (1 − (1−α)·z⁻¹)

    The smoothing coefficient α ∈ (0, 1]:
        α → 1  ≡ no smoothing (output = raw input)
        α → 0  ≡ maximum smoothing (output barely moves)
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._y: float = 0.0  # previous filtered output

    def filter(self, x: float, alpha: float) -> float:
        """
        Apply one filter step.

        Parameters
        ----------
        x     : raw input sample
        alpha : smoothing coefficient for this timestep
        """
        if not self._initialized:
            self._y = x
            self._initialized = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def last_value(self) -> float:
        return self._y


def _compute_alpha(cutoff_hz: float, dt: float) -> float:
    """
    Convert a cutoff frequency (Hz) to IIR alpha coefficient.

    Derivation from the bilinear approximation of a 1st-order RC low-pass:
        τ = 1 / (2π·f_c)
        α = dt / (τ + dt) = (2π·f_c·dt) / (2π·f_c·dt + 1)
    """
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return dt / (tau + dt)


class OneEuroFilter1D:
    """
    One Euro Filter for a single scalar signal.

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency (Hz). Lower → smoother at rest,
        but more lag when motion starts. Typical: 0.5–2.0 Hz.
    beta : float
        Speed coefficient. Higher → cutoff rises faster with speed,
        so the filter tracks fast motion better. Typical: 0.0–1.0.
    d_cutoff : float
        Cutoff frequency for the derivative estimator (Hz). Usually
        fixed at 1.0 Hz — this controls how quickly the speed estimate
        itself is smoothed.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Separate low-pass filters for signal and its derivative
        self._x_filter = _LowPassFilter1D()
        self._dx_filter = _LowPassFilter1D()

        self._prev_x: float | None = None

    def filter(self, x: float, dt: float) -> float:
        """
        Filter one sample.

        Parameters
        ----------
        x  : raw scalar value at current timestep
        dt : time elapsed since last call (seconds or frames, consistent with
             cutoff frequencies — if Hz is used, dt should be in seconds)

        Returns
        -------
        float : filtered value
        """
        # --- Step 1: Estimate derivative --------------------------------
        if self._prev_x is None:
            dx = 0.0
        else:
            dx = (x - self._prev_x) / dt

        # Smooth the derivative estimate with fixed d_cutoff
        alpha_d = _compute_alpha(self.d_cutoff, dt)
        dx_hat = self._dx_filter.filter(dx, alpha_d)

        # --- Step 2: Compute adaptive cutoff ----------------------------
        # f_c grows with |dx_hat| → faster movement → higher cutoff → less lag
        f_c = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = _compute_alpha(f_c, dt)

        # --- Step 3: Filter the signal with adaptive alpha --------------
        x_hat = self._x_filter.filter(x, alpha)

        self._prev_x = x
        return x_hat


class OneEuroFilter2D:
    """
    One Euro Filter for 2-D signals (e.g., (x, y) pixel positions).
    Applies independent 1-D filters to each dimension.

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency (Hz).
    beta : float
        Speed coefficient.
    d_cutoff : float
        Derivative estimator cutoff (Hz).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

        # History for evaluation
        self.history: list[np.ndarray] = []

    def filter(self, position: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Filter a 2-D position measurement.

        Parameters
        ----------
        position : ndarray of shape (2,)
            Raw (x, y) position.
        dt : float
            Time step (frames or seconds).

        Returns
        -------
        ndarray of shape (2,)
            Filtered (x, y) position.
        """
        x_hat = self._fx.filter(float(position[0]), dt)
        y_hat = self._fy.filter(float(position[1]), dt)
        result = np.array([x_hat, y_hat], dtype=np.float64)
        self.history.append(result.copy())
        return result