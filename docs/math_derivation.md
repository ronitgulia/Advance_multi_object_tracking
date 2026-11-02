# Mathematical Derivation of the Kalman Filter

> **Audience:** This document is intended for someone who understands
> multivariate Gaussian distributions and basic linear algebra. We derive
> every Kalman Filter equation step-by-step from Bayesian first principles.

---

## 1. The Problem: Bayesian State Estimation

We have a system whose **true state** $\mathbf{x}_k \in \mathbb{R}^n$ evolves over discrete time $k$, but we can only observe a **noisy, partial measurement** $\mathbf{z}_k \in \mathbb{R}^m$.

The goal is to compute the **posterior distribution**:
$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$$

That is: *what is our best belief about the current state, given everything we've seen?*

The Kalman Filter solves this exactly when:
1. The system evolves **linearly**
2. All noise is **Gaussian**

Under these two conditions, the posterior is always a Gaussian — and we only need to track its **mean** $\hat{\mathbf{x}}_k$ and **covariance** $\mathbf{P}_k$.

---

## 2. System Model

### State Transition (Process Model)

$$\mathbf{x}_k = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}$$

- $\mathbf{F}$: state transition matrix — encodes *how the state evolves*
- $\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$: process noise — *how uncertain is our model?*

### Measurement Model

$$\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k$$

- $\mathbf{H}$: measurement matrix — *which parts of the state do we observe?*
- $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$: measurement noise — *how noisy is our sensor?*

---

## 3. Constant-Velocity (CV) Motion Model

For 2D tracking (position + velocity), the state vector is:

$$\mathbf{x} = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \end{bmatrix}$$

With time step $\Delta t$, Newton's kinematics give:

$$p_x(k) = p_x(k-1) + v_x(k-1)\cdot\Delta t$$
$$v_x(k) = v_x(k-1) \quad \text{(constant velocity assumption)}$$

And identically for $y$. This gives:

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

We only observe position, not velocity:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

### Process Noise Matrix Q

The true system is not perfectly constant-velocity — acceleration $a_k \sim \mathcal{N}(0, \sigma_a^2)$ perturbs it. Using the **Discretized White Noise Acceleration (DWNA)** model:

$$\mathbf{G}_x = \begin{bmatrix} \Delta t^2/2 \\ 0 \\ \Delta t \\ 0 \end{bmatrix}, \quad \mathbf{G}_y = \begin{bmatrix} 0 \\ \Delta t^2/2 \\ 0 \\ \Delta t \end{bmatrix}$$

$$\mathbf{Q} = \sigma_a^2 \left( \mathbf{G}_x \mathbf{G}_x^T + \mathbf{G}_y \mathbf{G}_y^T \right)$$

**Why this form?** $\mathbf{G}$ maps scalar acceleration noise into the 4D state space. The outer product $\mathbf{G}\mathbf{G}^T$ gives the covariance of that noise in state space.

### Measurement Noise Matrix R

Assuming independent, isotropic sensor noise with std $\sigma_z$:

$$\mathbf{R} = \begin{bmatrix} \sigma_z^2 & 0 \\ 0 & \sigma_z^2 \end{bmatrix}$$

---

## 4. Bayesian Derivation

### 4.1 Prior: The Predict Step

At time $k-1$ we have the posterior $p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) = \mathcal{N}(\hat{\mathbf{x}}_{k-1}, \mathbf{P}_{k-1})$.

We want the **prior** before seeing measurement $\mathbf{z}_k$:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}) \, p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1}) \, d\mathbf{x}_{k-1}$$

Since both distributions are Gaussian and the transition is linear, the integral is analytically tractable:

$$\boxed{\hat{\mathbf{x}}_k^- = \mathbf{F} \hat{\mathbf{x}}_{k-1}}$$

$$\boxed{\mathbf{P}_k^- = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}}$$

**Intuition:** $\mathbf{F} \mathbf{P} \mathbf{F}^T$ propagates existing uncertainty through the dynamics. $\mathbf{Q}$ adds the new uncertainty introduced by the imperfect motion model.

### 4.2 Likelihood: The Measurement

Given the current state, the measurement likelihood is:

$$p(\mathbf{z}_k \mid \mathbf{x}_k) = \mathcal{N}(\mathbf{z}_k; \, \mathbf{H}\mathbf{x}_k, \, \mathbf{R})$$

### 4.3 Posterior: The Update Step (Bayes' Rule)

By Bayes' theorem:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) \propto p(\mathbf{z}_k \mid \mathbf{x}_k) \cdot p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})$$

Both factors are Gaussian. The product of two Gaussians is another (unnormalised) Gaussian. Completing the square in the exponent yields:

**Innovation (residual):**
$$\mathbf{y}_k = \mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_k^-$$

This is what the sensor observed *minus* what we predicted the sensor would observe.

**Innovation covariance:**
$$\mathbf{S}_k = \mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R}$$

This is the total uncertainty in the innovation: predicted state uncertainty (projected into measurement space) plus sensor noise.

**Kalman gain:**
$$\boxed{\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T \mathbf{S}_k^{-1}}$$

**Posterior mean (state update):**
$$\boxed{\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \mathbf{y}_k}$$

**Posterior covariance (Joseph form):**
$$\boxed{\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R} \mathbf{K}_k^T}$$

---

## 5. Interpreting the Kalman Gain

The Kalman gain $\mathbf{K}_k$ is the **optimal weighting** between the prediction and the measurement:

$$\hat{\mathbf{x}}_k = \underbrace{\hat{\mathbf{x}}_k^-}_{\text{prediction}} + \mathbf{K}_k \underbrace{(\mathbf{z}_k - \mathbf{H}\hat{\mathbf{x}}_k^-)}_{\text{innovation}}$$

### Limit Analysis

| Condition | Effect on K | Meaning |
|---|---|---|
| $\mathbf{R} \to 0$ | $\mathbf{K} \to \mathbf{H}^{-1}$ | Sensor is perfect → trust it completely |
| $\mathbf{P}^- \to 0$ | $\mathbf{K} \to 0$ | Model prediction is certain → ignore measurement |
| $\mathbf{Q} \to 0$ | $\mathbf{P}^-$ shrinks over time → $\mathbf{K} \to 0$ | System is deterministic → trust motion model |

---

## 6. Why the Joseph Form?

The simple covariance update is:
$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^-$$

This is **equivalent** to the Joseph form only when $\mathbf{K}_k$ is exactly optimal. In practice, floating-point arithmetic makes $\mathbf{K}_k$ slightly sub-optimal, causing $\mathbf{P}$ to lose symmetry and eventually become non-PSD.

The Joseph form is **algebraically equivalent** to the simple form when $\mathbf{K}_k$ is exact, but is **guaranteed to produce a symmetric, PSD matrix** for any $\mathbf{K}_k$. This is why we use it.

---

## 7. Summary of Equations

| Step | Equation | Name |
|---|---|---|
| **Predict** | $\hat{\mathbf{x}}_k^- = \mathbf{F} \hat{\mathbf{x}}_{k-1}$ | State prediction |
| **Predict** | $\mathbf{P}_k^- = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$ | Covariance prediction |
| **Update** | $\mathbf{y}_k = \mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_k^-$ | Innovation |
| **Update** | $\mathbf{S}_k = \mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R}$ | Innovation covariance |
| **Update** | $\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T \mathbf{S}_k^{-1}$ | Kalman gain |
| **Update** | $\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \mathbf{y}_k$ | State correction |
| **Update** | $\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R} \mathbf{K}_k^T$ | Covariance correction (Joseph) |

---

## 8. One Euro Filter — Algorithm

The One Euro Filter (Casiez et al., CHI 2012) takes a different approach: rather than a probabilistic motion model, it uses **adaptive cutoff frequency**:

$$\hat{\dot{x}}_k = \text{LPF}(\dot{x}_k, \alpha_d), \quad \dot{x}_k = \frac{x_k - x_{k-1}}{\Delta t}$$

$$f_c = f_{\min} + \beta \cdot |\hat{\dot{x}}_k|$$

$$\alpha = \frac{2\pi f_c \Delta t}{2\pi f_c \Delta t + 1}$$

$$\hat{x}_k = \alpha x_k + (1 - \alpha) \hat{x}_{k-1}$$

### Intuition

- **Slow motion** → small $|\hat{\dot{x}}|$ → low $f_c$ → low $\alpha$ → heavy smoothing (jitter reduced)
- **Fast motion** → large $|\hat{\dot{x}}|$ → high $f_c$ → high $\alpha$ → light smoothing (lag reduced)

This elegantly handles the jitter-lag tradeoff for **human-input signals** (mouse, touch, gestures) where the motion is unpredictable. The Kalman Filter is better when the **motion model is known** and **noise statistics are characterisable**.

---

## References

1. Kalman, R.E. (1960). "A new approach to linear filtering and prediction problems." *ASME Journal of Basic Engineering*, 82(1), 35-45.
2. Bar-Shalom, Y., Li, X.-R., Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
3. Casiez, G., Roussel, N., Vogel, D. (2012). "1€ Filter." *CHI 2012*. DOI: 10.1145/2207676.2208639.
4. Welch, G., Bishop, G. (2006). "An Introduction to the Kalman Filter." UNC Chapel Hill Technical Report TR 95-041.