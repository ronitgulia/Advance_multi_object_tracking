# Advanced Multi-Object Tracking (MOT) with Kalman & One Euro Filter Sensor Fusion

A high-performance, modular computer vision pipeline for Multi-Object Tracking (MOT). Combines deep learning object detection (YOLOv8) with Bayesian state estimation (**2D Constant-Velocity Kalman Filter** with Joseph-form covariance guarantees) and adaptive signal smoothing (**One Euro Filter**), associated via the **Hungarian Algorithm** on IoU cost matrices.

---

## 🚀 Key Features

- **Bayesian State Estimation (Kalman Filter CV):**
  - Continuous 2D constant-velocity motion model ($[\text{pos}_x, \text{pos}_y, \text{vel}_x, \text{vel}_y]$).
  - Discrete White Noise Acceleration (DWNA) continuous-discrete process noise matrix $\mathbf{Q}$.
  - **Joseph-form covariance update** guarantee: $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}\mathbf{K}^T$ to strictly preserve positive semi-definiteness under finite-precision floating-point arithmetic.
- **Adaptive Signal Smoothing (One Euro Filter):**
  - Dual 1D/2D implementation adapting cutoff frequency dynamically based on signal velocity (Casiez et al., CHI 2012).
  - Eliminates slow-speed tracking jitter while preventing high-speed tracking lag.
- **Data Association & Lifecycle Management:**
  - Optimal bipartite matching using the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`).
  - IoU cost metric ($1 - \text{IoU}$) naturally invariant to multi-scale bounding boxes.
  - Robust track state machine: instant confirmation, prediction propagation during sensor occlusions, and automated track pruning after $N$ missed frames.
- **Modular Object Detection:**
  - Decoupled `YOLOv8Detector` interface wrapping Ultralytics YOLOv8 with configurable confidence, NMS IoU, and COCO class filters.
- **Quantitative Benchmark Suite:**
  - Automated evaluation across synthetic scenarios (constant velocity, sudden turns, sensor blackouts, sinusoidal trajectories).
  - Evaluates RMSE against ground truth, frame-to-frame jitter, and tracking lag.

---

## 📁 Repository Structure

```text
Advance_multi_object_tracking/
├── docs/
│   └── math_derivation.md      # Rigorous step-by-step Bayesian derivation of KF equations
├── src/
│   ├── detector.py             # YOLOv8 deep learning detection wrapper
│   ├── kalman_filter.py        # 2D Constant-Velocity Kalman Filter (Joseph form)
│   ├── one_euro_filter.py      # Adaptive 1D and 2D One Euro filters
│   ├── tracker.py              # MultiObjectTracker with Hungarian IoU matching
│   └── evaluate.py             # Benchmark engine, metric calculator, and plot generator
├── tests/
│   └── test_kalman_filter.py   # Comprehensive test suite (convergence, PSD, lifecycle)
├── .gitignore                  # Standard Python and environment ignores
└── README.md                   # Project documentation
```

---

## 🔬 Mathematical Foundations

Full first-principles derivation of all filter equations is documented in [docs/math_derivation.md](docs/math_derivation.md):

$$\mathbf{x}_k = \mathbf{F}\mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$
$$\mathbf{z}_k = \mathbf{H}\mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R})^{-1}$$

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- Recommended: NVIDIA GPU with CUDA support for YOLOv8 inference (CPU also supported)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/ronitgulia/Advance_multi_object_tracking.git
cd Advance_multi_object_tracking

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib ultralytics pytest
```
