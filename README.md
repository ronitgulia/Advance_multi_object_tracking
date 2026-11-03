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

---

## ⚡ Quickstart & Usage

### 1. Real-time Object Tracking Pipeline
```python
import cv2
from detector import YOLOv8Detector
from tracker import MultiObjectTracker

# Initialize detector (YOLOv8n default) and tracker
detector = YOLOv8Detector(model_name="yolov8n.pt", conf_threshold=0.4)
tracker = MultiObjectTracker(max_missed=5, iou_threshold=0.3)

cap = cv2.VideoCapture("input_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect objects [x1, y1, x2, y2, confidence, class_id]
    detections = detector.detect(frame)

    # 2. Update multi-object tracker
    active_tracks = tracker.update(detections)

    # 3. Retrieve filtered positions and track states
    for track in active_tracks:
        track_id = track["track_id"]
        bbox = track["bbox"]        # [x1, y1, x2, y2]
        kf_pos = track["kf_pos"]    # [cx, cy] Kalman filtered position
        oef_pos = track["oef_pos"]  # [cx, cy] One Euro filtered position

cap.release()
```

### 2. Running Benchmarks & Generating Plots
Evaluate filter tracking performance across synthetic constant velocity, high-maneuver turns, and sensor dropout scenarios:

```bash
python src/evaluate.py
```
Outputs trajectory comparisons and error-over-time plots in `results/`.

### 3. Running Unit Tests
Verify mathematical convergence, gain limits, and covariance PSD guarantees:

```bash
pytest tests/ -v
```

---

## 📊 Benchmark Metrics Summary

| Metric | Target | Description |
| :--- | :--- | :--- |
| **RMSE** | Lower is better | Root Mean Squared Error between filtered state estimate and ground truth |
| **Jitter** | Lower is better | Mean variance of first-order frame-to-frame position displacement |
| **Lag** | Lower is better | Cross-correlation offset peak identifying phase delay behind dynamic motion |
| **Robustness** | Higher is better | State prediction error under simulated sensor occlusions & detection dropouts |

---

## 📌 Roadmap & Progress

- [x] 2D Constant-Velocity Kalman Filter with discrete white noise acceleration (DWNA)
- [x] Joseph-form covariance update to ensure numerical positive semi-definiteness
- [x] Adaptive 1D and 2D One Euro filters for speed-dependent jitter mitigation
- [x] Multi-object tracker state machine with Hungarian association (IoU cost metric)
- [x] YOLOv8 deep learning detector wrapper interface
- [x] Multi-scenario synthetic benchmarking suite with automated plot generation
- [x] Comprehensive test suite covering filter convergence and matrix guarantees
- [ ] Multi-camera coordinate fusion & re-identification (ReID) embedding support
- [ ] 3D bounding box orientation estimation with Extended Kalman Filter (EKF)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
