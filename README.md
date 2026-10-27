# Advance Multi-Object Tracking

A modular and scalable computer vision repository for advanced Multi-Object Tracking (MOT). Designed for high-accuracy object detection, real-time association, and trajectory analysis across video streams.

---

## 🚀 Features

- **Object Detection Integration:** Modular detector interface for deep learning detection models (e.g., YOLOv8, YOLOv11, RT-DETR).
- **Multi-Object Tracking:** Multi-target state estimation, trajectory association, and identity preservation (DeepSORT, ByteTrack, BoT-SORT).
- **Clean Architecture:** Organized modular components separating detectors, trackers, association metrics, and visualization utilities.
- **Configurable Pipeline:** Easily swap models, confidence thresholds, and tracking hyper-parameters.

---

## 📁 Project Structure

```text
Advance_multi_object_tracking/
├── src/
│   └── detector.py         # Object detection wrapper and model interface
├── .gitignore              # Python and machine learning ignore rules
└── README.md               # Project documentation
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- GPU with CUDA support (recommended for real-time inference)

### Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/ronitgulia/Advance_multi_object_tracking.git
cd Advance_multi_object_tracking

# Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
```

---

## 📌 Roadmap

- [ ] Complete `detector.py` inference pipeline (model loader, bounding box extraction, NMS)
- [ ] Implement multi-object tracker module (ByteTrack / DeepSORT / BoT-SORT integration)
- [ ] Add real-time video processing and webcam stream support
- [ ] Visualizer utilities for bounding boxes, trajectory trails, and track ID overlays
- [ ] Quantitative evaluation suite (MOTA, IDF1, HOTA metrics)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
