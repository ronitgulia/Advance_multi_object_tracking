"""
demo.py
=======
CLI entry point: run the multi-object tracker on a video file or webcam,
then produce benchmark plots.

Usage
─────
  # Track a video file
  python demo.py --video data/sample.mp4

  # Track from webcam (device 0) with custom output
  python demo.py --output results/webcam_tracked.mp4

  # Only run benchmarks (no video)
  python demo.py --benchmark-only

  # Skip benchmark after tracking
  python demo.py --video data/sample.mp4 --no-benchmark

Full option reference
─────────────────────
  --video         Path to input video (default: webcam 0)
  --output        Path for output annotated video (default: results/tracked.mp4)
  --conf          YOLO confidence threshold (default: 0.4)
  --device        PyTorch device: cpu | cuda | mps (default: cpu)
  --max-missed    Track deletion after N missed frames (default: 5)
  --classes       COCO class IDs to track (space-separated, default: all)
  --benchmark-only  Skip tracking, run only the KF-vs-OEF benchmark
  --no-benchmark  Skip the benchmark after tracking

Author: Portfolio project — Kalman Filter Sensor Fusion
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src/ to Python path so all modules resolve correctly
_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(_SRC))

from detector import YOLOv8Detector
from tracker import MultiObjectTracker

# Palette for bounding box colours (one per track ID, cycled)
_PALETTE = [
    (0,  255, 127),  # spring green
    (255, 107,  33),  # vivid orange
    (33, 150, 255),  # dodger blue
    (255,  50, 200),  # magenta
    (255, 230,   0),  # yellow
    (0,  210, 230),  # cyan
    (255,  80,  80),  # coral red
    (180, 100, 255),  # violet
]


def _track_colour(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _draw_tracks(frame: np.ndarray, track_outputs: list[dict]) -> np.ndarray:
    """
    Annotate a frame with bounding boxes, track IDs, and KF/OEF centroids.
    """
    out = frame.copy()

    for t in track_outputs:
        tid = t["track_id"]
        bbox = t["bbox"].astype(int)
        kf_pos = t["kf_position"].astype(int)
        oef_pos = t["oef_position"].astype(int)
        raw_c = t["raw_centroid"]

        colour = _track_colour(tid)

        # Bounding box (KF-smoothed position, original size)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        # Track ID label
        label = f"ID:{tid}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # KF centroid — green dot
        cv2.circle(out, tuple(kf_pos), 5, (0, 255, 100), -1)

        # OEF centroid — pink dot
        cv2.circle(out, tuple(oef_pos), 5, (255, 100, 180), -1)

        # Raw centroid — yellow cross (if available)
        if not np.isnan(raw_c).any():
            rx, ry = int(raw_c[0]), int(raw_c[1])
            cv2.drawMarker(out, (rx, ry), (0, 215, 255),
                           cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

    # Legend
    legend_items = [
        ((0, 255, 100),    "KF centroid"),
        ((255, 100, 180),  "OEF centroid"),
        ((0, 215, 255),    "Raw detection"),
    ]
    y_off = 20
    for colour, label in legend_items:
        cv2.circle(out, (15, y_off), 5, colour, -1)
        cv2.putText(out, label, (25, y_off + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        y_off += 20

    return out

