"""
tracker.py
==========
Multi-object tracker: detection → data association → Kalman Filter.

Architecture Overview
─────────────────────
                ┌──────────┐     detections (N×6)
  video frame → │ Detector │ ──────────────────────────────────────────┐
                └──────────┘                                            │
                                                                        ▼
  tracks (M predicted positions) ──────────────────→ ┌──────────────────────┐
                                                      │ Cost Matrix (M×N)    │
                                                      │  IoU or centroid dist│
                                                      └─────────┬────────────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │ Hungarian Algorithm   │
                                                    │ linear_sum_assignment  │
                                                    └─────────┬─────────────┘
                                                              │ matched pairs
                                         ┌────────────────────┼─────────────────────┐
                                         │                    │                     │
                                         ▼                    ▼                     ▼
                                   matched tracks      unmatched tracks      unmatched detections
                                   KF.update()         age++ (predict only)  new Track born

Track lifecycle
───────────────
  BIRTH  : A detection with no matching track creates a new Track.
  ALIVE  : Each frame — predict → if matched, update; if not, skip update.
  DEATH  : A track is deleted after `max_missed` consecutive unmatched frames.

Design decisions
────────────────
• IoU is used as the primary association metric (vs. centroid distance)
  because it naturally handles objects of different scales.
• IoU cost matrix: cost = 1 − IoU (so Hungarian minimises cost).
• Tracks are confirmed (output) immediately — no tentative-state delay —
  which works well for clean detectors like YOLOv8 at conf ≥ 0.4.
• Each track holds both a KalmanFilterCV and a OneEuroFilter2D so
  both filters run in parallel for benchmarking.

Author: Portfolio project — Kalman Filter Sensor Fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

from kalman_filter import KalmanFilterCV
from one_euro_filter import OneEuroFilter2D


# ──────────────────────────────────────────────────────────────────────────────
# IoU utilities
# ──────────────────────────────────────────────────────────────────────────────

def _iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """
    Compute Intersection over Union for two bounding boxes.

    Parameters
    ----------
    bbox_a, bbox_b : ndarray of shape (4,)  [x1, y1, x2, y2]

    Returns
    -------
    float in [0, 1]
    """
    # Intersection rectangle
    ix1 = max(bbox_a[0], bbox_b[0])
    iy1 = max(bbox_a[1], bbox_b[1])
    ix2 = min(bbox_a[2], bbox_b[2])
    iy2 = min(bbox_a[3], bbox_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = max(0.0, bbox_a[2] - bbox_a[0]) * max(0.0, bbox_a[3] - bbox_a[1])
    area_b = max(0.0, bbox_b[2] - bbox_b[0]) * max(0.0, bbox_b[3] - bbox_b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _build_iou_cost_matrix(
    predicted_bboxes: np.ndarray,  # shape (M, 4)
    detection_bboxes: np.ndarray,  # shape (N, 4)
) -> np.ndarray:
    """
    Build an M×N cost matrix where cost[i,j] = 1 − IoU(track_i, det_j).

    The Hungarian algorithm minimises cost, so we convert the similarity
    measure IoU into a cost by subtracting from 1.

    Returns
    -------
    ndarray of shape (M, N), float64, values in [0, 1]
    """
    M, N = len(predicted_bboxes), len(detection_bboxes)
    cost = np.ones((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            cost[i, j] = 1.0 - _iou(predicted_bboxes[i], detection_bboxes[j])
    return cost


# ──────────────────────────────────────────────────────────────────────────────
# Track dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    """
    A single tracked object — contains its Kalman Filter, One Euro Filter,
    bounding box state, and bookkeeping metadata.
    """

    track_id: int

    # Kalman Filter — carries state estimate
    kf: KalmanFilterCV

    # One Euro Filter — alternative smoother for benchmarking
    oef: OneEuroFilter2D

    # Most recent axis-aligned bounding box (from detection, not KF)
    bbox: np.ndarray                  # [x1, y1, x2, y2]

    # Current box size (width, height) — used to project KF position to bbox
    wh: np.ndarray                    # [w, h]

    # How many consecutive frames since last successful association
    missed: int = 0

    # Total frames this track has been alive
    age: int = 0

    # Whether at least one update has been applied (avoids KF output at birth)
    is_confirmed: bool = False

    # Full trajectory of centroid positions (px), stored for evaluation
    centroid_history_raw: list[np.ndarray] = field(default_factory=list)
    centroid_history_kf:  list[np.ndarray] = field(default_factory=list)
    centroid_history_oef: list[np.ndarray] = field(default_factory=list)

    # Per-frame Kalman position estimate (from predict or predict+update)
    kf_position: Optional[np.ndarray] = None
    oef_position: Optional[np.ndarray] = None

