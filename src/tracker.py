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


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Object Tracker
# ──────────────────────────────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    Multi-object tracker combining:
      - YOLOv8 detections (passed in from detector.py)
      - Hungarian algorithm for data association
      - Per-track Kalman Filter (CV model)
      - Per-track One Euro Filter (for benchmarking)

    Parameters
    ----------
    max_missed : int
        Maximum number of consecutive unmatched frames before a track is deleted.
    iou_threshold : float
        Minimum IoU to consider a detection–track pair as a valid match.
        Pairs below this threshold are treated as unmatched even if the
        Hungarian algorithm assigns them.
    kf_process_noise : float
        Process noise σ_a for all track KFs.
    kf_measurement_noise : float
        Measurement noise σ_z for all track KFs.
    oef_min_cutoff : float
        One Euro Filter minimum cutoff frequency.
    oef_beta : float
        One Euro Filter speed coefficient.
    """

    def __init__(
        self,
        max_missed: int = 5,
        iou_threshold: float = 0.15,
        kf_process_noise: float = 2.0,
        kf_measurement_noise: float = 5.0,
        oef_min_cutoff: float = 0.5,
        oef_beta: float = 0.007,
    ) -> None:
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.kf_process_noise = kf_process_noise
        self.kf_measurement_noise = kf_measurement_noise
        self.oef_min_cutoff = oef_min_cutoff
        self.oef_beta = oef_beta

        self._next_id: int = 1
        self.tracks: list[Track] = []
        self.frame_count: int = 0

    # ──────────────────────────────────────────────────────────────────
    def _create_track(self, detection: np.ndarray) -> Track:
        """
        Initialise a new Track from a fresh detection.

        Parameters
        ----------
        detection : ndarray of shape (6,)  [x1, y1, x2, y2, conf, cls]
        """
        x1, y1, x2, y2 = detection[:4]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        kf = KalmanFilterCV(
            process_noise_std=self.kf_process_noise,
            measurement_noise_std=self.kf_measurement_noise,
        )
        kf.initialise(np.array([cx, cy]))

        oef = OneEuroFilter2D(
            min_cutoff=self.oef_min_cutoff,
            beta=self.oef_beta,
        )

        t = Track(
            track_id=self._next_id,
            kf=kf,
            oef=oef,
            bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
            wh=np.array([w, h], dtype=np.float64),
        )
        self._next_id += 1
        return t

    # ──────────────────────────────────────────────────────────────────
    def _predict_all(self) -> np.ndarray:
        """
        Run KF predict step on every active track.

        Returns
        -------
        predicted_bboxes : ndarray of shape (M, 4)
            Axis-aligned boxes [x1, y1, x2, y2] derived from KF position
            prediction + last known box size.
        """
        predicted = []
        for t in self.tracks:
            pred_state = t.kf.predict()
            cx, cy = pred_state[0], pred_state[1]
            w, h = t.wh
            predicted.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return np.array(predicted, dtype=np.float64) if predicted else np.zeros((0, 4))

    # ──────────────────────────────────────────────────────────────────
    def update(self, detections: np.ndarray) -> list[dict]:
        """
        Process one frame of detections and update all tracks.

        Parameters
        ----------
        detections : ndarray of shape (N, 6)  [x1, y1, x2, y2, conf, cls]
            All detections in the current frame (from YOLOv8Detector.detect).

        Returns
        -------
        list of dicts, one per active track:
            {
              'track_id'    : int,
              'bbox'        : ndarray (4,)   [x1, y1, x2, y2],
              'kf_position' : ndarray (2,)   KF-filtered centroid,
              'oef_position': ndarray (2,)   OEF-filtered centroid,
              'raw_centroid': ndarray (2,)   raw detected centroid (or None if predicted),
              'age'         : int,
              'missed'      : int,
            }
        """
        self.frame_count += 1
        det_bboxes = detections[:, :4] if len(detections) > 0 else np.zeros((0, 4))

        # ── 1. Predict all tracks ──────────────────────────────────────
        predicted_bboxes = self._predict_all()

        # ── 2. Build cost matrix & run Hungarian matching ──────────────
        matched_track_idx: list[int] = []
        matched_det_idx:   list[int] = []
        unmatched_track_idx: list[int] = list(range(len(self.tracks)))
        unmatched_det_idx:   list[int] = list(range(len(detections)))

        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = _build_iou_cost_matrix(predicted_bboxes, det_bboxes)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                iou_val = 1.0 - cost_matrix[r, c]
                if iou_val >= self.iou_threshold:
                    matched_track_idx.append(r)
                    matched_det_idx.append(c)
                # below threshold: both remain unmatched

            unmatched_track_idx = [
                i for i in range(len(self.tracks)) if i not in matched_track_idx
            ]
            unmatched_det_idx = [
                j for j in range(len(detections)) if j not in matched_det_idx
            ]

        # ── 3. Update matched tracks ───────────────────────────────────
        for t_idx, d_idx in zip(matched_track_idx, matched_det_idx):
            t = self.tracks[t_idx]
            det = detections[d_idx]
            x1, y1, x2, y2 = det[:4]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Kalman update with observed centroid
            kf_state = t.kf.update(np.array([cx, cy]))
            kf_cx, kf_cy = kf_state[0], kf_state[1]

            # One Euro Filter update
            oef_pos = t.oef.filter(np.array([cx, cy]))

            # Update bounding box using KF-smoothed centroid + last known size
            w, h = x2 - x1, y2 - y1
            t.wh = np.array([w, h])
            t.bbox = np.array([kf_cx - w/2, kf_cy - h/2, kf_cx + w/2, kf_cy + h/2])

            t.kf_position = np.array([kf_cx, kf_cy])
            t.oef_position = oef_pos.copy()
            t.missed = 0
            t.age += 1
            t.is_confirmed = True

            # Record trajectory history
            t.centroid_history_raw.append(np.array([cx, cy]))
            t.centroid_history_kf.append(t.kf_position.copy())
            t.centroid_history_oef.append(t.oef_position.copy())

        # ── 4. Handle unmatched tracks (predict-only frames) ───────────
        for t_idx in unmatched_track_idx:
            t = self.tracks[t_idx]
            t.missed += 1
            t.age += 1
            # KF position already updated in predict step
            pos = t.kf.get_position()
            t.kf_position = pos.copy()
            # One Euro: repeat previous position (no detection available)
            if t.oef_position is None:
                t.oef_position = pos.copy()
            t.centroid_history_kf.append(t.kf_position.copy())
            t.centroid_history_oef.append(t.oef_position.copy())
            t.centroid_history_raw.append(np.full(2, np.nan))  # mark as missed

        # ── 5. Spawn new tracks for unmatched detections ───────────────
        for d_idx in unmatched_det_idx:
            new_track = self._create_track(detections[d_idx])
            cx = (detections[d_idx][0] + detections[d_idx][2]) / 2.0
            cy = (detections[d_idx][1] + detections[d_idx][3]) / 2.0
            new_track.kf_position = np.array([cx, cy])
            new_track.oef_position = np.array([cx, cy])
            new_track.centroid_history_raw.append(np.array([cx, cy]))
            new_track.centroid_history_kf.append(new_track.kf_position.copy())
            new_track.centroid_history_oef.append(new_track.oef_position.copy())
            new_track.is_confirmed = True
            new_track.age = 1
            self.tracks.append(new_track)

        # ── 6. Delete dead tracks ──────────────────────────────────────
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        # ── 7. Collect outputs ─────────────────────────────────────────
        outputs = []
        for t in self.tracks:
            if not t.is_confirmed:
                continue
            raw_c = t.centroid_history_raw[-1] if t.centroid_history_raw else np.array([np.nan, np.nan])
            outputs.append(
                {
                    "track_id": t.track_id,
                    "bbox": t.bbox.copy(),
                    "kf_position": t.kf_position.copy() if t.kf_position is not None else np.array([0.0, 0.0]),
                    "oef_position": t.oef_position.copy() if t.oef_position is not None else np.array([0.0, 0.0]),
                    "raw_centroid": raw_c,
                    "age": t.age,
                    "missed": t.missed,
                }
            )

        return outputs