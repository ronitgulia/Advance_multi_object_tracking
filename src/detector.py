"""
detector.py
===========
YOLOv8 object detection wrapper.

Wraps the Ultralytics YOLOv8 model to provide a clean, frame-by-frame
interface that returns normalised bounding box detections compatible with
the tracker's expected format.

Design decisions
────────────────
• We use YOLOv8n (nano) by default for real-time speed on CPU.
  Larger models (s, m, l, x) can be passed via the `model_name` argument.
• Confidence threshold defaults to 0.4 — low enough to catch objects
  early in tracks, high enough to avoid many false positives.
• Class filtering defaults to None (all classes). Pass a list of COCO
  class IDs to restrict detection to specific objects (e.g., [0] for person).

Author: Portfolio project — Kalman Filter Sensor Fusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class YOLOv8Detector:
    """
    Thin wrapper around Ultralytics YOLOv8 for per-frame detection.

    Parameters
    ----------
    model_name : str
        YOLOv8 weight name or path. E.g. 'yolov8n.pt', 'yolov8s.pt'.
        Ultralytics auto-downloads the weights on first run.
    conf_threshold : float
        Detection confidence threshold (0.0–1.0).
    iou_threshold : float
        NMS IoU threshold.
    class_filter : list[int] | None
        COCO class IDs to retain. None = all classes.
    device : str
        PyTorch device string: 'cpu', 'cuda', 'mps', etc.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        class_filter: Optional[list[int]] = None,
        device: str = "cpu",
    ) -> None:
        # Import here so that users without ultralytics can still use
        # the rest of the library for pure KF work.
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for detection. "
                "Install it with: pip install ultralytics"
            ) from e

        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = class_filter
        self.device = device

    # ──────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLOv8 inference on one BGR frame (as returned by cv2.VideoCapture).

        Parameters
        ----------
        frame : ndarray of shape (H, W, 3), uint8, BGR
            Raw video frame.

        Returns
        -------
        detections : ndarray of shape (N, 6), float32
            Each row is [x1, y1, x2, y2, confidence, class_id].
            - (x1, y1) : top-left corner (pixels)
            - (x2, y2) : bottom-right corner (pixels)
            - confidence: detection score ∈ (0, 1)
            - class_id  : COCO class index (int, stored as float)
            Returns empty array of shape (0, 6) when no detections.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        detections: list[list[float]] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                # Skip classes not in the filter list
                if self.class_filter is not None and cls_id not in self.class_filter:
                    continue
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append([x1, y1, x2, y2, conf, float(cls_id)])

        if not detections:
            return np.zeros((0, 6), dtype=np.float32)

        return np.array(detections, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def bbox_to_centroid(bbox: np.ndarray) -> np.ndarray:
        """
        Convert [x1, y1, x2, y2] bounding box to (cx, cy) centroid.

        Parameters
        ----------
        bbox : ndarray of shape (4,) or (N, 4)

        Returns
        -------
        ndarray of shape (2,) or (N, 2)
        """
        if bbox.ndim == 1:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            return np.array([cx, cy], dtype=np.float64)
        # Batch form
        cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
        cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        return np.stack([cx, cy], axis=1).astype(np.float64)