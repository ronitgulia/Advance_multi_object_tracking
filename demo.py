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


# ──────────────────────────────────────────────────────────────────────────────
# Main tracking loop
# ──────────────────────────────────────────────────────────────────────────────

def run_tracking(
    video_source: str | int,
    output_path: Path,
    conf_threshold: float = 0.4,
    device: str = "cpu",
    max_missed: int = 5,
    class_filter: list[int] | None = None,
) -> None:
    """
    Open a video source, run YOLOv8 detection + KF/OEF tracking,
    and write annotated output video.
    """
    print(f"\n{'='*60}")
    print(f"  Kalman Filter Multi-Object Tracker")
    print(f"{'='*60}")
    print(f"  Source  : {video_source}")
    print(f"  Output  : {output_path}")
    print(f"  Device  : {device}")
    print(f"  Conf    : {conf_threshold}")
    print(f"{'='*60}\n")

    # ── Detector ──────────────────────────────────────────────────────
    detector = YOLOv8Detector(
        model_name="yolov8n.pt",
        conf_threshold=conf_threshold,
        device=device,
        class_filter=class_filter,
    )

    # ── Tracker ───────────────────────────────────────────────────────
    tracker = MultiObjectTracker(max_missed=max_missed)

    # ── Video I/O ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        if isinstance(video_source, str):
            print(f"[ERROR] Cannot open video: {video_source}")
            print("[INFO]  Falling back to webcam (device 0).")
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open any video source. Exiting.")
            sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        detections = detector.detect(frame)

        # Track
        track_outputs = tracker.update(detections)

        # Annotate
        vis = _draw_tracks(frame, track_outputs)

        # Frame counter overlay
        progress = f"Frame {frame_idx+1}"
        if total > 0:
            progress += f"/{total}  ({(frame_idx+1)/total*100:.1f}%)"
        cv2.putText(vis, progress, (W - 280, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        writer.write(vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - t0
            fps_actual = frame_idx / elapsed
            n_tracks = len([t for t in track_outputs if t["missed"] == 0])
            print(f"  Frame {frame_idx:4d}  |  Active tracks: {n_tracks:2d}  |  {fps_actual:.1f} fps")

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t0
    print(f"\n[INFO] Tracking complete — {frame_idx} frames in {elapsed:.1f}s")
    print(f"[INFO] Output video → {output_path}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalman Filter Multi-Object Tracker with KF vs OEF Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", default=None,
                        help="Path to input video file. Defaults to webcam (device 0).")
    parser.add_argument("--output", default="results/tracked.mp4",
                        help="Output annotated video path.")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="YOLO detection confidence threshold (default: 0.4).")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: cpu | cuda | mps (default: cpu).")
    parser.add_argument("--max-missed", type=int, default=5,
                        help="Frames before a track is deleted (default: 5).")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="COCO class IDs to track (default: all classes).")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip tracking; run KF vs OEF benchmark only.")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip benchmark after tracking.")
    parser.add_argument("--benchmark-frames", type=int, default=300,
                        help="Number of synthetic frames for benchmark (default: 300).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Benchmark ─────────────────────────────────────────────────────
    if not args.no_benchmark:
        print("\n[INFO] Running KF vs One Euro Filter benchmark on synthetic data…")
        from evaluate import run_full_benchmark
        results = run_full_benchmark(n_frames=args.benchmark_frames, verbose=True)
        print("\n[INFO] All benchmark plots saved to results/")

    # ── Tracking ──────────────────────────────────────────────────────
    if not args.benchmark_only:
        video_source: str | int = args.video if args.video else 0
        run_tracking(
            video_source=video_source,
            output_path=Path(args.output),
            conf_threshold=args.conf,
            device=args.device,
            max_missed=args.max_missed,
            class_filter=args.classes,
        )

    print("\n[DONE] All tasks complete.")
    print("       Results and plots -> results/")


if __name__ == "__main__":
    main()