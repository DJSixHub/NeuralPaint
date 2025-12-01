from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
from typing import Optional, Sequence

from neuralpaint.app import run_interactive_app
from neuralpaint.calibration import ensure_calibration


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Markerless calibration and gesture-driven drawing.")
    parser.add_argument("--camera", type=int, default=0, help="Index of the capture device (default: 0).")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration") / "homography.npz",
        help="Path to save/read the homography calibration.",
    )
    parser.add_argument(
        "--surface-width",
        type=float,
        default=0.0,
        help="Logical canvas width in pixels (0 = usar resolución de pantalla).",
    )
    parser.add_argument(
        "--surface-height",
        type=float,
        default=0.0,
        help="Logical canvas height in pixels (0 = usar resolución de pantalla).",
    )
    parser.add_argument("--calibration-min-area", type=float, default=0.15, help="Minimum contour area ratio (0-1).")
    parser.add_argument(
        "--calibration-approx-eps",
        type=float,
        default=0.03,
        help="Polygon approximation factor relative to contour perimeter.",
    )
    parser.add_argument("--calibration-warp", action="store_true", help="Display the rectified surface during calibration.")
    parser.add_argument("--calibration-debug", action="store_true", help="Show edge map while calibrating.")
    parser.add_argument("--force-calibrate", action="store_true", help="Run calibration even if cached data exists.")
    parser.add_argument("--calibration-only", action="store_true", help="Perform calibration and exit without drawing mode.")
    parser.add_argument("--min-detection", type=float, default=0.6, help="MediaPipe detection confidence threshold.")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="MediaPipe tracking confidence threshold.")
    parser.add_argument("--smoothing", type=float, default=0.6, help="Exponential smoothing factor for pointer (0-1).")
    parser.add_argument("--max-strokes", type=int, default=200, help="Maximum number of stored strokes before oldest are dropped.")
    parser.add_argument("--command-hold-frames", type=int, default=6, help="Frames required to confirm a left-arm command.")
    parser.add_argument("--erase-radius", type=float, default=35.0, help="Radius in pixels for erasing strokes.")
    parser.add_argument("--preview-scale", type=float, default=0.25, help="Fraction of window height used for the camera preview (0-0.5).")
    parser.add_argument("--brush-thickness", type=float, default=4.0, help="Brush thickness for newly created strokes in surface pixels.")
    parser.add_argument("--mode-toggle-delay", type=float, default=3.0, help="Seconds to wait before a repeated draw/erase gesture returns to idle.")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirrored preview windows.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.flip_view = not args.no_flip
    if args.surface_width <= 0 or args.surface_height <= 0:
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        args.surface_width = float(max(1, screen_width))
        args.surface_height = float(max(1, screen_height))
    calibration = ensure_calibration(args)
    if calibration is None:
        return 1
    if args.calibration_only:
        print("Calibration ready. Restart without --calibration-only to draw.")
        return 0
    return run_interactive_app(args, calibration)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
