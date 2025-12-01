from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CalibrationData:
    homography: np.ndarray
    inverse_homography: np.ndarray
    surface_width: int
    surface_height: int


@dataclass
class CalibrationResult:
    polygon: np.ndarray
    homography: np.ndarray


def load_calibration(path: Path) -> CalibrationData:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found at {path}. Run calibration first.")
    data = np.load(path)
    homography = data["homography"].astype(np.float32)
    width = int(data["surface_width"])
    height = int(data["surface_height"])
    inverse = np.linalg.inv(homography)
    return CalibrationData(
        homography=homography,
        inverse_homography=inverse,
        surface_width=width,
        surface_height=height,
    )


def save_calibration(path: Path, homography: np.ndarray, width: float, height: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        homography=homography.astype(np.float32),
        surface_width=float(width),
        surface_height=float(height),
    )


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.equalizeHist(gray)


def order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    top_left_idx = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -top_left_idx, axis=0)
    return ordered.astype(np.float32)


def detect_surface_polygon(
    frame: np.ndarray,
    min_area_ratio: float,
    approx_epsilon_factor: float,
) -> Optional[np.ndarray]:
    processed = preprocess_frame(frame)
    edges = cv2.Canny(processed, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * np.clip(min_area_ratio, 0.0, 1.0)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        ordered = order_corners_clockwise(approx.reshape(4, 2))
        candidates.append((area, ordered))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def compute_homography(polygon: np.ndarray, width: float, height: float) -> np.ndarray:
    destination = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=np.float32,
    )
    homography, status = cv2.findHomography(polygon, destination, method=0)
    if homography is None or status is None or not status.all():
        raise RuntimeError("Failed to solve homography")
    return homography


def draw_calibration_overlay(frame: np.ndarray, polygon: Optional[np.ndarray], message: str) -> np.ndarray:
    overlay = frame.copy()
    if polygon is not None:
        cv2.polylines(overlay, [polygon.astype(int)], True, (0, 255, 0), 3)
        for idx, (x, y) in enumerate(polygon):
            cv2.circle(overlay, (int(x), int(y)), 8, (0, 255, 255), -1)
            cv2.putText(
                overlay,
                f"P{idx}",
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        overlay,
        message,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "Pulsa 's' para guardar la calibracion, 'q' para salir",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def run_auto_calibration(args: Namespace) -> Optional[CalibrationData]:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera}", file=sys.stderr)
        return None

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    if args.calibration_warp:
        cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
    if args.calibration_debug:
        cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)

    latest: Optional[CalibrationResult] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera", file=sys.stderr)
                latest = None
                break

            polygon = detect_surface_polygon(frame, args.calibration_min_area, args.calibration_approx_eps)
            if polygon is not None:
                try:
                    homography = compute_homography(polygon, args.surface_width, args.surface_height)
                except RuntimeError:
                    homography = None
                if homography is not None:
                    latest = CalibrationResult(polygon=polygon, homography=homography)
                    message = "Superficie detectada"
                    if args.calibration_warp:
                        warped = cv2.warpPerspective(
                            frame,
                            latest.homography,
                            (int(args.surface_width), int(args.surface_height)),
                        )
                        display_warped = cv2.flip(warped, 1) if getattr(args, "flip_view", False) else warped
                        cv2.imshow("Warped", display_warped)
                else:
                    message = "Rectangulo encontrado pero la homografia fallo"
            else:
                message = "Muestra un rectangulo brillante que cubra el area"

            overlay = draw_calibration_overlay(frame, polygon, message)
            display_overlay = cv2.flip(overlay, 1) if getattr(args, "flip_view", False) else overlay
            cv2.imshow("Calibration", display_overlay)

            if args.calibration_debug:
                edges_dbg = cv2.Canny(preprocess_frame(frame), 50, 150)
                if getattr(args, "flip_view", False):
                    edges_dbg = cv2.flip(edges_dbg, 1)
                cv2.imshow("Edges", edges_dbg)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                latest = None
                break
            if key == ord("s") and latest is not None:
                save_calibration(
                    args.calibration,
                    latest.homography,
                    args.surface_width,
                    args.surface_height,
                )
                print(f"Calibration saved to {args.calibration}")
                break
    finally:
        cap.release()
        cv2.destroyWindow("Calibration")
        if args.calibration_warp:
            cv2.destroyWindow("Warped")
        if args.calibration_debug:
            cv2.destroyWindow("Edges")

    if latest is None:
        return None
    return CalibrationData(
        homography=latest.homography.astype(np.float32),
        inverse_homography=np.linalg.inv(latest.homography.astype(np.float32)),
        surface_width=int(args.surface_width),
        surface_height=int(args.surface_height),
    )


def ensure_calibration(args: Namespace) -> Optional[CalibrationData]:
    needs_calibration = args.force_calibrate or args.calibration_only or not args.calibration.exists()
    if needs_calibration:
        calibration = run_auto_calibration(args)
        if calibration is None:
            print("Calibration failed or was cancelled.", file=sys.stderr)
            return None
        return calibration
    return load_calibration(args.calibration)
