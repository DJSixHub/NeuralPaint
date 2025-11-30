"""General helpers for NeuralPaint."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def clamp_point(point: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    x = int(np.clip(point[0], 0, width - 1))
    y = int(np.clip(point[1], 0, height - 1))
    return x, y


def is_inside_surface(point: Tuple[float, float], width: int, height: int) -> bool:
    x, y = point
    return 0 <= x < width and 0 <= y < height


def project_point(homography: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    pts = np.array([[xy]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, homography)
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


def exponential_smooth(
    current: Tuple[float, float],
    previous: Optional[Tuple[float, float]],
    alpha: float,
) -> Tuple[float, float]:
    if previous is None:
        return current
    alpha = float(np.clip(alpha, 0.0, 0.99))
    return (
        alpha * current[0] + (1 - alpha) * previous[0],
        alpha * current[1] + (1 - alpha) * previous[1],
    )
