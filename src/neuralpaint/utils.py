# Utilidades geométricas y de suavizado para proyecciones de la superficie.
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


# clamp_point limita un punto flotante al rectángulo entero de la superficie.
def clamp_point(point: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    x = int(np.clip(point[0], 0, width - 1))
    y = int(np.clip(point[1], 0, height - 1))
    return x, y


# is_inside_surface verifica si un punto está dentro del área de dibujo.
def is_inside_surface(point: Tuple[float, float], width: int, height: int) -> bool:
    x, y = point
    return 0 <= x < width and 0 <= y < height


# project_point aplica la homografía y devuelve el punto proyectado (x, y).
def project_point(homography: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    pts = np.array([[xy]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, homography)
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


# exponential_smooth combina el punto actual y previo según el factor alpha.
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


# surface_corners_in_camera proyecta los vértices del lienzo a coordenadas de cámara.
def surface_corners_in_camera(
    inverse_homography: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    corners = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    camera_points = cv2.perspectiveTransform(corners, inverse_homography.astype(np.float32))
    return camera_points.reshape(-1, 2)
