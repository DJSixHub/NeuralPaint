# Lógica de calibración basada en contornos y rejillas AprilTag.
from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


# CalibrationData encapsula matrices de homografía y dimensiones enteras.
@dataclass
class CalibrationData:
    homography: np.ndarray
    inverse_homography: np.ndarray
    surface_width: int
    surface_height: int


# CalibrationResult guarda la homografía temporal y el polígono detectado.
@dataclass
class CalibrationResult:
    polygon: np.ndarray
    homography: np.ndarray


# APRILTAG_GRID_LAYOUT define la posición lógica de cada marcador requerido.
APRILTAG_GRID_LAYOUT: Dict[int, np.ndarray] = {
    0: np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    1: np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    2: np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]], dtype=np.float32),
    3: np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]], dtype=np.float32),
}
# APRILTAG_GRID_COLUMNS y APRILTAG_GRID_ROWS indican la rejilla 2x2 usada.
APRILTAG_GRID_COLUMNS = 2.0
APRILTAG_GRID_ROWS = 2.0
# APRILTAG_REQUIRED_IDS contiene los IDs obligatorios que deben detectarse.
APRILTAG_REQUIRED_IDS = set(APRILTAG_GRID_LAYOUT.keys())


# load_calibration lee el archivo .npz, devuelve CalibrationData y lanza FileNotFoundError si falta.
def load_calibration(path: Path) -> CalibrationData:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró calibración en {path}. Ejecuta la calibración primero.")
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


# save_calibration guarda la homografía y dimensiones en formato .npz.
def save_calibration(path: Path, homography: np.ndarray, width: float, height: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        homography=homography.astype(np.float32),
        surface_width=float(width),
        surface_height=float(height),
    )
# preprocess_frame convierte la imagen a escala de grises, suaviza y ecualiza.
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.equalizeHist(gray)


# order_corners_clockwise ordena cuatro puntos en sentido horario y devuelve float32.
def order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    top_left_idx = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -top_left_idx, axis=0)
    return ordered.astype(np.float32)
# detect_surface_polygon busca un rectángulo brillante y devuelve sus vértices o None.
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
# compute_homography calcula la homografía que lleva el polígono a width x height.
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
        raise RuntimeError("No se pudo resolver la homografía")
    return homography


# polygon_from_homography regresa el polígono proyectado a partir de la homografía.
def polygon_from_homography(homography: np.ndarray, width: float, height: float) -> np.ndarray:
    inverse = np.linalg.inv(homography)
    target = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    polygon = cv2.perspectiveTransform(target, inverse).reshape(-1, 2)
    return polygon.astype(np.float32)
# draw_calibration_overlay dibuja el mensaje y el polígono sobre la imagen, con opción espejo.
def draw_calibration_overlay(
    frame: np.ndarray,
    polygon: Optional[np.ndarray],
    message: str,
    flip: bool,
) -> np.ndarray:
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
    if flip:
        overlay = cv2.flip(overlay, 1)
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
# prompt_calibration_mode pide al usuario el modo y devuelve 'contour' o 'apriltag'.
def prompt_calibration_mode() -> str:
    prompt_text = (
        "Selecciona modo de calibración:\n"
        "  1) Contorno brillante (rectángulo luminoso)\n"
        "  2) Rejilla AprilTag (marcadores 0-3)\n"
        "Opción [1-2]: "
    )
    while True:
        try:
            choice = input(prompt_text).strip()
        except EOFError:
            return "contour"
        if choice == "1":
            return "contour"
        if choice == "2":
            return "apriltag"
        print("Opción inválida, intenta de nuevo.")
# detect_contour_calibration aplica la detección basada en contornos y devuelve CalibrationResult o None.
def detect_contour_calibration(frame: np.ndarray, args: Namespace) -> Optional[CalibrationResult]:
    polygon = detect_surface_polygon(frame, args.calibration_min_area, args.calibration_approx_eps)
    if polygon is None:
        return None
    homography = compute_homography(polygon, args.surface_width, args.surface_height)
    return CalibrationResult(polygon=polygon, homography=homography)
# create_apriltag_state configura el diccionario ArUco necesario o devuelve None si falta soporte.
def create_apriltag_state() -> Optional[Dict[str, object]]:
    if not hasattr(cv2, "aruco"):
        return None
    aruco_module = cv2.aruco
    if not hasattr(aruco_module, "getPredefinedDictionary"):
        return None
    dict_id = getattr(aruco_module, "DICT_APRILTAG_36h11", None)
    if dict_id is None:
        return None
    dictionary = aruco_module.getPredefinedDictionary(dict_id)
    if hasattr(aruco_module, "DetectorParameters"):
        parameters = aruco_module.DetectorParameters()
    else:  # pragma: no cover - compatibilidad con OpenCV antiguos
        parameters = aruco_module.DetectorParameters_create()
    return {"dictionary": dictionary, "parameters": parameters}


# detect_apriltag_calibration resuelve homografía con la rejilla AprilTag y devuelve resultado/parámetros.
def detect_apriltag_calibration(
    frame: np.ndarray,
    surface_width: float,
    surface_height: float,
    state: Dict[str, object],
) -> Tuple[Optional[CalibrationResult], Tuple[Sequence[np.ndarray], Optional[np.ndarray]]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, state["dictionary"], parameters=state["parameters"])
    if ids is None or len(ids) == 0:
        return None, (corners, ids)

    ids_list = ids.flatten().tolist()
    img_points: list[np.ndarray] = []
    dst_points: list[np.ndarray] = []

    for tag_corners, tag_id in zip(corners, ids_list):
        if tag_id not in APRILTAG_GRID_LAYOUT:
            continue
        img_points.append(tag_corners.reshape(4, 2).astype(np.float32))
        layout = APRILTAG_GRID_LAYOUT[tag_id]
        scaled = layout.copy()
        scaled[:, 0] *= float(surface_width) / APRILTAG_GRID_COLUMNS
        scaled[:, 1] *= float(surface_height) / APRILTAG_GRID_ROWS
        dst_points.append(scaled.astype(np.float32))

    if not img_points:
        return None, (corners, ids)

    if len(dst_points) < 2:
        return None, (corners, ids)

    img_pts = np.concatenate(img_points, axis=0)
    dst_pts = np.concatenate(dst_points, axis=0)
    if img_pts.shape[0] < 4:
        return None, (corners, ids)

    homography, status = cv2.findHomography(img_pts, dst_pts, method=0)
    if homography is None or status is None or not status.all():
        return None, (corners, ids)

    polygon = polygon_from_homography(homography, surface_width, surface_height)
    return CalibrationResult(polygon=polygon, homography=homography.astype(np.float32)), (corners, ids)
# run_auto_calibration abre la cámara y ejecuta el lazo de calibración automática.
def run_auto_calibration(args: Namespace) -> Optional[CalibrationData]:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con índice {args.camera}", file=sys.stderr)
        return None

    mode = args.calibration_mode
    if mode == "prompt":
        mode = prompt_calibration_mode()

    apriltag_state: Optional[Dict[str, object]] = None
    if mode == "apriltag":
        apriltag_state = create_apriltag_state()
        if apriltag_state is None:
            print("Soporte ArUco/AprilTag no disponible en OpenCV; usando modo de contorno.", file=sys.stderr)
            mode = "contour"
    args.calibration_mode = mode

    windows_created = False
    try:
        cv2.namedWindow("Calibracion", cv2.WINDOW_NORMAL)
        if args.calibration_warp:
            cv2.namedWindow("Rectificado", cv2.WINDOW_NORMAL)
        if args.calibration_debug:
            cv2.namedWindow("Bordes", cv2.WINDOW_NORMAL)
        windows_created = True
    except cv2.error:
        print(
            "OpenCV fue instalado sin soporte de ventanas (cv2.imshow). Reinstala con 'pip install opencv-contrib-python'.",
            file=sys.stderr,
        )
        cap.release()
        return None

    latest: Optional[CalibrationResult] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fallo al capturar un cuadro de la cámara", file=sys.stderr)
                latest = None
                break

            detection_result: Optional[CalibrationResult] = None
            marker_info: Tuple[Sequence[np.ndarray], Optional[np.ndarray]] = ([], None)

            if mode == "apriltag" and apriltag_state is not None:
                detection_result, marker_info = detect_apriltag_calibration(
                    frame,
                    args.surface_width,
                    args.surface_height,
                    apriltag_state,
                )
                corners, ids = marker_info
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                if detection_result is not None:
                    latest = detection_result
                    message = "Rejilla AprilTag detectada"
                    if args.calibration_warp:
                        warped = cv2.warpPerspective(
                            frame,
                            latest.homography,
                            (int(args.surface_width), int(args.surface_height)),
                        )
                        display_warped = cv2.flip(warped, 1) if getattr(args, "flip_view", False) else warped
                        cv2.imshow("Rectificado", display_warped)
                else:
                    detected_ids = set()
                    if marker_info[1] is not None:
                        detected_ids = {int(x) for x in marker_info[1].flatten().tolist()}
                    missing = APRILTAG_REQUIRED_IDS.difference(detected_ids)
                    if missing:
                        missing_str = ", ".join(str(mid) for mid in sorted(missing))
                        message = f"Muestra la rejilla AprilTag (faltan IDs: {missing_str})"
                    else:
                        message = "No se pudo resolver la homografia con las AprilTags"
            else:
                try:
                    detection_result = detect_contour_calibration(frame, args)
                except RuntimeError:
                    detection_result = None
                if detection_result is not None:
                    latest = detection_result
                    message = "Superficie detectada"
                    if args.calibration_warp:
                        warped = cv2.warpPerspective(
                            frame,
                            latest.homography,
                            (int(args.surface_width), int(args.surface_height)),
                        )
                        display_warped = cv2.flip(warped, 1) if getattr(args, "flip_view", False) else warped
                        cv2.imshow("Rectificado", display_warped)
                else:
                    message = "Muestra un rectangulo brillante que cubra el area"

            polygon = detection_result.polygon if detection_result is not None else None

            display_overlay = draw_calibration_overlay(
                frame,
                polygon,
                message,
                getattr(args, "flip_view", False),
            )
            cv2.imshow("Calibracion", display_overlay)

            if args.calibration_debug:
                if mode == "apriltag" and marker_info[0]:
                    debug_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), marker_info[0], marker_info[1])
                    if getattr(args, "flip_view", False):
                        debug_frame = cv2.flip(debug_frame, 1)
                    cv2.imshow("Bordes", debug_frame)
                else:
                    edges_dbg = cv2.Canny(preprocess_frame(frame), 50, 150)
                    if getattr(args, "flip_view", False):
                        edges_dbg = cv2.flip(edges_dbg, 1)
                    cv2.imshow("Bordes", edges_dbg)

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
                print(f"Calibración guardada en {args.calibration}")
                break
    finally:
        cap.release()
        if windows_created:
            cv2.destroyWindow("Calibracion")
            if args.calibration_warp:
                cv2.destroyWindow("Rectificado")
            if args.calibration_debug:
                cv2.destroyWindow("Bordes")

    if latest is None:
        return None
    return CalibrationData(
        homography=latest.homography.astype(np.float32),
        inverse_homography=np.linalg.inv(latest.homography.astype(np.float32)),
        surface_width=int(args.surface_width),
        surface_height=int(args.surface_height),
    )
# ensure_calibration decide si recalibrar y devuelve CalibrationData o None.
def ensure_calibration(args: Namespace) -> Optional[CalibrationData]:
    needs_calibration = args.force_calibrate or args.calibration_only or not args.calibration.exists()
    if needs_calibration:
        calibration = run_auto_calibration(args)
        if calibration is None:
            print("La calibración falló o fue cancelada.", file=sys.stderr)
            return None
        return calibration
    return load_calibration(args.calibration)
