# Bucle principal de la aplicación interactiva con gestos, overlay y OCR.
from __future__ import annotations

import ctypes
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:  # pragma: no cover - lectura de teclado específica de Windows
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - compatibilidad con sistemas sin Windows
    msvcrt = None

from .calibration import CalibrationData
from .color_picker import ColorPicker
from .gestures import CommandType, InteractionMode, ArmGestureClassifier, classify_left_arm_command
from .overlay import HAS_OVERLAY_SUPPORT, OverlayWindow, draw_status_banner
from .recognition import RegionAnalyzer, RecognitionResult
from .strokes import StrokeCanvas
from .utils import (
    clamp_point,
    exponential_smooth,
    is_inside_surface,
    project_point,
    surface_corners_in_camera,
)

# MODE_LABELS mapea cada modo de interacción a una etiqueta en pantalla.
MODE_LABELS = {
    InteractionMode.IDLE: "INACTIVO",
    InteractionMode.DRAW: "DIBUJAR",
    InteractionMode.ERASE: "BORRAR",
    InteractionMode.COLOR_SELECT: "COLORES",
}


# extract_pointer_position toma el resultado de MediaPipe y devuelve un punto (x, y) o None.
def extract_pointer_position(
    holistic_result,
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.4,
) -> Optional[Tuple[float, float]]:
    right_hand = holistic_result.right_hand_landmarks
    if right_hand:
        index_tip = right_hand.landmark[8]
        return index_tip.x * frame_width, index_tip.y * frame_height

    pose_landmarks = holistic_result.pose_landmarks
    if pose_landmarks:
        wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        if wrist.visibility >= min_visibility:
            return wrist.x * frame_width, wrist.y * frame_height
    return None


# run_interactive_app recibe los argumentos CLI y la calibración y devuelve código de salida int.
def run_interactive_app(args, calibration: CalibrationData) -> int:
    if not HAS_OVERLAY_SUPPORT:
        print(
            "pywin32 es obligatorio para el modo de superposición. Instálalo con 'pip install pywin32'.",
        )
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con índice {args.camera}")
        return 1

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
    )
    command_classifier = ArmGestureClassifier(args.command_hold_frames)

    stroke_canvas = StrokeCanvas(
        calibration.surface_width,
        calibration.surface_height,
        thickness=max(1, int(args.brush_thickness)),
    )
    color_picker = ColorPicker()
    recognizer = RegionAnalyzer(model_dir=args.easyocr_models, use_gpu=args.easyocr_gpu)
    active_recognition: Optional[RecognitionResult] = None
    recognition_display_until = 0.0
    last_recognition_request = 0.0
    recognition_cooldown = 1.5
    recognition_display_duration = 8.0

    prev_pointer: Optional[Tuple[float, float]] = None
    drawing_active = False
    interaction_mode = InteractionMode.IDLE
    last_mode_before_picker = InteractionMode.DRAW
    toggle_delay = max(0.0, float(args.mode_toggle_delay))
    mode_activated_at = {
        InteractionMode.DRAW: 0.0,
        InteractionMode.ERASE: 0.0,
    }

    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    overlay = OverlayWindow(screen_width, screen_height)
    scale_x = screen_width / calibration.surface_width
    scale_y = screen_height / calibration.surface_height
    preview_scale = float(np.clip(args.preview_scale, 0.05, 0.5))
    surface_polygon_cam: Optional[np.ndarray] = None

    surface_dimensions = (
        int(calibration.surface_width),
        int(calibration.surface_height),
    )

    # extract_surface_roi devuelve un recorte del lienzo alrededor del centro dado o None.
    def extract_surface_roi(surface_img: np.ndarray, center: Tuple[int, int]) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:

        roi_width = min(420, surface_dimensions[0])
        roi_height = min(280, surface_dimensions[1])
        half_w = roi_width // 2
        half_h = roi_height // 2
        cx, cy = int(center[0]), int(center[1])
        x0 = int(np.clip(cx - half_w, 0, max(0, surface_dimensions[0] - roi_width)))
        y0 = int(np.clip(cy - half_h, 0, max(0, surface_dimensions[1] - roi_height)))
        x1 = x0 + roi_width
        y1 = y0 + roi_height
        patch = surface_img[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        return patch, (x0, y0)

    # try_request_recognition lanza OCR asincrónico si el puntero está dentro de la superficie.
    def try_request_recognition(surface_img: np.ndarray, center: Optional[Tuple[int, int]]) -> bool:
        nonlocal last_recognition_request
        if center is None or recognizer.busy:
            return False
        if time.perf_counter() - last_recognition_request < recognition_cooldown:
            return False
        roi = extract_surface_roi(surface_img, center)
        if roi is None:
            return False
        patch, origin = roi
        recognizer.submit(patch, origin)
        last_recognition_request = time.perf_counter()
        return True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fallo al capturar un cuadro de la cámara")
                break

            frame_height, frame_width = frame.shape[:2]
            if surface_polygon_cam is None:
                surface_polygon_cam = surface_corners_in_camera(
                    calibration.inverse_homography,
                    calibration.surface_width,
                    calibration.surface_height,
                )
            surface_view = cv2.warpPerspective(frame, calibration.homography, surface_dimensions)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            if result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
                )
            if result.left_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
            if result.right_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

            raw_command = classify_left_arm_command(result.pose_landmarks)
            command = command_classifier.update(raw_command)

            now = time.perf_counter()
            if command == CommandType.CLEAR_ALL:
                stroke_canvas.clear()
                drawing_active = False
                if not color_picker.active:
                    interaction_mode = InteractionMode.IDLE
            elif command == CommandType.COLOR_PICKER and not color_picker.active:
                color_picker.activate(frame.shape)
                last_mode_before_picker = interaction_mode if interaction_mode != InteractionMode.COLOR_SELECT else InteractionMode.DRAW
                interaction_mode = InteractionMode.COLOR_SELECT
                drawing_active = False
            elif not color_picker.active:
                if command == CommandType.DRAW_MODE:
                    if interaction_mode == InteractionMode.DRAW:
                        if now - mode_activated_at[InteractionMode.DRAW] >= toggle_delay:
                            interaction_mode = InteractionMode.IDLE
                            drawing_active = False
                    else:
                        interaction_mode = InteractionMode.DRAW
                        mode_activated_at[InteractionMode.DRAW] = now
                elif command == CommandType.ERASE_MODE:
                    if interaction_mode == InteractionMode.ERASE:
                        if now - mode_activated_at[InteractionMode.ERASE] >= toggle_delay:
                            interaction_mode = InteractionMode.IDLE
                    else:
                        interaction_mode = InteractionMode.ERASE
                        mode_activated_at[InteractionMode.ERASE] = now
                    drawing_active = False

            pointer_cam: Optional[Tuple[float, float]] = extract_pointer_position(
                result,
                frame_width,
                frame_height,
            )

            if pointer_cam is not None:
                smoothed = exponential_smooth(pointer_cam, prev_pointer, args.smoothing)
                prev_pointer = smoothed
            else:
                prev_pointer = None

            pointer_surface_point: Optional[Tuple[int, int]] = None
            pointer_screen_point: Optional[Tuple[int, int]] = None
            pointer_preview_point: Optional[Tuple[int, int]] = None

            if prev_pointer is not None:
                pointer_preview_point = (
                    int(np.clip(prev_pointer[0], 0, frame_width - 1)),
                    int(np.clip(prev_pointer[1], 0, frame_height - 1)),
                )
                if args.flip_view:
                    pointer_preview_point = (
                        frame_width - 1 - pointer_preview_point[0],
                        pointer_preview_point[1],
                    )

                pointer_surface = project_point(calibration.homography, prev_pointer)
                if is_inside_surface(pointer_surface, calibration.surface_width, calibration.surface_height):
                    pointer_surface_point = clamp_point(
                        pointer_surface,
                        calibration.surface_width,
                        calibration.surface_height,
                    )
                    pointer_screen_point = (
                        int(np.clip(pointer_surface_point[0] * scale_x, 0, screen_width - 1)),
                        int(np.clip(pointer_surface_point[1] * scale_y, 0, screen_height - 1)),
                    )

            if color_picker.active:
                selected_color = color_picker.update(pointer_preview_point)
                if selected_color is not None:
                    stroke_canvas.set_color(selected_color)
                    color_picker.deactivate()
                    interaction_mode = last_mode_before_picker
                    if interaction_mode in mode_activated_at:
                        mode_activated_at[interaction_mode] = time.perf_counter()

            pointer_color = stroke_canvas.brush_color
            if interaction_mode == InteractionMode.ERASE:
                pointer_color = (0, 0, 255)
            elif interaction_mode == InteractionMode.IDLE:
                pointer_color = (0, 215, 255)
            elif interaction_mode == InteractionMode.COLOR_SELECT:
                pointer_color = (255, 255, 255)

            if interaction_mode == InteractionMode.DRAW and pointer_surface_point is not None and not color_picker.active:
                if not drawing_active:
                    drawing_active = True
                    stroke_canvas.start_stroke(pointer_surface_point)
                else:
                    stroke_canvas.add_point(pointer_surface_point)
            elif interaction_mode == InteractionMode.ERASE and pointer_surface_point is not None:
                stroke_canvas.erase_at(pointer_surface_point, args.erase_radius)
                drawing_active = False
            else:
                drawing_active = False

            if len(stroke_canvas.strokes) > args.max_strokes:
                stroke_canvas.strokes = stroke_canvas.strokes[-args.max_strokes :]

            canvas = stroke_canvas.render()
            canvas_display = canvas.copy()
            if pointer_surface_point is not None and not color_picker.active:
                cv2.circle(canvas_display, pointer_surface_point, 10, pointer_color, -1, cv2.LINE_AA)

            overlay_rgb = cv2.resize(canvas_display, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
            overlay_frame = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
            mask = cv2.cvtColor(overlay_rgb, cv2.COLOR_BGR2GRAY)
            overlay_frame[..., :3] = overlay_rgb
            overlay_frame[..., 3] = np.where(mask > 10, 200, 0)

            overlay_color = overlay_frame[..., :3].copy()
            overlay_alpha = overlay_frame[..., 3].copy()

            recognition_result = recognizer.poll_result()
            if recognition_result is not None:
                active_recognition = recognition_result
                recognition_display_until = time.perf_counter() + recognition_display_duration
            display_now = time.perf_counter()

            if pointer_screen_point is not None:
                cv2.circle(overlay_color, pointer_screen_point, 14, pointer_color, -1, cv2.LINE_AA)
                cv2.circle(overlay_alpha, pointer_screen_point, 14, 255, -1, cv2.LINE_AA)

            preview_height = int(screen_height * preview_scale)
            preview_width = int(screen_width * preview_scale)
            if preview_height > 0 and preview_width > 0:
                preview_image = cv2.flip(frame, 1) if args.flip_view else frame.copy()
                if surface_polygon_cam is not None and np.isfinite(surface_polygon_cam).all():
                    polygon = surface_polygon_cam.copy()
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_height - 1)
                    if args.flip_view:
                        polygon[:, 0] = frame_width - 1 - polygon[:, 0]
                    polygon_int = polygon.astype(np.int32)
                    cv2.polylines(preview_image, [polygon_int], True, (0, 255, 255), 2, cv2.LINE_AA)
                if pointer_preview_point is not None:
                    cv2.circle(preview_image, pointer_preview_point, 10, pointer_color, -1, cv2.LINE_AA)

                if color_picker.active:
                    color_picker.draw(preview_image, pointer_preview_point)

                preview_resized = cv2.resize(preview_image, (preview_width, preview_height))
                y0 = 10
                x0 = 10
                y_end = min(y0 + preview_height, screen_height)
                x_end = min(x0 + preview_width, screen_width)
                patch_height = y_end - y0
                patch_width = x_end - x0
                if patch_height > 0 and patch_width > 0:
                    overlay_color[y0:y_end, x0:x_end] = preview_resized[:patch_height, :patch_width]
                    alpha_patch = np.full((patch_height, patch_width), 230, dtype=np.uint8)
                    existing_alpha = overlay_alpha[y0:y_end, x0:x_end]
                    np.maximum(existing_alpha, alpha_patch, out=existing_alpha)


            overlay_frame[..., :3] = overlay_color
            overlay_frame[..., 3] = overlay_alpha

            mode_label = MODE_LABELS.get(interaction_mode, interaction_mode.name)
            status_lines = [
                f"MODO: {mode_label}",
                "Brazo izquierdo -> horiz+arriba dibuja | horiz+abajo borra | arriba limpia | horiz mantiene color",
                "Teclas: q salir | c limpiar | r analizar región",
            ]
            if recognizer.busy:
                status_lines.append("Analizando... mantén el puntero quieto")
            elif active_recognition is not None and display_now <= recognition_display_until and active_recognition.has_items:
                status_lines.append(f"Recon: {active_recognition.kind.upper()} ({len(active_recognition.items)} elementos)")
            elif active_recognition is not None and active_recognition.error:
                status_lines.append("Recon: error, revisa la consola")
            draw_status_banner(overlay_frame, status_lines)

            overlay.update(np.ascontiguousarray(overlay_frame))
            overlay.pump_messages()

            key_pressed = None
            if msvcrt and msvcrt.kbhit():  # pragma: no branch - lectura de teclado Windows
                key_pressed = msvcrt.getwch().lower()
            if key_pressed == "q":
                break
            if key_pressed == "c":
                stroke_canvas.clear()
                prev_pointer = None
                interaction_mode = InteractionMode.IDLE
                drawing_active = False
            if key_pressed == "r":
                try_request_recognition(surface_view, pointer_surface_point)
                continue

    finally:
        cap.release()
        holistic.close()
        overlay.close()
        recognizer.close()

    return 0
