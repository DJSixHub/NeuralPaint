# Bucle principal de la aplicación interactiva con gestos, overlay y OCR.
from __future__ import annotations

import ctypes
import importlib
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import os

msvcrt = importlib.import_module("msvcrt") if importlib.util.find_spec("msvcrt") else None

from .calibration import CalibrationData
from .color_picker import ColorPicker
from .gestures import (
    ArmGestureClassifier,
    CommandType,
    InteractionMode,
    SelectionGestureTracker,
    both_index_fingers_up,
    classify_left_arm_command,
    extract_pointer_position,
    hand_index_point,
)
from .overlay import HAS_OVERLAY_SUPPORT, OverlayWindow, draw_status_banner, enumerate_monitors
from .recognition import RegionAnalyzer, RecognitionResult
from .segmentation import Segmenter
from .segmentation_options import apply_effect
from .strokes import StrokeCanvas

# MODE_LABELS mapea cada modo de interacción a una etiqueta en pantalla.
MODE_LABELS = {
    InteractionMode.IDLE: "INACTIVO",
    InteractionMode.DRAW: "DIBUJAR",
    InteractionMode.ERASE: "BORRAR",
    InteractionMode.COLOR_SELECT: "COLORES",
    InteractionMode.REGION_SELECT: "SELECCION",
}


# Ajusta un punto a límites de ancho/alto de superficie.
def clamp_point(point: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    x = int(np.clip(point[0], 0, width - 1))
    y = int(np.clip(point[1], 0, height - 1))
    return x, y


# Verifica si un punto está dentro de la superficie.
def is_inside_surface(point: Tuple[float, float], width: int, height: int) -> bool:
    x, y = point
    return 0 <= x < width and 0 <= y < height


# Proyecta un punto con la homografía dada.
def project_point(homography: np.ndarray, xy: Tuple[float, float]) -> Tuple[float, float]:
    pts = np.array([[xy]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, homography)
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


# Suaviza exponencialmente un punto actual respecto al previo.
def exponential_smooth(current: Tuple[float, float], previous: Optional[Tuple[float, float]], alpha: float) -> Tuple[float, float]:
    if previous is None:
        return current
    alpha = float(np.clip(alpha, 0.0, 0.99))
    return (
        alpha * current[0] + (1 - alpha) * previous[0],
        alpha * current[1] + (1 - alpha) * previous[1],
    )


# Devuelve los vértices de la superficie en coordenadas de cámara usando homografía inversa.
def surface_corners_in_camera(inverse_homography: np.ndarray, width: float, height: float) -> np.ndarray:
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


# Crea un frame BGRA para el proyector con canvas y efectos persistentes (sin UI ni video).
def render_projector_overlay(
    canvas_rgb: np.ndarray,
    screen_width: int,
    screen_height: int,
    applied_masks: list,
) -> np.ndarray:
    # Escalar canvas al tamaño del proyector
    canvas_display = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
    overlay_rgb = cv2.resize(canvas_display, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
    overlay_frame = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    overlay_color = overlay_frame[..., :3]
    overlay_alpha = overlay_frame[..., 3]
    
    # Renderizar canvas: solo píxeles con contenido visible (no negros)
    # Alpha proporcional a la intensidad del color para trazos suaves
    canvas_intensity = np.max(overlay_rgb, axis=2)  # Max RGB channel per pixel
    content_mask = canvas_intensity > 5  # Umbral bajo para detectar contenido
    
    overlay_color[:] = overlay_rgb
    # Alpha 0 donde no hay contenido, 200 donde sí hay (semi-transparente para ver debajo)
    overlay_alpha[content_mask] = 200
    
    # Renderizar efectos aplicados (BGRA con transparencia)
    for (sx, sy, sw, sh, _produced_path, effect_patch) in applied_masks:
        if effect_patch is None or effect_patch.ndim != 3 or effect_patch.shape[2] != 4:
            continue
            
        x0 = max(0, min(int(sx), screen_width - 1))
        y0 = max(0, min(int(sy), screen_height - 1))
        x1 = max(x0 + 1, min(int(sx + sw), screen_width))
        y1 = max(y0 + 1, min(int(sy + sh), screen_height))
        sub_w = x1 - x0
        sub_h = y1 - y0
        
        if sub_w <= 0 or sub_h <= 0:
            continue
        
        # Resize effect patch to match display region
        effect_resized = effect_patch
        if effect_patch.shape[:2] != (sub_h, sub_w):
            effect_resized = cv2.resize(effect_patch, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)
        
        # Composite the BGRA effect onto the overlay using alpha blending
        effect_bgr = effect_resized[:, :, :3]
        effect_alpha = effect_resized[:, :, 3].astype(np.float32) / 255.0
        
        # Blend color
        for c in range(3):
            overlay_color[y0:y1, x0:x1, c] = (
                overlay_color[y0:y1, x0:x1, c] * (1 - effect_alpha) + 
                effect_bgr[:, :, c] * effect_alpha
            ).astype(np.uint8)
        
        # Update alpha to show the effect is visible
        overlay_alpha[y0:y1, x0:x1] = np.maximum(
            overlay_alpha[y0:y1, x0:x1],
            (effect_alpha * 255).astype(np.uint8)
        )

    return overlay_frame


# run_interactive_app recibe los argumentos CLI y la calibración y devuelve código de salida int.
def run_interactive_app(args, calibration: CalibrationData) -> int:
    if not HAS_OVERLAY_SUPPORT:
        print(
            "pywin32 es obligatorio para el modo de superposición. Instálalo con 'pip install pywin32'.",
        )
        return 1

    # Asegura coords Win32 reales (evita offset con DPI scaling) antes de crear ventanas.
    if hasattr(ctypes.windll, "user32") and hasattr(ctypes.windll.user32, "SetProcessDPIAware"):
        ctypes.windll.user32.SetProcessDPIAware()

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
    # Selector de segmentación y segmenter con models/checkpoint_epoch_70.pth.
    segmenter = Segmenter()
    selection_tracker = SelectionGestureTracker()
    # Efectos aplicados almacenados en coords de pantalla (sx, sy, w, h, ruta, BGRA).
    applied_masks = []
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

    # Inicializa overlays en modo dual o monitor único.
    dual_mode = getattr(args, "dual_monitor", False)
    main_mon_idx = getattr(args, "main_monitor", 0)
    proj_mon_idx = getattr(args, "projector_monitor", 1)
    
    if dual_mode:
        # Modo dual: UI en principal, canvas en proyector.
        monitors = enumerate_monitors()
        if main_mon_idx >= len(monitors):
            print(f"Monitor principal {main_mon_idx} no existe. Disponibles: {len(monitors)}")
            return 1
        if proj_mon_idx >= len(monitors):
            print(f"Monitor proyector {proj_mon_idx} no existe. Disponibles: {len(monitors)}")
            return 1
        
        main_mon = monitors[main_mon_idx]
        proj_mon = monitors[proj_mon_idx]
        
        print(f"Modo dual-monitor:")
        print(f"  Monitor principal ({main_mon_idx}): {main_mon[2]}x{main_mon[3]} @ ({main_mon[0]}, {main_mon[1]})")
        print(f"  Monitor proyector ({proj_mon_idx}): {proj_mon[2]}x{proj_mon[3]} @ ({proj_mon[0]}, {proj_mon[1]})")
        
        main_overlay = OverlayWindow(
            main_mon[2], main_mon[3], main_mon[0], main_mon[1],
            title="NeuralPaint - Principal"
        )
        projector_overlay = OverlayWindow(
            proj_mon[2], proj_mon[3], proj_mon[0], proj_mon[1],
            title="NeuralPaint - Proyector"
        )
        
        screen_width = main_mon[2]
        screen_height = main_mon[3]
        proj_width = proj_mon[2]
        proj_height = proj_mon[3]
    else:
        # Modo monitor único (comportamiento por defecto).
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        main_overlay = OverlayWindow(screen_width, screen_height)
        projector_overlay = None
        proj_width = proj_height = 0
    
    scale_x = screen_width / calibration.surface_width
    scale_y = screen_height / calibration.surface_height
    preview_scale = float(np.clip(args.preview_scale, 0.05, 0.5))
    surface_polygon_cam: Optional[np.ndarray] = None

    surface_dimensions = (
        int(calibration.surface_width),
        int(calibration.surface_height),
    )

    # Extrae un recorte del lienzo alrededor de un centro dado o None.
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

    # Lanza OCR asincrónico si el puntero está dentro de la superficie.
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

    both_index_prev = False

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
            left_hand = getattr(result, "left_hand_landmarks", None)
            right_hand = getattr(result, "right_hand_landmarks", None)

            # Entra a REGION_SELECT solo en flanco de ambos índices arriba y si no estaba ya activo.
            both_now = both_index_fingers_up(left_hand, right_hand)
            if both_now and not both_index_prev and interaction_mode != InteractionMode.REGION_SELECT:
                selection_tracker.reset()
                command_classifier.reset()
                interaction_mode = InteractionMode.REGION_SELECT
                drawing_active = False
                print("[REGION] enter REGION_SELECT (both index edge)")
            both_index_prev = both_now

            # Procesa comandos de brazo en todos los modos; si hay selección activa ignora toggles salvo CLEAR_ALL.
            command = command_classifier.update(raw_command)
            if interaction_mode == InteractionMode.REGION_SELECT and selection_tracker.anchor is not None:
                if command != CommandType.CLEAR_ALL:
                    command = None

            now = time.perf_counter()
            if command == CommandType.CLEAR_ALL:
                stroke_canvas.clear()
                drawing_active = False
                if not color_picker.active:
                    interaction_mode = InteractionMode.IDLE
                    # clear any applied segmentation masks on clear-all gesture
                    try:
                        applied_masks = []
                    except Exception:
                        pass
                selection_tracker.reset()
                command_classifier.reset()
            elif command == CommandType.REGION_SELECT:
                # entrada al modo selección de región por gesto: pasa a REGION_SELECT
                selection_tracker.reset()
                command_classifier.reset()
                interaction_mode = InteractionMode.REGION_SELECT
                drawing_active = False
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

            # compute per-hand index tip positions (frame coords)
            pointer_cam_right = hand_index_point(right_hand, frame_width, frame_height)
            pointer_cam_left = hand_index_point(left_hand, frame_width, frame_height)

            # fallback pointer (wrist/right precedence) for legacy behaviour
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

            # also compute surface-space pointers per hand when their index is visible
            pointer_surface_right = None
            pointer_surface_left = None
            if pointer_cam_right is not None:
                psr = project_point(calibration.homography, pointer_cam_right)
                if is_inside_surface(psr, calibration.surface_width, calibration.surface_height):
                    pointer_surface_right = clamp_point(psr, calibration.surface_width, calibration.surface_height)
            if pointer_cam_left is not None:
                psl = project_point(calibration.homography, pointer_cam_left)
                if is_inside_surface(psl, calibration.surface_width, calibration.surface_height):
                    pointer_surface_left = clamp_point(psl, calibration.surface_width, calibration.surface_height)

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

            # Only draw in DRAW mode. REGION_SELECT must not create strokes.
            # Important: don't break a stroke just because the pointer disappears for a frame.
            if interaction_mode == InteractionMode.DRAW and not color_picker.active:
                if pointer_surface_point is not None:
                    if not drawing_active:
                        drawing_active = True
                        stroke_canvas.start_stroke(pointer_surface_point)
                    else:
                        stroke_canvas.add_point(pointer_surface_point)

            # REGION_SELECT: index fingers to size rectangle, finger gestures to confirm with effect
            if interaction_mode == InteractionMode.REGION_SELECT:
                result = selection_tracker.update(
                    left_hand,
                    right_hand,
                    pointer_surface_point,
                    pointer_surface_left,
                    pointer_surface_right,
                )

                if result is not None:
                    rect, effect_type = result
                    lx, ty, w, h = rect
                    lx = max(0, min(lx, int(calibration.surface_width) - 1))
                    ty = max(0, min(ty, int(calibration.surface_height) - 1))
                    w = max(1, min(w, int(calibration.surface_width) - lx))
                    h = max(1, min(h, int(calibration.surface_height) - ty))

                    # compute exact screen coordinates from surface coordinates
                    sx = int(lx * scale_x)
                    sy = int(ty * scale_y)
                    sw = max(1, int(w * scale_x))
                    sh = max(1, int(h * scale_y))

                    # clamp screen coords to valid screen bounds
                    sx = max(0, min(sx, screen_width - sw))
                    sy = max(0, min(sy, screen_height - sh))
                    sw = min(sw, screen_width - sx)
                    sh = min(sh, screen_height - sy)

                    print(f"[REGION] capture request: surface=({lx},{ty},{w},{h}) screen=({sx},{sy},{sw},{sh}) effect={effect_type}")

                    # capture ONLY what's literally on screen - no homography, no transformations
                    # this is the actual visible content the user selected
                    # Hide overlay before capture so the selection rectangle (and any overlay content)
                    # does not appear inside the captured patch.
                    main_overlay.set_visible(False)
                    main_overlay.pump_messages()
                    patch_bgr = main_overlay.capture_region(sx, sy, sw, sh)
                    main_overlay.set_visible(True)

                    print(f"[REGION] capture result: shape={getattr(patch_bgr, 'shape', None)} size={getattr(patch_bgr, 'size', None)}")
                    
                    # only proceed if we successfully captured the screen region
                    if patch_bgr.size > 0 and patch_bgr.shape[0] == sh and patch_bgr.shape[1] == sw:
                        # send patch to network exactly as captured from screen
                        print("[REGION] running segmenter...")
                        mask, produced_path = segmenter.run_on_patch(patch_bgr)
                        print(f"[REGION] segmenter done: produced_path={produced_path} mask_shape={getattr(mask, 'shape', None)} mask_size={getattr(mask, 'size', None)}")
                        if mask is not None and mask.size != 0:
                            if mask.shape[:2] != (sh, sw):
                                mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
                            
                            # Apply the selected effect to the captured patch
                            # Get current drawing color in BGR format (already in BGR)
                            current_color_bgr = stroke_canvas.brush_color
                            print(f"[REGION] applying effect '{effect_type}' with color {current_color_bgr}")
                            patch_with_effect = apply_effect(patch_bgr, mask, effect_type, current_color_bgr)

                            # Store the effect-applied patch (BGRA) for overlay rendering.
                            # Important: never display the produced mask image; keep it only as a saved artifact.
                            applied_masks.append((sx, sy, sw, sh, produced_path, patch_with_effect))
                    else:
                        print("[REGION] capture invalid or wrong size; skipping inference")

                    # Stay in REGION_SELECT mode for seamless multiple segmentations.
                    # Only reset tracker so user can immediately select another region.
                    selection_tracker.reset()
                    drawing_active = False
                    command_classifier.reset()
            elif interaction_mode == InteractionMode.ERASE:
                if pointer_surface_point is not None:
                    stroke_canvas.erase_at(pointer_surface_point, args.erase_radius)
                drawing_active = False
            elif interaction_mode != InteractionMode.DRAW:
                drawing_active = False

            if len(stroke_canvas.strokes) > args.max_strokes:
                stroke_canvas.strokes = stroke_canvas.strokes[-args.max_strokes :]

            canvas = stroke_canvas.render()
            canvas_display = canvas.copy()
            if pointer_surface_point is not None and not color_picker.active:
                cv2.circle(canvas_display, pointer_surface_point, 10, pointer_color, -1, cv2.LINE_AA)

            # draw selection preview rectangle (white) if in REGION_SELECT
            # must be drawn in screen coords on overlay_frame to match actual capture location
            selection_preview_rect = None
            if interaction_mode == InteractionMode.REGION_SELECT:
                anchor, candidate = selection_tracker.preview()
                if anchor is not None and candidate is not None:
                    # convert surface coords to screen coords for accurate preview
                    ax_screen = int(anchor[0] * scale_x)
                    ay_screen = int(anchor[1] * scale_y)
                    cx_screen = int(candidate[0] * scale_x)
                    cy_screen = int(candidate[1] * scale_y)
                    selection_preview_rect = (ax_screen, ay_screen, cx_screen, cy_screen)

            overlay_rgb = cv2.resize(canvas_display, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
            overlay_frame = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
            mask = cv2.cvtColor(overlay_rgb, cv2.COLOR_BGR2GRAY)
            overlay_frame[..., :3] = overlay_rgb
            overlay_frame[..., 3] = np.where(mask > 10, 200, 0)

            overlay_color = overlay_frame[..., :3].copy()
            overlay_alpha = overlay_frame[..., 3].copy()

            # apply effects produced by the segmenter (persistent, already in screen coords)
            if applied_masks:
                for entry in applied_masks:
                    if len(entry) != 6:
                        continue
                    sx, sy, sw, sh, _produced_path, effect_patch = entry

                    x0 = max(0, min(int(sx), screen_width - 1))
                    y0 = max(0, min(int(sy), screen_height - 1))
                    x1 = max(x0 + 1, min(int(sx + sw), screen_width))
                    y1 = max(y0 + 1, min(int(sy + sh), screen_height))
                    sub_w = x1 - x0
                    sub_h = y1 - y0
                    if sub_w <= 0 or sub_h <= 0:
                        continue

                    if effect_patch is None or effect_patch.ndim != 3 or effect_patch.shape[2] != 4:
                        continue

                    effect_resized = effect_patch
                    if effect_patch.shape[:2] != (sub_h, sub_w):
                        effect_resized = cv2.resize(effect_patch, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)

                    effect_bgr = effect_resized[:, :, :3]
                    effect_alpha = effect_resized[:, :, 3].astype(np.float32) / 255.0

                    # Blend color
                    for c in range(3):
                        overlay_color[y0:y1, x0:x1, c] = (
                            overlay_color[y0:y1, x0:x1, c] * (1 - effect_alpha)
                            + effect_bgr[:, :, c] * effect_alpha
                        ).astype(np.uint8)

                    # Update alpha so the effect is visible
                    overlay_alpha[y0:y1, x0:x1] = np.maximum(
                        overlay_alpha[y0:y1, x0:x1],
                        effect_resized[:, :, 3],
                    )

            

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

            # draw selection rectangle preview on final overlay in screen coords
            if selection_preview_rect is not None:
                ax_s, ay_s, cx_s, cy_s = selection_preview_rect
                cv2.rectangle(overlay_frame, (ax_s, ay_s), (cx_s, cy_s), (255, 255, 255, 255), 3, cv2.LINE_AA)

            mode_label = MODE_LABELS.get(interaction_mode, interaction_mode.name)
            status_lines = [
                f"MODO: {mode_label}",
                "Brazo izquierdo -> horiz+arriba dibuja | horiz+abajo borra | arriba limpia | horiz mantiene color",
                "Teclas: q salir | c limpiar | r analizar región",
            ]
            # show which hand is used for gesture/selection when in region mode
            if interaction_mode == InteractionMode.REGION_SELECT:
                initiator = selection_tracker.initiator_hand
                if initiator == "right":
                    status_lines.insert(1, "Seleccion: gesto en mano DERECHA; usa IZQUIERDA para definir rect.")
                elif initiator == "left":
                    status_lines.insert(1, "Seleccion: gesto en mano IZQUIERDA; usa DERECHA para definir rect.")
                else:
                    status_lines.insert(1, "Seleccion: gesto detectado; usa otra mano para definir rect.")
            if recognizer.busy:
                status_lines.append("Analizando... mantén el puntero quieto")
            elif active_recognition is not None and display_now <= recognition_display_until and active_recognition.has_items:
                status_lines.append(f"Recon: {active_recognition.kind.upper()} ({len(active_recognition.items)} elementos)")
            elif active_recognition is not None and active_recognition.error:
                status_lines.append("Recon: error, revisa la consola")
            draw_status_banner(overlay_frame, status_lines)

            # Actualizar overlay principal
            main_overlay.update(np.ascontiguousarray(overlay_frame))
            main_overlay.pump_messages()
            
            # Si modo dual-monitor, actualizar overlay del proyector (solo canvas)
            if dual_mode and projector_overlay is not None:
                projector_frame = render_projector_overlay(
                    stroke_canvas.get_canvas(),
                    proj_width,
                    proj_height,
                    applied_masks,
                )
                projector_overlay.update(np.ascontiguousarray(projector_frame))
                projector_overlay.pump_messages()

            key_pressed = None
            if msvcrt and msvcrt.kbhit():  # pragma: no branch - lectura de teclado Windows
                key_pressed = msvcrt.getwch().lower()
            if key_pressed == "q":
                break
            if key_pressed == "c":
                stroke_canvas.clear()
                prev_pointer = None
                interaction_mode = InteractionMode.IDLE
                # clear applied masks on manual clear
                try:
                    applied_masks = []
                except Exception:
                    pass
                selection_tracker.reset()
                drawing_active = False
            if key_pressed == "r":
                try_request_recognition(surface_view, pointer_surface_point)
                continue

    finally:
        cap.release()
        holistic.close()
        main_overlay.close()
        if dual_mode and projector_overlay is not None:
            projector_overlay.close()
        recognizer.close()

    return 0
