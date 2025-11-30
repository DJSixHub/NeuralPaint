from __future__ import annotations

import ctypes
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:  # Windows-only keyboard polling
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None

from .calibration import CalibrationData
from .color_picker import ColorPicker
from .gestures import CommandType, InteractionMode, ArmGestureClassifier, classify_left_arm_command
from .overlay import HAS_OVERLAY_SUPPORT, OverlayWindow, draw_status_banner
from .strokes import StrokeCanvas
from .utils import clamp_point, exponential_smooth, is_inside_surface, project_point


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


def run_interactive_app(args, calibration: CalibrationData) -> int:
    if not HAS_OVERLAY_SUPPORT:
        print(
            "pywin32 is required for the overlay mode. Install it via 'pip install pywin32'.",
        )
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera}")
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

    prev_pointer: Optional[Tuple[float, float]] = None
    drawing_active = False
    interaction_mode = InteractionMode.IDLE
    last_mode_before_picker = InteractionMode.DRAW

    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    overlay = OverlayWindow(screen_width, screen_height)
    scale_x = screen_width / calibration.surface_width
    scale_y = screen_height / calibration.surface_height
    preview_scale = float(np.clip(args.preview_scale, 0.05, 0.5))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame grab failed")
                break

            frame_height, frame_width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            if result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
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
                        interaction_mode = InteractionMode.IDLE
                        drawing_active = False
                    else:
                        interaction_mode = InteractionMode.DRAW
                elif command == CommandType.ERASE_MODE:
                    if interaction_mode == InteractionMode.ERASE:
                        interaction_mode = InteractionMode.IDLE
                    else:
                        interaction_mode = InteractionMode.ERASE
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

            if pointer_screen_point is not None:
                cv2.circle(
                    overlay_frame,
                    pointer_screen_point,
                    14,
                    (*pointer_color, 255),
                    -1,
                    cv2.LINE_AA,
                )

            preview_height = int(screen_height * preview_scale)
            preview_width = int(screen_width * preview_scale)
            if preview_height > 0 and preview_width > 0:
                preview_image = cv2.flip(frame, 1) if args.flip_view else frame.copy()
                if pointer_preview_point is not None:
                    cv2.circle(preview_image, pointer_preview_point, 10, pointer_color, -1, cv2.LINE_AA)

                if color_picker.active:
                    color_picker.draw(preview_image, pointer_preview_point)

                preview_resized = cv2.resize(preview_image, (preview_width, preview_height))
                preview_patch = np.zeros((preview_height, preview_width, 4), dtype=np.uint8)
                preview_patch[..., :3] = preview_resized
                preview_patch[..., 3] = 230
                y_end = min(10 + preview_height, screen_height)
                x_end = min(10 + preview_width, screen_width)
                overlay_frame[10:y_end, 10:x_end] = preview_patch[: y_end - 10, : x_end - 10]

            status_lines = [
                f"MODE: {interaction_mode.name}",
                "Left arm → horiz+up draw | horiz+down erase | up clear | horiz hold color",
                "Keys: q quit | c clear",
            ]
            draw_status_banner(overlay_frame, status_lines)

            overlay.update(np.ascontiguousarray(overlay_frame))
            overlay.pump_messages()

            key_pressed = None
            if msvcrt and msvcrt.kbhit():  # pragma: no branch - Windows keyboard polling
                key_pressed = msvcrt.getwch().lower()
            if key_pressed == "q":
                break
            if key_pressed == "c":
                stroke_canvas.clear()
                prev_pointer = None
                interaction_mode = InteractionMode.IDLE
                drawing_active = False

    finally:
        cap.release()
        holistic.close()
        overlay.close()

    return 0
