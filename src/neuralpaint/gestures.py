# Clasificación de gestos de brazo y modos de interacción.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import mediapipe as mp
import numpy as np


# devuelve el punto del índice en coords de frame o None
def hand_index_point(
    hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.0,
) -> Optional[tuple[float, float]]:
    if hand_landmarks is None:
        return None
    tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    if hasattr(tip, "visibility") and tip.visibility < min_visibility:
        return None
    return tip.x * frame_width, tip.y * frame_height


# CommandType enumera los comandos detectados del brazo izquierdo.
class CommandType(Enum):
    NONE = auto()
    DRAW_MODE = auto()
    ERASE_MODE = auto()
    CLEAR_ALL = auto()
    REGION_SELECT = auto()
    COLOR_PICKER = auto()


# InteractionMode define el estado actual del flujo de dibujo.
class InteractionMode(Enum):
    IDLE = auto()
    DRAW = auto()
    ERASE = auto()
    COLOR_SELECT = auto()
    REGION_SELECT = auto()


# ArmGestureClassifier estabiliza comandos cronometrando cuadros consecutivos.
@dataclass
class ArmGestureClassifier:
    hold_frames: int

    # __post_init__ garantiza un mínimo de cuadros y reinicia el estado interno.
    def __post_init__(self) -> None:
        self.hold_frames = max(1, self.hold_frames)
        self._last_raw: CommandType = CommandType.NONE
        self._counter: int = 0

    def reset(self) -> None:
        self._last_raw = CommandType.NONE
        self._counter = 0

    # update recibe un CommandType crudo y devuelve uno confirmado o None.
    def update(self, raw: CommandType) -> Optional[CommandType]:
        if raw == self._last_raw:
            self._counter += 1
        else:
            self._last_raw = raw
            self._counter = 1

        if raw == CommandType.NONE:
            return None

        if self._counter >= self.hold_frames:
            self._counter = 0
            return raw
        return None


# _to_vec convierte un landmark de MediaPipe a un vector numpy (x, y).
def _to_vec(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y], dtype=np.float32)


# extrae el puntero principal: índice derecho o muñeca derecha visible
def extract_pointer_position(
    holistic_result,
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.4,
) -> Optional[tuple[float, float]]:
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


# classify_left_arm_command analiza el brazo izquierdo y devuelve un CommandType.
def classify_left_arm_command(
    pose_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    min_visibility: float = 0.5,
) -> CommandType:
    if pose_landmarks is None:
        return CommandType.NONE

    pose = pose_landmarks.landmark
    required = (
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    )

    for idx in required:
        if pose[idx].visibility < min_visibility:
            return CommandType.NONE

    shoulder = _to_vec(pose[required[0]])
    elbow = _to_vec(pose[required[1]])
    wrist = _to_vec(pose[required[2]])

    upper = elbow - shoulder
    forearm = wrist - elbow

    if np.linalg.norm(upper) < 1e-4 or np.linalg.norm(forearm) < 1e-4:
        return CommandType.NONE

    upper = upper / np.linalg.norm(upper)
    forearm = forearm / np.linalg.norm(forearm)

    horizontal = abs(upper[1]) <= 0.2 and abs(upper[0]) >= 0.4
    forearm_up = forearm[1] <= -0.4
    forearm_down = forearm[1] >= 0.4
    forearm_horizontal = abs(forearm[1]) <= 0.25 and abs(forearm[0]) >= 0.6
    arm_up = (
        upper[1] <= -0.6
        and forearm[1] <= -0.6
        and abs(upper[0]) <= 0.4
        and abs(forearm[0]) <= 0.4
    )

    if arm_up:
        return CommandType.CLEAR_ALL
    if horizontal and forearm_up:
        return CommandType.DRAW_MODE
    if horizontal and forearm_down:
        return CommandType.ERASE_MODE
    if horizontal and forearm_horizontal:
        return CommandType.COLOR_PICKER
    return CommandType.NONE


def is_left_thumb_up(
    hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    min_visibility: float = 0.5,
) -> bool:
    """Heurística simple para detectar pulgar hacia arriba en la mano izquierda.

    Requiere que la punta del pulgar esté por encima (y menor y) de la muñeca
    y que las otras puntas de dedos estén plegadas (más abajo que sus PIP).
    """
    if hand_landmarks is None:
        return False

    lm = hand_landmarks.landmark
    required = (
        mp.solutions.hands.HandLandmark.WRIST,
        mp.solutions.hands.HandLandmark.THUMB_TIP,
        mp.solutions.hands.HandLandmark.THUMB_IP,
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
        mp.solutions.hands.HandLandmark.PINKY_TIP,
        mp.solutions.hands.HandLandmark.PINKY_PIP,
    )

    for idx in required:
        if lm[idx].visibility < min_visibility:
            return False

    wrist_y = lm[mp.solutions.hands.HandLandmark.WRIST].y
    thumb_tip_y = lm[mp.solutions.hands.HandLandmark.THUMB_TIP].y
    thumb_ip_y = lm[mp.solutions.hands.HandLandmark.THUMB_IP].y

    # pulgar arriba: tip por encima de la muñeca y por encima del IP
    if not (thumb_tip_y + 0.03 < wrist_y and thumb_tip_y + 0.01 < thumb_ip_y):
        return False

    # otros dedos plegados: tip más abajo que su PIP
    idx_cond = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y + 0.01
    mid_cond = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y + 0.01
    ring_cond = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y + 0.01
    pinky_cond = lm[mp.solutions.hands.HandLandmark.PINKY_TIP].y > lm[mp.solutions.hands.HandLandmark.PINKY_PIP].y + 0.01

    if idx_cond and mid_cond and ring_cond and pinky_cond:
        return True
    return False


def is_index_finger_up(
    hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    min_visibility: float = 0.5,
) -> bool:
    """Detect if the index finger is raised (tip above its PIP and other fingers folded)."""
    if hand_landmarks is None:
        return False
    lm = hand_landmarks.landmark
    idx_up = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y - 0.02
    mid_fold = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y + 0.01
    ring_fold = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y + 0.01
    pinky_fold = lm[mp.solutions.hands.HandLandmark.PINKY_TIP].y > lm[mp.solutions.hands.HandLandmark.PINKY_PIP].y + 0.01
    return bool(idx_up and mid_fold and ring_fold and pinky_fold)


def both_index_fingers_up(
    left_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    right_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
) -> bool:
    """Return True if both hands are present and both have the index finger up."""
    if left_hand is None or right_hand is None:
        return False
    return is_index_finger_up(left_hand) and is_index_finger_up(right_hand)


def is_hand_fist(hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList], fold_threshold: float = 0.02) -> bool:
    if hand_landmarks is None:
        return False

    lm = hand_landmarks.landmark
    H = mp.solutions.hands.HandLandmark

    return (
        lm[H.INDEX_FINGER_TIP].y > lm[H.INDEX_FINGER_PIP].y + fold_threshold and
        lm[H.MIDDLE_FINGER_TIP].y > lm[H.MIDDLE_FINGER_PIP].y + fold_threshold and
        lm[H.RING_FINGER_TIP].y > lm[H.RING_FINGER_PIP].y + fold_threshold and
        lm[H.PINKY_TIP].y > lm[H.PINKY_PIP].y + fold_threshold
    )


@dataclass
class SelectionGestureTracker:
    anchor: Optional[Tuple[int, int]] = None
    initiator_hand: Optional[str] = None
    candidate: Optional[Tuple[int, int]] = None
    last_left_fist: bool = False
    last_right_fist: bool = False
    prev_index_up_left: bool = False
    prev_index_up_right: bool = False
    _confirm_counter: int = 0
    _confirm_hold_frames: int = 2
    _debug_last_event: Optional[str] = None

    def reset(self) -> None:
        self.anchor = None
        self.initiator_hand = None
        self.candidate = None
        self.last_left_fist = False
        self.last_right_fist = False
        self.prev_index_up_left = False
        self.prev_index_up_right = False
        self._confirm_counter = 0
        self._debug_last_event = None

    def update(
        self,
        left_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
        right_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
        pointer_surface_point: Optional[Tuple[int, int]],
        pointer_surface_left: Optional[Tuple[int, int]],
        pointer_surface_right: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        left_fist = is_hand_fist(left_hand)
        right_fist = is_hand_fist(right_hand)

        # detect fist rising edge to set anchor
        left_event = left_fist and not self.last_left_fist
        right_event = right_fist and not self.last_right_fist
        self.last_left_fist = left_fist
        self.last_right_fist = right_fist

        if (left_event or right_event) and self.anchor is None:
            if left_event and pointer_surface_left is not None:
                self.anchor = (int(pointer_surface_left[0]), int(pointer_surface_left[1]))
                self.initiator_hand = "left"
                self._confirm_counter = 0
                msg = f"[REGION] anchor set by LEFT fist at {self.anchor}"
                if self._debug_last_event != msg:
                    print(msg)
                    self._debug_last_event = msg
            elif right_event and pointer_surface_right is not None:
                self.anchor = (int(pointer_surface_right[0]), int(pointer_surface_right[1]))
                self.initiator_hand = "right"
                self._confirm_counter = 0

                msg = f"[REGION] anchor set by RIGHT fist at {self.anchor}"
                if self._debug_last_event != msg:
                    print(msg)
                    self._debug_last_event = msg
        # candidate is the opposite hand's pointer.
        # Fallback to the generic pointer if the opposite hand isn't reliably tracked,
        # otherwise confirmation/capture can get stuck with candidate=None.
        if self.initiator_hand == "left":
            self.candidate = pointer_surface_right or pointer_surface_point
        elif self.initiator_hand == "right":
            self.candidate = pointer_surface_left or pointer_surface_point
        else:
            self.candidate = pointer_surface_point

        # Confirm when initiator hand is NOT a fist and its index is up.
        # Use a short hold counter (few frames) instead of a rising-edge requirement,
        # because on subsequent selections the index can already be up and we'd
        # otherwise miss the edge and never confirm.
        idx_up_right = is_index_finger_up(right_hand)
        idx_up_left = is_index_finger_up(left_hand)

        confirm = False
        if self.anchor is not None and self.candidate is not None:
            can_confirm = False
            if self.initiator_hand == "right":
                can_confirm = idx_up_right and not right_fist
            elif self.initiator_hand == "left":
                can_confirm = idx_up_left and not left_fist

            if can_confirm:
                self._confirm_counter += 1
            else:
                self._confirm_counter = 0

            if self._confirm_counter >= max(1, int(self._confirm_hold_frames)):
                confirm = True

        self.prev_index_up_right = idx_up_right
        self.prev_index_up_left = idx_up_left

        if confirm and self.anchor is not None and self.candidate is not None:
            p0 = self.anchor
            p1 = self.candidate
            lx, rx = sorted((int(p0[0]), int(p1[0])))
            ty, by = sorted((int(p0[1]), int(p1[1])))
            w = max(1, rx - lx)
            h = max(1, by - ty)
            rect = (lx, ty, w, h)
            msg = f"[REGION] confirm by {self.initiator_hand} index -> rect={rect}"
            if self._debug_last_event != msg:
                print(msg)
                self._debug_last_event = msg
            self.reset()
            return rect
        return None

    def preview(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        return self.anchor, self.candidate


