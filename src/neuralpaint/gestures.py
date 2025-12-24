# Clasificación de gestos de brazo y modos de interacción.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import mediapipe as mp
import numpy as np


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


