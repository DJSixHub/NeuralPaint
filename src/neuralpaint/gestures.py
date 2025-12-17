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
    COLOR_PICKER = auto()


# InteractionMode define el estado actual del flujo de dibujo.
class InteractionMode(Enum):
    IDLE = auto()
    DRAW = auto()
    ERASE = auto()
    COLOR_SELECT = auto()


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


