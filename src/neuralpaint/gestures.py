from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import mediapipe as mp
import numpy as np


class CommandType(Enum):
    NONE = auto()
    DRAW_MODE = auto()
    ERASE_MODE = auto()
    CLEAR_ALL = auto()
    COLOR_PICKER = auto()


class InteractionMode(Enum):
    IDLE = auto()
    DRAW = auto()
    ERASE = auto()
    COLOR_SELECT = auto()


@dataclass
class ArmGestureClassifier:
    hold_frames: int

    def __post_init__(self) -> None:
        self.hold_frames = max(1, self.hold_frames)
        self._last_raw: CommandType = CommandType.NONE
        self._counter: int = 0

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


def _to_vec(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y], dtype=np.float32)


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
