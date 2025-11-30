from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


DEFAULT_PALETTE: Sequence[Color] = (
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 128, 0),
    (128, 0, 255),
)


@dataclass
class ColorPicker:
    palette: Sequence[Color] = field(default_factory=lambda: DEFAULT_PALETTE)
    hold_seconds: float = 3.0

    def __post_init__(self) -> None:
        self.active: bool = False
        self.center: Tuple[int, int] = (0, 0)
        self.radius: int = 0
        self.inner_radius: int = 0
        self._current_index: Optional[int] = None
        self._hold_started: Optional[float] = None

    def activate(self, frame_shape: Tuple[int, int, int]) -> None:
        height, width = frame_shape[:2]
        self.radius = max(60, min(height, width) // 4)
        self.inner_radius = int(self.radius * 0.45)
        self.center = (width // 2, height // 2)
        self.active = True
        self._current_index = None
        self._hold_started = None

    def deactivate(self) -> None:
        self.active = False
        self._current_index = None
        self._hold_started = None

    def draw(self, image: np.ndarray, pointer: Optional[Tuple[int, int]] = None) -> None:
        if not self.active or self.radius <= 0:
            return

        overlay = image.copy()
        num_colors = len(self.palette)
        sweep = 360.0 / num_colors
        highlight = self._current_index if pointer is not None else None

        for idx, color in enumerate(self.palette):
            start_angle = sweep * idx
            end_angle = start_angle + sweep
            thickness = -1
            cv2.ellipse(
                overlay,
                self.center,
                (self.radius, self.radius),
                0,
                start_angle,
                end_angle,
                color,
                thickness,
                cv2.LINE_AA,
            )

        cv2.circle(overlay, self.center, self.inner_radius, (0, 0, 0), -1, cv2.LINE_AA)

        if highlight is not None:
            start_angle = sweep * highlight
            end_angle = start_angle + sweep
            cv2.ellipse(
                overlay,
                self.center,
                (self.radius, self.radius),
                0,
                start_angle,
                end_angle,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        if highlight is not None:
            cv2.putText(
                image,
                "Hold to select",
                (self.center[0] - self.radius, self.center[1] + self.radius + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if self._hold_started:
                elapsed = time.perf_counter() - self._hold_started
                progress = min(1.0, elapsed / self.hold_seconds)
                cv2.ellipse(
                    image,
                    self.center,
                    (self.inner_radius - 10, self.inner_radius - 10),
                    0,
                    0,
                    360 * progress,
                    (255, 255, 255),
                    4,
                    cv2.LINE_AA,
                )

    def update(self, pointer: Optional[Tuple[int, int]]) -> Optional[Color]:
        if not self.active:
            return None

        now = time.perf_counter()

        if pointer is None:
            self._current_index = None
            self._hold_started = None
            return None

        index = self._hit_test(pointer)
        if index is None:
            self._current_index = None
            self._hold_started = None
            return None

        if self._current_index != index:
            self._current_index = index
            self._hold_started = now
            return None

        if self._hold_started is None:
            self._hold_started = now
            return None

        if now - self._hold_started >= self.hold_seconds:
            selected = self.palette[self._current_index]
            self.deactivate()
            return selected
        return None

    def _hit_test(self, pointer: Tuple[int, int]) -> Optional[int]:
        dx = pointer[0] - self.center[0]
        dy = pointer[1] - self.center[1]
        distance = math.hypot(dx, dy)
        if distance < self.inner_radius or distance > self.radius:
            return None

        angle = math.degrees(math.atan2(-dy, dx)) % 360.0
        sweep = 360.0 / len(self.palette)
        return int(angle // sweep)
