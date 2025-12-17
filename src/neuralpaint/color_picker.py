# Rueda de colores para seleccionar tonos mediante la mano detectada.

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Color representa un color en formato BGR de OpenCV.
Color = Tuple[int, int, int]

# DEFAULT_PALETTE define los colores disponibles en el selector en formato BGR.
DEFAULT_PALETTE: Sequence[Color] = (
    (180, 0, 180),  # morado
    (255, 0, 255),  # magenta
    (255, 0, 0),  # azul
    (255, 255, 0),  # cian
    (0, 255, 0),  # verde
    (0, 255, 255),  # amarillo
    (0, 128, 255),  # naranja
    (0, 0, 255),  # rojo
    (20, 20, 20),  # casi negro
    (255, 255, 255),  # blanco
)


# ColorPicker maneja la activación y lectura de la rueda de colores.
@dataclass
class ColorPicker:
    # palette define los colores disponibles y hold_seconds controla el tiempo para confirmar.
    palette: Sequence[Color] = field(default_factory=lambda: DEFAULT_PALETTE)
    hold_seconds: float = 3.0

    # __post_init__ prepara los campos internos después de crear la instancia.
    def __post_init__(self) -> None:
        self.active: bool = False
        self.center: Tuple[int, int] = (0, 0)
        self.radius: int = 0
        self.inner_radius: int = 0
        self._current_index: Optional[int] = None
        self._hold_started: Optional[float] = None
        self._display_colors: List[Color] = list(self.palette)

    # activate calcula el centro y radio a partir de la forma del cuadro (alto, ancho, canales).
    def activate(self, frame_shape: Tuple[int, int, int]) -> None:
        height, width = frame_shape[:2]
        self.radius = max(60, min(height, width) // 4)
        self.inner_radius = int(self.radius * 0.45)
        self.center = (width // 2, height // 2)
        self.active = True
        self._current_index = None
        self._hold_started = None

    # deactivate apaga la rueda y limpia el estado interno.
    def deactivate(self) -> None:
        self.active = False
        self._current_index = None
        self._hold_started = None

    # draw pinta la rueda en la imagen y resalta el sector cercano al puntero.
    def draw(self, image: np.ndarray, pointer: Optional[Tuple[int, int]] = None) -> None:
        if not self.active or self.radius <= 0:
            return

        overlay = image.copy()
        num_colors = len(self._display_colors)
        if num_colors == 0:
            return
        sweep = 360.0 / num_colors
        highlight = self._current_index if pointer is not None else None

        for idx, color in enumerate(self._display_colors):
            start_angle = sweep * idx
            end_angle = start_angle + sweep
            cv2.ellipse(
                overlay,
                self.center,
                (self.radius, self.radius),
                0,
                start_angle,
                end_angle,
                color,
                -1,
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
                "Mantener para seleccionar",
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

    # update recibe el puntero en pixeles y devuelve un color BGR seleccionado o None.
    def update(self, pointer: Optional[Tuple[int, int]]) -> Optional[Color]:
        if not self.active:
            return None

        now = time.perf_counter()

        if pointer is None:
            self._current_index = None
            self._hold_started = None
            return None

        sector = self._hit_test(pointer)
        if sector is None:
            self._current_index = None
            self._hold_started = None
            return None

        if self._current_index != sector:
            self._current_index = sector
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

    # _hit_test calcula el índice del sector donde cae el puntero y devuelve None si está fuera.
    def _hit_test(self, pointer: Tuple[int, int]) -> Optional[int]:
        if not self._display_colors:
            return None

        dx = pointer[0] - self.center[0]
        dy = pointer[1] - self.center[1]
        distance = math.hypot(dx, dy)
        if distance < self.inner_radius or distance > self.radius:
            return None

        # cv2.ellipse usa el eje X hacia la derecha y recorre los ángulos en sentido horario
        # (debido a que las coordenadas de imagen crecen hacia abajo). Calculamos el ángulo
        # en el mismo sistema para que el sector coincida con la vista.
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        sweep = 360.0 / len(self._display_colors)
        return int(angle // sweep) % len(self._display_colors)
