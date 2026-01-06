# Clasificación de gestos del brazo y modos de interacción.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import mediapipe as mp
import numpy as np


# Devuelve la posición (x,y) de la punta del índice en coordenadas del frame, o None si no hay mano/visibilidad.
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


# Enumera comandos detectados en el brazo izquierdo.
class CommandType(Enum):
    NONE = auto()
    DRAW_MODE = auto()
    ERASE_MODE = auto()
    CLEAR_ALL = auto()
    REGION_SELECT = auto()
    COLOR_PICKER = auto()


# Define el estado actual del flujo de interacción/dibujo.
class InteractionMode(Enum):
    IDLE = auto()
    DRAW = auto()
    ERASE = auto()
    COLOR_SELECT = auto()
    REGION_SELECT = auto()


# Estabiliza comandos del brazo acumulando cuadros consecutivos del mismo gesto.
@dataclass
class ArmGestureClassifier:
    hold_frames: int

    # Inicializa el clasificador garantizando un mínimo de cuadros y reiniciando el estado interno.
    def __post_init__(self) -> None:
        self.hold_frames = max(1, self.hold_frames)
        self._last_raw: CommandType = CommandType.NONE
        self._counter: int = 0

    # Reinicia el estado interno del clasificador (último comando y contador).
    def reset(self) -> None:
        self._last_raw = CommandType.NONE
        self._counter = 0

    # Consume un comando crudo y devuelve el comando confirmado cuando se sostuvo suficientes cuadros.
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


# Convierte un landmark de MediaPipe a un vector numpy (x, y).
def _to_vec(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y], dtype=np.float32)


# Extrae el puntero principal: punta del índice derecho si hay mano; si no, muñeca derecha del pose si es visible.
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


# Clasifica el gesto del brazo izquierdo (pose) y devuelve el comando correspondiente.
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


 # Detecta si el pulgar izquierdo está arriba y los otros dedos están plegados (heurística basada en landmarks).
def is_left_thumb_up(
    hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    min_visibility: float = 0.5,
) -> bool:
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

    # Pulgar arriba: la punta está por encima de la muñeca y por encima del IP.
    if not (thumb_tip_y + 0.03 < wrist_y and thumb_tip_y + 0.01 < thumb_ip_y):
        return False

    # Otros dedos plegados: la punta está por debajo de su PIP.
    idx_cond = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y + 0.01
    mid_cond = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y + 0.01
    ring_cond = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y + 0.01
    pinky_cond = lm[mp.solutions.hands.HandLandmark.PINKY_TIP].y > lm[mp.solutions.hands.HandLandmark.PINKY_PIP].y + 0.01

    if idx_cond and mid_cond and ring_cond and pinky_cond:
        return True
    return False


 # Detecta si el índice está levantado (punta por encima de su PIP) y los demás dedos están plegados.
def is_index_finger_up(
    hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    min_visibility: float = 0.5,
) -> bool:
    if hand_landmarks is None:
        return False
    lm = hand_landmarks.landmark
    idx_up = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y - 0.02
    mid_fold = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y + 0.01
    ring_fold = lm[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y > lm[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y + 0.01
    pinky_fold = lm[mp.solutions.hands.HandLandmark.PINKY_TIP].y > lm[mp.solutions.hands.HandLandmark.PINKY_PIP].y + 0.01
    return bool(idx_up and mid_fold and ring_fold and pinky_fold)


 # Detecta si el dedo medio está levantado (punta por encima de su PIP).
def is_middle_finger_up(
    hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    min_visibility: float = 0.5,
) -> bool:
    if hand_landmarks is None:
        return False
    lm = hand_landmarks.landmark
    mid_up = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y - 0.02
    return bool(mid_up)


 # Detecta si el meñique está levantado (punta por encima de su PIP).
def is_pinky_finger_up(
    hand_landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    min_visibility: float = 0.5,
) -> bool:
    if hand_landmarks is None:
        return False
    lm = hand_landmarks.landmark
    pinky_up = lm[mp.solutions.hands.HandLandmark.PINKY_TIP].y < lm[mp.solutions.hands.HandLandmark.PINKY_PIP].y - 0.02
    return bool(pinky_up)


 # Devuelve True si ambas manos existen y ambas tienen el índice levantado.
def both_index_fingers_up(
    left_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
    right_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
) -> bool:
    if left_hand is None or right_hand is None:
        return False
    return is_index_finger_up(left_hand) and is_index_finger_up(right_hand)


 # Detecta si la mano está en puño (todas las puntas por debajo de sus PIP por un umbral).
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
    initiator_hand: Optional[str] = None  # Which hand moves the rectangle
    candidate: Optional[Tuple[int, int]] = None
    prev_middle_up_left: bool = False
    prev_middle_up_right: bool = False
    prev_pinky_up_left: bool = False
    prev_pinky_up_right: bool = False
    _confirm_counter: int = 0
    _confirm_hold_frames: int = 2
    _debug_last_event: Optional[str] = None

    # Reinicia el estado del flujo de selección de región (ancla, candidato, mano iniciadora y contadores).
    def reset(self) -> None:
        self.anchor = None
        self.initiator_hand = None
        self.candidate = None
        self.prev_middle_up_left = False
        self.prev_middle_up_right = False
        self.prev_pinky_up_left = False
        self.prev_pinky_up_right = False
        self._confirm_counter = 0
        self._debug_last_event = None

    # Actualiza el estado de selección y detecta confirmación con dedos extra para escoger el efecto.
    def update(
        self,
        left_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
        right_hand: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList],
        pointer_surface_point: Optional[Tuple[int, int]],
        pointer_surface_left: Optional[Tuple[int, int]],
        pointer_surface_right: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[Tuple[int, int, int, int], str]]:
        # Inicializa la selección: cuando hay dos punteros (izq/der), fija ancla y candidato.
        if self.anchor is None and pointer_surface_left is not None and pointer_surface_right is not None:
            # Convención: izquierda ancla, derecha dimensiona; se actualizan ambas con el tiempo.
            self.anchor = (int(pointer_surface_left[0]), int(pointer_surface_left[1]))
            self.candidate = (int(pointer_surface_right[0]), int(pointer_surface_right[1]))
            self.initiator_hand = "right"  # right hand moves the rectangle
            msg = f"[REGION] selection started: anchor={self.anchor} (left), moving=right"
            if self._debug_last_event != msg:
                print(msg)
                self._debug_last_event = msg
        
        # Actualiza posiciones: ancla sigue la izquierda; candidato sigue la derecha.
        if self.anchor is not None:
            if pointer_surface_left is not None:
                self.anchor = (int(pointer_surface_left[0]), int(pointer_surface_left[1]))
            if pointer_surface_right is not None:
                self.candidate = (int(pointer_surface_right[0]), int(pointer_surface_right[1]))

        # Detecta flancos de subida de dedos adicionales para elegir el tipo de efecto.
        # Nota: durante selección ya están los índices arriba, así que se miran dedos extra.
        mid_up_left = is_middle_finger_up(left_hand)
        mid_up_right = is_middle_finger_up(right_hand)
        pinky_up_left = is_pinky_finger_up(left_hand)
        pinky_up_right = is_pinky_finger_up(right_hand)

        effect_type = None
        if self.anchor is not None and self.candidate is not None:
            # Medio derecho → colorear texto.
            if mid_up_right and not self.prev_middle_up_right:
                effect_type = "colorize"
            # Medio izquierdo → subrayar.
            elif mid_up_left and not self.prev_middle_up_left:
                effect_type = "underline"
            # Meñique izquierdo → aura.
            elif pinky_up_left and not self.prev_pinky_up_left:
                effect_type = "aura"

        # Actualiza estados previos para detección por flanco.
        self.prev_middle_up_left = mid_up_left
        self.prev_middle_up_right = mid_up_right
        self.prev_pinky_up_left = pinky_up_left
        self.prev_pinky_up_right = pinky_up_right

        if effect_type and self.anchor is not None and self.candidate is not None:
            p0 = self.anchor
            p1 = self.candidate
            lx, rx = sorted((int(p0[0]), int(p1[0])))
            ty, by = sorted((int(p0[1]), int(p1[1])))
            w = max(1, rx - lx)
            h = max(1, by - ty)
            rect = (lx, ty, w, h)
            msg = f"[REGION] confirm with effect '{effect_type}' -> rect={rect}"
            if self._debug_last_event != msg:
                print(msg)
                self._debug_last_event = msg
            self.reset()
            return (rect, effect_type)
        return None

    # Devuelve (ancla, candidato) actuales para previsualizar el rectángulo de selección.
    def preview(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        return self.anchor, self.candidate


