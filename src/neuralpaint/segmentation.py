from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

from .gestures import is_index_finger_up


@dataclass
class SelectionState:
    active: bool = False
    anchor: Optional[Tuple[int, int]] = None
    candidate: Optional[Tuple[int, int]] = None
    hold_start: Optional[float] = None
    progress: float = 0.0


class RegionSelector:
    """Selector visual que usa el índice hacia arriba para entrar en modo selección.

    - `start_on_gesture`: when True, caller should call `try_start(hand_landmarks, pointer)`
      frequently; selector will activate when `is_index_finger_up` is True.
    - `update(pointer)` updates candidate and returns rect (x,y,w,h) when committed.
    """

    def __init__(self, hold_time: float = 3.0, move_threshold: float = 8.0) -> None:
        self.hold_time = float(hold_time)
        self.move_threshold = float(move_threshold)
        self._state = SelectionState()

    def reset(self) -> None:
        self._state = SelectionState()

    def is_active(self) -> bool:
        return self._state.active

    def get_preview(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], float]:
        return self._state.anchor, self._state.candidate, self._state.progress

    def try_start(self, hand_landmarks, pointer: Optional[Tuple[int, int]]) -> bool:
        if pointer is None:
            return False
        # allow starting selection even if we don't have hand landmarks (gesture hand already triggered)
        if hand_landmarks is None or is_index_finger_up(hand_landmarks):
            self._state.active = True
            self._state.anchor = (int(pointer[0]), int(pointer[1]))
            self._state.candidate = (int(pointer[0]), int(pointer[1]))
            self._state.hold_start = None
            self._state.progress = 0.0
            return True
        return False

    def update(self, pointer: Optional[Tuple[int, int]], now: Optional[float] = None) -> Optional[Tuple[int, int, int, int]]:
        if not self._state.active:
            return None
        now = now if now is not None else time.perf_counter()
        if pointer is None:
            self._state.hold_start = None
            self._state.progress = 0.0
            self._state.candidate = None
            return None

        px, py = int(pointer[0]), int(pointer[1])
        if self._state.anchor is None:
            self._state.anchor = (px, py)
        old_candidate = self._state.candidate
        # if pointer moved significantly, restart hold
        if self._state.hold_start is None:
            self._state.candidate = (px, py)
            self._state.hold_start = now
            self._state.progress = 0.0
            return None

        # compute movement relative to previous candidate
        if old_candidate is None:
            dist = 0.0
        else:
            dist = np.hypot(px - (old_candidate[0]), py - (old_candidate[1]))
        self._state.candidate = (px, py)
        if dist > self.move_threshold:
            self._state.hold_start = now
            self._state.progress = 0.0
            return None

        elapsed = now - (self._state.hold_start or now)
        self._state.progress = min(1.0, elapsed / self.hold_time)
        if elapsed >= self.hold_time:
            x0, y0 = self._state.anchor
            x1, y1 = self._state.candidate
            lx, rx = sorted((int(x0), int(x1)))
            ty, by = sorted((int(y0), int(y1)))
            w = max(1, rx - lx)
            h = max(1, by - ty)
            rect = (lx, ty, w, h)
            self.reset()
            return rect
        return None


class Segmenter:
    """Wrapper that runs `Reconocimiento de Caracteres/testing.py` on a single image patch.

    Uses checkpoint at `models/checkpoint_epoch_70.pth` by default (relative to repo root).
    """

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path is None:
            checkpoint_path = os.path.join("models", "checkpoint_epoch_70.pth")
        self.checkpoint = str(checkpoint_path)

    def run_on_patch(self, patch_bgr: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return np.zeros((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "in")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            img_path = os.path.join(in_dir, "patch.png")
            # write BGR image
            cv2.imwrite(img_path, patch_bgr)
            script = os.path.join(os.getcwd(), "Reconocimiento de Caracteres", "testing.py")
            cmd = [sys.executable, script, "--model", self.checkpoint, "--testing-dir", in_dir, "--out-dir", out_dir, "--stride", "0"]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                return np.zeros((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8)
            masks = glob.glob(os.path.join(out_dir, "*_mask.png"))
            if not masks:
                return np.zeros((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8)
            mask = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return np.zeros((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8)
            if mask.shape[0] != patch_bgr.shape[0] or mask.shape[1] != patch_bgr.shape[1]:
                mask = cv2.resize(mask, (patch_bgr.shape[1], patch_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return binary
