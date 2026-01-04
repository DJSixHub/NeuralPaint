from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .gestures import is_index_finger_up


# Neural refinement network classes
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RefinementUNet(nn.Module):
    """Lightweight U-Net for mask refinement: binary mask -> smooth mask.
    
    Input:  1-channel binary/rough mask
    Output: 1-channel smooth mask with anti-aliasing
    """
    def __init__(self, base: int = 12):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(1, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)

        u2 = self.up2(e3)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.outc(d1)


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


def refine_mask(
    binary_mask: np.ndarray,
    erode_iter: int = 3,
    open_kernel_size: int = 3,
    dilate_iter: int = 1,
    blur_kernel_size: int = 9,
    interior_smooth_depth: int = 6,
    edge_blur_sigma: float = 2.5,
) -> np.ndarray:
    """Refine a binary mask by removing halo and applying aggressive inward smoothing.
    
    The network captures blur/aliasing/halo as part of characters, making them rough.
    This function:
    1. Aggressively erodes to remove the thick captured halo
    2. Applies strong inward blur/anti-aliasing from the edge into the character interior
    3. Creates smooth gradients that mimic font rendering anti-aliasing
    
    Args:
        binary_mask: Binary mask (0=background, 255=character) with thick halo from network
        erode_iter: Number of erosion iterations to remove captured halo (default 3, more aggressive)
        open_kernel_size: Kernel size for separating merged strokes (default 3)
        dilate_iter: Dilation iterations to restore thickness (default 1)
        blur_kernel_size: Gaussian blur kernel for anti-aliasing (default 9, must be odd)
        interior_smooth_depth: Pixels from edge to smooth inward (default 6, more aggressive)
        edge_blur_sigma: Gaussian blur sigma for edge smoothing (default 2.5, stronger)
    
    Returns:
        Refined mask with smooth inward gradients (grayscale 0-255, not binary)
    """
    if binary_mask.size == 0:
        return binary_mask
    
    # 1. AGGRESSIVE EROSION: Remove the thick halo captured by the network
    # Use more iterations to strip away the blur/aliasing that was classified as character
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_thin = cv2.erode(binary_mask, kernel_erode, iterations=erode_iter)
    
    # 2. Morphological opening to separate merged strokes
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    mask_separated = cv2.morphologyEx(mask_thin, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Slight dilation to restore base thickness
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_base = cv2.dilate(mask_separated, kernel_dilate, iterations=dilate_iter)
    
    # 4. AGGRESSIVE INWARD SMOOTHING: Apply blur from character edge inward
    # This recreates the natural font anti-aliasing that was wrongly included in the mask
    
    # Distance transform: distance of each white pixel from the nearest background pixel
    dist_transform = cv2.distanceTransform(mask_base, cv2.DIST_L2, 5)
    
    # Apply strong Gaussian blur to create smooth gradients
    blur_k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    mask_blurred = cv2.GaussianBlur(mask_base.astype(np.float32), (blur_k, blur_k), edge_blur_sigma)
    
    # Create gradient falloff from edge inward
    # Pixels at the edge get maximum blur, deep interior pixels stay solid
    edge_proximity = np.clip(dist_transform, 0, interior_smooth_depth) / interior_smooth_depth
    # Invert: 1.0 at edge (full blur), 0.0 deep inside (no blur)
    blur_weight = 1.0 - edge_proximity
    
    # Blend: smoothly transition from blurred edge to solid interior
    mask_base_f = mask_base.astype(np.float32)
    mask_smooth = blur_weight * mask_blurred + (1.0 - blur_weight) * mask_base_f
    
    # Apply additional bilateral filter for even smoother edges while preserving structure
    # Bilateral filter smooths while preserving edges - perfect for anti-aliasing
    mask_smooth_uint8 = np.clip(mask_smooth, 0, 255).astype(np.uint8)
    mask_bilateral = cv2.bilateralFilter(mask_smooth_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Final gentle Gaussian blur to create soft halo effect (like subpixel rendering)
    mask_final = cv2.GaussianBlur(mask_bilateral.astype(np.float32), (7, 7), 1.0)
    mask_final = np.clip(mask_final, 0, 255).astype(np.uint8)
    
    # 5. Remove tiny noise (but keep grayscale for anti-aliasing)
    # Use connected components to remove small isolated regions
    # First threshold to find components
    _, binary_temp = cv2.threshold(mask_final, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_temp, connectivity=8)
    
    # Keep only components larger than minimum size
    min_component_size = 20  # pixels
    mask_cleaned = np.zeros_like(mask_final)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
            # Copy original grayscale values for this component
            mask_cleaned[labels == i] = mask_final[labels == i]
    
    return mask_cleaned


def refine_mask_neural(
    mask: np.ndarray,
    model: RefinementUNet,
    device: torch.device
) -> np.ndarray:
    """Apply neural network refinement to add natural anti-aliasing.
    
    Args:
        mask: Input mask (grayscale, 0-255)
        model: Trained refinement network
        device: torch device (cuda/cpu)
    
    Returns:
        Refined mask with neural anti-aliasing (grayscale, 0-255)
    """
    if mask.size == 0:
        return mask
    
    # Prepare input tensor
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
    mask_t = mask_t.to(device)
    
    # Run inference
    with torch.no_grad():
        refined_t = model(mask_t)
        refined_t = torch.sigmoid(refined_t)  # Ensure [0, 1] range
    
    # Convert back to numpy
    refined = refined_t.squeeze().cpu().numpy()
    refined = (refined * 255.0).clip(0, 255).astype(np.uint8)
    
    return refined


class Segmenter:
    """Wrapper that runs `Reconocimiento de Caracteres/scripts/inference/testing.py` on a single image patch.

    Uses checkpoint at `Reconocimiento de Caracteres/models/segmentation/checkpoint_epoch_70.pth` by default.
    Also applies neural refinement network for smooth anti-aliasing.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        refinement_checkpoint: Optional[str] = None,
        use_neural_refinement: bool = True
    ) -> None:
        if checkpoint_path is None:
            checkpoint_path = os.path.join("Reconocimiento de Caracteres", "models", "segmentation", "checkpoint_epoch_70.pth")
        self.checkpoint = str(checkpoint_path)
        
        # Load refinement network if enabled
        self.refinement_model = None
        self.refinement_device = None
        if use_neural_refinement:
            try:
                if refinement_checkpoint is None:
                    refinement_checkpoint = os.path.join(
                        "Reconocimiento de Caracteres",
                        "models",
                        "refinement",
                        "best_model.pth"
                    )
                
                if os.path.exists(refinement_checkpoint):
                    self.refinement_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.refinement_model = RefinementUNet(base=12).to(self.refinement_device)
                    state_dict = torch.load(refinement_checkpoint, map_location=self.refinement_device, weights_only=True)
                    self.refinement_model.load_state_dict(state_dict)
                    self.refinement_model.eval()
                    print(f"[SEG] Loaded neural refinement model from {refinement_checkpoint}")
                else:
                    print(f"[SEG] Neural refinement checkpoint not found: {refinement_checkpoint}")
                    print("[SEG] Will use morphological refinement only")
            except Exception as e:
                print(f"[SEG] Failed to load refinement model: {e}")
                print("[SEG] Will use morphological refinement only")
                self.refinement_model = None

    def run_on_patch(self, patch_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        print(f"[SEG] run_on_patch start: patch_shape={getattr(patch_bgr, 'shape', None)}")
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "in")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            img_path = os.path.join(in_dir, "patch.png")
            # Use the exact patch as captured from the screen (no rotation/flip)
            transformed = patch_bgr
            # write BGR image (transformed) into the temp input dir
            cv2.imwrite(img_path, transformed)
            print(f"[SEG] wrote temp input: {img_path}")
            # save the exact transformed input sent to the network into masc_produced (single file)
            in_path = None
            try:
                out_dir_prod = os.path.join(os.getcwd(), "masc_produced")
                os.makedirs(out_dir_prod, exist_ok=True)
                ts_in = int(time.time() * 1000)
                in_name = f"{ts_in}_input.png"
                in_path = os.path.join(out_dir_prod, in_name)
                cv2.imwrite(in_path, transformed)
                print(f"[SEG] saved input to masc_produced: {in_path}")
            except Exception as e:
                print(f"[SEG] failed saving input to masc_produced: {e}")
                in_path = None
            script = os.path.join(os.getcwd(), "Reconocimiento de Caracteres", "scripts", "inference", "testing.py")
            cmd = [sys.executable, script, "--model", self.checkpoint, "--testing-dir", in_dir, "--out-dir", out_dir, "--stride", "0"]
            try:
                print(f"[SEG] running: {cmd}")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[SEG] subprocess failed: {e}")
                return np.zeros((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8), None
            masks = glob.glob(os.path.join(out_dir, "*_mask.png"))
            overlays = glob.glob(os.path.join(out_dir, "*_overlay.png"))
            print(f"[SEG] produced files: masks={len(masks)} overlays={len(overlays)}")
            if not masks:
                # return empty mask and no saved path
                print("[SEG] no mask found in out_dir")
                return (np.zeros((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8), None)
            # load produced mask as-is (preserve image appearance) and also produce a binary mask for internal use
            mask_raw = cv2.imread(masks[0], cv2.IMREAD_UNCHANGED)
            if mask_raw is None:
                print(f"[SEG] failed to read mask: {masks[0]}")
                return (np.zeros((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8), None)
            # save the produced mask image "tal cual" into masc_produced (single file)
            out_path = None
            try:
                out_dir_prod = os.path.join(os.getcwd(), "masc_produced")
                os.makedirs(out_dir_prod, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(masks[0]))[0]
                ts = int(time.time() * 1000)
                out_name = f"{ts}_{base_name}_mask_produced.png"
                out_path = os.path.join(out_dir_prod, out_name)
                # write mask_raw preserving channels exactly as produced by the model
                cv2.imwrite(out_path, mask_raw)
                print(f"[SEG] saved produced mask to masc_produced: {out_path}")
            except Exception as e:
                print(f"[SEG] failed saving produced mask: {e}")
                out_path = None

            # ensure mask is single-channel grayscale for binary thresholding
            if mask_raw.ndim == 3:
                try:
                    mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
                except Exception:
                    mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGRA2GRAY)
            else:
                mask_gray = mask_raw

            if mask_gray.shape[0] != patch_bgr.shape[0] or mask_gray.shape[1] != patch_bgr.shape[1]:
                mask_gray = cv2.resize(mask_gray, (patch_bgr.shape[1], patch_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
            
            # Refine the mask: aggressively remove captured halo and apply inward smoothing
            print("[SEG] refining mask with aggressive morphological operations...")
            binary_refined = refine_mask(
                binary,
                erode_iter=3,              # More aggressive halo removal
                open_kernel_size=3,        # Separate merged strokes
                dilate_iter=1,             # Restore base thickness
                blur_kernel_size=9,        # Larger blur kernel for smoother anti-aliasing
                interior_smooth_depth=6,   # Deep inward smoothing (6 pixels from edge)
                edge_blur_sigma=2.5        # Strong blur sigma for soft edges
            )
            print(f"[SEG] morphological refinement complete: shape={binary_refined.shape}, dtype={binary_refined.dtype}")
            
            # Apply neural refinement if available
            if self.refinement_model is not None:
                print("[SEG] applying neural anti-aliasing refinement...")
                try:
                    binary_refined = refine_mask_neural(
                        binary_refined,
                        self.refinement_model,
                        self.refinement_device
                    )
                    print(f"[SEG] neural refinement complete: shape={binary_refined.shape}, dtype={binary_refined.dtype}")
                except Exception as e:
                    print(f"[SEG] neural refinement failed: {e}, using morphological result")

            # try to load overlay image produced by the prediction script (original patch with mask overlay)
            overlay_img = None
            # ignore overlay images from the script; return refined binary and path to produced file
            print("[SEG] run_on_patch complete")
            return (binary_refined, out_path)
