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


# Bloque de dos convoluciones ReLU para el U-Net de refinamiento.
class DoubleConv(nn.Module):
    # Inicializa el bloque de dos convs 3x3 con ReLU.
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    # Aplica las dos convoluciones consecutivas.
    def forward(self, x):
        return self.net(x)


# U-Net liviano para refinar máscaras (binary -> suavizada con anti-aliasing).
class RefinementUNet(nn.Module):
    # Construye encoder/decoder de 3 niveles para refinar máscaras.
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

    # Ejecuta la pasada hacia adelante del U-Net de refinamiento.
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

# Estado interno del selector de región (ancla, candidato y progreso de confirmación).
@dataclass
class SelectionState:
    active: bool = False
    anchor: Optional[Tuple[int, int]] = None
    candidate: Optional[Tuple[int, int]] = None
    hold_start: Optional[float] = None
    progress: float = 0.0


class RegionSelector:
    # Selector visual que arranca con índice arriba y confirma tras mantener puntero estable.

    def __init__(self, hold_time: float = 3.0, move_threshold: float = 8.0) -> None:
        self.hold_time = float(hold_time)
        self.move_threshold = float(move_threshold)
        self._state = SelectionState()

    # Resetea el estado de selección.
    def reset(self) -> None:
        self._state = SelectionState()

    # Indica si el selector está activo.
    def is_active(self) -> bool:
        return self._state.active

    # Devuelve (ancla, candidato, progreso) para previsualización.
    def get_preview(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], float]:
        return self._state.anchor, self._state.candidate, self._state.progress

    # Intenta iniciar selección si hay puntero y el índice está arriba (o sin landmarks).
    def try_start(self, hand_landmarks, pointer: Optional[Tuple[int, int]]) -> bool:
        if pointer is None:
            return False
        # Permite iniciar selección aunque falten landmarks de la mano (gesto ya confirmado).
        if hand_landmarks is None or is_index_finger_up(hand_landmarks):
            self._state.active = True
            self._state.anchor = (int(pointer[0]), int(pointer[1]))
            self._state.candidate = (int(pointer[0]), int(pointer[1]))
            self._state.hold_start = None
            self._state.progress = 0.0
            return True
        return False

    # Actualiza candidato y confirma rectángulo tras mantener estable el puntero.
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


# Refina una máscara binaria eliminando halo y aplicando suavizado interno agresivo.
def refine_mask(
    binary_mask: np.ndarray,
    erode_iter: int = 3,
    open_kernel_size: int = 3,
    dilate_iter: int = 1,
    blur_kernel_size: int = 9,
    interior_smooth_depth: int = 6,
    edge_blur_sigma: float = 2.5,
) -> np.ndarray:
    if binary_mask.size == 0:
        return binary_mask
    
    # 1. Erosión agresiva para quitar halo grueso detectado por la red.
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_thin = cv2.erode(binary_mask, kernel_erode, iterations=erode_iter)
    
    # 2. Apertura morfológica para separar trazos pegados.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    mask_separated = cv2.morphologyEx(mask_thin, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Ligera dilatación para recuperar grosor base.
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_base = cv2.dilate(mask_separated, kernel_dilate, iterations=dilate_iter)
    
    # 4. Suavizado hacia adentro: aplica blur desde el borde hacia el interior.
    
    # Transformada de distancia para saber qué tan profundo está cada píxel.
    dist_transform = cv2.distanceTransform(mask_base, cv2.DIST_L2, 5)
    
    # Blur Gaussiano fuerte para gradientes suaves.
    blur_k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    mask_blurred = cv2.GaussianBlur(mask_base.astype(np.float32), (blur_k, blur_k), edge_blur_sigma)
    
    # Gradiente de caída desde el borde hacia adentro.
    edge_proximity = np.clip(dist_transform, 0, interior_smooth_depth) / interior_smooth_depth
    # Invertir: 1.0 en borde (máximo blur), 0.0 en interior.
    blur_weight = 1.0 - edge_proximity
    
    # Mezcla: transición suave de borde borroso a interior sólido.
    mask_base_f = mask_base.astype(np.float32)
    mask_smooth = blur_weight * mask_blurred + (1.0 - blur_weight) * mask_base_f
    
    # Filtro bilateral adicional para suavizar sin perder bordes.
    mask_smooth_uint8 = np.clip(mask_smooth, 0, 255).astype(np.uint8)
    mask_bilateral = cv2.bilateralFilter(mask_smooth_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Blur Gaussiano final suave para efecto de halo.
    mask_final = cv2.GaussianBlur(mask_bilateral.astype(np.float32), (7, 7), 1.0)
    mask_final = np.clip(mask_final, 0, 255).astype(np.uint8)
    
    # 5. Quita ruido pequeño usando componentes conectados.
    _, binary_temp = cv2.threshold(mask_final, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_temp, connectivity=8)
    
    # Conserva solo componentes grandes; copia valores originales para mantener escala de grises.
    min_component_size = 20  # pixels
    mask_cleaned = np.zeros_like(mask_final)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
            # Copy original grayscale values for this component
            mask_cleaned[labels == i] = mask_final[labels == i]
    
    return mask_cleaned


# Refina una máscara con el modelo U-Net de anti-aliasing si está disponible.
def refine_mask_neural(
    mask: np.ndarray,
    model: RefinementUNet,
    device: torch.device
) -> np.ndarray:
    if mask.size == 0:
        return mask
    
    # Prepara tensor de entrada.
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
    mask_t = mask_t.to(device)
    
    # Inferencia.
    with torch.no_grad():
        refined_t = model(mask_t)
        refined_t = torch.sigmoid(refined_t)  # Ensure [0, 1] range
    
    # Convierte de vuelta a numpy en 0-255.
    refined = refined_t.squeeze().cpu().numpy()
    refined = (refined * 255.0).clip(0, 255).astype(np.uint8)
    
    return refined


class Segmenter:
    # Ejecuta testing.py sobre un patch y aplica refinamiento (morfológico o neural).

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        refinement_checkpoint: Optional[str] = None,
        use_neural_refinement: bool = True
    ) -> None:
        if checkpoint_path is None:
            checkpoint_path = os.path.join("Reconocimiento de Caracteres", "models", "segmentation", "fine_tuning_checkpoint_epoch_10.pth")
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

    # Corre segmentación sobre un patch BGR y devuelve máscara (grayscale) y ruta guardada (opcional).
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
            # Updated testing.py saves files as *_output.png (raw network output, no overlay)
            masks = glob.glob(os.path.join(out_dir, "*_output.png"))
            print(f"[SEG] produced files: masks={len(masks)}")
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

            # ensure mask is single-channel grayscale (keep original 0-255 intensity for effects)
            if mask_raw.ndim == 3:
                try:
                    mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
                except Exception:
                    mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGRA2GRAY)
            else:
                mask_gray = mask_raw

            if mask_gray.shape[0] != patch_bgr.shape[0] or mask_gray.shape[1] != patch_bgr.shape[1]:
                # Preserve gray levels / anti-aliasing when scaling
                mask_gray = cv2.resize(mask_gray, (patch_bgr.shape[1], patch_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

            # For effects, return the raw network output (0-255). Do not threshold here.
            print(
                f"[SEG] using raw network output: shape={mask_gray.shape}, dtype={mask_gray.dtype}, non-zero pixels={np.count_nonzero(mask_gray)}"
            )
            
            # Skip neural refinement as well - use raw mask for effects
            # # Refine the mask: aggressively remove captured halo and apply inward smoothing
            # print("[SEG] refining mask with aggressive morphological operations...")
            # binary_refined = refine_mask(
            #     binary,
            #     erode_iter=3,              # More aggressive halo removal
            #     open_kernel_size=3,        # Separate merged strokes
            #     dilate_iter=1,             # Restore base thickness
            #     blur_kernel_size=9,        # Larger blur kernel for smoother anti-aliasing
            #     interior_smooth_depth=6,   # Deep inward smoothing (6 pixels from edge)
            #     edge_blur_sigma=2.5        # Strong blur sigma for soft edges
            # )
            # print(f"[SEG] morphological refinement complete: shape={binary_refined.shape}, dtype={binary_refined.dtype}")
            
            # Apply neural refinement if available
            # if self.refinement_model is not None:
            #     print("[SEG] applying neural anti-aliasing refinement...")
            #     try:
            #         binary_refined = refine_mask_neural(
            #             binary_refined,
            #             self.refinement_model,
            #             self.refinement_device
            #         )
            #         print(f"[SEG] neural refinement complete: shape={binary_refined.shape}, dtype={binary_refined.dtype}")
            #     except Exception as e:
            #         print(f"[SEG] neural refinement failed: {e}, using morphological result")

            # try to load overlay image produced by the prediction script (original patch with mask overlay)
            overlay_img = None
            # ignore overlay images from the script; return raw grayscale mask and path to produced file
            print("[SEG] run_on_patch complete")
            return (mask_gray, out_path)
