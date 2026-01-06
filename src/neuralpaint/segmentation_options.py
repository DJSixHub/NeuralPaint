from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# Coloriza los píxeles de texto usando la máscara como intensidad/alpha (salida BGRA con fondo transparente).
def colorize_text(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    threshold: int = 5
) -> np.ndarray:
    h, w = mask.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA with alpha channel
    
    # Filtra intensidades muy bajas para evitar colorear fondo/ruido.
    mask_filtered = np.where(mask >= threshold, mask, 0).astype(np.uint8)
    
    # Aplicación directa: intensidad de color y alpha = intensidad de la máscara.
    # Esto preserva cada píxel de la máscara por encima del umbral.
    result[:, :, 0] = ((color_bgr[0] / 255.0) * mask_filtered).astype(np.uint8)  # B
    result[:, :, 1] = ((color_bgr[1] / 255.0) * mask_filtered).astype(np.uint8)  # G
    result[:, :, 2] = ((color_bgr[2] / 255.0) * mask_filtered).astype(np.uint8)  # R
    result[:, :, 3] = mask_filtered  # Alpha = mask value directly
    
    return result


# Dibuja subrayados (BGRA transparente) agrupando contornos cercanos para subrayar por línea, no por letra.
def underline_text(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    threshold: int = 127,
    line_thickness: int = 2,
    offset: int = 5
) -> np.ndarray:
    h, w = mask.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA with alpha
    
    # Crea máscara binaria de texto.
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    # Encuentra contornos de regiones de texto.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return result
    
    # Obtiene bounding boxes de cada contorno.
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    
    # Agrupa cajas en líneas según cercanía/solape vertical; ordena por coordenada y.
    boxes = sorted(boxes, key=lambda b: b[1])
    
    lines = []
    current_line = [boxes[0]]
    
    for box in boxes[1:]:
        x, y, w_box, h_box = box
        # Decide si la caja pertenece a la línea actual comparando su y con el promedio.
        prev_boxes = current_line
        avg_y = sum(b[1] for b in prev_boxes) / len(prev_boxes)
        avg_h = sum(b[3] for b in prev_boxes) / len(prev_boxes)
        
        # Si está suficientemente cerca en vertical (<= 0.5 * altura promedio), se agrega.
        if abs(y - avg_y) < avg_h * 0.5:
            current_line.append(box)
        else:
            # Empieza una nueva línea.
            lines.append(current_line)
            current_line = [box]
    
    lines.append(current_line)
    
    # Dibuja un subrayado por línea agrupada.
    for line_boxes in lines:
        # Calcula el rango horizontal y la base (y_max) de la línea completa.
        x_min = min(b[0] for b in line_boxes)
        x_max = max(b[0] + b[2] for b in line_boxes)
        y_max = max(b[1] + b[3] for b in line_boxes)
        
        # Calcula la posición del subrayado debajo de la base.
        y_underline = min(y_max + offset, h - 1)
        
        # Dibuja el subrayado directamente sobre el BGRA resultante.
        pt1 = (x_min, y_underline)
        pt2 = (x_max, y_underline)
        cv2.line(result, pt1, pt2, (*color_bgr, 255), line_thickness)
    
    return result


# Crea un halo (aura) suave alrededor del texto: dilata + desenfoca y lo deja solo hacia afuera del texto.
def add_aura(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    threshold: int = 127,
    aura_size: int = 15,
    blur_strength: int = 31,
    alpha: float = 0.5
) -> np.ndarray:
    h, w = mask.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA with alpha
    
    # Crea máscara binaria de texto.
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    # Dilata la máscara para generar la región externa del aura.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (aura_size, aura_size))
    aura_mask = cv2.dilate(binary, kernel, iterations=1)
    
    # Desenfoque fuerte para un brillo suave.
    aura_blurred = cv2.GaussianBlur(aura_mask, (blur_strength, blur_strength), 0)
    
    # Resta el texto original para que el halo quede solo hacia afuera (el texto queda legible).
    binary_blurred = cv2.GaussianBlur(binary, (11, 11), 0)
    aura_only = np.maximum(0, aura_blurred.astype(np.int16) - binary_blurred.astype(np.int16)).astype(np.uint8)
    
    # Normaliza y aplica el factor alpha base.
    aura_alpha = (aura_only.astype(np.float32) / 255.0 * alpha * 255).astype(np.uint8)
    
    # Aplica el color del aura.
    for c in range(3):
        result[:, :, c] = color_bgr[c]
    
    # El canal alpha controla la intensidad del brillo (caída suave, transparente dentro).
    result[:, :, 3] = aura_alpha
    
    return result


# Agrupa constantes de nombres de efectos visuales soportados.
class EffectType:
    NONE = "none"
    COLORIZE = "colorize"
    UNDERLINE = "underline"
    AURA = "aura"


# Aplica un efecto visual (colorize/underline/aura) según la máscara y el color, devolviendo la imagen resultante.
def apply_effect(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    effect_type: str,
    color_bgr: Tuple[int, int, int]
) -> np.ndarray:
    # Debug: estadísticas rápidas de máscara.
    mask_min = mask.min()
    mask_max = mask.max()
    mask_mean = mask.mean()
    text_pixels = (mask > 127).sum()
    print(f"[EFFECT] mask stats: min={mask_min}, max={mask_max}, mean={mask_mean:.2f}, text_pixels={text_pixels}")
    
    if effect_type == EffectType.COLORIZE:
        return colorize_text(image_bgr, mask, color_bgr)
    elif effect_type == EffectType.UNDERLINE:
        return underline_text(image_bgr, mask, color_bgr)
    elif effect_type == EffectType.AURA:
        return add_aura(image_bgr, mask, color_bgr)
    else:
        return image_bgr.copy()
