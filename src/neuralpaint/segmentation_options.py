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
    
    # Filtra cajas muy pequeñas (probablemente ruido).
    boxes = [b for b in boxes if b[2] >= 3 and b[3] >= 3]
    
    if not boxes:
        return result
    
    # Ordena cajas por coordenada y (verticalmente).
    boxes = sorted(boxes, key=lambda b: b[1])
    
    # Agrupa cajas en líneas basándose en superposición vertical.
    lines = []
    current_line = [boxes[0]]
    
    for box in boxes[1:]:
        x, y, w_box, h_box = box
        
        # Calcula el rango vertical de la línea actual.
        y_min_line = min(b[1] for b in current_line)
        y_max_line = max(b[1] + b[3] for b in current_line)
        
        # Centro y de la caja actual.
        y_center = y + h_box / 2
        
        # Si el centro de la caja actual está dentro del rango vertical de la línea (con margen),
        # o si hay superposición significativa, la agrega a la línea actual.
        overlap_threshold = 0.3  # 30% de superposición mínima
        
        # Calcula superposición vertical entre la caja y el rango de la línea.
        overlap_start = max(y, y_min_line)
        overlap_end = min(y + h_box, y_max_line)
        overlap = max(0, overlap_end - overlap_start)
        
        # Altura de referencia para calcular porcentaje de superposición.
        ref_height = (y_max_line - y_min_line + h_box) / 2
        
        if y_center >= y_min_line and y_center <= y_max_line:
            # Centro dentro del rango → claramente misma línea.
            current_line.append(box)
        elif overlap > 0 and overlap / ref_height >= overlap_threshold:
            # Superposición significativa → probablemente misma línea.
            current_line.append(box)
        else:
            # Nueva línea.
            lines.append(current_line)
            current_line = [box]
    
    lines.append(current_line)
    
    # Dibuja un subrayado por línea agrupada (solo UNA línea por grupo).
    drawn_regions = []  # Rastrea regiones ya dibujadas para evitar duplicados.
    
    for line_boxes in lines:
        # Calcula el rango horizontal y la base (y_max) de la línea completa.
        x_min = min(b[0] for b in line_boxes)
        x_max = max(b[0] + b[2] for b in line_boxes)
        y_max = max(b[1] + b[3] for b in line_boxes)
        
        # Calcula la posición del subrayado debajo de la base.
        y_underline = min(y_max + offset, h - 1)
        
        # Verifica si ya dibujamos un subrayado muy cercano en esta región.
        # Esto evita duplicados si la agrupación es ambigua.
        skip = False
        for prev_x_min, prev_x_max, prev_y in drawn_regions:
            # Si hay superposición horizontal y vertical muy cercana → skip.
            x_overlap = min(x_max, prev_x_max) - max(x_min, prev_x_min)
            if x_overlap > 0 and abs(y_underline - prev_y) <= line_thickness + 1:
                skip = True
                break
        
        if skip:
            continue
        
        # Dibuja el subrayado directamente sobre el BGRA resultante.
        pt1 = (x_min, y_underline)
        pt2 = (x_max, y_underline)
        cv2.line(result, pt1, pt2, (*color_bgr, 255), line_thickness)
        
        # Registra esta región como dibujada.
        drawn_regions.append((x_min, x_max, y_underline))
    
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
