from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
from fontTools.ttLib import TTFont


# ================================================================================================
# CONFIGURACIÓN - Parámetros ajustables
# ================================================================================================

CHARSET = ( # conjunto de caracteres para generación sintética
    "0123456789"
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    + "abcdefghijklmnopqrstuvwxyz"
    + "áéíóúñÁÉÍÓÚÑüÜ"
    + ".,;:!?()[]{}}<>-+/=\\@%&'\"«»"""
    + "_"
    + "∀∃¬∧∨→⇒⇔↔"
    + "+-×÷=≈≠≤≥±∑∫√πµφθψ"
    + "αβγΑΒΓΔδΩωΛλ"
    + "āēīōūĀĒĪŌŪ‾"
    + "âêîôûÂÊÎÔÛ"
    + "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    + "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃⁿᵐᵏᵗ"
)

FONT_SIZES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 34, 48, 96] # tamaños de fuente optimizados para antialiasing (énfasis en fuentes pequeñas 4-12pt)
FONT_SIZE_WEIGHTS = [20, 30, 40, 40, 30, 15, 12, 10, 5, 4, 3, 2, 1, 1] # pesos de probabilidad para cada tamaño

IMAGE_SIZE = 256 # tamaño fijo de imagen para red neuronal (256x256)

# parámetros de generación de texto
MIN_CHARS = 2
MAX_CHARS = 12
MIN_LINES = 3
MAX_LINES = 8
MIN_CHARS_PER_LINE = 15
MAX_CHARS_PER_LINE = 35

BG_PALETTES = { # paletas de colores de fondo
    "white": [(255, 255, 255), (250, 250, 250), (245, 245, 245)],
    "gray": [(245, 245, 245), (230, 230, 230), (200, 200, 200), (180, 180, 180)],
    "beige": [(250, 245, 235), (240, 230, 210), (235, 225, 200)],
    "brown": [(200, 180, 160), (170, 140, 120), (140, 110, 90)],
    "black": [(40, 40, 40), (20, 20, 20), (0, 0, 0)],
}

SUPERSAMPLE_FACTOR = 2 # factor de sobremuestreo para preservar antialiasing


# ================================================================================================
# FUNCIONES AUXILIARES
# ================================================================================================

# get_font carga una fuente con cache para evitar recargas repetidas
def get_font(font_path: Path, size: int, cache: dict) -> ImageFont.FreeTypeFont:
    key = (str(font_path), int(size))
    if key not in cache:
        try:
            cache[key] = ImageFont.truetype(str(font_path), size=int(size))
        except Exception:
            cache[key] = ImageFont.load_default()
    return cache[key]


# font_supports_text verifica si una fuente puede renderizar todos los caracteres del texto sin fallback
def font_supports_text(font_path: Path, text: str, font_cache: dict) -> bool:
    try:
        # Check if we've already analyzed this font
        cache_key = f"_cmap_{font_path}"
        if cache_key not in font_cache:
            # Load font's character map using fontTools
            try:
                ttfont = TTFont(str(font_path))
                # Get the best cmap (character to glyph mapping)
                cmap = ttfont.getBestCmap()
                if cmap:
                    font_cache[cache_key] = set(cmap.keys())  # Set of Unicode code points
                else:
                    font_cache[cache_key] = set()
                ttfont.close()
            except:
                # If fontTools fails, assume font supports nothing special
                font_cache[cache_key] = set()
        
        supported_chars = font_cache[cache_key]
        
        # Check each character in text
        for char in set(text):
            if not char.isprintable() or char.isspace():
                continue  # Skip whitespace
            
            # Check if character's Unicode code point is in the font's cmap
            if ord(char) not in supported_chars:
                return False
        
        return True
    except Exception:
        return False


def filter_compatible_fonts_fast(text: str, char_to_fonts: dict) -> list[Path]:
    """
    Fast font filtering using pre-built character index.
    Returns fonts that support ALL printable characters in text.
    Ignores whitespace/control characters like newlines.
    """
    # Filter to only printable characters (exclude \n, \t, etc.)
    printable_chars = set(char for char in text if char.isprintable() and not char.isspace())
    
    # Get sets of fonts supporting each character
    font_sets = [set(char_to_fonts.get(char, [])) for char in printable_chars]
    
    # Intersection: fonts supporting all characters
    if not font_sets:
        return []
    
    compatible_fonts_set = font_sets[0]
    for font_set in font_sets[1:]:
        compatible_fonts_set &= font_set
    
    return list(compatible_fonts_set)


def luminance(rgb: tuple[int, int, int]) -> float:
    # Calcula la luminancia relativa (0-1).
    r, g, b = [c / 255.0 for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def random_bg_color() -> tuple[int, int, int]:
    # Elige un color de fondo aleatorio de las paletas.
    palette = random.choice(list(BG_PALETTES.values()))
    return random.choice(palette)


def random_text_color(bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
    # Elige color de texto con buen contraste frente al fondo.
    bg_lum = luminance(bg_color)
    if bg_lum > 0.5:  # Light background
        return (0, 0, 0) if random.random() < 0.9 else (50, 50, 50)
    else:  # Dark background
        return (255, 255, 255) if random.random() < 0.9 else (200, 200, 200)


def add_noise(image: Image.Image, strength: float = 0.1) -> Image.Image:
    # Añade ruido gaussiano a la imagen.
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, strength * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================================
# Text Rendering with Supersampling & Conflict Resolution
# ============================================================================

def render_text_with_antialiasing(
    text: str,
    font_path: Path,
    font_size: int,
    text_color: tuple[int, int, int],
    font_cache: dict,
    layout: str = "centered",
) -> tuple[Image.Image, Image.Image]:
    # Renderiza texto a 256x256 preservando anti-aliasing vía supermuestreo.
    canvas_w = canvas_h = IMAGE_SIZE
    
    # Render at higher resolution
    large_w = canvas_w * SUPERSAMPLE_FACTOR
    large_h = canvas_h * SUPERSAMPLE_FACTOR
    large_font_size = font_size * SUPERSAMPLE_FACTOR
    
    # Create high-res canvas
    large_img = Image.new("RGBA", (large_w, large_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(large_img)
    font = get_font(font_path, large_font_size, font_cache)
    
    # Get text bounding box to measure dimensions
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        # Fallback if textbbox fails
        text_w, text_h = draw.textsize(text, font=font)
    
    # Position text based on layout mode
    margin = 10 * SUPERSAMPLE_FACTOR
    
    if layout == "full_width":
        # Left-aligned, vertically centered
        x = margin
        y = (large_h - text_h) // 2 + random.randint(-30, 30) * SUPERSAMPLE_FACTOR
        y = max(margin, min(y, large_h - text_h - margin))
        
    elif layout == "full_height":
        # Top-aligned, horizontally centered
        x = (large_w - text_w) // 2 + random.randint(-30, 30) * SUPERSAMPLE_FACTOR
        y = margin
        x = max(margin, min(x, large_w - text_w - margin))
        
    else:  # centered
        # Center with random offset
        x = (large_w - text_w) // 2 + random.randint(-40, 40) * SUPERSAMPLE_FACTOR
        y = (large_h - text_h) // 2 + random.randint(-40, 40) * SUPERSAMPLE_FACTOR
        x = max(margin, min(x, large_w - text_w - margin))
        y = max(margin, min(y, large_h - text_h - margin))
    
    # Draw text at high resolution
    draw.text((x, y), text, font=font, fill=text_color + (255,))
    
    # Downsample with LANCZOS for smooth anti-aliasing
    char_img = large_img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
    
    # Extract alpha channel as mask (preserves anti-aliasing)
    mask_arr = np.array(char_img.split()[3], dtype=np.uint8)
    mask = Image.fromarray(mask_arr)
    
    return char_img, mask


def generate_sample(
    text: str,
    font_path: Path,
    font_size: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    font_cache: dict,
    layout: str = "centered",
) -> tuple[Image.Image, Image.Image]:
    # Genera un sample 256x256 con texto y su máscara AA.
    # Render de texto con anti-aliasing (siempre 256x256).
    char_img, mask = render_text_with_antialiasing(
        text, font_path, font_size, text_color, font_cache, layout
    )
    
    # Crea fondo 256x256.
    bg = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg_color)
    
    # Compone texto sobre el fondo.
    composed = bg.copy()
    composed.paste(char_img, mask=char_img.split()[3])
    
    # Opcional: añade ruido/desenfoque para realismo.
    if random.random() < 0.3:
        composed = add_noise(composed, strength=random.uniform(0.02, 0.12))
        composed = composed.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))
    
    return composed, mask


def generate_negative_sample(bg_color: tuple[int, int, int]) -> tuple[Image.Image, Image.Image]:
    # Genera sample negativo con bordes fuertes pero sin texto (máscara vacía).
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Escoge tipo de muestra negativa.
    sample_type = random.choice([
        "rectangles",      # Bordes rectangulares fuertes
        "circles",         # Bordes circulares
        "lines",           # Líneas diagonales/horizontales/verticales
        "gradients",       # Transiciones de gradiente fuertes
        "noise_patterns",  # Ruido aleatorio con bordes
    ])
    
    if sample_type == "rectangles":
        # Dibuja 3-8 rectángulos con bordes marcados.
        num_rects = random.randint(3, 8)
        for _ in range(num_rects):
            x1 = random.randint(0, IMAGE_SIZE - 20)
            y1 = random.randint(0, IMAGE_SIZE - 20)
            x2 = x1 + random.randint(20, 80)
            y2 = y1 + random.randint(20, 80)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
    
    elif sample_type == "circles":
        # Dibuja 3-8 círculos aleatorios.
        num_circles = random.randint(3, 8)
        for _ in range(num_circles):
            x = random.randint(20, IMAGE_SIZE - 20)
            y = random.randint(20, IMAGE_SIZE - 20)
            r = random.randint(10, 40)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=None)
    
    elif sample_type == "lines":
        # Dibuja 5-15 líneas aleatorias con grosor variable.
        num_lines = random.randint(5, 15)
        for _ in range(num_lines):
            x1 = random.randint(0, IMAGE_SIZE)
            y1 = random.randint(0, IMAGE_SIZE)
            x2 = random.randint(0, IMAGE_SIZE)
            y2 = random.randint(0, IMAGE_SIZE)
            color = tuple(random.randint(0, 255) for _ in range(3))
            width = random.randint(1, 5)
            draw.line([x1, y1, x2, y2], fill=color, width=width)
    
    elif sample_type == "gradients":
        # Create strong gradient transitions
        arr = np.array(img)
        # Random gradient direction
        if random.random() < 0.5:
            # Horizontal gradient
            gradient = np.linspace(0, 255, IMAGE_SIZE, dtype=np.uint8)
            for i in range(3):  # RGB channels
                arr[:, :, i] = gradient[np.newaxis, :]
        else:
            # Vertical gradient
            gradient = np.linspace(0, 255, IMAGE_SIZE, dtype=np.uint8)
            for i in range(3):
                arr[:, :, i] = gradient[:, np.newaxis]
        img = Image.fromarray(arr)
    
    else:  # noise_patterns
        # Random noise with strong edges
        arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        # Add some structure with blur
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    
    # Add noise for realism
    if random.random() < 0.5:
        img = add_noise(img, strength=random.uniform(0.05, 0.15))
    
    # Empty mask (no text)
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    
    return img, mask


# ============================================================================
# Text Generation
# ============================================================================

def generate_text(chars: str, single_ratio: float) -> tuple[str, str]:
    """
    Generate random text (single char, phrase, or multi-line).
    
    Returns:
        - text: Generated text string
        - layout: "centered", "full_width", or "full_height"
    """
    r = random.random()
    
    if r < single_ratio:  # Single character (5%)
        return random.choice(chars), "centered"
    
    elif r < single_ratio + 0.10:  # Short phrase (10%)
        length = random.randint(MIN_CHARS, MAX_CHARS)
        return ''.join(random.choices(chars, k=length)), "centered"
    
    else:  # Multi-line paragraph (85%)
        num_lines = random.randint(MIN_LINES, MAX_LINES)
        
        # Choose layout mode (70% full_width, 20% full_height, 10% centered)
        layout_mode = random.choices(
            ["full_width", "full_height", "centered"],
            weights=[0.70, 0.20, 0.10]
        )[0]
        
        if layout_mode == "full_width":
            # Long lines for left-to-right coverage
            lines = []
            for _ in range(num_lines):
                line_length = random.randint(MIN_CHARS_PER_LINE, MAX_CHARS_PER_LINE)
                lines.append(''.join(random.choices(chars, k=line_length)))
            return '\n'.join(lines), "full_width"
        
        elif layout_mode == "full_height":
            # More lines for top-to-bottom coverage
            num_lines = random.randint(MAX_LINES, MAX_LINES + 3)
            lines = []
            for _ in range(num_lines):
                line_length = random.randint(8, 18)
                lines.append(''.join(random.choices(chars, k=line_length)))
            return '\n'.join(lines), "full_height"
        
        else:  # centered paragraphs
            lines = []
            for _ in range(num_lines):
                line_length = random.randint(10, 25)
                lines.append(''.join(random.choices(chars, k=line_length)))
            return '\n'.join(lines), "centered"


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_dataset(
    fonts_dir: Path,
    output_dir: Path,
    num_samples: int,
    charset: str = CHARSET,
    single_ratio: float = 0.05,
    seed: int = 42,
    fine_tune_mode: bool = False,
) -> None:
    """
    Generate synthetic dataset with anti-aliasing.
    
    Args:
        fonts_dir: Directory containing .ttf/.otf font files
        output_dir: Output directory for images/masks/binary
        num_samples: Number of samples to generate
        charset: Characters to use
        single_ratio: Probability of single-character samples
        seed: Random seed for reproducibility
        fine_tune_mode: If True, generate 70% normal + 30% negative samples (Stage 2)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Setup directories
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    binary_dir = output_dir / 'binary'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect fonts
    fonts = []
    for ext in ['*.ttf', '*.otf']:
        fonts.extend(list(fonts_dir.rglob(ext)))
    
    if not fonts:
        raise FileNotFoundError(f"No fonts found in {fonts_dir}")
    
    print(f"Found {len(fonts)} fonts")
    
    # Font cache and compatibility cache
    font_cache = {}
    compat_cache = {}  # Cache for font-character compatibility checks
    
    # Pre-filter fonts: build per-character font support map
    print("Building font compatibility index...")
    char_to_fonts = {}  # char -> list of fonts that support it
    for char in tqdm(set(charset), desc="Indexing"):
        char_to_fonts[char] = []
        for font_path in fonts:
            if font_supports_text(font_path, char, font_cache):
                char_to_fonts[char].append(font_path)
    
    # Report coverage
    min_coverage = min(len(fonts_list) for fonts_list in char_to_fonts.values())
    print(f"Character coverage: {min_coverage}-{len(fonts)} fonts per character")
    
    print(f"Generating {num_samples} samples...")
    
    # Metadata
    metadata = []
    
    # Generate samples
    generated = 0
    attempts = 0
    max_attempts = num_samples * 3  # Allow retries for empty masks
    
    # Stage 2 fine-tuning: 70% normal text samples + 30% negative samples
    num_text_samples = int(num_samples * 0.7) if fine_tune_mode else num_samples
    num_negative_samples = num_samples - num_text_samples if fine_tune_mode else 0
    
    print(f"Mode: {'Fine-tune (Stage 2)' if fine_tune_mode else 'Stage 1'}")
    if fine_tune_mode:
        print(f"  - Text samples (70%): {num_text_samples}")
        print(f"  - Negative samples (30%): {num_negative_samples}")
    
    with tqdm(total=num_samples, desc="Generating") as pbar:
        while generated < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Decide sample type
            is_negative = fine_tune_mode and generated >= num_text_samples
            
            if is_negative:
                # Generate negative sample (no text, empty mask)
                bg_color = random_bg_color()
                image, mask = generate_negative_sample(bg_color)
                mask_np = np.array(mask, dtype=np.uint8)
                sample_type = "negative"
            else:
                # Generate normal text sample
                font_size = random.choices(FONT_SIZES, weights=FONT_SIZE_WEIGHTS)[0]
                text, layout = generate_text(charset, single_ratio)
                
                # Filter fonts that support all characters in text (fast lookup)
                compatible_fonts = filter_compatible_fonts_fast(text, char_to_fonts)
                
                # Skip if no fonts support this text (shouldn't happen with proper charset)
                if not compatible_fonts:
                    continue
                
                # Pick compatible font
                font_path = random.choice(compatible_fonts)
                bg_color = random_bg_color()
                text_color = random_text_color(bg_color)
                
                # Generate sample (always 256x256)
                image, mask = generate_sample(
                    text, font_path, font_size,
                    bg_color, text_color, font_cache, layout
                )
                
                # Validate: skip if mask is empty (no characters rendered)
                mask_np = np.array(mask, dtype=np.uint8)
                if mask_np.max() < 10:  # Too faint or empty
                    continue
                sample_type = "text"
            
            generated += 1
            idx = generated
            
            # Shard into subdirectories (1000 samples per shard)
            shard = (idx - 1) // 1000
            shard_name = f"{shard:04d}"
            img_shard = images_dir / shard_name
            mask_shard = masks_dir / shard_name
            bin_shard = binary_dir / shard_name
            img_shard.mkdir(exist_ok=True)
            mask_shard.mkdir(exist_ok=True)
            bin_shard.mkdir(exist_ok=True)
            
            # Save PNG files (for visualization)
            fname = f"img_{idx:06d}.png"
            image.save(img_shard / fname, compress_level=1)
            mask.save(mask_shard / fname, compress_level=1)
            
            # Save binary .pt file (for fast training)
            img_np = np.array(image, dtype=np.uint8)  # (256, 256, 3)
            
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()  # (3, 256, 256)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).contiguous()  # (1, 256, 256)
            ignore_t = torch.zeros_like(mask_t)  # No ignore mask
            
            bin_fname = f"img_{idx:06d}.pt"
            torch.save({'img': img_t, 'mask': mask_t, 'ignore': ignore_t}, 
                      bin_shard / bin_fname)
            
            # Metadata
            if is_negative:
                metadata.append({
                    "file": f"{shard_name}/{fname}",
                    "type": "negative",
                    "text": "",
                    "width": IMAGE_SIZE,
                    "height": IMAGE_SIZE,
                })
            else:
                metadata.append({
                    "file": f"{shard_name}/{fname}",
                    "type": "text",
                    "text": text,
                    "font": font_path.stem,
                    "font_size": font_size,
                    "width": IMAGE_SIZE,
                    "height": IMAGE_SIZE,
                })
            
            pbar.update(1)
    
    # Save metadata
    with open(output_dir / 'metadata.jsonl', 'w', encoding='utf-8') as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Generated {num_samples} samples")
    print(f"  - Images: {images_dir}")
    print(f"  - Masks: {masks_dir}")
    print(f"  - Binary cache: {binary_dir}")
    print(f"  - Metadata: {output_dir / 'metadata.jsonl'}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic character dataset with anti-aliasing"
    )
    parser.add_argument(
        '--fonts', type=Path, default=Path('assets/fonts/extracted'),
        help='Font directory'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('datasets/synthetic'),
        help='Output directory'
    )
    parser.add_argument(
        '--num-samples', type=int, default=50000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--single-ratio', type=float, default=0.05,
        help='Probability of single-character samples (vs multi-char/multi-line)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--fine-tune', action='store_true',
        help='Stage 2 mode: generate 70%% text + 30%% negative samples (no text, strong edges)'
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        fonts_dir=args.fonts,
        output_dir=args.output,
        num_samples=args.num_samples,
        single_ratio=args.single_ratio,
        seed=args.seed,
        fine_tune_mode=args.fine_tune,
    )


if __name__ == '__main__':
    main()
