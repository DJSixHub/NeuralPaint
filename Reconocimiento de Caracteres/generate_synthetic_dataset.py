"""Generador sintético simple de caracteres.
Produce pares de `image` y `mask` (PNG) usando las fuentes bajo `assets/fonts/extracted`.
Uso básico:
    python generate_synthetic_dataset.py --fonts "assets/fonts/extracted" --out "datasets/synthetic" --samples 3 --size 128

Requisitos: Pillow, numpy
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops


CHARSET = (
    "0123456789"
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    + "abcdefghijklmnopqrstuvwxyz"
    + "áéíóúñÁÉÍÓÚÑüÜ"
    + ".,;:!?()[]{}<>-+/=\\@%&'\"«»“”"
    + "_"
    + "∀∃¬∧∨→⇒⇔↔"
    + "+-×÷=≈≠≤≥±∑∫√πµφθψ"
    + "αβγΑΒΓ"
    + "ΔδΩωΛλ"
    + "āēīōūĀĒĪŌŪ"
    + "‾"
    + "âêîôûÂÊÎÔÛ"
    + "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    + "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃⁿᵐᵏᵗ"
)

# Tamaños de fuente (píxeles) y una distribución no uniforme
SIZES = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
# Probabilidades relativas para los tamaños anteriores (suman implícitamente a 1 tras normalizar)
SIZE_PROBS = [11, 11, 16, 13, 11, 9, 8, 7, 5, 4]
# Porcentaje de ejemplos fuera de la distribución (tamaños grandes/raros)
OOD_RATIO = 0.05
OOD_SIZES = [48, 64, 96]

# We render directly at the target size (no DPR/sampling/downsample pipeline)

# multi-character settings (por defecto mezcla single/multi)
MIN_CHARS = 2
MAX_CHARS = 12

# Paletas ampliadas para fondos (blancos, grises, beige, marrones, negros)
BG_PALETTES = {
    "white": [(255, 255, 255), (250, 250, 250), (245, 245, 245)],
    "gray": [(245, 245, 245), (230, 230, 230), (200, 200, 200), (180, 180, 180)],
    "beige": [(250, 245, 235), (240, 230, 210), (235, 225, 200)],
    "brown": [(200, 180, 160), (170, 140, 120), (140, 110, 90)],
    "black": [(40, 40, 40), (20, 20, 20), (0, 0, 0)],
}


def random_solid_color() -> tuple[int, int, int]:
    group = random.choice(list(BG_PALETTES.values()))
    return random.choice(group)


def lum(rgb: tuple[int, int, int]) -> float:
    r, g, b = [c / 255.0 for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ok(c1: tuple[int, int, int], c2: tuple[int, int, int], min_delta: float = 0.35) -> bool:
    return abs(lum(c1) - lum(c2)) >= min_delta


def random_text_color(bg_color: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    # choose varied saturated colors and ensure contrast with background if given
    candidates = [
        (0, 0, 0),
        (255, 255, 255),
        (200, 30, 30),
        (30, 80, 160),
        (10, 120, 80),
        (120, 10, 120),
        (200, 120, 30),
    ]
    if bg_color is None:
        return random.choice(candidates)
    for _ in range(10):
        c = random.choice(candidates)
        if contrast_ok(c, bg_color):
            return c
    # fallback b/w
    return (0, 0, 0) if lum(bg_color) > 0.5 else (255, 255, 255)


def random_gradient(size: int) -> Image.Image:
    w = h = size
    base = Image.new("RGB", (w, h))
    top = tuple(random.randint(200, 255) for _ in range(3))
    bottom = tuple(random.randint(180, 240) for _ in range(3))
    for y in range(h):
        t = y / max(h - 1, 1)
        row = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        for x in range(w):
            base.putpixel((x, y), row)
    return base


def random_image_background(size: int, bg_dir: Path) -> Image.Image:
    imgs = [p for p in bg_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if not imgs:
        return Image.new("RGB", (size, size), random_solid_color())
    chosen = random.choice(imgs)
    try:
        im = Image.open(chosen).convert("RGB")
        im = im.resize((size, size), Image.LANCZOS)
        return im
    except Exception:
        return Image.new("RGB", (size, size), random_solid_color())


def find_font_files(fonts_root: Path) -> List[Path]:
    exts = (".ttf", ".otf")
    files = [p for p in fonts_root.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)






def render_line_simple(font_path: Path, text: str, base_font_size: int, image_size: int, fill: tuple[int, int, int], x_offset: int = 0) -> tuple[Image.Image, Image.Image, list]:
    # Render a sequence of characters at target size, support newline for multiline.
    # Return composed image, composite mask, and per-instance masks
    try:
        font = ImageFont.truetype(str(font_path), size=int(base_font_size))
    except Exception:
        font = ImageFont.load_default()

    img = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    lines = text.split("\n")
    line_heights = []
    line_widths = []
    for line in lines:
        # measure line
        w = 0
        h = 0
        for ch in line:
            bb = draw.textbbox((0, 0), ch, font=font)
            w += (bb[2] - bb[0]) + int(base_font_size * 0.05)
            h = max(h, bb[3] - bb[1])
        if w > 0:
            w -= int(base_font_size * 0.05)
        line_widths.append(w)
        line_heights.append(h)

    total_h = sum(line_heights) + int((len(lines) - 1) * (base_font_size * 0.2))
    y0 = (image_size - total_h) // 2
    instance_masks = []
    y = y0
    for li, line in enumerate(lines):
        w = line_widths[li]
        x = max(0, (image_size - w) // 2) + x_offset
        for ch in line:
            jitter = random.randint(-int(base_font_size * 0.02), int(base_font_size * 0.02))
            xi = int(x + jitter)
            draw.text((xi, y), ch, fill=(*fill, 255), font=font)
            m = Image.new("L", (image_size, image_size), 0)
            md = ImageDraw.Draw(m)
            md.text((xi, y), ch, fill=255, font=font)
            m = m.point(lambda p: 255 if p > 0 else 0).convert("L")
            instance_masks.append(m)
            bb = draw.textbbox((0, 0), ch, font=font)
            x += (bb[2] - bb[0]) + int(base_font_size * 0.05)
        y += line_heights[li] + int(base_font_size * 0.2)

    mask = Image.new("L", (image_size, image_size), 0)
    for m in instance_masks:
        mask = ImageChops.add(mask, m)
    mask = mask.point(lambda p: 255 if p > 0 else 0).convert("L")
    instances = []
    for idx, m in enumerate(instance_masks):
        bbox = m.getbbox()
        instances.append({"mask": m, "bbox": bbox, "char": '?'})

    return img, mask, instances


def compose_and_save_sample(
    font_path: Path,
    text: str,
    font_size: int,
    is_partial: bool,
    render_size: int,
    images_dir: Path,
    masks_dir: Path,
    masks_ignore_dir: Path,
    bg_mode: str,
    bg_dir: Path | None,
    idx: int,
    force_non_partial: bool = False,
) -> dict:
    """Compose a sample (render, build masks, augment, save) and return the metadata record.

    This consolidates the duplicated generation logic used for required_specs and the main loop.
    """
    font_name = font_path.stem

    # background selection
    if bg_mode == "solid":
        bg_color = random_solid_color()
        bg = Image.new("RGB", (render_size, render_size), bg_color)
        bg_info = {"bg_type": "solid"}
    elif bg_mode == "gradient":
        bg = random_gradient(render_size)
        bg_color = tuple(bg.getpixel((render_size // 2, render_size // 2)))
        bg_info = {"bg_type": "gradient"}
    elif bg_mode == "image" and bg_dir is not None:
        bg = random_image_background(render_size, bg_dir)
        bg_color = tuple(bg.getpixel((render_size // 2, render_size // 2)))
        bg_info = {"bg_type": "image"}
    else:
        bg = Image.new("RGB", (render_size, render_size), (255, 255, 255))
        bg_color = (255, 255, 255)
        bg_info = {"bg_type": "white"}

    text_color = random_text_color(bg_color)

    attempts = 0
    max_attempts = 12
    while True:
        x_off = 0
        if is_partial:
            x_off = random.randint(-render_size // 2, render_size // 2)
        img_comp, comp_mask, instances = render_line_simple(font_path, text, font_size, render_size, text_color, x_offset=x_off)

        main_mask = Image.new("L", (render_size, render_size), 0)
        mask_ignore = Image.new("L", (render_size, render_size), 0)
        partial_found = False
        for inst in instances:
            im = inst["mask"] if isinstance(inst, dict) else inst
            bbox = im.getbbox()
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            touches_border = (x0 <= 0 or y0 <= 0 or x1 >= render_size or y1 >= render_size)
            if touches_border:
                partial_found = True
                mask_ignore = ImageChops.lighter(mask_ignore, im)
            else:
                main_mask = ImageChops.add(main_mask, im)

        # Decision logic: if force_non_partial, require no instance touching borders
        if force_non_partial:
            if not partial_found or attempts >= max_attempts:
                char_rgba = img_comp
                mask = main_mask.point(lambda p: 255 if p > 0 else 0).convert("L")
                mask_ignore = mask_ignore.point(lambda p: 255 if p > 0 else 0).convert("L")
                break
            attempts += 1
            continue

        # Default behavior: if not requested partial, accept; if requested partial ensure at least one partial instance
        if not is_partial or partial_found or attempts >= max_attempts:
            char_rgba = img_comp
            mask = main_mask.point(lambda p: 255 if p > 0 else 0).convert("L")
            mask_ignore = mask_ignore.point(lambda p: 255 if p > 0 else 0).convert("L")
            break
        attempts += 1

    composed = bg.copy()
    composed.paste(char_rgba, mask=char_rgba.split()[3])

    # synthetic occlusion / complex overlay sometimes
    if random.random() < 0.12:
        occ = Image.new("RGBA", (render_size, render_size), (0, 0, 0, 0))
        od = ImageDraw.Draw(occ)
        for _ in range(random.randint(1, 3)):
            x0 = random.randint(0, render_size - 1)
            y0 = random.randint(0, render_size - 1)
            x1 = random.randint(x0, render_size)
            y1 = random.randint(y0, render_size)
            od.ellipse((x0, y0, x1, y1), fill=(0, 0, 0, random.randint(30, 110)))
        composed = Image.alpha_composite(composed.convert("RGBA"), occ).convert("RGB")

    if random.random() < 0.3:
        composed = add_noise_background(composed, strength=random.uniform(0.02, 0.12))
        composed = composed.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))

    fname = f"img_{idx:06d}.png"
    composed.save(images_dir / fname)
    mask.save(masks_dir / fname)
    mask_ignore.save(masks_ignore_dir / fname)

    rec = {
        "file": fname,
        "text": text,
        "font": font_name,
        "font_path": str(font_path),
        "font_size": font_size,
        "render_size": int(render_size),
        "num_chars": len(text),
        "partial": bool(mask_ignore.getbbox()),
    }
    rec.update(bg_info)
    return rec


def add_noise_background(image: Image.Image, strength: float = 0.2) -> Image.Image:
    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.randn(*arr.shape) * strength
    arr = np.clip(arr + noise, 0.0, 1.0)
    out = Image.fromarray((arr * 255).astype(np.uint8))
    return out


def generate(
    fonts_dir: Path,
    out_dir: Path,
    samples_per_pair: int = 3,
    image_size: int = 128,
    chars: str = CHARSET,
    seed: int | None = None,
    bg_mode: str = "solid",
    bg_dir: Path | None = None,
    max_images: int | None = None,
    single_ratio: float = 0.4,
    partial_ratio: float = 0.1,
    size_dist: tuple[float, float, float] = (0.12, 0.55, 0.33),
    size_choices: list[int] | None = None,
    multi_subdist: tuple[float, float, float] = (0.7, 0.25, 0.05),
) -> None:
    random.seed(seed)
    np.random.seed(seed if seed is not None else 0)
    fonts = find_font_files(fonts_dir)
    if not fonts:
        raise SystemExit(f"No se encontraron fuentes en {fonts_dir}")

    out_dir = out_dir.expanduser().resolve()
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    masks_ignore_dir = out_dir / "masks_ignore"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    masks_ignore_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    counter = 0

    # prepare size buckets
    small_sizes = [s for s in SIZES if s <= 12]
    medium_sizes = [s for s in SIZES if 14 <= s <= 32]
    large_sizes = OOD_SIZES[:] if OOD_SIZES else [48, 64, 96]

    # normalize size_dist
    sd = list(size_dist)
    ssum = sum(sd)
    if ssum <= 0:
        sd = [0.15, 0.6, 0.25]
        ssum = sum(sd)
    sd = [v / ssum for v in sd]

    if size_choices is None:
        # default choices and weights per user preference: 50% 512, 30% 256, 5% 720, 5% 128
        size_choices = [512, 256, 720, 128]
        size_weights = [0.5, 0.3, 0.05, 0.05]
    else:
        size_weights = None

    # normalize multi_subdist
    msd = list(multi_subdist)
    msum = sum(msd)
    if msum <= 0:
        msd = [0.7, 0.25, 0.05]
        msum = sum(msd)
    msd = [v / msum for v in msd]

    # helper to pick font size according to size_dist or forced bucket
    def pick_size(bucket: str | None = None) -> int:
        if bucket == "small":
            return int(random.choice(small_sizes)) if small_sizes else int(random.choice(SIZES))
        if bucket == "medium":
            return int(random.choice(medium_sizes)) if medium_sizes else int(random.choice(SIZES))
        if bucket == "large":
            return int(random.choice(large_sizes)) if large_sizes else int(random.choice(OOD_SIZES))
        # sample by distribution
        r = random.random()
        if r < sd[0]:
            return int(random.choice(small_sizes)) if small_sizes else int(random.choice(SIZES))
        if r < sd[0] + sd[1]:
            return int(random.choice(medium_sizes)) if medium_sizes else int(random.choice(SIZES))
        return int(random.choice(large_sizes)) if large_sizes else int(random.choice(OOD_SIZES))

    def choose_font_size_for_render(render_size: int) -> int:
        """Choose a font size biased by output render resolution.

        - For 512x512: favor sizes commonly used (10,11,12,14,16,18,20,22,24) with emphasis on 12-16.
        - For 256x256: favor 12,16,18,20.
        - For others: fall back to the usual distribution with some OOD probability.
        """
        if render_size == 512:
            cand = [10, 11, 12, 14, 16, 18, 20, 22, 24]
            # slightly increase probability for 12 and 14
            weights = [5, 4, 28, 30, 20, 8, 5, 2, 1]
            return int(min(random.choices(cand, weights=weights, k=1)[0], render_size))
        if render_size == 256:
            cand = [12, 16, 18, 20]
            weights = [10, 35, 35, 20]
            return int(min(random.choices(cand, weights=weights, k=1)[0], render_size))
        # default: keep previous OOD logic
        if random.random() < OOD_RATIO:
            return int(random.choice(OOD_SIZES))
        return int(random.choices(SIZES, weights=SIZE_PROBS, k=1)[0])

    # if max_images large, prepare required coverage list
    required_specs: list[dict] = []
    if max_images is not None and max_images > 200:
        # Base guarantees per character: single non-partial, multi (single-line or multiline), small/medium/large bucket, partial
        for ch in chars:
            required_specs.append({"type": "single_nonpartial", "char": ch})
            required_specs.append({"type": "multiline", "char": ch})
            required_specs.append({"type": "small", "char": ch})
            required_specs.append({"type": "medium", "char": ch})
            required_specs.append({"type": "large", "char": ch})
            required_specs.append({"type": "partial", "char": ch})

        # Additional stronger guarantees when user requests larger corpus
        if max_images is not None and max_images > 500:
            # ensure each character appears at least once in each render resolution and each font size
            for ch in chars:
                for rs in size_choices:
                    required_specs.append({"type": "per_resolution", "char": ch, "render_size": int(rs)})
                for fs in SIZES + OOD_SIZES:
                    required_specs.append({"type": "per_fontsize", "char": ch, "font_size": int(fs)})

        # sanity check: if we exceed budget, abort with clear message
        if len(required_specs) > (max_images if max_images is not None else 0):
            raise SystemExit(
                f"Requested max_images={max_images} is too small to guarantee required coverage ({len(required_specs)} specs). "
                "Increase --max-images or reduce charset/resolution/font-size requirements."
            )

    # generate required coverage samples first (if any)
    if required_specs:
        random.shuffle(required_specs)
        for spec in required_specs:
            if counter >= max_images:
                break
            ch = spec["char"]
            typ = spec["type"]
            font_path = random.choice(fonts)
            font_name = font_path.stem

            # defaults
            render_size_for_spec = spec.get("render_size", image_size)
            font_size_for_spec = spec.get("font_size", None)
            is_partial = False
            force_non_partial = False

            # build text and font size depending on spec type
            if typ == "single_nonpartial":
                text = ch
                font_size = pick_size(None) if font_size_for_spec is None else int(font_size_for_spec)
                is_partial = False
                force_non_partial = True
            elif typ == "multiline":
                # create paragraph-like multiline text that includes the char
                # bias long lines for small font sizes
                if font_size_for_spec is None:
                    font_size = pick_size(None)
                else:
                    font_size = int(font_size_for_spec)
                # choose lines between 2 and 7
                lines = random.randint(2, 7)
                if font_size in (8, 10, 12, 14):
                    # paragraph-like: more chars per line
                    total_chars = random.randint(40, 120)
                    lines = min(lines, 6)
                else:
                    total_chars = random.randint(12, 48)
                # ensure the special char appears at least once
                chars_list = [random.choice(chars) for _ in range(total_chars - 1)]
                insert_at = random.randint(0, len(chars_list))
                chars_list.insert(insert_at, ch)
                # split into lines
                parts = []
                remaining = len(chars_list)
                for i in range(lines - 1):
                    take = max(1, remaining // (lines - i))
                    parts.append(''.join(chars_list[:take]))
                    chars_list = chars_list[take:]
                    remaining -= take
                parts.append(''.join(chars_list))
                text = '\n'.join(parts)
                is_partial = False
                force_non_partial = True
            elif typ in ("small", "medium", "large"):
                text = ch
                font_size = pick_size(typ) if font_size_for_spec is None else int(font_size_for_spec)
                is_partial = False
                force_non_partial = True
            elif typ == "partial":
                text = ch
                font_size = pick_size(None) if font_size_for_spec is None else int(font_size_for_spec)
                is_partial = True
                force_non_partial = False
            elif typ == "per_resolution":
                rs = int(spec.get("render_size", image_size))
                render_size_for_spec = rs
                # choose font size suitable for this render size
                font_size = choose_font_size_for_render(rs) if font_size_for_spec is None else int(font_size_for_spec)
                is_partial = False
                force_non_partial = True
            elif typ == "per_fontsize":
                fs = int(spec.get("font_size", SIZES[0]))
                font_size = fs
                # pick a render_size that can accommodate this font size (prefer larger choices)
                suitable = sorted([int(x) for x in size_choices if int(x) >= font_size])
                render_size_for_spec = suitable[-1] if suitable else image_size
                is_partial = False
                force_non_partial = True
            else:
                text = ch
                font_size = pick_size(None)
                is_partial = False

            # render sample using render_size_for_spec and force_non_partial
            rec = compose_and_save_sample(
                font_path,
                text,
                font_size,
                is_partial,
                render_size_for_spec,
                images_dir,
                masks_dir,
                masks_ignore_dir,
                bg_mode,
                bg_dir,
                counter + 1,
                force_non_partial=force_non_partial,
            )
            meta.append(rec)
            counter += 1

    for font_path in fonts:
        font_name = font_path.stem
        for _ch in chars:
            for s in range(samples_per_pair):
                counter += 1
                # pick output resolution for this sample
                # choose curr_size with weights if provided, else uniform
                if size_weights is not None:
                    # normalize weights
                    total_w = sum(size_weights)
                    if total_w <= 0:
                        probs = None
                    else:
                        probs = [w / total_w for w in size_weights]
                    curr_size = random.choices(size_choices, weights=probs, k=1)[0]
                else:
                    curr_size = random.choice(size_choices)

                # seleccionar un tamaño de fuente según la resolución de render (sesgos para 512/256)
                font_size = choose_font_size_for_render(curr_size)
                # garantizar que el tamaño no exceda la imagen
                font_size = min(font_size, curr_size)

                # decide single vs multi according to single_ratio
                if random.random() < single_ratio:
                    text = _ch
                else:
                    n = random.randint(MIN_CHARS, min(MAX_CHARS, len(chars)))
                    # decide multi vs multiline (50% of multi are multiline)
                    if random.random() < 0.5:
                        # multiline: choose number of lines 2..min(7,n)
                        lines = random.randint(2, min(7, n))
                        # ensure at least one char per line
                        # generate n chars
                        chars_list = [random.choice(chars) for _ in range(n)]
                        parts = []
                        remaining = n
                        for i in range(lines - 1):
                            take = random.randint(1, remaining - (lines - i - 1))
                            parts.append(''.join(chars_list[:take]))
                            chars_list = chars_list[take:]
                            remaining -= take
                        parts.append(''.join(chars_list))
                        text = '\n'.join(parts)
                    else:
                        text = ''.join(random.choice(chars) for _ in range(n))

                # create background at curr_size
                if bg_mode == "solid":
                    bg_color = random_solid_color()
                    bg = Image.new("RGB", (curr_size, curr_size), bg_color)
                    bg_info = {"bg_type": "solid"}
                elif bg_mode == "gradient":
                    bg = random_gradient(curr_size)
                    # sample center pixel as approximate bg color
                    bg_color = tuple(bg.getpixel((curr_size // 2, curr_size // 2)))
                    bg_info = {"bg_type": "gradient"}
                elif bg_mode == "image" and bg_dir is not None:
                    bg = random_image_background(curr_size, bg_dir)
                    bg_color = tuple(bg.getpixel((curr_size // 2, curr_size // 2)))
                    bg_info = {"bg_type": "image"}
                else:
                    bg = Image.new("RGB", (curr_size, curr_size), (255, 255, 255))
                    bg_color = (255, 255, 255)
                    bg_info = {"bg_type": "white"}

                text_color = random_text_color(bg_color)

                # decide partial
                is_partial = random.random() < partial_ratio

                # attempt to render; if partial requested, try offsets until at least one partial instance exists
                rec = compose_and_save_sample(
                    font_path,
                    text,
                    font_size,
                    is_partial,
                    curr_size,
                    images_dir,
                    masks_dir,
                    masks_ignore_dir,
                    bg_mode,
                    bg_dir,
                    counter,
                )
                meta.append(rec)
                # if we've reached the requested limit, write metadata and exit early
                if max_images is not None and counter >= max_images:
                    with open(out_dir / "metadata.json", "w", encoding="utf-8") as fh:
                        json.dump(meta, fh, ensure_ascii=False, indent=2)
                    print(f"Generadas {counter} pares en {out_dir} (límite alcanzado)")
                    return

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"Generadas {counter} pares en {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generador sintético simple de caracteres usando fuentes descargadas.")
    p.add_argument("--fonts", type=Path, default=Path("assets/fonts/extracted"), help="Carpeta donde están las fuentes extraídas.")
    p.add_argument("--out", type=Path, default=Path("datasets/synthetic"), help="Carpeta destino para imágenes y máscaras.")
    p.add_argument("--samples", type=int, default=3, help="Muestras por (fuente,carácter).")
    p.add_argument("--size", type=int, default=128, help="Tamaño de la imagen (px).")
    p.add_argument("--chars", type=str, default=CHARSET, help="Conjunto de caracteres a renderizar.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad.")
    p.add_argument("--bg-mode", choices=["solid", "gradient", "image"], default="solid", help="Modo de fondo: solid, gradient, image (usar --bg-dir para image).")
    p.add_argument("--bg-dir", type=Path, default=Path("assets/backgrounds"), help="Carpeta con imágenes de fondo (solo para --bg-mode image).")
    p.add_argument("--max-images", type=int, default=0, help="Máximo número de imágenes a generar (0 = sin límite).")
    p.add_argument("--single-ratio", type=float, default=0.4, help="Probabilidad de generar muestras single-char (vs multi).")
    p.add_argument("--partial-ratio", type=float, default=0.1, help="Probabilidad de generar muestras con caracteres parciales fuera del borde.")
    p.add_argument("--size-dist", type=float, nargs=3, default=[0.15, 0.6, 0.25], help="Distribución (small medium large) de tamaños.")
    p.add_argument("--size-choices", type=int, nargs='+', default=[64,128,256,512], help="Resoluciones candidatas para cada muestra.")
    p.add_argument("--multi-subdist", type=float, nargs=3, default=[0.7,0.25,0.05], help="Distribución (short,phrase,multiline) dentro del multi samples.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    bg_dir = args.bg_dir if args.bg_mode == "image" else None
    generate(
        args.fonts,
        args.out,
        samples_per_pair=args.samples,
        image_size=args.size,
        chars=args.chars,
        seed=args.seed,
        bg_mode=args.bg_mode,
        bg_dir=bg_dir,
        max_images=(args.max_images if args.max_images > 0 else None),
        single_ratio=args.single_ratio,
        partial_ratio=args.partial_ratio,
        size_dist=tuple(args.size_dist),
        size_choices=list(args.size_choices),
        multi_subdist=tuple(args.multi_subdist),
    )
