from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, total=None, **kwargs):
        if iterable is None:
            class Dummy:
                def update(self, n=1):
                    return
                def close(self):
                    return
            return Dummy()
        return iterable

# Devuelve una `ImageFont.FreeTypeFont` para (font_path: Path, size: int); usa `font_cache` (dict) para cache.
def get_font(font_path: Path, size: int, font_cache: dict) -> ImageFont.FreeTypeFont:
    key = (str(font_path), int(size))
    f = font_cache.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(str(font_path), size=int(size))
        except Exception:
            f = ImageFont.load_default()
        font_cache[key] = f
    return f


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

SIZES = [8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 40]
# weights: very high probability for 10/12/14; lower for 16; much lower for 18/8/20;
# very small probability for 24,28,30,32,40
SIZE_PROBS = [3, 30, 40, 35, 10, 5, 4, 2, 1, 1, 1, 1]
OOD_RATIO = 0.05
OOD_SIZES = [48, 64, 96]



MIN_CHARS = 2
MAX_CHARS = 12

BG_PALETTES = {
    "white": [(255, 255, 255), (250, 250, 250), (245, 245, 245)],
    "gray": [(245, 245, 245), (230, 230, 230), (200, 200, 200), (180, 180, 180)],
    "beige": [(250, 245, 235), (240, 230, 210), (235, 225, 200)],
    "brown": [(200, 180, 160), (170, 140, 120), (140, 110, 90)],
    "black": [(40, 40, 40), (20, 20, 20), (0, 0, 0)],
}

# Devuelve un color RGB (tuple[int,int,int]) elegido aleatoriamente de `BG_PALETTES`.
def random_solid_color() -> tuple[int, int, int]:
    group = random.choice(list(BG_PALETTES.values()))
    return random.choice(group)

# Calcula la luminancia relativa (float) de un color RGB (tuple[int,int,int]) en rango 0..1.
def lum(rgb: tuple[int, int, int]) -> float:
    r, g, b = [c / 255.0 for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# Devuelve True (bool) si la diferencia de luminancia entre dos colores RGB excede `min_delta`.
def contrast_ok(c1: tuple[int, int, int], c2: tuple[int, int, int], min_delta: float = 0.35) -> bool:
    return abs(lum(c1) - lum(c2)) >= min_delta


# Devuelve un color RGB (tuple[int,int,int]) para texto; intenta contrastar con `bg_color` si se da.
def random_text_color(bg_color: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
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
    return (0, 0, 0) if lum(bg_color) > 0.5 else (255, 255, 255)


# Genera y devuelve un degradado vertical como PIL.Image RGB de tamaño `size x size`.
def random_gradient(size: int) -> Image.Image:
    top = np.array([random.randint(200, 255) for _ in range(3)], dtype=np.uint8)
    bottom = np.array([random.randint(180, 240) for _ in range(3)], dtype=np.uint8)
    t = np.linspace(0.0, 1.0, num=size)[:, None]
    grad = (top * (1.0 - t) + bottom * t).astype(np.uint8)
    img = np.repeat(grad[:, None, :], size, axis=1)
    return Image.fromarray(img, mode="RGB")


# Devuelve un `Image` de fondo: toma una imagen aleatoria de `bg_dir` (cacheada) o genera un color sólido.
def random_image_background(size: int, bg_dir: Path) -> Image.Image:
    if not hasattr(random_image_background, "_cache") or random_image_background._cache_dir != str(bg_dir):
        imgs = [p for p in bg_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        random_image_background._cache = imgs
        random_image_background._cache_dir = str(bg_dir)
    imgs = random_image_background._cache
    if not imgs:
        return Image.new("RGB", (size, size), random_solid_color())
    chosen = random.choice(imgs)
    try:
        im = Image.open(chosen).convert("RGB")
        im = im.resize((size, size), Image.LANCZOS)
        return im
    except Exception:
        return Image.new("RGB", (size, size), random_solid_color())


# Busca y devuelve lista ordenada de archivos de fuente (.ttf/.otf) bajo `fonts_root` (Path).
def find_font_files(fonts_root: Path) -> List[Path]:
    exts = (".ttf", ".otf", ".ttc")
    fonts_root = fonts_root.expanduser().resolve()
    files = []
    try:
        if fonts_root.exists():
            for ext in exts:
                files.extend(list(fonts_root.rglob(f"*{ext}")))
    except Exception:
        pass

    # fallback: common subfolders inside fonts_root
    if not files:
        for sub in ("extracted", "downloads", "fonts", ""):
            try:
                p = (fonts_root / sub).resolve()
                if p.exists():
                    for ext in exts:
                        files.extend(list(p.rglob(f"*{ext}")))
            except Exception:
                continue

    # fallback: project assets/fonts
    if not files:
        try:
            alt = Path(__file__).parent / "assets" / "fonts"
            if alt.exists():
                for ext in exts:
                    files.extend(list(alt.rglob(f"*{ext}")))
        except Exception:
            pass

    files = [p for p in files if p.is_file()]
    # dedupe and sort
    unique = sorted({p.resolve() for p in files})
    return unique

# Renderiza texto en un canvas y devuelve `(RGBA_image, mask_L, instances)`.
# Parámetros: `font_path:Path, text:str, base_font_size:int, image_w:int, image_h:int, fill:tuple, x_offset:int=0, font_cache:dict|None`.
# Devuelve: `tuple[Image.Image, Image.Image, list]` donde `instances` contiene `(x0i:int,y0i:int,arr_clip:np.ndarray,touches_border:bool)`.
def render_line_simple(
    font_path: Path,
    text: str,
    base_font_size: int,
    image_w: int,
    image_h: int,
    fill: tuple[int, int, int],
    x_offset: int = 0,
    font_cache: dict | None = None,
) -> tuple[Image.Image, Image.Image, list]:
    if font_cache is None:
        font_cache = {}
    font = get_font(font_path, base_font_size, font_cache)

    img = Image.new("RGBA", (image_w, image_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    bbox_cache: dict[str, tuple[int, int, int, int]] = {}
    adv_cache: dict[str, float] = {}

    raw_lines = text.split("\n")
    lines: list[str] = []
    for rl in raw_lines:
        if not rl:
            lines.append(rl)
            continue
        cur = ""
        cur_w = 0
        for ch in rl:
            if ch in adv_cache:
                aw = adv_cache[ch]
            else:
                try:
                    aw = draw.textlength(ch, font=font)
                except Exception:
                    bbch = draw.textbbox((0, 0), ch, font=font)
                    aw = bbch[2] - bbch[0]
                adv_cache[ch] = aw
            test_w = cur_w + aw
            if test_w > image_w - 4:
                if cur == "":
                    lines.append(ch)
                    cur = ""
                    cur_w = 0
                else:
                    lines.append(cur)
                    cur = ch
                    cur_w = aw
            else:
                cur += ch
                cur_w = test_w
        if cur:
            lines.append(cur)

    line_heights = []
    line_widths = []
    for line in lines:
        w = 0
        h = 0
        for ch in line:
            if ch in adv_cache:
                ch_w = adv_cache[ch]
            else:
                try:
                    ch_w = draw.textlength(ch, font=font)
                except Exception:
                    bb = draw.textbbox((0, 0), ch, font=font)
                    ch_w = bb[2] - bb[0]
                adv_cache[ch] = ch_w
            w += ch_w + int(base_font_size * 0.05)
            if ch in bbox_cache:
                ch_h = bbox_cache[ch][3] - bbox_cache[ch][1]
            else:
                bb = draw.textbbox((0, 0), ch, font=font)
                bbox_cache[ch] = bb
                ch_h = bb[3] - bb[1]
            h = max(h, ch_h)
        if w > 0:
            w -= int(base_font_size * 0.05)
        line_widths.append(w)
        line_heights.append(h)

    total_h = sum(line_heights) + int((len(lines) - 1) * (base_font_size * 0.2))
    y0 = max(0, (image_h - total_h) // 2)
    instances = []
    y = y0
    for li, line in enumerate(lines):
        w = line_widths[li]
        x = max(2, 0 + x_offset)
        for ch in line:
            jitter = random.randint(-int(base_font_size * 0.02), int(base_font_size * 0.02))
            xi = int(x + jitter)
            draw.text((xi, y), ch, fill=(*fill, 255), font=font)
            if ch in bbox_cache:
                bbch = bbox_cache[ch]
            else:
                bbch = draw.textbbox((0, 0), ch, font=font)
                bbox_cache[ch] = bbch
            x0 = xi + bbch[0]
            y0c = y + bbch[1]
            x1 = xi + bbch[2]
            y1 = y + bbch[3]
            wch = bbch[2] - bbch[0]
            hch = bbch[3] - bbch[1]
            if wch <= 0 or hch <= 0:
                x += (bbch[2] - bbch[0]) + int(base_font_size * 0.05)
                continue
            else:
                m_small = Image.new("L", (wch, hch), 0)
                md = ImageDraw.Draw(m_small)
                md.text((-bbch[0], -bbch[1]), ch, fill=255, font=font)
                if base_font_size <= 14:
                    m_small = m_small.point(lambda p: 255 if p > 64 else 0).convert("L")
                else:
                    m_small = m_small.point(lambda p: 255 if p > 0 else 0).convert("L")
                arr_crop = (np.asarray(m_small) > 0)
            x0i = max(0, int(x0))
            y0i = max(0, int(y0c))
            x1i = min(image_w, int(x1))
            y1i = min(image_h, int(y1))
            if x1i <= x0i or y1i <= y0i:
                x += (bbch[2] - bbch[0]) + int(base_font_size * 0.05)
                continue
            sx0 = max(0, -int(x0))
            sy0 = max(0, -int(y0c))
            sx1 = sx0 + (x1i - x0i)
            sy1 = sy0 + (y1i - y0i)
            arr_clip = arr_crop[sy0:sy1, sx0:sx1]
            if arr_clip.size == 0:
                x += (bbch[2] - bbch[0]) + int(base_font_size * 0.05)
                continue
            touches_border = (int(x0) < 0) or (int(y0c) < 0) or (int(x1) > image_w) or (int(y1) > image_h)
            instances.append((x0i, y0i, arr_clip, touches_border))
            x += (bbch[2] - bbch[0]) + int(base_font_size * 0.05)
        y += line_heights[li] + int(base_font_size * 0.2)

    mask_arr = np.zeros((image_h, image_w), dtype=np.uint8)
    for it in instances:
        x0i, y0i, arr_clip, _touch = it
        if arr_clip.size == 0:
            continue
        hsub, wsub = arr_clip.shape
        mask_arr[y0i : y0i + hsub, x0i : x0i + wsub] = np.maximum(
            mask_arr[y0i : y0i + hsub, x0i : x0i + wsub], arr_clip.astype(np.uint8) * 255
        )
    mask = Image.fromarray(mask_arr)
    return img, mask, instances


# Compone una muestra: renderiza texto sobre fondo, construye máscaras (main y ignore), guarda archivos y devuelve metadata (dict).
# Parámetros: ver firma; devuelve `dict` con campos de metadata.
def compose_and_save_sample(
    font_path: Path,
    text: str,
    font_size: int,
    is_partial: bool,
    render_w: int,
    render_h: int,
    images_dir: Path,
    masks_dir: Path,
    masks_ignore_dir: Path,
    bg_mode: str,
    bg_dir: Path | None,
    idx: int,
    force_non_partial: bool = False,
) -> dict:
    # Consolida la lógica de generación usada por required_specs y el bucle principal.

# Selección de modo de fondo: solid/gradient/image/white
    if bg_mode == "solid":
        bg_color = random_solid_color()
        bg = Image.new("RGB", (render_w, render_h), bg_color)
        bg_info = {"bg_type": "solid"}
    elif bg_mode == "gradient":
        base = random_gradient(max(render_w, render_h))
        bg = base.resize((render_w, render_h), Image.LANCZOS)
        bg_color = tuple(bg.getpixel((render_w // 2, render_h // 2)))
        bg_info = {"bg_type": "gradient"}
    elif bg_mode == "image" and bg_dir is not None:
        bg = random_image_background(max(render_w, render_h), bg_dir).resize((render_w, render_h), Image.LANCZOS)
        bg_color = tuple(bg.getpixel((render_w // 2, render_h // 2)))
        bg_info = {"bg_type": "image"}
    else:
        bg = Image.new("RGB", (render_w, render_h), (255, 255, 255))
        bg_color = (255, 255, 255)
        bg_info = {"bg_type": "white"}

    text_color = random_text_color(bg_color)

    attempts = 0
    max_attempts = 12
    font_cache_local: dict = {}
    while True:
        x_off = 0
        if is_partial:
            x_off = random.randint(-render_w // 2, render_w // 2)
        img_comp, comp_mask, instances = render_line_simple(
            font_path, text, font_size, render_w, render_h, text_color, x_offset=x_off, font_cache=font_cache_local
        )

        h, w = render_h, render_w
        owner = -1 * np.ones((h, w), dtype=np.int32)
        count = np.zeros((h, w), dtype=np.int32)
        mask_ignore_arr = np.zeros((h, w), dtype=bool)
        partial_found = False

        for idx_inst, inst in enumerate(instances):
            x0i, y0i, arr_clip, touches_border = inst
            if arr_clip is None or arr_clip.size == 0:
                continue
            hsub, wsub = arr_clip.shape
            dy0 = int(y0i)
            dx0 = int(x0i)
            dy1 = dy0 + hsub
            dx1 = dx0 + wsub
            if dy1 <= dy0 or dx1 <= dx0:
                continue
            pos_mask = arr_clip.astype(bool)
            if touches_border and pos_mask.any():
                partial_found = True
                mask_ignore_arr[dy0:dy1, dx0:dx1] |= pos_mask
            slice_count = count[dy0:dy1, dx0:dx1]
            slice_owner = owner[dy0:dy1, dx0:dx1]
            slice_count[pos_mask] += 1
            newly = pos_mask & (slice_owner == -1)
            slice_owner[newly] = idx_inst

        main_mask_arr = (count == 1)

        if owner.size > 0:
            op = np.pad(owner, pad_width=1, mode="constant", constant_values=-1)
            center = op[1:-1, 1:-1]
            conflict = np.zeros_like(center, dtype=bool)
            shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dy, dx in shifts:
                neigh = op[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
                c = (center != -1) & (neigh != -1) & (neigh != center)
                conflict |= c
            main_mask_arr[conflict] = False

        main_mask_arr[count > 1] = False

        main_mask = Image.fromarray((main_mask_arr.astype(np.uint8) * 255).astype(np.uint8))
        mask_ignore = Image.fromarray((mask_ignore_arr.astype(np.uint8) * 255).astype(np.uint8))

# Reintentos hasta obtener muestra no parcial cuando force_non_partial es True
        # if: requiere que la muestra no tenga caracteres parciales en el borde
        if force_non_partial:
            if not partial_found or attempts >= max_attempts:
                char_rgba = img_comp
                mask = main_mask.convert("L")
                mask_ignore = mask_ignore.convert("L")
                if mask.getbbox() is None and attempts < max_attempts:
                    attempts += 1
                    continue
                break
            attempts += 1
            continue

        if not is_partial or partial_found or attempts >= max_attempts:
            char_rgba = img_comp
            mask = main_mask.convert("L")
            mask_ignore = mask_ignore.convert("L")
            if mask.getbbox() is None and attempts < max_attempts:
                attempts += 1
                continue
            break
        attempts += 1

    composed = bg.copy()
    composed.paste(img_comp, mask=img_comp.split()[3])

    if random.random() < 0.12:
        occ = Image.new("RGBA", (render_w, render_h), (0, 0, 0, 0))
        od = ImageDraw.Draw(occ)
        for _ in range(random.randint(1, 3)):
            x0 = random.randint(0, max(0, render_w - 1))
            y0 = random.randint(0, max(0, render_h - 1))
            x1 = random.randint(x0, render_w)
            y1 = random.randint(y0, render_h)
            od.ellipse((x0, y0, x1, y1), fill=(0, 0, 0, random.randint(30, 110)))
        composed = Image.alpha_composite(composed.convert("RGBA"), occ).convert("RGB")

    if random.random() < 0.3:
        composed = add_noise_background(composed, strength=random.uniform(0.02, 0.12))
        composed = composed.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))

    fname = f"img_{idx:06d}.png"
    shard = (int(idx) - 1) // 1000
    shard_name = f"{shard:04d}"
    images_subdir = images_dir / shard_name
    masks_subdir = masks_dir / shard_name
    masks_ignore_subdir = masks_ignore_dir / shard_name
    images_subdir.mkdir(parents=True, exist_ok=True)
    masks_subdir.mkdir(parents=True, exist_ok=True)
    masks_ignore_subdir.mkdir(parents=True, exist_ok=True)

    composed.save(images_subdir / fname, compress_level=1)
    mask.save(masks_subdir / fname, compress_level=1)
    mask_ignore.save(masks_ignore_subdir / fname, compress_level=1)

    font_name = font_path.stem
    rec = {
        "file": f"{shard_name}/{fname}",
        "text": text,
        "font": font_name,
        "font_path": str(font_path),
        "font_size": font_size,
        "render_w": int(render_w),
        "render_h": int(render_h),
        "num_chars": len(text),
        "partial": bool(mask_ignore.getbbox()),
    }
    rec.update(bg_info)
    return rec


# Desempaqueta parámetros y llama a `compose_and_save_sample(params...)`.
# Parámetro: `params: tuple` con la forma documentada; devuelve `dict` metadata.
def _compose_task(params: tuple) -> dict:
    (
        font_path,
        text,
        font_size,
        is_partial,
        render_w,
        render_h,
        images_dir,
        masks_dir,
        masks_ignore_dir,
        bg_mode,
        bg_dir,
        idx,
        force_non_partial,
        master_seed,
    ) = params
    return compose_and_save_sample(
        font_path,
        text,
        font_size,
        is_partial,
        render_w,
        render_h,
        images_dir,
        masks_dir,
        masks_ignore_dir,
        bg_mode,
        bg_dir,
        idx,
        force_non_partial=force_non_partial,
    )


# Procesa una lista de parámetros en un solo worker y devuelve lista de metadata dicts.
# Si `master_seed` no es None, reseeda determinísticamente por `master_seed + idx` antes de cada imagen.
def _compose_chunk(params_list: list, master_seed: int | None = None) -> list:
    recs: list[dict] = []
    for params in params_list:
        idx = params[11] if len(params) > 11 else None
        if master_seed is not None and idx is not None:
            random.seed(int(master_seed) + int(idx))
        rec = _compose_task(params)
        recs.append(rec)
    return recs


# Añade ruido gaussiano al `Image` de entrada.
# Parámetros: `image: Image.Image`, `strength: float` (0..1). Devuelve `Image.Image`.
def add_noise_background(image: Image.Image, strength: float = 0.2) -> Image.Image:
    arr = np.asarray(image, dtype=np.int16)
    noise = np.random.randn(*arr.shape) * (strength * 255.0)
    out_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(out_arr)
    return out


# Genera el dataset sintético y escribe imágenes, máscaras y metadata JSONL en `out_dir`.
# Parámetros: `fonts_dir:Path, out_dir:Path, samples_per_pair:int=3, image_size:int=128, chars:str=..., seed:int|None=None, bg_mode:str='solid', bg_dir:Path|None=None, max_images:int|None=None, single_ratio:float=0.15, partial_ratio:float=0.06, size_dist:tuple=..., size_choices:list|None=None, multi_subdist:tuple=..., workers:int=1, chunk_size:int=1`.
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
    single_ratio: float = 0.15,
    partial_ratio: float = 0.06,
    size_dist: tuple[float, float, float] = (0.12, 0.55, 0.33),
    size_choices: list[int] | None = None,
    multi_subdist: tuple[float, float, float] = (0.15, 0.25, 0.6),
    workers: int = 1,
    chunk_size: int = 1,
    force_render_size: int | None = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed if seed is not None else 0)
    fonts = find_font_files(fonts_dir)
    # if: comprueba que existan fuentes en `fonts_dir` y aborta si no hay ninguna.
    if not fonts:
        raise SystemExit(f"No se encontraron fuentes en {fonts_dir}")

    out_dir = out_dir.expanduser().resolve()
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    masks_ignore_dir = out_dir / "masks_ignore"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    masks_ignore_dir.mkdir(parents=True, exist_ok=True)
    metadata_jsonl = out_dir / "metadata.jsonl"
    metadata_fh = open(metadata_jsonl, "a", encoding="utf-8")

    meta_buffer: list[str] = []
    # Función interna: flush_meta_buffer() -> None; vacía buffer de metadata al archivo abierto.
    def flush_meta_buffer():
        if not meta_buffer:
            return
        try:
            metadata_fh.write("\n".join(meta_buffer) + "\n")
            metadata_fh.flush()
        except Exception:
            pass
        meta_buffer.clear()

    # Función interna: write_recs(recs: list[dict]) -> None; añade recs al buffer y actualiza contadores/progress.
    def write_recs(recs: list[dict]):
        nonlocal counter, single_ct, multiline_ct, partial_ct
        for rec in recs:
            try:
                meta_buffer.append(json.dumps(rec, ensure_ascii=False))
            except Exception:
                continue
            counter += 1
            if rec.get("num_chars", 0) == 1:
                single_ct += 1
            if "\n" in (rec.get("text") or ""):
                multiline_ct += 1
            if rec.get("partial"):
                partial_ct += 1
            try:
                pbar.update(1)
            except Exception:
                pass
        if len(meta_buffer) >= 100:
            flush_meta_buffer()

    counter = 0
    single_ct = 0
    multiline_ct = 0
    partial_ct = 0
    next_idx = 1
    total_estimate = max_images if max_images is not None else None
    pbar = tqdm(total=total_estimate)

    # if: forzar un `chunk_size` mínimo para reducir overhead de IPC y mejorar rendimiento
    if chunk_size < 8:
        chunk_size = 8

    MAX_IN_FLIGHT = max(2, int(workers) * 2)

    small_sizes = [s for s in SIZES if s <= 12]
    medium_sizes = [s for s in SIZES if 14 <= s <= 32]
    large_sizes = OOD_SIZES[:] if OOD_SIZES else [48, 64, 96]

    sd = list(size_dist)
    ssum = sum(sd)
    if ssum <= 0:
        sd = [0.15, 0.6, 0.25]
        ssum = sum(sd)
    sd = [v / ssum for v in sd]

    if size_choices is None:
        size_choices = [512, 256, 720, 128]
        size_weights = [0.5, 0.3, 0.05, 0.05]
    else:
        size_weights = None

    # Elige dimensiones (w,h) para la imagen sintética; favorece horizontales; devuelve tuple[int,int].
    def pick_dimensions() -> tuple[int, int]:
        # If user requested a forced fixed render size, return square of that size
        if force_render_size is not None:
            s = int(force_render_size)
            return s, s
        orient_probs = [0.65, 0.20, 0.15]
        orientations = ["horizontal", "vertical", "square"]

        # Muestra principal en rango central: devuelve (w,h) con orientación preferida; tipos: int,int.
        def sample_in_core_bucket() -> tuple[int, int]:
            orient = random.choices(orientations, weights=orient_probs, k=1)[0]
            if orient == "horizontal":
                h = random.randint(127, 420)
                ratio = max(1.2, min(3.5, random.gauss(1.85, 0.35)))
                w = int(round(h * ratio))
                w = max(127, min(1280, w))
                return w, h
            if orient == "vertical":
                w = random.randint(127, 360)
                ratio = max(1.2, min(3.0, random.gauss(1.85, 0.35)))
                h = int(round(w * ratio))
                h = max(127, min(720, h))
                return w, h
            s = random.randint(127, 720)
            return s, s

        # Muestra amplia/broad: permite relaciones y tamaños extremos; devuelve (w,h) int,int.
        def sample_broad() -> tuple[int, int]:
            orient = random.choices(orientations, weights=orient_probs, k=1)[0]
            if orient == "horizontal":
                h = random.randint(10, 720)
                ratio = max(1.2, min(6.0, random.gauss(1.85, 0.6)))
                w = int(round(h * ratio))
                w = max(11, min(1280, w))
                return w, h
            if orient == "vertical":
                w = random.randint(10, 720)
                ratio = max(1.2, min(4.0, random.gauss(1.85, 0.6)))
                h = int(round(w * ratio))
                h = max(11, min(720, h))
                return w, h
            s = random.randint(10, 720)
            return s, s

        if random.random() < 0.6:
            return sample_in_core_bucket()
        return sample_broad()

    msd = list(multi_subdist)
    msum = sum(msd)
    if msum <= 0:
        msd = [0.7, 0.25, 0.05]
        msum = sum(msd)
    msd = [v / msum for v in msd]

    # Elige tamaño de fuente (int) según bucket opcional o la distribución configurada.
    def pick_size(bucket: str | None = None) -> int:
        if bucket == "small":
            return int(random.choice(small_sizes)) if small_sizes else int(random.choice(SIZES))
        if bucket == "medium":
            return int(random.choice(medium_sizes)) if medium_sizes else int(random.choice(SIZES))
        if bucket == "large":
            return int(random.choice(large_sizes)) if large_sizes else int(random.choice(OOD_SIZES))
        r = random.random()
        if r < sd[0]:
            return int(random.choice(small_sizes)) if small_sizes else int(random.choice(SIZES))
        if r < sd[0] + sd[1]:
            return int(random.choice(medium_sizes)) if medium_sizes else int(random.choice(SIZES))
        return int(random.choice(large_sizes)) if large_sizes else int(random.choice(OOD_SIZES))

    # Selecciona tamaño de fuente (int) preferente según la resolución de render `render_size`.
    def choose_font_size_for_render(render_size: int) -> int:
        if render_size == 512:
            cand = [10, 11, 12, 14, 16, 18, 20, 22, 24]
            weights = [5, 4, 28, 30, 20, 8, 5, 2, 1]
            return int(min(random.choices(cand, weights=weights, k=1)[0], render_size))
        if render_size == 256:
            cand = [12, 16, 18, 20]
            weights = [10, 35, 35, 20]
            return int(min(random.choices(cand, weights=weights, k=1)[0], render_size))
        if random.random() < OOD_RATIO:
            return int(random.choice(OOD_SIZES))
        return int(random.choices(SIZES, weights=SIZE_PROBS, k=1)[0])

    required_specs: list[dict] = []
    # if: construir lista `required_specs` para garantizar cobertura de caracteres/tamaños cuando corpus grande
    if max_images is not None and max_images > 200:
        num_chars = len(chars)
        max_forced = min(num_chars, max(50, int((max_images or 1000) // 200)))
        sample_chars = random.sample(list(chars), k=max_forced)
        for ch in sample_chars:
            required_specs.append({"type": "multiline", "char": ch})
            required_specs.append({"type": "partial", "char": ch})
            required_specs.append({"type": "small", "char": ch})
            required_specs.append({"type": "medium", "char": ch})
            required_specs.append({"type": "large", "char": ch})

        if max_images is not None and max_images > 500:
            subset_k = min(len(chars), max(50, int((max_images or 1000) // 20), 100))
            subset_k = min(subset_k, 100)
            sample_subset = random.sample(list(chars), k=subset_k)
            rep_resolutions = list(dict.fromkeys([int(x) for x in size_choices]))[:3]
            rep_fontsizes = (SIZES[:5] if len(SIZES) >= 5 else SIZES) + (OOD_SIZES[:1] if OOD_SIZES else [])
            for ch in sample_subset:
                for rs in rep_resolutions:
                    required_specs.append({"type": "per_resolution", "char": ch, "render_size": int(rs)})
                for fs in rep_fontsizes:
                    required_specs.append({"type": "per_fontsize", "char": ch, "font_size": int(fs)})

        if len(required_specs) > (max_images if max_images is not None else 0):
            raise SystemExit(
                f"Requested max_images={max_images} is too small to guarantee required coverage ({len(required_specs)} specs). "
                "Increase --max-images or reduce charset/resolution/font-size requirements."
            )

    # Si `required_specs` no está vacío, generar esas muestras primero para asegurar cobertura.
    if required_specs:
        random.shuffle(required_specs)
        if True:
            tasks = []
            params_buffer: list = []
            with ProcessPoolExecutor(max_workers=workers) as exc:
                for spec in required_specs:
                    if max_images is not None and counter >= max_images:
                        break
                    ch = spec["char"]
                    typ = spec["type"]
                    font_path = random.choice(fonts)

                    render_size_for_spec = spec.get("render_size", image_size)
                    font_size_for_spec = spec.get("font_size", None)
                    is_partial = False
                    force_non_partial = False

                    if typ == "single_nonpartial":
                        text = ch
                        font_size = pick_size(None) if font_size_for_spec is None else int(font_size_for_spec)
                        is_partial = False
                        force_non_partial = True
                    elif typ == "multiline":
                        if font_size_for_spec is None:
                            font_size = pick_size(None)
                        else:
                            font_size = int(font_size_for_spec)
                        lines = random.randint(2, 7)
                        if font_size in (8, 10, 12, 14):
                            total_chars = random.randint(40, 120)
                            lines = min(lines, 6)
                        else:
                            total_chars = random.randint(12, 48)
                        chars_list = [random.choice(chars) for _ in range(total_chars - 1)]
                        insert_at = random.randint(0, len(chars_list))
                        chars_list.insert(insert_at, ch)
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
                        font_size = choose_font_size_for_render(rs) if font_size_for_spec is None else int(font_size_for_spec)
                        text = ch
                        is_partial = False
                        force_non_partial = True
                    elif typ == "per_fontsize":
                        fs = int(spec.get("font_size", SIZES[0]))
                        font_size = fs
                        suitable = sorted([int(x) for x in size_choices if int(x) >= font_size])
                        render_size_for_spec = suitable[-1] if suitable else image_size
                        text = ch
                        is_partial = False
                        force_non_partial = True
                    else:
                        text = ch
                        font_size = pick_size(None)
                        is_partial = False

                    try:
                        rw, rh = pick_dimensions()
                    except Exception:
                        rw = int(render_size_for_spec)
                        rh = int(render_size_for_spec)

                    params = (
                        font_path,
                        text,
                        font_size,
                        is_partial,
                        rw,
                        rh,
                        images_dir,
                        masks_dir,
                        masks_ignore_dir,
                        bg_mode,
                        bg_dir,
                        next_idx,
                        force_non_partial,
                        seed,
                    )
                    params_buffer.append(params)
                    next_idx += 1
                    if len(params_buffer) >= chunk_size:
                        tasks.append(exc.submit(_compose_chunk, params_buffer.copy(), seed))
                        params_buffer.clear()
                    while len(tasks) >= MAX_IN_FLIGHT:
                        done, _ = wait(tasks, timeout=0.1, return_when=FIRST_COMPLETED)
                        if not done:
                            time.sleep(0.01)
                            continue
                        for fut in list(done):
                            try:
                                recs = fut.result()
                                write_recs(recs)
                            except Exception:
                                pass
                            try:
                                tasks.remove(fut)
                            except ValueError:
                                pass
                if params_buffer:
                    tasks.append(exc.submit(_compose_chunk, params_buffer.copy(), seed))
                    params_buffer.clear()
                while tasks:
                    done, _ = wait(tasks, return_when=FIRST_COMPLETED)
                    for fut in list(done):
                        try:
                            recs = fut.result()
                            write_recs(recs)
                        except Exception:
                            pass
                        try:
                            tasks.remove(fut)
                        except ValueError:
                            pass

                flush_meta_buffer()
                try:
                    if total_estimate is None:
                        pbar.refresh()
                except Exception:
                    pass
        else:
            for spec in required_specs:
                if counter >= max_images:
                    break
                ch = spec["char"]
                typ = spec["type"]
                font_path = random.choice(fonts)

                render_size_for_spec = spec.get("render_size", image_size)
                font_size_for_spec = spec.get("font_size", None)
                is_partial = False
                force_non_partial = False

                if typ == "single_nonpartial":
                    text = ch
                    font_size = pick_size(None) if font_size_for_spec is None else int(font_size_for_spec)
                    is_partial = False
                    force_non_partial = True
                elif typ == "multiline":
                    if font_size_for_spec is None:
                        font_size = pick_size(None)
                    else:
                        font_size = int(font_size_for_spec)
                    lines = random.randint(2, 7)
                    if font_size in (8, 10, 12, 14):
                        total_chars = random.randint(40, 120)
                        lines = min(lines, 6)
                    else:
                        total_chars = random.randint(12, 48)
                    chars_list = [random.choice(chars) for _ in range(total_chars - 1)]
                    insert_at = random.randint(0, len(chars_list))
                    chars_list.insert(insert_at, ch)
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
                    font_size = choose_font_size_for_render(rs) if font_size_for_spec is None else int(font_size_for_spec)
                    text = ch
                    is_partial = False
                    force_non_partial = True
                elif typ == "per_fontsize":
                    fs = int(spec.get("font_size", SIZES[0]))
                    font_size = fs
                    suitable = sorted([int(x) for x in size_choices if int(x) >= font_size])
                    render_size_for_spec = suitable[-1] if suitable else image_size
                    text = ch
                    is_partial = False
                    force_non_partial = True
                else:
                    text = ch
                    font_size = pick_size(None)
                    is_partial = False

                try:
                    rw, rh = pick_dimensions()
                except Exception:
                    rw = int(render_size_for_spec)
                    rh = int(render_size_for_spec)
                idx_for_save = next_idx
                next_idx += 1
                rec = compose_and_save_sample(
                    font_path,
                    text,
                    font_size,
                    is_partial,
                    rw,
                    rh,
                    images_dir,
                    masks_dir,
                    masks_ignore_dir,
                    bg_mode,
                    bg_dir,
                    idx_for_save,
                    force_non_partial=force_non_partial,
                )
                write_recs([rec])

    # Si `workers>1`, usar ProcessPoolExecutor para generar en paralelo (batches por chunk_size).
    if workers is not None and workers > 1:
        futures = []
        params_buffer = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for font_path in fonts:
                for _ch in chars:
                    for s in range(samples_per_pair):
                        if max_images is not None and counter >= max_images:
                            break
                        if size_weights is not None:
                            total_w = sum(size_weights)
                            probs = None if total_w <= 0 else [w / total_w for w in size_weights]
                            nominal = int(random.choices(size_choices, weights=probs, k=1)[0])
                            curr_w, curr_h = pick_dimensions()
                        else:
                            curr_w, curr_h = pick_dimensions()

                        font_size = choose_font_size_for_render(min(curr_w, curr_h))
                        font_size = min(font_size, min(curr_w, curr_h))

                        if random.random() < single_ratio:
                            text = _ch
                        else:
                            w, h = curr_w, curr_h
                            approx_char_w = max(4, int(font_size * 0.6))
                            max_chars_per_line = max(2, (w - 20) // approx_char_w)
                            max_lines_possible = max(2, (h - 10) // max(8, int(font_size * 1.0)))
                            if w > h:
                                total_chars = random.randint(max(20, max_chars_per_line * 3), max(60, max_chars_per_line * 8))
                            else:
                                total_chars = random.randint(12, max(20, max_chars_per_line * 3))
                            total_chars = min(total_chars, max_chars_per_line * max_lines_possible)
                            choice = random.choices([0, 1, 2], weights=list(multi_subdist), k=1)[0]
                            if choice == 2:
                                lines = random.randint(2, min(max_lines_possible, 7))
                                chars_list = [random.choice(chars) for _ in range(total_chars)]
                                parts: list[str] = []
                                remaining = total_chars
                                for i in range(lines - 1):
                                    min_take = 2
                                    max_take = max(2, remaining - 2 * (lines - i - 1))
                                    take = random.randint(min_take, max_take)
                                    take = min(take, remaining - 2 * (lines - i - 1))
                                    parts.append(''.join(chars_list[:take]))
                                    chars_list = chars_list[take:]
                                    remaining -= take
                                parts.append(''.join(chars_list))
                                j = 0
                                while j < len(parts):
                                    if len(parts[j]) < 2 and j < len(parts) - 1:
                                        parts[j] += parts[j+1]
                                        parts.pop(j+1)
                                    else:
                                        j += 1
                                text = '\n'.join(parts)
                            elif choice == 1:
                                lines = random.randint(2, min(3, max_lines_possible))
                                chars_list = [random.choice(chars) for _ in range(total_chars)]
                                parts = []
                                remaining = total_chars
                                for i in range(lines - 1):
                                    take = max(2, remaining // (lines - i))
                                    parts.append(''.join(chars_list[:take]))
                                    chars_list = chars_list[take:]
                                    remaining -= take
                                parts.append(''.join(chars_list))
                                text = '\n'.join(parts)
                            else:
                                n = max(2, min(MAX_CHARS, total_chars))
                                text = ''.join(random.choice(chars) for _ in range(n))

                        is_partial = random.random() < partial_ratio

                        params = (
                            font_path,
                            text,
                            font_size,
                            is_partial,
                            curr_w,
                            curr_h,
                            images_dir,
                            masks_dir,
                            masks_ignore_dir,
                            bg_mode,
                            bg_dir,
                            next_idx,
                            False,
                            seed,
                        )
                        params_buffer.append(params)
                        next_idx += 1
                        if len(params_buffer) >= chunk_size:
                            futures.append(pool.submit(_compose_chunk, params_buffer.copy(), seed))
                            params_buffer.clear()
                        while len(futures) >= MAX_IN_FLIGHT:
                            done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                            if not done:
                                time.sleep(0.01)
                                continue
                            for fut in list(done):
                                try:
                                    recs = fut.result()
                                    write_recs(recs)
                                except Exception:
                                    pass
                                try:
                                    futures.remove(fut)
                                except ValueError:
                                    pass
                    if max_images is not None and counter >= max_images:
                        break
                if max_images is not None and counter >= max_images:
                    break
            if params_buffer:
                futures.append(pool.submit(_compose_chunk, params_buffer.copy(), seed))
                params_buffer.clear()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in list(done):
                    try:
                        recs = fut.result()
                        write_recs(recs)
                    except Exception:
                        pass
                    try:
                        futures.remove(fut)
                    except ValueError:
                        pass
    else:
        for font_path in fonts:
            for _ch in chars:
                for s in range(samples_per_pair):
                    counter += 1
                    if size_weights is not None:
                        total_w = sum(size_weights)
                        if total_w <= 0:
                            probs = None
                        else:
                            probs = [w / total_w for w in size_weights]
                        nominal = int(random.choices(size_choices, weights=probs, k=1)[0])
                        curr_w, curr_h = pick_dimensions()
                    else:
                        curr_w, curr_h = pick_dimensions()

                    font_size = choose_font_size_for_render(min(curr_w, curr_h))
                    font_size = min(font_size, min(curr_w, curr_h))

                    if random.random() < single_ratio:
                        text = _ch
                    else:
                        w, h = curr_w, curr_h
                        approx_char_w = max(4, int(font_size * 0.6))
                        max_chars_per_line = max(2, (w - 20) // approx_char_w)
                        max_lines_possible = max(2, (h - 10) // max(8, int(font_size * 1.0)))

                        if w > h:
                            total_chars = random.randint(max(20, max_chars_per_line * 3), max(60, max_chars_per_line * 8))
                        else:
                            total_chars = random.randint(12, max(20, max_chars_per_line * 3))

                        total_chars = min(total_chars, max_chars_per_line * max_lines_possible)

                        choice = random.choices([0, 1, 2], weights=list(multi_subdist), k=1)[0]
                        if choice == 2:
                            lines = random.randint(2, min(max_lines_possible, 7))
                            chars_list = [random.choice(chars) for _ in range(total_chars)]
                            parts: list[str] = []
                            remaining = total_chars
                            for i in range(lines - 1):
                                min_take = 2
                                max_take = max(2, remaining - 2 * (lines - i - 1))
                                take = random.randint(min_take, max_take)
                                take = min(take, remaining - 2 * (lines - i - 1))
                                parts.append(''.join(chars_list[:take]))
                                chars_list = chars_list[take:]
                                remaining -= take
                            parts.append(''.join(chars_list))
                            j = 0
                            while j < len(parts):
                                if len(parts[j]) < 2 and j < len(parts) - 1:
                                    parts[j] += parts[j+1]
                                    parts.pop(j+1)
                                else:
                                    j += 1
                            text = '\n'.join(parts)
                        elif choice == 1:
                            lines = random.randint(2, min(3, max_lines_possible))
                            chars_list = [random.choice(chars) for _ in range(total_chars)]
                            parts = []
                            remaining = total_chars
                            for i in range(lines - 1):
                                take = max(2, remaining // (lines - i))
                                parts.append(''.join(chars_list[:take]))
                                chars_list = chars_list[take:]
                                remaining -= take
                            parts.append(''.join(chars_list))
                            text = '\n'.join(parts)
                        else:
                            n = max(2, min(MAX_CHARS, total_chars))
                            text = ''.join(random.choice(chars) for _ in range(n))

    # Selección de modo de fondo (solid/gradient/image/white) y creación/redimensionado del `bg`.
                    if bg_mode == "solid":
                        bg_color = random_solid_color()
                        bg = Image.new("RGB", (curr_w, curr_h), bg_color)
                        bg_info = {"bg_type": "solid"}
                    elif bg_mode == "gradient":
                        base = random_gradient(max(curr_w, curr_h))
                        bg = base.resize((curr_w, curr_h), Image.LANCZOS)
                        bg_color = tuple(bg.getpixel((curr_w // 2, curr_h // 2)))
                        bg_info = {"bg_type": "gradient"}
                    elif bg_mode == "image" and bg_dir is not None:
                        bg = random_image_background(max(curr_w, curr_h), bg_dir).resize((curr_w, curr_h), Image.LANCZOS)
                        bg_color = tuple(bg.getpixel((curr_w // 2, curr_h // 2)))
                        bg_info = {"bg_type": "image"}
                    else:
                        bg = Image.new("RGB", (curr_w, curr_h), (255, 255, 255))
                        bg_color = (255, 255, 255)
                        bg_info = {"bg_type": "white"}

                    text_color = random_text_color(bg_color)

                    is_partial = random.random() < partial_ratio

                    rec = compose_and_save_sample(
                        font_path,
                        text,
                        font_size,
                        is_partial,
                        curr_w,
                        curr_h,
                        images_dir,
                        masks_dir,
                        masks_ignore_dir,
                        bg_mode,
                        bg_dir,
                        next_idx,
                    )
                    write_recs([rec])
                    next_idx += 1
                    if max_images is not None and counter >= max_images:
                        flush_meta_buffer()
                        try:
                            metadata_fh.close()
                        except Exception:
                            pass
                        summary = {
                            "total": counter,
                            "single": single_ct,
                            "multiline": multiline_ct,
                            "partials": partial_ct,
                        }
                        with open(out_dir / "metadata_summary.json", "w", encoding="utf-8") as fh:
                            json.dump(summary, fh, ensure_ascii=False, indent=2)
                        print(f"Generadas {counter} pares en {out_dir} (límite alcanzado)")
                        return
    try:
        flush_meta_buffer()
        metadata_fh.close()
    except Exception:
        pass
    total = counter
    print(f"Generadas {counter} pares en {out_dir}")
    if total > 0:
        print(f"Resumen: total={total}, single={single_ct} ({single_ct/total:.2%}), multiline_paragraphs={multiline_ct} ({multiline_ct/total:.2%}), partials={partial_ct} ({partial_ct/total:.2%})")
    else:
        print("Resumen: no se generaron imágenes.")


# Construye y devuelve un `argparse.ArgumentParser` para la CLI del generador; devuelve argparse.ArgumentParser.
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
    p.add_argument("--single-ratio", type=float, default=0.05, help="Probabilidad de generar muestras single-char (vs multi).")
    p.add_argument("--partial-ratio", type=float, default=0.06, help="Probabilidad de generar muestras con caracteres parciales fuera del borde.")
    p.add_argument("--size-dist", type=float, nargs=3, default=[0.15, 0.6, 0.25], help="Distribución (small medium large) de tamaños.")
    p.add_argument("--size-choices", type=int, nargs='+', default=[64,128,256,512], help="Resoluciones candidatas para cada muestra.")
    p.add_argument("--multi-subdist", type=float, nargs=3, default=[0.05,0.15,0.8], help="Distribución (short,phrase,multiline) dentro del multi samples.")
    p.add_argument("--workers", type=int, default=1, help="Número de procesos worker para generar imágenes en paralelo (0 = auto-detect CPUs).")
    p.add_argument("--chunk-size", type=int, default=1, help="Número de imágenes que cada worker procesará en una sola tarea (1 = comportamiento por-imagen).")
    p.add_argument("--force-size", type=int, default=0, help="Forzar todas las imágenes generadas a ser cuadradas de tamaño N (ej: --force-size 256). 0 = deshabilitado.")
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
        workers=(args.workers if args.workers and args.workers > 0 else 1),
        chunk_size=(args.chunk_size if args.chunk_size and args.chunk_size > 0 else 1),
        force_render_size=(args.force_size if getattr(args, 'force_size', 0) and args.force_size > 0 else None),
    )