# Script para descargar fuentes populares usadas en editores, IDEs y textos academicos.
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from tqdm import tqdm

import zipfile

# FONT_SOURCES define nombre, url y sugerencia de subcarpeta destino.
FONT_SOURCES: List[Dict[str, str]] = [
    {"name": "JetBrainsMono", "url": "https://download.jetbrains.com/fonts/JetBrainsMono-2.304.zip", "extension": ".zip"},
    {"name": "FiraCode", "url": "https://github.com/tonsky/FiraCode/releases/download/6.2/Fira_Code_v6.2.zip", "extension": ".zip"},
    {"name": "CascadiaCode", "url": "https://github.com/microsoft/cascadia-code/releases/download/v2404.23/CascadiaCode-2404.23.zip", "extension": ".zip"},
    {"name": "SourceCodePro", "url": "https://github.com/adobe-fonts/source-code-pro/archive/refs/heads/release.zip", "extension": ".zip"},
    {"name": "SourceSans3", "url": "https://github.com/adobe-fonts/source-sans-pro/archive/refs/heads/release.zip", "extension": ".zip"},
    {"name": "Inter", "url": "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip", "extension": ".zip"},
    {"name": "Ubuntu", "url": "https://assets.ubuntu.com/v1/0cef8205-ubuntu-font-family-0.83.zip", "extension": ".zip"},
    {"name": "LatinModern", "url": "https://mirrors.ctan.org/fonts/lm.zip", "extension": ".zip"},
    {"name": "NotoSans", "url": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"},
    {"name": "NotoSansMath", "url": "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSansMath/NotoSansMath-Regular.ttf"},
    {"name": "DejaVu", "url": "http://sourceforge.net/projects/dejavu/files/dejavu/2.37/dejavu-fonts-ttf-2.37.zip", "extension": ".zip"},
    {"name": "Arimo", "url": "https://github.com/google/fonts/raw/main/apache/arimo/Arimo-Regular.ttf"},
    # Tinos family (4 variants, apache/)
    {"name": "Tinos-Regular", "url": "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Regular.ttf"},
    {"name": "Tinos-Bold", "url": "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Bold.ttf"},
    {"name": "Tinos-Italic", "url": "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Italic.ttf"},
    {"name": "Tinos-BoldItalic", "url": "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-BoldItalic.ttf"},
    # Cousine family (4 variants, apache/)
    {"name": "Cousine-Regular", "url": "https://github.com/google/fonts/raw/main/apache/cousine/Cousine-Regular.ttf"},
    {"name": "Cousine-Bold", "url": "https://github.com/google/fonts/raw/main/apache/cousine/Cousine-Bold.ttf"},
    {"name": "Cousine-Italic", "url": "https://github.com/google/fonts/raw/main/apache/cousine/Cousine-Italic.ttf"},
    {"name": "Cousine-BoldItalic", "url": "https://github.com/google/fonts/raw/main/apache/cousine/Cousine-BoldItalic.ttf"},
    # Carlito family (4 variants)
    {"name": "Carlito-Regular", "url": "https://github.com/google/fonts/raw/main/ofl/carlito/Carlito-Regular.ttf"},
    {"name": "Carlito-Bold", "url": "https://github.com/google/fonts/raw/main/ofl/carlito/Carlito-Bold.ttf"},
    {"name": "Carlito-Italic", "url": "https://github.com/google/fonts/raw/main/ofl/carlito/Carlito-Italic.ttf"},
    {"name": "Carlito-BoldItalic", "url": "https://github.com/google/fonts/raw/main/ofl/carlito/Carlito-BoldItalic.ttf"},
    # Caladea family (4 variants)
    {"name": "Caladea-Regular", "url": "https://github.com/google/fonts/raw/main/ofl/caladea/Caladea-Regular.ttf"},
    {"name": "Caladea-Bold", "url": "https://github.com/google/fonts/raw/main/ofl/caladea/Caladea-Bold.ttf"},
    {"name": "Caladea-Italic", "url": "https://github.com/google/fonts/raw/main/ofl/caladea/Caladea-Italic.ttf"},
    {"name": "Caladea-BoldItalic", "url": "https://github.com/google/fonts/raw/main/ofl/caladea/Caladea-BoldItalic.ttf"},
    # STIX Two (usar source zip del tag v2.13)
    {"name": "STIXTwo_v2.13_source", "url": "https://github.com/stipub/stixfonts/archive/refs/tags/v2.13.zip", "extension": ".zip"},
    {"name": "GFSDidot", "url": "https://github.com/google/fonts/raw/main/ofl/gfsdidot/GFSDidot-Regular.ttf"},
    # GFS Neohellenic (archivos individuales)
    {"name": "GFSNeohellenic-Regular", "url": "https://github.com/google/fonts/raw/main/ofl/gfsneohellenic/GFSNeohellenic.ttf"},
    {"name": "GFSNeohellenic-Bold", "url": "https://github.com/google/fonts/raw/main/ofl/gfsneohellenic/GFSNeohellenicBold.ttf"},
    {"name": "GFSNeohellenic-Italic", "url": "https://github.com/google/fonts/raw/main/ofl/gfsneohellenic/GFSNeohellenicItalic.ttf"},
    {"name": "GFSNeohellenic-BoldItalic", "url": "https://github.com/google/fonts/raw/main/ofl/gfsneohellenic/GFSNeohellenicBoldItalic.ttf"},
]


# get_filename deduce el nombre final basado en la URL.
def get_filename(url: str) -> str:
    filename = url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]
    return filename


# ensure_directories crea las carpetas necesarias.
def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    downloads = base_dir / "downloads"
    extracted = base_dir / "extracted"
    downloads.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    return {"downloads": downloads, "extracted": extracted}


# stream_download guarda la respuesta manejando reanudaciones simples.
def stream_download(url: str, target: Path, retries: int = 4, timeout: int = 30) -> None:
    temp_path = target.with_suffix(target.suffix + ".part")
    chunk_size = 1024 * 256
    for attempt in range(1, retries + 1):
        resume_from = temp_path.stat().st_size if temp_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else None
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                # Manejo de errores HTTP explícitos
                if response.status_code == 404:
                    print(f"ERROR 404: No encontrado: {url}")
                    if temp_path.exists():
                        temp_path.unlink()
                    return
                if response.status_code == 403:
                    print(f"ERROR 403: Acceso prohibido: {url}")
                    if temp_path.exists():
                        temp_path.unlink()
                    return
                if 400 <= response.status_code < 600 and response.status_code not in (200, 206):
                    print(f"ERROR HTTP {response.status_code}: {url}")
                    if temp_path.exists():
                        temp_path.unlink()
                    return
                if resume_from and response.status_code == 416:
                    temp_path.unlink(missing_ok=True)
                    resume_from = 0
                    continue
                if resume_from and response.status_code == 200:
                    temp_path.unlink(missing_ok=True)
                    resume_from = 0
                    continue
                response.raise_for_status()
                total_bytes = response.headers.get("Content-Length")
                if total_bytes is not None:
                    total_bytes = int(total_bytes) + (resume_from if response.status_code == 206 else 0)
                mode = "ab" if resume_from else "wb"
                with open(temp_path, mode) as file_obj, tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    initial=resume_from,
                    desc=target.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        file_obj.write(chunk)
                        progress.update(len(chunk))
            if target.exists():
                target.unlink()
            temp_path.rename(target)
            return
        except requests.exceptions.Timeout:
            print(f"Timeout (espera agotada) en intento {attempt}/{retries} para {url}")
        except requests.exceptions.ConnectionError as exc:
            print(f"Error de conexión en intento {attempt}/{retries} para {url}: {exc}")
        except requests.exceptions.RequestException as exc:
            print(f"Error inesperado en intento {attempt}/{retries} para {url}: {exc}")
        if attempt == retries:
            if temp_path.exists():
                temp_path.unlink()
            print(f"Descarga fallida tras {retries} intentos: {url}")
            return
        time.sleep(min(5 * attempt, 30))


# extract_zip descomprime el archivo zip si no fue extraido previamente.
def extract_zip(archive_path: Path, target_dir: Path) -> None:
    marker = target_dir / ".extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(archive_path, "r") as zipper:
        zipper.extractall(target_dir)
    marker.touch()


# copy_single_font copia archivos individuales ttf al directorio destino.
def copy_single_font(source: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / source.name
    if destination.exists():
        return
    shutil.copy2(source, destination)


# download_all procesa la lista completa respetando archivos ya existentes.
def download_all(fonts: Iterable[Dict[str, str]], base_dir: Path) -> None:
    dirs = ensure_directories(base_dir)
    for entry in fonts:
        url = entry["url"]
        name = entry["name"]
        filename = entry.get("filename") or get_filename(url)
        extension = entry.get("extension")
        if extension and not filename.lower().endswith(extension.lower()):
            filename = f"{filename}{extension}"
        download_path = dirs["downloads"] / filename
        extract_dir = dirs["extracted"] / name
        if extract_dir.exists():
            print(f"{name}: ya extraido, se omite")
            continue
        if not download_path.exists():
            print(f"{name}: descargando {url}")
            stream_download(url, download_path)
        else:
            print(f"{name}: descarga existente, se omite")
        if not download_path.exists():
            print(f"{name}: descarga fallida, se omite extracción")
            continue
        if download_path.suffix.lower() == ".zip":
            extract_dir.mkdir(parents=True, exist_ok=True)
            extract_zip(download_path, extract_dir)
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            copy_single_font(download_path, extract_dir)


# build_argument_parser define CLI sencillo para indicar carpeta destino.
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Descarga fuentes populares para reconocimiento de caracteres.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("assets/fonts"),
        help="Carpeta base donde se almacenaran las fuentes descargadas.",
    )
    return parser


# main ejecuta la descarga basada en los argumentos proporcionados.
def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    download_all(FONT_SOURCES, args.dest)


if __name__ == "__main__":  # pragma: no cover
    main()
