# Punto de entrada CLI para calibración y modo interactivo de NeuralPaint.
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
from typing import Optional, Sequence

from neuralpaint.app import run_interactive_app
from neuralpaint.calibration import ensure_calibration


# parse_args construye el parser CLI y devuelve argparse.Namespace.
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibración sin marcadores y dibujo controlado por gestos.")
    parser.add_argument("--camera", type=int, default=0, help="Índice del dispositivo de captura (por defecto 0).")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration") / "homography.npz",
        help="Ruta para guardar o cargar la homografía de calibración.",
    )
    parser.add_argument(
        "--surface-width",
        type=float,
        default=0.0,
        help="Logical canvas width in pixels (0 = usar resolución de pantalla).",
    )
    parser.add_argument(
        "--surface-height",
        type=float,
        default=0.0,
        help="Logical canvas height in pixels (0 = usar resolución de pantalla).",
    )
    parser.add_argument("--calibration-min-area", type=float, default=0.15, help="Proporción mínima de área de contorno (0-1).")
    parser.add_argument(
        "--calibration-approx-eps",
        type=float,
        default=0.03,
        help="Factor de aproximación poligonal relativo al perímetro.",
    )
    parser.add_argument("--calibration-warp", action="store_true", help="Muestra la superficie rectificada durante la calibración.")
    parser.add_argument("--calibration-debug", action="store_true", help="Muestra el mapa de bordes mientras calibras.")
    parser.add_argument("--force-calibrate", action="store_true", help="Forzar calibración incluso si ya hay datos guardados.")
    parser.add_argument("--calibration-only", action="store_true", help="Realiza la calibración y sale sin entrar al modo de dibujo.")
    parser.add_argument(
        "--calibration-mode",
        choices=("prompt", "contour", "apriltag"),
        default="prompt",
        help="Modo de calibración: 'contour' (rectángulo brillante), 'apriltag' (rejilla AprilTag), 'prompt' pregunta al iniciar.",
    )
    parser.add_argument("--min-detection", type=float, default=0.6, help="Umbral de confianza de detección para MediaPipe.")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="Umbral de seguimiento para MediaPipe.")
    parser.add_argument("--smoothing", type=float, default=0.6, help="Factor de suavizado exponencial del puntero (0-1).")
    parser.add_argument("--max-strokes", type=int, default=200, help="Número máximo de trazos almacenados antes de descartar los más antiguos.")
    parser.add_argument("--command-hold-frames", type=int, default=6, help="Cuadros necesarios para confirmar un comando del brazo izquierdo.")
    parser.add_argument("--erase-radius", type=float, default=35.0, help="Radio en pixeles para borrar trazos.")
    parser.add_argument("--preview-scale", type=float, default=0.25, help="Fracción de alto de ventana usada para la vista previa (0-0.5).")
    parser.add_argument("--brush-thickness", type=float, default=4.0, help="Grosor del pincel en pixeles de la superficie.")
    parser.add_argument("--mode-toggle-delay", type=float, default=3.0, help="Segundos antes de volver a inactivo tras repetir gesto de dibujar/borrar.")
    parser.add_argument("--no-flip", action="store_true", help="Desactiva las vistas previas en espejo.")
    parser.add_argument(
        "--easyocr-models",
        type=Path,
        default=None,
        help="Carpeta con los pesos de EasyOCR (no se descargan en tiempo de ejecución).",
    )
    parser.add_argument(
        "--easyocr-cpu",
        action="store_true",
        help="Forzar EasyOCR en CPU (por defecto usa GPU si está disponible).",
    )
    return parser.parse_args(argv)


# main aplica valores por defecto, asegura calibración y arranca la app; devuelve int de salida.
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.flip_view = not args.no_flip
    if args.easyocr_models is not None:
        args.easyocr_models = args.easyocr_models.expanduser().resolve()
    args.easyocr_gpu = not args.easyocr_cpu
    if args.surface_width <= 0 or args.surface_height <= 0:
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        args.surface_width = float(max(1, screen_width))
        args.surface_height = float(max(1, screen_height))
    calibration = ensure_calibration(args)
    if calibration is None:
        return 1
    if args.calibration_only:
        print("Calibración lista. Reinicia sin --calibration-only para dibujar.")
        return 0
    return run_interactive_app(args, calibration)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
