# Reconocimiento OCR asincrónico de regiones seleccionadas.
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from easyocr import Reader


# RegionRequest describe un análisis pendiente con metadata de superficie.
@dataclass
class RegionRequest:

    image: np.ndarray
    surface_origin: Tuple[int, int]
    surface_extent: Tuple[int, int]
    requested_at: float
    coordinate_space: str = "surface"


# RecognizedItem representa un elemento detectado con geometría y confianza.
@dataclass
class RecognizedItem:

    polygon: np.ndarray
    text: str
    confidence: float
    role: str
    color_bgr: Tuple[int, int, int]


# RecognitionResult encapsula el resultado devuelto por el trabajador OCR.
@dataclass
class RecognitionResult:

    kind: str
    items: List[RecognizedItem]
    surface_origin: Tuple[int, int]
    surface_extent: Tuple[int, int]
    requested_at: float
    completed_at: float
    error: Optional[str] = None
    coordinate_space: str = "surface"

    @property
    def has_items(self) -> bool:
        # has_items indica si se detectó al menos un elemento.
        return bool(self.items)


# RegionAnalyzer gestiona la cola OCR y resalta fórmulas de manera asincrónica.
class RegionAnalyzer:

    # ROLE_COLORS relaciona cada rol con un color BGR para resaltar.
    ROLE_COLORS = {
        "text": (40, 180, 255),
        "operator": (64, 128, 255),
        "variable": (162, 89, 255),
        "number": (0, 200, 140),
        "default": (255, 200, 60),
    }

    # FORMULA_INDICATORS agrupa símbolos que revelan contenido matemático.
    FORMULA_INDICATORS = set("∑∫√π∞±=<>^*/∂∇ΣΠΔΩθλμ→←⇒⇐≤≥≠≈×÷·|{}[]")
    # FORMULA_KEYWORDS incluye cadenas clave para heurística de fórmulas.
    FORMULA_KEYWORDS = {
        "lim",
        "der",
        "det",
        "grad",
        "curl",
        "div",
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "log",
        "ln",
        "exp",
        "sqrt",
        "sum",
        "prod",
        "int",
        "integral",
        "matrix",
        "mat",
        "trace",
        "tr",
        "rank",
        "eig",
    }

    # __init__ configura colas, hilos y valida el uso de GPU y modelos OCR.
    def __init__(
        self,
        languages: Sequence[str] = ("en", "es"),
        queue_size: int = 1,
        use_gpu: bool = True,
        model_dir: Optional[Path] = None,
    ) -> None:
        self._languages = list(languages)
        self._requests: "queue.Queue[Optional[RegionRequest]]" = queue.Queue(maxsize=max(1, queue_size))
        self._latest_result: Optional[RecognitionResult] = None
        self._result_lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, name="RegionAnalyzerWorker", daemon=True)
        self._shutdown = threading.Event()
        self._busy = threading.Event()
        self._reader: Optional["Reader"] = None
        if use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU solicitado para EasyOCR pero CUDA no está disponible. "
                    "Ejecuta con --easyocr-cpu para forzar modo CPU."
                )
        self._use_gpu = use_gpu
        self._model_dir = Path(model_dir) if model_dir is not None else None
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        # busy indica si el hilo trabajador está atendiendo una tarea.
        return self._busy.is_set()

    # submit coloca una región BGR en la cola y descarta peticiones antiguas.
    def submit(
        self,
        patch_bgr: np.ndarray,
        surface_origin: Tuple[int, int],
        *,
        coordinate_space: str = "surface",
    ) -> None:
        if patch_bgr.size == 0:
            return
        try:
            while True:
                self._requests.get_nowait()
        except queue.Empty:
            pass

        request = RegionRequest(
            image=patch_bgr.copy(),
            surface_origin=surface_origin,
            surface_extent=(patch_bgr.shape[1], patch_bgr.shape[0]),
            requested_at=time.perf_counter(),
            coordinate_space=coordinate_space,
        )
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            # Debería evitarse tras vaciar, pero se protege por seguridad.
            discarded = self._requests.get_nowait()
            del discarded
            self._requests.put_nowait(request)

    # poll_result devuelve el reconocimiento más reciente y lo limpia.
    def poll_result(self) -> Optional[RecognitionResult]:
        with self._result_lock:
            result = self._latest_result
            self._latest_result = None
        return result

    # close marca el cierre, inserta un centinela y espera al hilo trabajador.
    def close(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Worker implementation
    # ------------------------------------------------------------------

    # _worker_loop atiende la cola hasta recibir un centinela None.
    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                request = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break
            self._busy.set()
            result = self._execute_request(request)
            self._busy.clear()
            with self._result_lock:
                self._latest_result = result

    # _execute_request ejecuta EasyOCR y crea un RecognitionResult.
    def _execute_request(self, request: RegionRequest) -> RecognitionResult:
        try:
            reader = self._ensure_reader()
            raw_items = reader.readtext(request.image, detail=1, paragraph=False)
            result = self._postprocess(raw_items, request)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[RegionAnalyzer] No se pudo analizar la región: {exc}")
            result = RecognitionResult(
                kind="error",
                items=[],
                surface_origin=request.surface_origin,
                surface_extent=request.surface_extent,
                requested_at=request.requested_at,
                completed_at=time.perf_counter(),
                error=str(exc),
                coordinate_space=request.coordinate_space,
            )
        else:
            result.requested_at = request.requested_at
            result.completed_at = time.perf_counter()
            result.coordinate_space = request.coordinate_space
        return result

    # _ensure_reader crea o reutiliza el lector EasyOCR según la configuración.
    def _ensure_reader(self) -> "Reader":
        if self._reader is not None:
            return self._reader
        model_dir_str: Optional[str] = None
        if self._model_dir is not None:
            model_dir_str = str(self._model_dir)
            self._model_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._reader = Reader(
                self._languages,
                gpu=self._use_gpu,
                model_storage_directory=model_dir_str,
                download_enabled=False,
                verbose=False,
            )
        except FileNotFoundError as err:
            location = model_dir_str or "~/.EasyOCR"
            raise RuntimeError(
                "Modelos de EasyOCR no encontrados. Coloca los pesos requeridos en "
                f"'{location}' antes de iniciar el reconocimiento."
            ) from err
        return self._reader

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    # _postprocess interpreta la salida de EasyOCR y clasifica cada elemento.
    def _postprocess(self, raw_items, request: RegionRequest) -> RecognitionResult:
        items: List[RecognizedItem] = []
        text_votes = 0
        formula_votes = 0
        for entry in raw_items:
            if len(entry) < 3:
                continue
            polygon, text, confidence = entry
            if not text:
                continue
            polygon_arr = np.array(polygon, dtype=np.float32)
            label = text.strip()
            if not label:
                continue

            role = self._classify_role(label)
            if role in ("operator", "variable", "number"):
                formula_votes += 1
            elif self._looks_like_formula(label):
                formula_votes += 1
                if role == "text":
                    role = "default"
            else:
                text_votes += 1

            color = self.ROLE_COLORS.get(role, self.ROLE_COLORS["default"])
            items.append(
                RecognizedItem(
                    polygon=polygon_arr,
                    text=label,
                    confidence=float(confidence),
                    role=role,
                    color_bgr=color,
                )
            )

        if not items:
            kind = "desconocido"
        elif formula_votes > text_votes:
            kind = "formula"
        else:
            kind = "texto"

        return RecognitionResult(
            kind=kind,
            items=items,
            surface_origin=request.surface_origin,
            surface_extent=request.surface_extent,
            requested_at=request.requested_at,
            completed_at=time.perf_counter(),
            coordinate_space=request.coordinate_space,
        )

    @staticmethod
    # _looks_like_formula evalúa si el texto parece una expresión matemática.
    def _looks_like_formula(text: str) -> bool:
        normalized = text.replace(" ", "")
        if not normalized:
            return False
        if any(char in RegionAnalyzer.FORMULA_INDICATORS for char in normalized):
            return True
        lowered = normalized.lower()
        if any(keyword in lowered for keyword in RegionAnalyzer.FORMULA_KEYWORDS):
            return True
        if re.search(r"(?:d|∂)[a-z]?/d[a-z]", lowered):  # notación de derivadas
            return True
        if re.search(r"∂/[a-z]", normalized):
            return True
        if re.search(r"\[[^\]]+\]", normalized):  # notación matricial o vectorial
            return True
        if re.search(r"\|[^|]+\|", normalized):  # determinante o norma
            return True
        if re.search(r"[A-Za-z0-9]\^[A-Za-z0-9]", normalized):  # exponente
            return True
        if re.search(r"\b(?:dx|dy|dz|dt)\b", lowered):
            return True
        if re.search(r"(?:→|←|⇒|⇐)", normalized):
            return True
        # Heurística: mezcla de letras, dígitos y operadores indica fórmula.
        has_letter = bool(re.search(r"[A-Za-z]", normalized))
        has_digit = bool(re.search(r"\d", normalized))
        has_operator = bool(re.search(r"[=+\-*/^]", normalized))
        return (has_letter and has_digit) or (has_letter and has_operator) or (has_digit and has_operator)

    @staticmethod
    # _classify_role devuelve el rol semántico (text, operator, number, variable).
    def _classify_role(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "default"
        if re.fullmatch(r"[=+\-*/^()\[\]{}:,.;]+", stripped):
            return "operator"
        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", stripped):
            return "number"
        if re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", stripped):
            return "variable"
        return "text"


# Note: segmentation/Model invocation moved to src/neuralpaint/segmentation.py
