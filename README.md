# NeuralPaint

Herramienta de dibujo interactivo que usa solo una cámara para seguir tus manos y proyectar sobre pantallas. Incluye reconocimiento de caracteres mediante redes neuronales U-Net para segmentación avanzada.

## Instalación rápida

### Requisitos base
- **Python 3.10 o superior**
- **pip** actualizado
- **Sistema operativo**: Windows (para overlay con `pywin32`)

### Instalación de dependencias

1. **Crear entorno virtual** (recomendado):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias principales**:
   ```powershell
   pip install opencv-contrib-python mediapipe numpy pywin32 easyocr torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   > **Nota GPU**: Si tu GPU no soporta CUDA o prefieres CPU, instala PyTorch sin CUDA:
   > ```powershell
   > pip install opencv-contrib-python mediapipe numpy pywin32 easyocr torch torchvision torchaudio
   > ```

3. **Verificar instalación**:
   ```powershell
   python src/neural_paint.py --help
   ```

## Uso rápido

### Primera vez: Calibración

Antes de usar NeuralPaint, debes calibrar la superficie de proyección. Existen dos métodos:

#### Método 1: Contorno brillante (recomendado para principiantes)

1. **Prepara tu superficie**: proyecta un rectángulo blanco en pantalla completa o usa un papel/cartulina blanco sobre fondo oscuro.

2. **Ejecuta calibración**:
   ```powershell
   python src/neural_paint.py --camera 0 --calibration-only --calibration-mode contour --calibration-warp
   ```

3. **Durante la calibración**:
   - Ajusta la cámara para que el rectángulo brillante sea claramente visible
   - Usa `--calibration-debug` si necesitas ver los bordes detectados
   - Presiona `s` para guardar cuando veas el contorno verde correcto
   - Presiona `q` para cancelar

#### Método 2: Rejilla AprilTag (más preciso)

1. **Imprime o proyecta** una rejilla de AprilTags (disponible en `assets/` o genera una con bibliotecas AprilTag estándar)

2. **Ejecuta calibración**:
   ```powershell
   python src/neural_paint.py --camera 0 --calibration-only --calibration-mode apriltag
   ```

3. **Requisitos**:
   - La rejilla debe tener al menos 4 tags visibles
   - Colócala alineada con la superficie de proyección
   - Presiona `s` para guardar cuando se detecten suficientes tags

#### Opciones útiles de calibración

- `--calibration-warp`: Muestra vista previa de la superficie rectificada
- `--calibration-debug`: Visualiza mapa de bordes (útil con iluminación difícil)
- `--force-calibrate`: Fuerza nueva calibración aunque ya exista una guardada
- `--calibration-min-area 0.15`: Ajusta área mínima del contorno (0-1)

La calibración se guarda en `calibration/homography.npz` y se carga automáticamente en siguientes sesiones.

### Uso normal (después de calibrar)

```powershell
python src/neural_paint.py --camera 0
```

## Uso con dos pantallas

NeuralPaint soporta configuraciones de doble monitor donde:
- **Monitor principal**: muestra UI completa (vista de cámara, estado, cursor)
- **Proyector/monitor secundario**: muestra solo el canvas con dibujos (sin UI)

### Configuración básica

1. **Configura Windows en modo extendido**:
   - Presiona `Win + P` → selecciona "Extender"
   - Abre Configuración de pantalla (`Win + I` → Sistema → Pantalla)
   - Arrastra y ordena los monitores según tu setup físico
   - Anota qué número tiene cada monitor (1, 2, etc.)

2. **Lanza con dual-monitor**:
   ```powershell
   python src/neural_paint.py --camera 0 --dual-monitor
   ```

   Por defecto usa:
   - Monitor 0: UI principal
   - Monitor 1: Proyector

3. **Personalizar monitores**:
   ```powershell
   python src/neural_paint.py --camera 0 --dual-monitor --main-monitor 1 --projector-monitor 0
   ```

### Ejemplos de configuración

**Setup 1: Monitor izquierdo = principal, derecho = proyector**
```powershell
python src/neural_paint.py --camera 0 --dual-monitor
```

**Setup 2: Tres monitores (centro = principal, derecha = proyector)**
```powershell
python src/neural_paint.py --camera 0 --dual-monitor --main-monitor 1 --projector-monitor 2
```

**Setup 3: Calibración + dual-monitor**
```powershell
# Primero calibra (usa single-monitor automáticamente)
python src/neural_paint.py --camera 0 --calibration-only --calibration-mode apriltag

# Luego ejecuta con dual-monitor
python src/neural_paint.py --camera 0 --dual-monitor
```

### Qué muestra cada pantalla

**Monitor principal**:
- ✅ Vista previa de cámara (esquina superior izquierda)
- ✅ Canvas con dibujos
- ✅ Cursor/puntero de mano
- ✅ Mensajes de estado (modo actual, instrucciones)
- ✅ Resultados de OCR y segmentación

**Proyector/monitor secundario**:
- ✅ Canvas con dibujos
- ✅ Máscaras y efectos de segmentación aplicados
- ❌ Sin vista de cámara
- ❌ Sin cursor
- ❌ Sin texto de estado

> **Consejo**: El proyector muestra una versión limpia ideal para presentaciones y arte en vivo.

## Gestos y controles

NeuralPaint usa **MediaPipe Holistic** para detectar pose y manos. Los gestos se controlan con:
- **Mano derecha**: Puntero (índice) para dibujar/interactuar
- **Brazo izquierdo**: Comandos de modo

### Gestos del brazo izquierdo

| Gesto | Descripción | Acción |
|-------|-------------|--------|
| **Brazo horizontal, antebrazo arriba** | ![](assets/draw.png) | Activa modo **DIBUJAR** |
| **Brazo horizontal, antebrazo abajo** | ![](assets/erase.png) | Activa modo **BORRAR** (radio configurable con `--erase-radius`) |
| **Brazo horizontal, antebrazo horizontal** | ![](assets/color.png) | Abre selector de **COLORES** (mantén mano derecha 3s sobre un color) |
| **Brazo extendido verticalmente arriba** | ![](assets/clear.png) | **LIMPIA TODO** el canvas |
| **Ambos índices arriba** | ![](assets/region.png) | Entra a modo **SELECCIÓN DE REGIÓN** (para OCR/segmentación) |

### Flujo de selección de región (OCR/Segmentación)

1. **Activar**: Levanta ambos índices → modo `REGION_SELECT`
2. **Anclar esquina**: Cierra una mano en puño → se fija la esquina inicial
3. **Ajustar tamaño**: Mueve la otra mano para dimensionar el rectángulo
4. **Confirmar**: Levanta el índice de la **misma mano que hizo el puño** → se captura y procesa

El sistema:
- Captura la región de pantalla en las coordenadas seleccionadas
- La envía a la red U-Net de segmentación (`models/checkpoint_epoch_70.pth`)
- Produce una máscara de caracteres/trazos
- La superpone coloreada en el canvas (verde para texto, rojo para ruido)

### Atajos de teclado

| Tecla | Acción |
|-------|--------|
| `q` | Salir de la aplicación |
| `c` | Limpiar todo el canvas (igual que gesto de brazo arriba) |
| `r` | Forzar reconocimiento OCR en región del puntero |

### Parámetros configurables

```powershell
python src/neural_paint.py --camera 0 \
  --brush-thickness 6.0 \             # Grosor del trazo (px)
  --erase-radius 50.0 \               # Radio de borrado (px)
  --command-hold-frames 8 \           # Cuadros para confirmar gesto (evita falsos positivos)
  --mode-toggle-delay 4.0 \           # Segundos para volver a inactivo tras gesto repetido
  --smoothing 0.7 \                   # Suavizado del puntero (0-1, mayor = más suave)
  --min-detection 0.7 \               # Umbral de confianza MediaPipe
  --preview-scale 0.3                 # Escala de vista previa (0-0.5)
```

## Cómo funciona: Pipeline interno

NeuralPaint integra visión por computadora, procesamiento de gestos y redes neuronales en un flujo en tiempo real:

### 1. Captura y detección
- **OpenCV** captura frames BGR del dispositivo (`--camera`)
- Se convierte a RGB para **MediaPipe Holistic**
- MediaPipe devuelve landmarks de:
  - Pose (33 puntos): cuerpo completo, brazos
  - Manos (21 puntos × 2): izquierda y derecha

### 2. Calibración y proyección
- **Homografía guardada** (`calibration/homography.npz`) mapea cámara → superficie virtual
- Se calculó previamente con:
  - **Contorno brillante**: detecta rectángulo blanco, ajusta 4 esquinas
  - **AprilTags**: detecta rejilla de marcadores fiduciales, calcula homografía precisa
- Cada landmark se proyecta con `cv2.perspectiveTransform()` a coordenadas del canvas

### 3. Extracción del puntero
- **Puntero primario**: punta del índice derecho (landmark 8)
- **Fallback**: muñeca derecha si no hay mano detectada (pose landmark 16)
- Se aplica **suavizado exponencial** (`--smoothing`) para estabilizar jitter

### 4. Clasificación de gestos
- **`classify_left_arm_command()`**: analiza geometría del brazo izquierdo (hombro-codo-muñeca)
  - Brazo horizontal + antebrazo arriba → `DRAW_MODE`
  - Brazo horizontal + antebrazo abajo → `ERASE_MODE`
  - Brazo horizontal + antebrazo horizontal → `COLOR_PICKER`
  - Brazo vertical arriba → `CLEAR_ALL`
- **`ArmGestureClassifier`**: acumula `--command-hold-frames` consecutivos del mismo gesto para confirmar (evita falsos positivos)
- **Selección de región**: `both_index_fingers_up()` detecta ambos índices levantados
  - `SelectionGestureTracker` maneja el flujo: ancla puño → arrastra → confirma con índice

### 5. Gestión de modos e interacción
- **Estado global**: `InteractionMode` (IDLE, DRAW, ERASE, COLOR_SELECT, REGION_SELECT)
- **Transiciones**:
  - Gesto confirmado → cambia modo
  - Gesto repetido en modo activo → vuelve a IDLE tras `--mode-toggle-delay` segundos
- **Modo DRAW**: agrega puntos al trazo activo; finaliza al salir del modo
- **Modo ERASE**: itera trazos y borra segmentos que intersectan círculo de radio `--erase-radius`
- **Modo COLOR_SELECT**: muestra `ColorPicker` (rueda HSV); detecta hover 3s → cambia color activo

### 6. Canvas y renderizado
- **`StrokeCanvas`**: lista de trazos (secuencias de puntos `(x,y)` + color)
- Renderiza con `cv2.polylines()` usando grosor `--brush-thickness`
- Límite de `--max-strokes` (por defecto 200); descarta los más antiguos al exceder
- Canvas es RGB de tamaño `surface_width × surface_height`

### 7. Segmentación y OCR
- **Captura de región**:
  - Modo REGION_SELECT → usuario define rectángulo
  - Al confirmar: captura píxeles de pantalla con Win32 API (`OverlayWindow.capture_screen_region()`)
  - Se escribe como PNG temporal en `masc_produced/input_*.png`
- **Red U-Net** (`Segmenter`):
  - Modelo: `models/checkpoint_epoch_70.pth` (entrenado en Stage 1 con BCE + edge loss)
  - Input: región capturada (256×256 normalizado)
  - Output: máscara binaria (probabilidad de texto/trazo)
  - Genera: `*_mask.png`, `*_overlay.png` (coloreado)
- **Clasificación** (`RegionAnalyzer`):
  - Detecta si es texto legible o solo trazos
  - Aplica efecto coloreado (verde=texto, rojo=ruido, azul=matemáticas)
  - Se superpone en el canvas en las coordenadas originales

### 8. Overlay en pantalla completa
- **`OverlayWindow`** (requiere `pywin32` en Windows):
  - Crea ventana transparente (WS_EX_LAYERED, WS_EX_TRANSPARENT)
  - Usa `SetLayeredWindowAttributes()` para hacer fondo transparente
  - Renderiza sobre el escritorio:
    - Canvas con trazos
    - Cursor circular (color actual + radio erase si aplica)
    - Vista previa de cámara (esquina superior izquierda, `--preview-scale`)
    - Banner de estado (modo actual, mensajes OCR, instrucciones)
- **Dual-monitor**:
  - Monitor principal: overlay completo (UI + canvas)
  - Proyector: overlay limpio (solo canvas + máscaras aplicadas, sin UI)

### 9. Retroalimentación visual
- **Mensajes de estado**:
  - "Calibrando..." / "Guardado con 's', Cancelar con 'q'" (durante calibración)
  - "MODO: DIBUJAR / BORRAR / COLORES / SELECCIÓN" (en overlay)
  - "Reconociendo región..." / "Texto detectado: X caracteres" (post-OCR)
- **Indicadores**:
  - Cursor: círculo relleno del color activo (o rojo con radio en modo ERASE)
  - Rectángulo de selección: borde amarillo animado durante ajuste
  - Máscaras aplicadas: overlays coloreados sobre el canvas

### 10. Flujo de actualización (loop principal)

```
mientras running:
  1. Capturar frame de cámara
  2. Procesar con MediaPipe → landmarks
  3. Extraer puntero derecho + clasificar brazo izquierdo
  4. Proyectar puntero a superficie con homografía
  5. Actualizar clasificador de gestos → comando confirmado
  6. Gestionar transiciones de modo según comando
  7. Ejecutar lógica del modo activo:
     - DRAW: agregar puntos al trazo
     - ERASE: borrar trazos en radio
     - COLOR_SELECT: detectar hover en rueda
     - REGION_SELECT: manejar ancla + resize + confirmación
  8. Renderizar canvas + overlay
  9. Actualizar ventana(s) overlay (principal + proyector si dual-monitor)
  10. Manejar input de teclado ('q', 'c', 'r')
```

### Componentes clave del código

| Archivo | Responsabilidad |
|---------|----------------|
| `src/neural_paint.py` | Entry point CLI, parsing args, lanza calibración + app |
| `src/neuralpaint/app.py` | Loop principal, orquesta MediaPipe, gestos, canvas, overlay, OCR |
| `src/neuralpaint/calibration.py` | Detección de contorno/AprilTags, cálculo de homografía |
| `src/neuralpaint/gestures.py` | Clasificación de comandos de brazo, detección de dedos, SelectionGestureTracker |
| `src/neuralpaint/strokes.py` | `StrokeCanvas`: gestión de trazos, renderizado, borrado |
| `src/neuralpaint/overlay.py` | `OverlayWindow`: ventana transparente Win32, captura de pantalla, dual-monitor |
| `src/neuralpaint/segmentation.py` | `Segmenter`: wrapper de red U-Net, invoca subprocess con modelo checkpoint |
| `src/neuralpaint/recognition.py` | `RegionAnalyzer`: wrapper de EasyOCR, clasifica texto vs fórmulas |
| `src/neuralpaint/color_picker.py` | `ColorPicker`: rueda HSV, detección de hover, selección de color |
| `Reconocimiento de Caracteres/scripts/inference/testing.py` | Script de segmentación U-Net (llamado por `Segmenter`) |

### Archivos de modelos y calibración

- **`calibration/homography.npz`**: matriz 3×3 de homografía cámara→superficie
- **`models/checkpoint_epoch_70.pth`**: pesos de U-Net entrenada (Stage 1: BCE + edge loss)
- **`models/easyocr/`**: modelos de EasyOCR (CRAFT, reconocimiento de texto)
- **`masc_produced/`**: inputs/outputs temporales de segmentación (debug)

## Extensibilidad

### Añadir nuevos gestos
1. Edita `classify_left_arm_command()` en `gestures.py`
2. Añade un nuevo `CommandType` al enum
3. Maneja el comando en `app.py` dentro del loop principal

### Cambiar motor de OCR
1. Reemplaza `RegionAnalyzer` en `recognition.py`
2. Mantén la interfaz: `analyze_region(image_path) → RecognitionResult`

### Modificar red de segmentación
1. Entrena nuevo modelo con `Reconocimiento de Caracteres/scripts/training/train_segmentation_clean.py`
2. Actualiza ruta en `Segmenter.__init__()` (`segmentation.py`)
3. Asegura compatibilidad de entrada 256×256 y salida de máscara

### Personalizar overlay
1. `overlay.py`: modifica `render_projector_overlay()` para proyector
2. `app.py`: ajusta `draw_status_banner()` para mensajes custom
3. Cambia colores en `StrokeCanvas.render()` (`strokes.py`)

---

**Documentación completa**: Ver `.github/copilot-instructions.md` y `DUAL_MONITOR.md` para detalles internos.

**Entrenamiento de modelos**: Ver `Reconocimiento de Caracteres/README.md` y `STRUCTURE.md` para pipeline de datos y HPO.
