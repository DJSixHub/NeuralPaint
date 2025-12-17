# NeuralPaint
Herramienta para dibujar sobre televisores o pantallas proyectadas usando solo una cámara para seguir tus manos.

## Instalación rápida

- Python 3.10 o superior
- [pip](https://pip.pypa.io/) actualizado
- Recomendado: crear y activar un entorno virtual (`python -m venv .venv`, luego `.\.venv\Scripts\Activate.ps1`)

Instala las dependencias esenciales:

```powershell
pip install opencv-contrib-python mediapipe numpy pywin32 easyocr torch torchvision torchaudio
```

> Si tu GPU no soporta CUDA, omite los paquetes de Torch o usa una versión solo CPU.

## Uso rápido del programa

1. **Calibrar la superficie** (solo la primera vez o cuando cambies la geometría):

    ```powershell
    python src/neural_paint.py --camera 0 --calibration-only --calibration-warp
    ```

    - Ajusta `--camera` si usas otra entrada.
    - El modo `--calibration-warp` muestra la superficie rectificada para verificar el encuadre.
    - Agrega `--calibration-debug` para visualizar bordes útiles al ajustar luces.
    - Al iniciar se solicitará el modo (contorno brillante o rejilla AprilTag). Puedes forzarlo con `--calibration-mode contour` o `--calibration-mode apriltag`.
    - Guarda la homografía con `s`; cancela con `q`. El archivo se almacena en `calibration/homography.npz`.

2. **Iniciar el modo de dibujo** una vez calibrado:

    ```powershell
    python src/neural_paint.py --camera 0
    ```

    - Usa `--no-flip` si no deseas que la vista previa se muestre en espejo.
    - Los parámetros `--min-detection`, `--min-tracking` y `--smoothing` te permiten afinar la estabilidad del puntero.

3. **Gestos y controles principales** (mano derecha para apuntar, brazo izquierdo para comandos):
    - **Modo dibujar**: brazo izquierdo horizontal con antebrazo apuntando arriba.
    - **Modo borrar**: brazo izquierdo horizontal con antebrazo apuntando abajo (borrado selectivo dentro del radio `--erase-radius`).
    - **Limpiar todo**: brazo izquierdo extendido verticalmente hacia arriba.
    - **Selector de color**: brazo izquierdo horizontal y estable; aparecerá la rueda de colores. Mantén la mano derecha sobre un color 3 s para seleccionarlo.
    - **Teclado**: `q` salir, `c` limpiar lienzo, `r` forzar reconocimiento OCR en la región del puntero.
    - El gesto repetido para dibujar/borrar devuelve a modo inactivo tras el tiempo `--mode-toggle-delay` (por defecto 3 s).

4. **Opciones útiles del CLI** (se pueden combinar):
    - `--surface-width` y `--surface-height`: fuerzan dimensiones lógicas del lienzo.
    - `--preview-scale`: escala la ventana de cámara (0.05 a 0.5).
    - `--command-hold-frames`: cuadros consecutivos para aceptar gestos (sube si hay falsos positivos).
    - `--easyocr-cpu`: fuerza OCR en CPU; combinado con `--easyocr-models` para indicar la carpeta de modelos.

## Pipeline interno y componentes

1. **Captura de cámara**: OpenCV obtiene cuadros BGR del dispositivo elegido y los convierte a RGB para MediaPipe Holistic.
2. **Detección corporal**: MediaPipe proporciona landmarks de pose y manos. Se extrae la punta del índice derecho como puntero y el brazo izquierdo para clasificar gestos.
3. **Calibración y proyección**: la homografía guardada convierte el puntero de coordenadas de cámara a coordenadas del lienzo virtual. Si no existe calibración, se lanza automáticamente el asistente (contorno brillante o AprilTags).
4. **Suavizado y validación**: la posición del puntero se filtra con un suavizado exponencial (`--smoothing`) y se comprueba si cae dentro de la superficie.
5. **Gestión de modos**: se interpretan los gestos (dibujar, borrar, limpiar, selector de color) mediante el clasificador de brazo. El modo determina si se agregan puntos a un trazo, se borra un área o se muestra la rueda de colores.
6. **Lienzo y trazos**: `StrokeCanvas` conserva la lista de trazos, dibuja nuevas líneas y aplica borrado preciso mediante intersecciones segmento-círculo.
7. **Overlay en pantalla completa**: con pywin32 se crea una ventana transparente que combina el lienzo, el puntero, mensajes de estado y una vista previa de cámara en la esquina superior izquierda.
8. **Reconocimiento OCR opcional**: al presionar `r`, un segmento del lienzo se envía a EasyOCR en segundo plano. El resultado se clasifica entre texto o fórmula y se colorean los elementos detectados.
9. **Retroalimentación visual**: la ventana de calibración y el overlay muestran mensajes en español, incluyendo instrucciones para guardar, estados de reconocimiento y recordatorios de atajos.

Comprender este flujo ayuda a extender el proyecto: por ejemplo, puedes añadir nuevos gestos dentro del clasificador, introducir otros motores de reconocimiento sustituyendo el `RegionAnalyzer`, o cambiar la lógica de overlay reutilizando la homografía calculada.
