# NeuralPaint
A piece of software for you to "draw" into TVs or projected screens using only a camera to track your movements.

## Prerequisitos

- Python 3.10 o superior
- [pip](https://pip.pypa.io/)
- Paquetes Python: `opencv-contrib-python`, `numpy`, `mediapipe`, `pywin32`

```powershell
pip install opencv-contrib-python mediapipe numpy pywin32
```

## Calibración de la superficie de proyección (integrada)

1. Proyecta una diapositiva de alto contraste donde la zona útil sea un **rectángulo brillante** sobre un fondo más oscuro.
2. Lanza el modo de calibración:

	```powershell
	python src/neural_paint.py --camera 0 --calibration-only --calibration-warp
	```

	Ajusta `--camera` si utilizas una cámara diferente y, opcionalmente, `--surface-width` y `--surface-height` para definir el lienzo lógico. Usa `--calibration-debug` para ver el mapa de bordes si necesitas afinar parámetros. La vista se muestra espejada por defecto; agrega `--no-flip` si prefieres la perspectiva directa de la cámara.
3. Cuando el contorno verde coincida con la superficie y aparezca “Surface detected”, presiona `s` para guardar `calibration/homography.npz`. Presiona `q` para salir sin guardar.

El archivo guardado contiene la homografía y los metadatos necesarios para mapear la posición de la mano al plano de dibujo.

## Sesión interactiva de dibujo

1. Asegúrate de haber generado `calibration/homography.npz` (si no existe, el programa iniciará la calibración automáticamente).
2. Ejecuta la aplicación principal (crea un overlay transparente encima de tu escritorio):

	```powershell
	python src/neural_paint.py --camera 0
	```

	Ajusta `--camera` o los umbrales `--min-detection`, `--min-tracking` según tu hardware. Usa `--no-flip` si no deseas espejar el recuadro de cámara.
3. Interfaz y gestos (mano derecha para apuntar, brazo izquierdo para comandos):
	- La ventana principal muestra la superficie rectificada a pantalla completa y un recuadro con la cámara en la esquina superior izquierda.
	- **Modo Dibujar**: brazo izquierdo horizontal (aprox. a la altura del hombro) con el antebrazo apuntando hacia arriba. El modo persiste hasta cambiar de gesto.
	- **Modo Borrar**: brazo izquierdo horizontal con el antebrazo apuntando hacia abajo. El puntero rojo actúa como una goma selectiva y elimina solo los segmentos que atraviesa dentro del radio configurado (`--erase-radius`).
	- **Limpiar todo**: brazo izquierdo completamente extendido hacia arriba. Borra todos los trazos y regresa a modo inactivo.
	- **Selector de color**: brazo izquierdo totalmente extendido en horizontal (antebrazo alineado con el suelo). Aparece una rueda de 10 colores sobre la imagen de la cámara; coloca la mano derecha encima del color deseado y mantenla 3 s para aplicarlo al pincel.
	- El puntero se apoya en la punta del índice derecho; si no se distingue, recurre a la muñeca.
4. Atajos de teclado: `q` salir, `c` limpiar el lienzo actual.

Parámetros adicionales útiles: `--command-hold-frames` (frames consecutivos necesarios para aceptar un gesto), `--erase-radius` (radio de borrado en píxeles), `--preview-scale` (tamaño relativo del recuadro de cámara) y `--brush-thickness` (ancho del trazo en píxeles del lienzo lógico).

El programa crea un overlay transparente encima de tu escritorio; cierra con `q` desde la consola.
