Generador sintético de caracteres

Archivos:
- `generate_synthetic_dataset.py`: renderiza caracteres usando las fuentes en `assets/fonts/extracted` y guarda pares `images` + `masks` en `datasets/synthetic`.
- `requirements.txt`: dependencias mínimas.

Uso rápido:

```powershell
python generate_synthetic_dataset.py --fonts "Reconocimiento de Caracteres/assets/fonts/extracted" --out "Reconocimiento de Caracteres/datasets/synthetic" --samples 3 --size 128
```

Salida:
- `images/` y `masks/` (PNGs)
- `metadata.json` con lista de pares y metadatos.
