# Este paquete expone las funciones de arranque y calibración principales.
from .app import run_interactive_app
from .calibration import CalibrationData, ensure_calibration

# __all__ controla qué símbolos se exportan en importaciones comodín.
__all__ = ["run_interactive_app", "CalibrationData", "ensure_calibration"]
