# Ventana de superposición transparente sobre Windows para mostrar el lienzo.
from __future__ import annotations

import ctypes
import importlib.util
from typing import Optional, Sequence

import cv2
import numpy as np

if importlib.util.find_spec("win32api") is not None:
    import win32api  # type: ignore
    import win32con  # type: ignore
    import win32gui  # type: ignore
    import win32ui  # type: ignore
else:  # pragma: no cover - pywin32 ausente
    win32api = win32con = win32gui = win32ui = None


# HAS_OVERLAY_SUPPORT indica si pywin32 está disponible para crear la ventana.
HAS_OVERLAY_SUPPORT = all(module is not None for module in (win32api, win32con, win32gui, win32ui))


# POINT define una estructura Win32 con coordenadas enteras.
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


# SIZE define un tamaño ancho/alto para llamadas Win32.
class SIZE(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]


# BLENDFUNCTION controla la operación alpha blending de GDI.
class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat", ctypes.c_byte),
    ]


if HAS_OVERLAY_SUPPORT:
    # BITMAPINFOHEADER describe el formato de un bitmap compatible con GDI.
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]


    # RGBQUAD almacena componentes de color para paletas.
    class RGBQUAD(ctypes.Structure):
        _fields_ = [
            ("rgbBlue", ctypes.c_byte),
            ("rgbGreen", ctypes.c_byte),
            ("rgbRed", ctypes.c_byte),
            ("rgbReserved", ctypes.c_byte),
        ]


    # BITMAPINFO agrupa cabecera y colores para CreateDIBSection.
    class BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", BITMAPINFOHEADER),
            ("bmiColors", RGBQUAD * 1),
        ]


# OverlayWindow genera una ventana siempre visible con soporte BGRA y alpha.
class OverlayWindow:

    _class_atom: Optional[int] = None

    def __init__(self, width: int, height: int, title: str = "Superposicion NeuralPaint") -> None:
        if not HAS_OVERLAY_SUPPORT:
            raise RuntimeError(
                "pywin32 es obligatorio para renderizar la superposición. Instálalo con 'pip install pywin32'."
            )

        self.width = int(width)
        self.height = int(height)
        self.title = title
        self._hinstance = win32api.GetModuleHandle(None)
        self._bitmap_info = BITMAPINFO()
        header = self._bitmap_info.bmiHeader
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = self.width
        header.biHeight = self.height
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = win32con.BI_RGB
        header.biSizeImage = 0
        header.biXPelsPerMeter = 0
        header.biYPelsPerMeter = 0
        header.biClrUsed = 0
        header.biClrImportant = 0

        if OverlayWindow._class_atom is None:
            wnd_class = win32gui.WNDCLASS()
            wnd_class.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
            wnd_class.lpfnWndProc = self._wnd_proc
            wnd_class.hInstance = self._hinstance
            wnd_class.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)
            wnd_class.lpszClassName = "NeuralPaintOverlayClass"
            OverlayWindow._class_atom = win32gui.RegisterClass(wnd_class)

        ex_style = (
            win32con.WS_EX_LAYERED
            | win32con.WS_EX_TOPMOST
            | win32con.WS_EX_TOOLWINDOW
            | win32con.WS_EX_TRANSPARENT
        )
        style = win32con.WS_POPUP

        self.hwnd = win32gui.CreateWindowEx(
            ex_style,
            OverlayWindow._class_atom,
            self.title,
            style,
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self._hinstance,
            None,
        )

        win32gui.SetWindowPos(
            self.hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            self.width,
            self.height,
            win32con.SWP_SHOWWINDOW,
        )

        self.screen_dc = win32gui.GetDC(0)
        self.mfc_dc = win32ui.CreateDCFromHandle(self.screen_dc)
        self.mem_dc = self.mfc_dc.CreateCompatibleDC()

        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        self._closed = False

    def set_visible(self, visible: bool) -> None:
        if self._closed:
            return
        cmd = win32con.SW_SHOWNA if visible else win32con.SW_HIDE
        win32gui.ShowWindow(self.hwnd, cmd)
        if visible:
            win32gui.SetWindowPos(
                self.hwnd,
                win32con.HWND_TOPMOST,
                0,
                0,
                self.width,
                self.height,
                win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
            )

    @staticmethod
    def _wnd_proc(hwnd, msg, wparam, lparam):  # pragma: no cover - GUI message loop
        if msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    # update recibe un frame BGRA (alto, ancho, 4) y lo envía a la ventana.
    def update(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width or frame.shape[2] != 4:
            raise ValueError("El frame debe coincidir en tamaño con la ventana y ser BGRA (H, W, 4)")

        # Convertimos a alpha pre-multiplicado e invertimos verticalmente para GDI.
        premultiplied = frame.astype(np.uint8)
        alpha = premultiplied[..., 3:4].astype(np.float32) / 255.0
        premultiplied[..., :3] = (premultiplied[..., :3].astype(np.float32) * alpha).astype(np.uint8)
        dib_data = np.ascontiguousarray(premultiplied[::-1])  # orientación inferior a superior
        bitmap = self._create_bitmap_from_buffer(dib_data)
        old_bitmap = self.mem_dc.SelectObject(bitmap)

        size = SIZE(self.width, self.height)
        position = POINT(0, 0)
        source = POINT(0, 0)
        blend = BLENDFUNCTION(win32con.AC_SRC_OVER, 0, 255, win32con.AC_SRC_ALPHA)

        ctypes.windll.user32.UpdateLayeredWindow(
            self.hwnd,
            self.screen_dc,
            ctypes.byref(position),
            ctypes.byref(size),
            self.mem_dc.GetSafeHdc(),
            ctypes.byref(source),
            0,
            ctypes.byref(blend),
            win32con.ULW_ALPHA,
        )

        self.mem_dc.SelectObject(old_bitmap)
        win32gui.DeleteObject(bitmap.GetHandle())

    # pump_messages vacía la cola de eventos de la ventana.
    def pump_messages(self) -> None:
        while win32gui.PumpWaitingMessages():  # pragma: no branch - procesa eventos
            pass

    # capture_region toma una captura BGR del escritorio en las coordenadas indicadas.
    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        # Get actual desktop dimensions for validation
        desktop_width = ctypes.windll.user32.GetSystemMetrics(0)
        desktop_height = ctypes.windll.user32.GetSystemMetrics(1)
        print(f"[CAP] desktop={desktop_width}x{desktop_height} request=({x},{y},{width},{height})")
        
        if width <= 0 or height <= 0:
            print(f"[CAP] invalid dimensions: width={width} height={height}")
            return np.zeros((0, 0, 3), dtype=np.uint8)
        
        # Validate and clamp coordinates to actual desktop bounds
        if x < 0 or y < 0 or x >= desktop_width or y >= desktop_height:
            print(f"[CAP] coordinates out of bounds: x={x} y={y} vs desktop {desktop_width}x{desktop_height}")
            return np.zeros((0, 0, 3), dtype=np.uint8)
        
        # Clamp width/height to stay within desktop
        if x + width > desktop_width:
            width = desktop_width - x
            print(f"[CAP] clamped width to {width}")
        if y + height > desktop_height:
            height = desktop_height - y
            print(f"[CAP] clamped height to {height}")
        
        if width <= 0 or height <= 0:
            print(f"[CAP] invalid dimensions after clamping")
            return np.zeros((0, 0, 3), dtype=np.uint8)
        
        screen_dc = None
        capture_dc = None
        mem_dc = None
        bitmap = None
        try:
            # Acquire a fresh desktop DC every capture. Reusing a cached DC can become unreliable
            # after hide/show cycles and can lead to empty captures on subsequent attempts.
            screen_dc = win32gui.GetDC(0)
            if not screen_dc:
                print("[CAP] GetDC(0) returned null")
                return np.zeros((0, 0, 3), dtype=np.uint8)
            print(f"[CAP] screen_dc={screen_dc}")
            
            capture_dc = win32ui.CreateDCFromHandle(screen_dc)
            if not capture_dc:
                print("[CAP] CreateDCFromHandle failed")
                return np.zeros((0, 0, 3), dtype=np.uint8)
            print(f"[CAP] capture_dc created")
            
            mem_dc = capture_dc.CreateCompatibleDC()
            if not mem_dc:
                print("[CAP] CreateCompatibleDC failed")
                return np.zeros((0, 0, 3), dtype=np.uint8)
            print(f"[CAP] mem_dc created")
            
            # Create bitmap - must use GetSafeHdc() to get the raw HDC for Win32 GDI calls
            bitmap = win32ui.CreateBitmap()
            if not bitmap:
                print("[CAP] CreateBitmap failed")
                return np.zeros((0, 0, 3), dtype=np.uint8)
            
            # CreateCompatibleBitmap needs a valid HDC handle, not the Python DC wrapper
            try:
                bitmap.CreateCompatibleBitmap(capture_dc, width, height)
                print(f"[CAP] CreateCompatibleBitmap succeeded for {width}x{height}")
            except Exception as e:
                print(f"[CAP] CreateCompatibleBitmap exception: {e}")
                return np.zeros((0, 0, 3), dtype=np.uint8)
            
            old_obj = mem_dc.SelectObject(bitmap)
            print(f"[CAP] SelectObject done")
            
            # BitBlt using direct Win32 API via ctypes (pywin32's wrapper returns None unreliably)
            # Get raw HDC handles for the ctypes call
            h_mem_dc = mem_dc.GetSafeHdc()
            h_capture_dc = capture_dc.GetSafeHdc()
            
            # Call BitBlt directly via ctypes: BOOL BitBlt(HDC hdc, int x, int y, int cx, int cy, HDC hdcSrc, int x1, int y1, DWORD rop)
            result = ctypes.windll.gdi32.BitBlt(
                h_mem_dc,      # destination DC
                0, 0,          # destination x, y
                width, height, # width, height
                h_capture_dc,  # source DC
                x, y,          # source x, y
                win32con.SRCCOPY  # raster operation
            )
            print(f"[CAP] BitBlt (ctypes) result={result}")
            
            if not result:
                print(f"[CAP] BitBlt failed")
                return np.zeros((0, 0, 3), dtype=np.uint8)
            
            print(f"[CAP] BitBlt succeeded, reading bitmap bits...")
            
            bits = bitmap.GetBitmapBits(True)
            img = np.frombuffer(bits, dtype=np.uint8)
            if img.size != width * height * 4:
                print(f"[CAP] bitmap size mismatch: got={img.size} expected={width * height * 4}")
                return np.zeros((0, 0, 3), dtype=np.uint8)

            img = img.reshape((height, width, 4))
            bgr = img[..., :3]
            return np.ascontiguousarray(bgr)
        except Exception as e:
            print(f"[CAP] exception: {e}")
            return np.zeros((0, 0, 3), dtype=np.uint8)
        finally:
            if screen_dc is not None:
                try:
                    win32gui.ReleaseDC(0, screen_dc)
                except Exception:
                    pass
            if bitmap is not None:
                try:
                    handle = bitmap.GetHandle()
                except Exception:
                    handle = None
                if handle:
                    try:
                        win32gui.DeleteObject(handle)
                    except Exception:
                        pass
            if mem_dc is not None:
                try:
                    mem_dc.DeleteDC()
                except Exception:
                    pass
            if capture_dc is not None:
                try:
                    capture_dc.DeleteDC()
                except Exception:
                    pass

    # close libera los recursos Win32 y destruye la ventana.
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        win32gui.DestroyWindow(self.hwnd)
        try:
            self.mem_dc.DeleteDC()
        except Exception:
            pass
        try:
            self.mfc_dc.DeleteDC()
        except Exception:
            pass
        win32gui.ReleaseDC(0, self.screen_dc)

    # _create_bitmap_from_buffer crea un bitmap Win32 desde datos BGRA contiguos.
    def _create_bitmap_from_buffer(self, dib_data: np.ndarray):
        dib_bytes = dib_data.tobytes()

        if hasattr(win32ui, "CreateBitmapFromBuffer"):
            return win32ui.CreateBitmapFromBuffer(self.width, self.height, dib_bytes, 32)

        bits_pointer = ctypes.c_void_p()
        CreateDIBSection = ctypes.windll.gdi32.CreateDIBSection
        CreateDIBSection.restype = ctypes.c_void_p
        CreateDIBSection.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(BITMAPINFO),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]

        hbmp = CreateDIBSection(
            self.mem_dc.GetSafeHdc(),
            ctypes.byref(self._bitmap_info),
            win32con.DIB_RGB_COLORS,
            ctypes.byref(bits_pointer),
            None,
            0,
        )

        if not hbmp or not bits_pointer.value:
            raise ctypes.WinError()

        ctypes.memmove(bits_pointer.value, dib_bytes, len(dib_bytes))
        return win32ui.CreateBitmapFromHandle(hbmp)


# draw_status_banner pinta una banda semi-transparente con líneas de texto centradas.
def draw_status_banner(
    image: np.ndarray,
    lines: Sequence[str],
    top_margin: int = 10,
    padding: int = 12,
    line_spacing: int = 6,
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness = 1

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(width for width, _ in text_sizes)
    total_height = sum(height for _, height in text_sizes) + line_spacing * (len(lines) - 1)

    box_width = max_width + padding * 2
    box_height = total_height + padding * 2
    x = max(10, (image.shape[1] - box_width) // 2)
    y = top_margin

    banner_rgb = np.zeros((box_height, box_width, 3), dtype=np.uint8)

    text_y = padding
    for (width, height), line in zip(text_sizes, lines):
        text_x = (box_width - width) // 2
        text_y += height
        cv2.putText(
            banner_rgb,
            line,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_spacing

    banner = np.zeros((box_height, box_width, 4), dtype=np.uint8)
    banner[..., :3] = banner_rgb
    banner[..., 3] = 180
    text_mask = cv2.cvtColor(banner_rgb, cv2.COLOR_BGR2GRAY)
    banner[text_mask > 0, 3] = 255

    y_end = min(y + box_height, image.shape[0])
    x_end = min(x + box_width, image.shape[1])
    image[y:y_end, x:x_end] = banner[: y_end - y, : x_end - x]
