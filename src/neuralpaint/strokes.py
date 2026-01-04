# Gestión de trazos de dibujo y operaciones de borrado sobre un lienzo.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# Stroke almacena color BGR y puntos enteros pertenecientes a un trazo.
@dataclass
class Stroke:
    color: Tuple[int, int, int]
    points: List[Tuple[int, int]]


# StrokeCanvas gestiona trazos y renderiza el lienzo final.
class StrokeCanvas:
    def __init__(
        self,
        width: int,
        height: int,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 4,
    ) -> None:
        self.width = width
        self.height = height
        self.brush_color = color
        self.thickness = thickness
        self.strokes: List[Stroke] = []

    # set_color actualiza el color BGR del pincel activo.
    def set_color(self, color: Tuple[int, int, int]) -> None:
        self.brush_color = color

    # start_stroke inicia un nuevo trazo con un punto inicial.
    def start_stroke(self, point: Tuple[int, int]) -> None:
        self.strokes.append(Stroke(color=self.brush_color, points=[point]))

    # add_point agrega un punto al trazo actual o inicia uno nuevo si no existe.
    def add_point(self, point: Tuple[int, int]) -> None:
        if not self.strokes:
            self.start_stroke(point)
            return
        stroke = self.strokes[-1]
        if stroke.points and point == stroke.points[-1]:
            return
        if stroke.points:
            last_x, last_y = stroke.points[-1]
            dx = point[0] - last_x
            dy = point[1] - last_y
            dist = math.hypot(dx, dy)
            # insert intermediate points to keep strokes visually continuous
            # use a smaller step so the stroke looks smoother even at lower FPS
            step = max(0.75, self.thickness * 0.25)
            if dist > step:
                segments = max(2, int(dist / step))
                for i in range(1, segments):
                    t = i / segments
                    ix = int(round(last_x + dx * t))
                    iy = int(round(last_y + dy * t))
                    if (ix, iy) != stroke.points[-1]:
                        stroke.points.append((ix, iy))
        stroke.points.append(point)

    # erase_at elimina secciones de trazos dentro de un círculo con centro y radio dados.
    def erase_at(self, point: Tuple[int, int], radius: float) -> None:
        if not self.strokes:
            return

        px, py = point
        remaining: List[Stroke] = []

        for stroke in self.strokes:
            if not stroke.points:
                continue

            new_segments: List[List[Tuple[int, int]]] = []
            current_segment: List[Tuple[int, int]] = []

            prev_x, prev_y = stroke.points[0]
            prev_inside = self._point_inside_circle(prev_x, prev_y, px, py, radius)
            if not prev_inside:
                current_segment.append((prev_x, prev_y))

            for curr_x, curr_y in stroke.points[1:]:
                curr_inside = self._point_inside_circle(curr_x, curr_y, px, py, radius)

                if prev_inside and curr_inside:
                    pass  # entire segment removed
                elif not prev_inside and not curr_inside:
                    current_segment.append((curr_x, curr_y))
                else:
                    intersections = self._segment_circle_intersections(
                        (prev_x, prev_y),
                        (curr_x, curr_y),
                        (px, py),
                        radius,
                    )
                    if intersections:
                        if not prev_inside and curr_inside:
                            entry = intersections[0]
                            current_segment.append(entry)
                            if current_segment:
                                new_segments.append(current_segment)
                            current_segment = []
                        elif prev_inside and not curr_inside:
                            exit_point = intersections[-1]
                            current_segment = [exit_point, (curr_x, curr_y)]
                        else:
                            for pt in intersections:
                                current_segment.append(pt)
                    else:
                        if not prev_inside:
                            current_segment.append((curr_x, curr_y))

                prev_inside = curr_inside
                prev_x, prev_y = curr_x, curr_y

            if current_segment:
                new_segments.append(current_segment)

            for segment in new_segments:
                if segment:
                    remaining.append(Stroke(color=stroke.color, points=segment.copy()))

        self.strokes = remaining

    # clear borra todos los trazos almacenados.
    def clear(self) -> None:
        self.strokes.clear()

    # render devuelve una imagen BGR con los trazos dibujados.
    def render(self) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for stroke in self.strokes:
            if len(stroke.points) == 1:
                cv2.circle(canvas, stroke.points[0], max(1, self.thickness // 2), stroke.color, -1, cv2.LINE_AA)
                continue
            for p0, p1 in zip(stroke.points[:-1], stroke.points[1:]):
                cv2.line(canvas, p0, p1, stroke.color, self.thickness, cv2.LINE_AA)
        return canvas

    # _point_inside_circle evalúa si un punto entero cae dentro del radio especificado.
    @staticmethod
    def _point_inside_circle(x: int, y: int, cx: int, cy: int, radius: float) -> bool:
        dx = x - cx
        dy = y - cy
        return dx * dx + dy * dy <= radius * radius

    # _segment_circle_intersections calcula intersecciones entre un segmento y un círculo.
    @staticmethod
    def _segment_circle_intersections(
        p0: Tuple[int, int],
        p1: Tuple[int, int],
        center: Tuple[int, int],
        radius: float,
    ) -> List[Tuple[int, int]]:
        (x0, y0), (x1, y1) = p0, p1
        cx, cy = center
        dx = x1 - x0
        dy = y1 - y0
        fx = x0 - cx
        fy = y0 - cy

        a = dx * dx + dy * dy
        if a == 0:
            return []
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        intersections: List[Tuple[int, int]] = []
        for t in sorted((t1, t2)):
            if 0.0 <= t <= 1.0:
                ix = x0 + t * dx
                iy = y0 + t * dy
                intersections.append((int(round(ix)), int(round(iy))))

        return intersections

    # ------------------------------------------------------------------
    # Self-intersection detection and inscribed rectangle extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _segments_intersect(a0, a1, b0, b1):
        # Return (intersect, x, y) using parametric intersection
        (x1, y1), (x2, y2) = a0, a1
        (x3, y3), (x4, y4) = b0, b1
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return (False, 0.0, 0.0)
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        # check within both segments
        def within(xa, ya, xb, yb, x, y):
            return min(xa, xb) - 1e-6 <= x <= max(xa, xb) + 1e-6 and min(ya, yb) - 1e-6 <= y <= max(ya, yb) + 1e-6

        if within(x1, y1, x2, y2, px, py) and within(x3, y3, x4, y4, px, py):
            return (True, px, py)
        return (False, 0.0, 0.0)

    def detect_self_intersection_inscribed_rect(self) -> Tuple[int, int, int, int] | None:
        """Detecta auto-intersección en el trazo actual y devuelve un rect (x,y,w,h) inscrito.

        Algoritmo:
          - busca la primera intersección entre segmentos no adyacentes del último trazo
          - construye el polígono cerrado entre los índices y rasteriza la máscara
          - toma el bounding box y lo reduce hasta quedar totalmente dentro de la máscara
        """
        if not self.strokes:
            return None
        stroke = self.strokes[-1]
        pts = stroke.points
        n = len(pts)
        if n < 6:
            return None

        # find first pair of non-adjacent segments that intersect
        for i in range(n - 3):
            a0 = pts[i]
            a1 = pts[i + 1]
            for j in range(i + 2, n - 1):
                # skip adjacent endpoints
                if j == i:
                    continue
                b0 = pts[j]
                b1 = pts[j + 1]
                intersect, ix, iy = self._segments_intersect(a0, a1, b0, b1)
                if intersect:
                    # build polygon from intersection point -> points j+1 back to i+1
                    poly = []
                    poly.append((int(round(ix)), int(round(iy))))
                    k = j + 1
                    while True:
                        poly.append(pts[k])
                        if k == i + 1:
                            break
                        k = (k + 1) % n
                    # create mask
                    mask = np.zeros((self.height, self.width), dtype=np.uint8)
                    try:
                        cv2.fillPoly(mask, [(np.array(poly, dtype=np.int32))], 255)
                    except Exception:
                        return None

                    ys, xs = np.where(mask > 0)
                    if ys.size == 0:
                        return None
                    x0 = int(xs.min())
                    x1 = int(xs.max())
                    y0 = int(ys.min())
                    y1 = int(ys.max())

                    # shrink bbox until fully inside mask
                    left, top, right, bottom = x0, y0, x1, y1
                    while left < right and top < bottom:
                        region = mask[top:bottom + 1, left:right + 1]
                        if region.size == 0:
                            break
                        # if all ones, done
                        if region.all():
                            break
                        # shrink by 1 on the side with zeros present
                        if not region[:, 0].all():
                            left += 1
                        if not region[:, -1].all():
                            right -= 1
                        if not region[0, :].all():
                            top += 1
                        if not region[-1, :].all():
                            bottom -= 1
                    if left >= right or top >= bottom:
                        return None
                    w = right - left + 1
                    h = bottom - top + 1
                    return (left, top, w, h)
        return None
