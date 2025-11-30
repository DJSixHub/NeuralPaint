from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Stroke:
    color: Tuple[int, int, int]
    points: List[Tuple[int, int]]


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

    def set_color(self, color: Tuple[int, int, int]) -> None:
        self.brush_color = color

    def start_stroke(self, point: Tuple[int, int]) -> None:
        self.strokes.append(Stroke(color=self.brush_color, points=[point]))

    def add_point(self, point: Tuple[int, int]) -> None:
        if not self.strokes:
            self.start_stroke(point)
            return
        stroke = self.strokes[-1]
        if stroke.points and point == stroke.points[-1]:
            return
        stroke.points.append(point)

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

    def clear(self) -> None:
        self.strokes.clear()

    def render(self) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for stroke in self.strokes:
            if len(stroke.points) == 1:
                cv2.circle(canvas, stroke.points[0], max(1, self.thickness // 2), stroke.color, -1, cv2.LINE_AA)
                continue
            for p0, p1 in zip(stroke.points[:-1], stroke.points[1:]):
                cv2.line(canvas, p0, p1, stroke.color, self.thickness, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _point_inside_circle(x: int, y: int, cx: int, cy: int, radius: float) -> bool:
        dx = x - cx
        dy = y - cy
        return dx * dx + dy * dy <= radius * radius

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
