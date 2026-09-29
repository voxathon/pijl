"""Drawable counterparts of sim objects.

pyglet is retained-mode: shapes are created once and live in a Batch until
deleted. Each view owns the shapes for one sim object, knows how to move
them, and `sync()` pushes sim state (on/off) into colors -- only when it
actually changed, because every property write costs Python time.
"""

from __future__ import annotations

import math

import pyglet
from pyglet import shapes

from ..sim import Chip, Pin, Wire
from . import theme as T
from .sdf_text import SDFText

Point = tuple[float, float]


class Layers:
    """Draw order inside the world batch (lower order draws first)."""

    def __init__(self) -> None:
        self.wires = pyglet.graphics.Group(order=0)
        self.bodies = pyglet.graphics.Group(order=1)
        self.pins = pyglet.graphics.Group(order=2)
        self.text_order = 3  # SDFText makes its own group at this order
        self.overlay = pyglet.graphics.Group(order=4)


class Polyline:
    """Thick line through several points, with round joints so corners have no gaps."""

    def __init__(self, points: list[Point], color, batch: pyglet.graphics.Batch,
                 group: pyglet.graphics.Group, thickness: float = T.WIRE_THICKNESS) -> None:
        self.batch, self.group, self.thickness = batch, group, thickness
        self._color = color
        self.segments: list[shapes.Line] = []
        self.joints: list[shapes.Circle] = []
        self.points: list[Point] = []
        self.set_points(points)

    def set_points(self, points: list[Point]) -> None:
        n_seg = max(len(points) - 1, 0)
        n_joint = max(len(points) - 2, 0)
        while len(self.segments) < n_seg:
            self.segments.append(shapes.Line(0, 0, 0, 0, thickness=self.thickness, color=self._color,
                                             batch=self.batch, group=self.group))
        while len(self.segments) > n_seg:
            self.segments.pop().delete()
        while len(self.joints) < n_joint:
            self.joints.append(shapes.Circle(0, 0, self.thickness / 2, segments=T.JOINT_SEGMENTS,
                                             color=self._color, batch=self.batch, group=self.group))
        while len(self.joints) > n_joint:
            self.joints.pop().delete()

        for seg, (a, b) in zip(self.segments, zip(points, points[1:])):
            seg.x, seg.y = a
            seg.x2, seg.y2 = b
        for joint, p in zip(self.joints, points[1:-1]):
            joint.position = p
        self.points = list(points)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value) -> None:
        self._color = value
        for s in self.segments:
            s.color = value
        for j in self.joints:
            j.color = value

    def distance_to(self, wx: float, wy: float) -> float:
        return min((_segment_distance(wx, wy, a, b) for a, b in zip(self.points, self.points[1:])),
                   default=math.inf)

    def delete(self) -> None:
        for s in self.segments:
            s.delete()
        for j in self.joints:
            j.delete()
        self.segments.clear()
        self.joints.clear()


def _segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


class Box:
    """Sharp-edged box with a solid border.

    Built from a fill rectangle plus four non-overlapping border strips.
    (pyglet's BorderedRectangle interpolates color between its inner and outer
    vertices, so its border smears into a gradient once you zoom in.)
    """

    def __init__(self, w: float, h: float, border: float, fill, border_color,
                 batch: pyglet.graphics.Batch, group: pyglet.graphics.Group) -> None:
        self.w, self.h, self.b = w, h, border
        self.fill = shapes.Rectangle(0, 0, w - 2 * border, h - 2 * border, color=fill,
                                     batch=batch, group=group)
        # bottom, top, left, right (left/right sit between top and bottom)
        self.edges = [shapes.Rectangle(0, 0, w, border, color=border_color, batch=batch, group=group),
                      shapes.Rectangle(0, 0, w, border, color=border_color, batch=batch, group=group),
                      shapes.Rectangle(0, 0, border, h - 2 * border, color=border_color, batch=batch, group=group),
                      shapes.Rectangle(0, 0, border, h - 2 * border, color=border_color, batch=batch, group=group)]

    @property
    def position(self) -> Point:
        return self.edges[0].position

    @property
    def x(self) -> float:
        return self.edges[0].x

    @property
    def y(self) -> float:
        return self.edges[0].y

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @position.setter
    def position(self, xy: Point) -> None:
        x, y = xy
        w, h, b = self.w, self.h, self.b
        self.fill.position = (x + b, y + b)
        for edge, pos in zip(self.edges, ((x, y), (x, y + h - b), (x, y + b), (x + w - b, y + b))):
            edge.position = pos

    @property
    def color(self):
        return self.fill.color

    @color.setter
    def color(self, value) -> None:
        self.fill.color = value

    @property
    def border_color(self):
        return self.edges[0].color

    @border_color.setter
    def border_color(self, value) -> None:
        for edge in self.edges:
            edge.color = value

    @property
    def opacity(self) -> int:
        return self.fill.opacity

    @opacity.setter
    def opacity(self, value: int) -> None:
        self.fill.opacity = value
        for edge in self.edges:
            edge.opacity = value

    def delete(self) -> None:
        self.fill.delete()
        for edge in self.edges:
            edge.delete()


class ChipView:
    def __init__(self, chip: Chip, x: float, y: float, batch: pyglet.graphics.Batch,
                 layers: Layers, text: SDFText) -> None:
        self.chip = chip
        self.x, self.y = x, y
        io = chip.kind in ("IN", "OUT")
        n = max(len(chip.inputs), len(chip.outputs), 1)
        self.w = T.IO_WIDTH if io else T.CHIP_WIDTH
        self.h = n * T.PIN_SPACING + 12

        self.body = Box(self.w, self.h, T.CHIP_BORDER, *T.CHIP_BODY, batch, layers.bodies)
        self.label = text.label(chip.kind, 0, 0, size=10 if io else 12, color=T.CHIP_TEXT)
        self.pin_dots = [shapes.Circle(0, 0, T.PIN_RADIUS, segments=T.PIN_SEGMENTS, color=T.PIN_OFF,
                                       batch=batch, group=layers.pins)
                         for _ in chip.pins]
        self._last_state: tuple[bool, ...] | None = None
        self.move_to(x, y)
        self.sync()

    # ---- geometry --------------------------------------------------------

    def pin_pos(self, pin: Pin) -> Point:
        side = self.chip.inputs if pin.is_input else self.chip.outputs
        n = len(side)
        px = self.x if pin.is_input else self.x + self.w
        # index 0 at the top, pins centered vertically
        py = self.y + self.h / 2 + ((n - 1) / 2 - pin.index) * T.PIN_SPACING
        return px, py

    def move_to(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        self.body.position = (x, y)
        self.label.move_to(x + self.w / 2, y + self.h / 2)
        for dot, pin in zip(self.pin_dots, self.chip.pins):
            dot.position = self.pin_pos(pin)

    def contains(self, wx: float, wy: float) -> bool:
        return self.x <= wx <= self.x + self.w and self.y <= wy <= self.y + self.h

    def pin_at(self, wx: float, wy: float, slop: float) -> Pin | None:
        r = T.PIN_RADIUS + slop
        for pin in self.chip.pins:
            px, py = self.pin_pos(pin)
            if (px - wx) ** 2 + (py - wy) ** 2 <= r * r:
                return pin
        return None

    def set_ghost(self, ghost: bool) -> None:
        """Semi-transparent while being carried around before placement."""
        a = T.GHOST_OPACITY if ghost else 255
        self.body.opacity = a
        self.label.opacity = a
        for dot in self.pin_dots:
            dot.opacity = a

    # ---- state -> visuals --------------------------------------------------

    def sync(self) -> None:
        state = tuple(p.state for p in self.chip.pins)
        if state == self._last_state:
            return
        self._last_state = state
        # 3-component colors keep the current opacity (matters for ghosts)
        for dot, on in zip(self.pin_dots, state):
            dot.color = T.PIN_ON if on else T.PIN_OFF
        if self.chip.kind == "IN":
            self._set_body(T.SWITCH_ON if state[0] else T.SWITCH_OFF)
        elif self.chip.kind == "OUT":
            self._set_body(T.LED_ON if state[0] else T.LED_OFF)

    def _set_body(self, colors: tuple[tuple[int, int, int], tuple[int, int, int]]) -> None:
        self.body.color, self.body.border_color = colors

    def delete(self) -> None:
        self.body.delete()
        self.label.delete()
        for dot in self.pin_dots:
            dot.delete()


class WireView:
    """A wire drawn from its src pin, through user-placed bend points, to its dst pin.

    Bend points are layout data, so they live here and not in the sim.
    """

    def __init__(self, wire: Wire, src: Point, bends: list[Point], dst: Point,
                 batch: pyglet.graphics.Batch, layers: Layers) -> None:
        self.wire = wire
        self.bends = list(bends)
        self.line = Polyline([src, *self.bends, dst], T.WIRE_OFF, batch, layers.wires)
        self._last_state: bool | None = None
        self.sync()

    def set_ends(self, src: Point, dst: Point) -> None:
        self.line.set_points([src, *self.bends, dst])

    def sync(self) -> None:
        on = self.wire.src.state
        if on != self._last_state:
            self._last_state = on
            self.line.color = T.WIRE_ON if on else T.WIRE_OFF

    def distance_to(self, wx: float, wy: float) -> float:
        return self.line.distance_to(wx, wy)

    def delete(self) -> None:
        self.line.delete()
