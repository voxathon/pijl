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


class Layers:
    """Draw order inside the world batch (lower order draws first)."""

    def __init__(self) -> None:
        self.wires = pyglet.graphics.Group(order=0)
        self.bodies = pyglet.graphics.Group(order=1)
        self.pins = pyglet.graphics.Group(order=2)
        self.text = pyglet.graphics.Group(order=3)
        self.overlay = pyglet.graphics.Group(order=4)


class ChipView:
    def __init__(self, chip: Chip, x: float, y: float,
                 batch: pyglet.graphics.Batch, layers: Layers) -> None:
        self.chip = chip
        self.x, self.y = x, y
        io = chip.kind in ("IN", "OUT")
        n = max(len(chip.inputs), len(chip.outputs), 1)
        self.w = T.IO_WIDTH if io else T.CHIP_WIDTH
        self.h = n * T.PIN_SPACING + 12

        self.body = shapes.RoundedRectangle(x, y, self.w, self.h, radius=6,
                                            color=T.CHIP_BODY, batch=batch, group=layers.bodies)
        self.label = pyglet.text.Label(chip.kind, font_name="Consolas", font_size=10 if io else 12,
                                       color=T.CHIP_TEXT, anchor_x="center", anchor_y="center",
                                       batch=batch, group=layers.text)
        self.pin_dots = [shapes.Circle(0, 0, T.PIN_RADIUS, color=T.PIN_OFF, batch=batch, group=layers.pins)
                         for _ in chip.pins]
        self._last_state: tuple[bool, ...] | None = None
        self.move_to(x, y)
        self.sync()

    # ---- geometry --------------------------------------------------------

    def pin_pos(self, pin: Pin) -> tuple[float, float]:
        side = self.chip.inputs if pin.is_input else self.chip.outputs
        n = len(side)
        px = self.x if pin.is_input else self.x + self.w
        # index 0 at the top, pins centered vertically
        py = self.y + self.h / 2 + ((n - 1) / 2 - pin.index) * T.PIN_SPACING
        return px, py

    def move_to(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        self.body.position = (x, y)
        self.label.position = (x + self.w / 2, y + self.h / 2, 0)
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

    # ---- state -> visuals --------------------------------------------------

    def sync(self) -> None:
        state = tuple(p.state for p in self.chip.pins)
        if state == self._last_state:
            return
        self._last_state = state
        for dot, on in zip(self.pin_dots, state):
            dot.color = T.PIN_ON if on else T.PIN_OFF
        if self.chip.kind == "IN":
            self.body.color = T.SWITCH_ON if state[0] else T.SWITCH_OFF
        elif self.chip.kind == "OUT":
            self.body.color = T.LED_ON if state[0] else T.LED_OFF

    def delete(self) -> None:
        self.body.delete()
        self.label.delete()
        for dot in self.pin_dots:
            dot.delete()


class WireView:
    def __init__(self, wire: Wire, a: tuple[float, float], b: tuple[float, float],
                 batch: pyglet.graphics.Batch, layers: Layers) -> None:
        self.wire = wire
        self.line = shapes.Line(*a, *b, thickness=T.WIRE_THICKNESS, color=T.WIRE_OFF,
                                batch=batch, group=layers.wires)
        self._last_state: bool | None = None
        self.sync()

    def set_ends(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        self.line.x, self.line.y = a
        self.line.x2, self.line.y2 = b

    def sync(self) -> None:
        on = self.wire.src.state
        if on != self._last_state:
            self._last_state = on
            self.line.color = T.WIRE_ON if on else T.WIRE_OFF

    def distance_to(self, wx: float, wy: float) -> float:
        """Distance from a point to the wire's segment."""
        x1, y1, x2, y2 = self.line.x, self.line.y, self.line.x2, self.line.y2
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((wx - x1) * dx + (wy - y1) * dy) / length_sq))
        return math.hypot(wx - (x1 + t * dx), wy - (y1 + t * dy))

    def delete(self) -> None:
        self.line.delete()
