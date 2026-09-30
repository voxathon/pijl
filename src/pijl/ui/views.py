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

from ..sim import Part, Pin, Wire
from . import theme as T
from .sdf_shapes import Dot, Segment, WireDot
from .paint import Pair, Rgb, mix, sample, with_hue
from .sdf_text import SDFText

Point = tuple[float, float]


class Layers:
    """Draw order inside the world batch (lower order draws first)."""

    def __init__(self) -> None:
        self.wire_halo = pyglet.graphics.Group(order=-1)  # glow under selected / edited wires
        self.wires = pyglet.graphics.Group(order=0)
        self.selection = pyglet.graphics.Group(order=1)   # part outlines, just under the bodies
        self.bodies = pyglet.graphics.Group(order=2)
        self.pins = pyglet.graphics.Group(order=3)
        self.tags = pyglet.graphics.Group(order=4)  # pin name tag backgrounds: over wires and parts
        self.text_order = 5  # SDFText makes its own group at this order
        self.overlay = pyglet.graphics.Group(order=6)


class _GradientLine(shapes.Line):
    """A Line whose two ends can have different colors (the GPU blends between them).
    For screen-space bits (menu swatches); wires use sdf_shapes.Segment."""

    def __init__(self, *args, **kwargs) -> None:
        self._rgba2: tuple[int, int, int] | None = None  # end color; None: same as the start
        super().__init__(*args, **kwargs)

    def set_colors(self, start: tuple, end: tuple) -> None:
        """Both ends at once; 3-component colors keep the current opacity."""
        self._rgba = (*start[:3], start[3] if len(start) > 3 else self._rgba[3])
        self._rgba2 = None if end[:3] == start[:3] else tuple(end[:3])
        self._update_color()

    def _create_vertex_list(self) -> None:
        super()._create_vertex_list()
        self._update_color()

    def _update_color(self) -> None:
        if self._rgba2 is None:
            super()._update_color()
            return
        a, b = self._rgba, (*self._rgba2, self._rgba[3])
        self._vertex_list.colors[:] = a + b + b + a + b + a  # vertex order of shapes.Line: start, end, end, ...


_STALE = object()  # Polyline._layout_key when the points changed: lay out again
GRADIENT_STEPS = 8  # pieces per stretch between two gradient stops (OKLab isn't linear in RGB)


class Polyline:
    """Thick line through several points, rounded at the corners (the segments meeting
    there have round caps) so they have no gaps. One color, or a gradient along its
    length (set_gradient)."""

    def __init__(self, points: list[Point], color, batch: pyglet.graphics.Batch,
                 group: pyglet.graphics.Group, thickness: float = T.WIRE_THICKNESS) -> None:
        self.batch, self.group, self.thickness = batch, group, thickness
        self._color = color
        self._stops: list[tuple[float, tuple]] | None = None  # gradient: (fraction of the length, rgb)
        self._opacity: int | None = None  # None: whatever alpha the color carries
        self.segments: list[Segment] = []
        self.points: list[Point] = []
        # What the segments are laid out for, so a gradient that only changes colors
        # (a wire switching on / off) recolors them instead of laying them out again.
        self._layout_key: object = _STALE
        self._plain = True  # laid out for one color (no gradient, or nothing to spread it over)
        self._vfracs: list[float] = []  # fraction of the length at each segment vertex
        self._colors: dict[tuple, list] = {}  # gradient stops -> color per vertex, for this layout
        self.set_points(points)

    def set_points(self, points: list[Point]) -> None:
        self.points = list(points)
        self._layout_key = _STALE
        self._build()

    def set_gradient(self, stops: list[tuple[float, tuple]]) -> None:
        """Colors at fractions of the length, ascending; the line blends between them."""
        if all(c[:3] == stops[0][1][:3] for _, c in stops):
            self.color = stops[0][1]  # one color after all: plain line
            return
        if stops != self._stops:
            self._stops = list(stops)
            self._build()

    def _build(self) -> None:
        """Lay segments along the points, unless they already are, then color them."""
        stops = self._stops
        # The layout depends on where the stops are and which neighbors differ, not on
        # the colors themselves (see _lay_out).
        key = None if stops is None else (tuple(f for f, _ in stops),
                                          tuple(c0 != c1 for (_, c0), (_, c1) in zip(stops, stops[1:])))
        if key != self._layout_key:
            self._lay_out()
            self._layout_key = key
        if self._plain:
            colors = [self._color] * len(self._vfracs)
        else:
            k = tuple(stops)
            colors = self._colors.get(k)
            if colors is None:
                colors = self._colors[k] = [_sample_rgb(stops, f) for f in self._vfracs]
        for seg, c0, c1 in zip(self.segments, colors, colors[1:]):
            seg.set_colors(c0, c1)

    def _lay_out(self) -> None:
        """Place segments along the points. A gradient splits them further: at every stop,
        and in GRADIENT_STEPS pieces between stops, so each piece blends only a little."""
        self._colors = {}
        pts = self.points
        cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            cum.append(cum[-1] + math.dist(a, b))
        total = cum[-1]
        if self._stops is None or total == 0:
            verts, corner_idx = pts, list(range(1, len(pts) - 1))
            vfracs = [0.0] * len(pts)  # only their count matters: one color everywhere
        else:
            fracs = [c / total for c in cum]
            cuts = set()
            for (f0, c0), (f1, c1) in zip(self._stops, self._stops[1:]):
                cuts.add(f0)
                if c0 != c1:
                    cuts.update(f0 + (f1 - f0) * i / GRADIENT_STEPS for i in range(1, GRADIENT_STEPS))
            cuts = sorted(f for f in cuts if 0 < f < 1 and all(abs(f - g) > 1e-9 for g in fracs))
            verts, vfracs, k, corner_idx = [pts[0]], [0.0], 0, []
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                while k < len(cuts) and cuts[k] < fracs[i + 1]:
                    t = (cuts[k] - fracs[i]) / (fracs[i + 1] - fracs[i])
                    verts.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                    vfracs.append(cuts[k])
                    k += 1
                if i + 1 < len(pts) - 1:
                    corner_idx.append(len(verts))
                verts.append(b)
                vfracs.append(fracs[i + 1])
        self._vfracs = vfracs
        self._plain = self._stops is None or total == 0
        n_seg = max(len(verts) - 1, 0)
        while len(self.segments) < n_seg:
            self.segments.append(self._styled(Segment(self.thickness, self._color, self.batch, self.group)))
        while len(self.segments) > n_seg:
            self.segments.pop().delete()
        # round caps only at the real corners (pieces of one straight segment need none, and the
        # line's own ends stay square: they sit under a pin or junction dot, or on the cursor)
        corners = set(corner_idx)
        for k, (seg, a, b) in enumerate(zip(self.segments, verts, verts[1:])):
            seg.place(a, b, k in corners, k + 1 in corners)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value) -> None:
        """One color for the whole line (drops any gradient)."""
        self._color = value
        if self._stops is not None:
            self._stops = None
            self._build()
            return
        for s in self.segments:
            s.set_colors(value, value)

    @property
    def opacity(self) -> int:
        return 255 if self._opacity is None else self._opacity

    @opacity.setter
    def opacity(self, value: int) -> None:
        self._opacity = value
        for shape in self.segments:
            shape.opacity = value

    def _styled(self, shape):
        if self._opacity is not None:
            shape.opacity = self._opacity  # new segments match existing ones (e.g. ghosts)
        return shape

    def distance_to(self, wx: float, wy: float) -> float:
        return min((_segment_distance(wx, wy, a, b) for a, b in zip(self.points, self.points[1:])),
                   default=math.inf)

    def delete(self) -> None:
        for s in self.segments:
            s.delete()
        self.segments.clear()


def _sample_rgb(stops: list[tuple[float, tuple]], f: float) -> tuple:
    if f <= stops[0][0]:
        return stops[0][1]
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
        if f <= f1:
            return mix(c0, c1, 0.0 if f1 == f0 else (f - f0) / (f1 - f0))
    return stops[-1][1]


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


class PartView:
    def __init__(self, part: Part, x: float, y: float, batch: pyglet.graphics.Batch,
                 layers: Layers, text: SDFText, pin_labels: bool = True) -> None:
        self.part = part
        self.x, self.y = x, y
        self.text = text
        self.look = look = part.type.look
        title = part.type.title or part.kind
        title_size = T.IO_TITLE_SIZE if look.narrow else T.TITLE_SIZE
        n = max(len(part.inputs), len(part.outputs), 1)
        self.w = T.IO_WIDTH if look.narrow else T.PART_WIDTH
        if not look.narrow:  # long titles (macro names) widen the body, in grid steps
            need = text.measure(title, title_size) + 2 * T.TITLE_PAD
            self.w = max(self.w, math.ceil(need / (2 * T.GRID)) * 2 * T.GRID)
        self.h = (n + 1) * T.PIN_SPACING  # multiple of GRID, see theme.py

        self.body = Box(self.w, self.h, T.PART_BORDER, *theme_color(look.body), batch, layers.bodies)
        self.kind_text = text.label(title, 0, 0, size=title_size, color=T.PART_TEXT)
        self.batch, self.layers = batch, layers
        self.pin_tags: list = []  # (background, SDF label) per pin, while shown (see set_pin_labels)
        self.opacity = 255
        # The user's label sits outside the body: left of IN switches, right of
        # OUT LEDs (so it reads like a pin name at the board edge), below gates.
        anchor = {"left": "right", "right": "left"}.get(look.label, "center")
        self.name = text.label(part.label, 0, 0, size=T.LABEL_SIZE, color=T.LABEL_TEXT, anchor_x=anchor)
        self.pin_dots = [Dot(0, 0, T.PIN_RADIUS, T.PIN_OFF, batch, layers.pins) for _ in part.pins]
        # Selection outline: a ring just outside the body, drawn under it and the pins.
        # Only while selected (see set_selected): a hidden one still costs vertices.
        self.outline: shapes.Box | None = None
        self._last_state: tuple[bool, ...] | None = None
        self.pin_tints: list[Rgb | None] = [None] * len(self.pin_dots)  # hue for each lit pin (paint.py)
        self.body_tint: Rgb | None = None  # hue for a lit body (switches, LEDs)
        self.set_pin_labels(pin_labels)
        self.move_to(x, y)
        self.sync()

    @property
    def pin_labels_shown(self) -> bool:
        return bool(self.pin_tags)

    def set_pin_labels(self, on: bool) -> None:
        """Show / hide the pin name tags: outside the body, next to each pin, on a dark
        backing so they read over wires and other parts. Only parts whose look asks."""
        on = on and self.look.pin_labels
        if on == self.pin_labels_shown:
            return
        for bg, label in self.pin_tags:
            bg.delete()
            label.delete()
        self.pin_tags = []
        if on:
            t = self.part.type
            for pin in self.part.pins:
                name = (t.ins if pin.is_input else t.outs)[pin.index]
                label = self.text.label(name, 0, 0, size=T.PIN_LABEL_SIZE, color=T.LABEL_TEXT,
                                        anchor_x="right" if pin.is_input else "left")
                label.opacity = self.opacity
                bg = shapes.Rectangle(0, 0, 1, 1, color=T.PIN_TAG_BG, batch=self.batch, group=self.layers.tags)
                self.pin_tags.append((bg, label))
            self._place_pin_tags()

    def _place_pin_tags(self) -> None:
        pad_x, pad_y = T.PIN_TAG_PAD
        off = T.PIN_RADIUS + T.PIN_TAG_GAP + pad_x
        for (bg, label), pin in zip(self.pin_tags, self.part.pins):
            px, py = self.pin_pos(pin)
            x = px - off if pin.is_input else px + off  # inputs: tag to the left; outputs: right
            label.move_to(x, py)
            h = label.cap_height + 2 * pad_y
            bg.width, bg.height = label.width + 2 * pad_x, h
            bg.position = (x - label.width - pad_x if pin.is_input else x - pad_x, py - h / 2)

    @property
    def selected(self) -> bool:
        return self.outline is not None

    def set_selected(self, on: bool) -> None:
        if on and self.outline is None:
            o = T.SELECT_OUTSET
            self.outline = shapes.Box(self.x - o, self.y - o, self.w + 2 * o, self.h + 2 * o,
                                      thickness=T.SELECT_THICKNESS, color=T.SELECT,
                                      batch=self.batch, group=self.layers.selection)
        elif not on and self.outline is not None:
            self.outline.delete()
            self.outline = None

    def intersects(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Does the body overlap the world-space rectangle (x0, y0)-(x1, y1)?"""
        return self.x <= x1 and x0 <= self.x + self.w and self.y <= y1 and y0 <= self.y + self.h

    # ---- geometry --------------------------------------------------------

    def pin_pos(self, pin: Pin) -> Point:
        side = self.part.inputs if pin.is_input else self.part.outputs
        n = len(side)
        px = self.x if pin.is_input else self.x + self.w
        # index 0 at the top, pins centered vertically
        py = self.y + self.h / 2 + ((n - 1) / 2 - pin.index) * T.PIN_SPACING
        return px, py

    def move_to(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        self.body.position = (x, y)
        if self.outline is not None:
            self.outline.position = (x - T.SELECT_OUTSET, y - T.SELECT_OUTSET)
        self.kind_text.move_to(x + self.w / 2, y + self.h / 2)
        self.name.move_to(*self.name_pos())
        for dot, pin in zip(self.pin_dots, self.part.pins):
            dot.position = self.pin_pos(pin)
        self._place_pin_tags()

    def name_pos(self) -> Point:
        if self.look.label == "left":
            return self.x - T.LABEL_GAP, self.y + self.h / 2
        if self.look.label == "right":
            return self.x + self.w + T.LABEL_GAP, self.y + self.h / 2
        return self.x + self.w / 2, self.y - T.LABEL_GAP - self.name.cap_height / 2

    def refresh_name(self) -> None:
        """Show part.label (after it was edited)."""
        self.name.set_text(self.part.label)

    def contains(self, wx: float, wy: float) -> bool:
        return self.x <= wx <= self.x + self.w and self.y <= wy <= self.y + self.h

    def pin_at(self, wx: float, wy: float, slop: float) -> Pin | None:
        r = T.PIN_RADIUS + slop
        for pin in self.part.pins:
            px, py = self.pin_pos(pin)
            if (px - wx) ** 2 + (py - wy) ** 2 <= r * r:
                return pin
        return None

    def set_ghost(self, ghost: bool) -> None:
        """Semi-transparent while being carried around before placement."""
        a = self.opacity = T.GHOST_OPACITY if ghost else 255
        self.body.opacity = a
        self.kind_text.opacity = a
        self.name.opacity = a
        for bg, label in self.pin_tags:
            label.opacity = a
            bg.opacity = T.PIN_TAG_BG[3] * a // 255
        for dot in self.pin_dots:
            dot.opacity = a

    # ---- state -> visuals --------------------------------------------------

    def sync(self) -> None:
        state = tuple(p.state for p in self.part.pins)
        if state == self._last_state:
            return
        self._last_state = state
        # 3-component colors keep the current opacity (matters for ghosts)
        for dot, on, tint in zip(self.pin_dots, state, self.pin_tints):
            dot.color = with_hue(T.PIN_ON, tint) if on else T.PIN_OFF
        if self.look.lit and state:  # body color follows the first pin (switches, LEDs)
            fill, border = theme_color(self.look.lit[1] if state[0] else self.look.lit[0])
            self._set_body((with_hue(fill, self.body_tint), with_hue(border, self.body_tint)))

    def set_tints(self, pins: list[Rgb | None], body: Rgb | None) -> None:
        if pins == self.pin_tints and body == self.body_tint:
            return
        self.pin_tints, self.body_tint = pins, body
        self._last_state = None
        self.sync()

    def _set_body(self, colors: tuple[tuple[int, int, int], tuple[int, int, int]]) -> None:
        self.body.color, self.body.border_color = colors

    def delete(self) -> None:
        self.body.delete()
        self.set_selected(False)
        self.kind_text.delete()
        self.name.delete()
        for dot in self.pin_dots:
            dot.delete()
        for bg, label in self.pin_tags:
            bg.delete()
            label.delete()


class WireView:
    """A wire drawn from its src pin, through user-placed bend points, to its dst pin.

    Bend points are layout data, so they live here and not in the sim. So is color:
    `color` is what the user picked (None: Default), `stops` the gradient that
    paint.py worked out from it and from what the wire connects to.
    """

    def __init__(self, wire: Wire, src: Point, bends: list[Point], dst: Point,
                 batch: pyglet.graphics.Batch, layers: Layers, color: str | None = None) -> None:
        self.wire = wire
        self.src, self.dst = src, dst
        self.bends = list(bends)
        self.color = color  # a T.WIRE_COLORS name; None = Default (inherit from the ends)
        self.stops: list[tuple[float, Pair]] = []  # empty: neutral, the classic grey / red
        self.batch, self.layers = batch, layers
        self.line = Polyline(self.points, T.WIRE_OFF, batch, layers.wires)
        self.highlight: Polyline | None = None  # selection glow, only while selected
        # A dot on each end that attaches to another wire (a junction), like on schematics.
        # (In the wires layer, right after the line: a wire crossing the junction covers it.)
        self.dots = {end: WireDot(0, 0, T.JUNCTION_RADIUS, T.WIRE_OFF, batch, layers.wires)
                     for end in ("src", "dst") if not isinstance(getattr(wire, end), Pin)}
        self._last_state: tuple[bool, bool] | None = None
        self._redraw()
        self.sync((False, False))

    @property
    def points(self) -> list[Point]:
        """Every vertex: src pin, bends..., dst pin. Segment k runs points[k] -> points[k+1]."""
        return [self.src, *self.bends, self.dst]

    def set_ends(self, src: Point, dst: Point) -> None:
        self.src, self.dst = src, dst
        self._redraw()

    def set_bends(self, bends: list[Point]) -> None:
        self.bends = list(bends)
        self._redraw()

    def _redraw(self) -> None:
        self.line.set_points(self.points)
        if self.highlight is not None:
            self.highlight.set_points(self.points)
        for end, dot in self.dots.items():
            dot.position = getattr(self, end)

    def set_ghost(self, ghost: bool) -> None:
        """Semi-transparent while being carried around before placement (paste)."""
        a = T.GHOST_OPACITY if ghost else 255
        self.line.opacity = a
        for dot in self.dots.values():
            dot.opacity = a

    @property
    def selected(self) -> bool:
        return self.highlight is not None

    def set_selected(self, on: bool) -> None:
        if on and self.highlight is None:
            self.highlight = Polyline(self.points, T.SELECT_WIRE, self.batch, self.layers.wire_halo,
                                      thickness=T.WIRE_THICKNESS + 6)
        elif not on and self.highlight is not None:
            self.highlight.delete()
            self.highlight = None

    def inside(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Is the whole wire within the world-space rectangle (x0, y0)-(x1, y1)?"""
        return all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in self.points)

    def sync(self, state: tuple[bool, bool]) -> None:
        """`state` is the net's (value, conflict), from Circuit.wire_state."""
        if state == self._last_state:
            return
        self._last_state = state
        on, conflict = state
        if conflict or not self.stops:
            color = T.WIRE_CONFLICT if conflict else T.WIRE_ON if on else T.WIRE_OFF
            self.line.color = color
            for dot in self.dots.values():
                dot.color = color
            return
        self.line.set_gradient([(f, c[on]) for f, c in self.stops])
        for end, dot in self.dots.items():
            dot.color = sample(self.stops, 0.0 if end == "src" else 1.0)[on]

    def set_stops(self, stops: list[tuple[float, Pair]]) -> None:
        if stops == self.stops:
            return
        self.stops = stops
        state, self._last_state = self._last_state, None
        self.sync(state or (False, False))

    def color_at(self, p: Point) -> Pair | None:
        """The gradient at the spot nearest to `p` (where a branch attaches); None if neutral."""
        if not self.stops:
            return None
        pts = self.points
        total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        return sample(self.stops, arc_length_at(pts, p) / total if total else 0.0)

    def distance_to(self, wx: float, wy: float) -> float:
        return self.line.distance_to(wx, wy)

    def delete(self) -> None:
        self.set_selected(False)
        self.line.delete()
        for dot in self.dots.values():
            dot.delete()


def arc_length_at(points: list[Point], p: Point) -> float:
    """How far along the polyline (from points[0]) the point nearest to `p` is."""
    best_s, best_d, walked = 0.0, math.inf, 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg = math.hypot(dx, dy)
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / (seg * seg)))
        d = math.hypot(x1 + t * dx - p[0], y1 + t * dy - p[1])
        if d < best_d:
            best_s, best_d = walked + t * seg, d
        walked += seg
    return best_s


def points_before(points: list[Point], s: float) -> list[Point]:
    """The vertices strictly before arc length `s` (points[0] included)."""
    kept, walked = [points[0]], 0.0
    for a, b in zip(points, points[1:]):
        walked += math.dist(a, b)
        if walked >= s - 1e-9:
            break
        kept.append(b)
    return kept


def project_onto(points: list[Point], p: Point) -> Point:
    """Nearest point to `p` on the polyline. Returns `p` itself (bit for bit) if it
    already lies on the line: junctions get re-projected after every edit, and
    float noise there would make identical boards compare unequal (see document.py)."""
    best, best_d = p, math.inf
    for a, b in zip(points, points[1:]):
        (x1, y1), (x2, y2) = a, b
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / length_sq))
        q = (x1 + t * dx, y1 + t * dy)
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        if d < best_d:
            best, best_d = q, d
    return p if best_d < 1e-9 else best


def theme_color(name: str) -> tuple:
    """A part look's color by its theme.py name (looks come from part scripts, so
    they name colors instead of importing the theme). Unknown names: plain body."""
    value = getattr(T, name, None)
    return value if isinstance(value, tuple) else T.PART_BODY
