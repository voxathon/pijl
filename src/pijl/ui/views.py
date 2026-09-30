"""Drawable counterparts of sim objects.

Each view owns the shapes for one sim object (canvas instances, see canvas.py and
sdf_shapes.py), knows how to move them, and `sync()` pushes sim state into them.
Shapes carry both their off and on colors, so syncing is flipping state flags;
colors are only worked out again when they change (paint.py's tints, gradients).
"""

from __future__ import annotations

import gc
import itertools
import math
from contextlib import contextmanager

import numpy as np
import pyglet
from pyglet import shapes

from ..sim import Part, Pin, Wire
from . import theme as T
from .canvas import Canvas
from .paint import Pair, Rgb, sample, with_hue
from .sdf_shapes import (DOT, RECT, SEGMENT, SHOW_BY_CODE, SHOW_OFF, Dot, Rect, Segment, WireDot, _rgba,
                         show)
from .sdf_text import SDFLabel, SDFText
from .spatial import SpatialIndex, polyline_boxes

Point = tuple[float, float]

_seq = itertools.count()  # creation order of views: newer ones are drawn (and hit) on top


class Touched:
    """The parts and wires (by uid) whose board data changed since the editor last took
    them: created, deleted, moved, relabeled, re-bent, re-ended, recolored. Only those get
    written into the undo history (see Editor._record). Views report their own changes;
    code that changes part props directly reports it itself.

    `paint_parts` / `paint_wires`: the ones among them whose change can change colors
    (paint.py), i.e. everything but moving a part and shifting a wire whole. Colors
    follow the wiring and wire shapes, and a rigid move changes neither."""
    parts: set[int] = set()
    wires: set[int] = set()
    paint_parts: set[int] = set()
    paint_wires: set[int] = set()

    @classmethod
    def part(cls, uid: int) -> None:
        """A change to a part that can change colors."""
        cls.parts.add(uid)
        cls.paint_parts.add(uid)

    @classmethod
    def wire(cls, uid: int) -> None:
        cls.wires.add(uid)
        cls.paint_wires.add(uid)

    @classmethod
    def part_many(cls, uids) -> None:
        uids = set(uids)
        cls.parts |= uids
        cls.paint_parts |= uids

    @classmethod
    def wire_many(cls, uids) -> None:
        uids = set(uids)
        cls.wires |= uids
        cls.paint_wires |= uids

    @classmethod
    def take(cls) -> tuple[set[int], set[int], set[int], set[int]]:
        """(parts, wires, paint_parts, paint_wires), and start over."""
        out = cls.parts, cls.wires, cls.paint_parts, cls.paint_wires
        cls.parts, cls.wires, cls.paint_parts, cls.paint_wires = set(), set(), set(), set()
        return out


class Layers:
    """Draw order in the world (lower order draws first). The canvas sorts its instance
    buffers by these; the few pyglet shapes left (overlay) draw after all of them."""

    def __init__(self) -> None:
        self.wire_halo = pyglet.graphics.Group(order=-1)  # glow under selected / edited wires
        self.wires = pyglet.graphics.Group(order=0)
        self.selection = pyglet.graphics.Group(order=1)   # part outlines, just under the bodies
        self.bodies = pyglet.graphics.Group(order=2)
        self.pins = pyglet.graphics.Group(order=3)
        self.tags = pyglet.graphics.Group(order=4)  # pin name tag backgrounds: over wires and parts
        self.text_order = 5  # SDFText's layer
        self.overlay = pyglet.graphics.Group(order=6)


class _GradientLine(shapes.Line):
    """A Line whose two ends can have different colors (the GPU blends between them).
    For the HUD (menu swatches); wires use sdf_shapes.Segment."""

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
    length (set_gradient). Colors are (off, on) pairs; `state` picks which one shows
    (or a pattern: X, Z, a conflict; see sdf_shapes.show)."""

    def __init__(self, points: list[Point], color, canvas: Canvas,
                 group: pyglet.graphics.Group, thickness: float = T.WIRE_THICKNESS) -> None:
        self._init(color, canvas, group, thickness)
        self.set_points(points)

    def _init(self, color, canvas: Canvas, group: pyglet.graphics.Group, thickness: float) -> None:
        """Everything but the points and segments."""
        self.canvas, self.group, self.thickness = canvas, group, thickness
        self.buf = canvas.buffer(SEGMENT, group)
        self._pair: Pair = (color, color)
        self._stops: list[tuple[float, Pair]] | None = None  # gradient: (fraction of the length, pair)
        self._on = SHOW_OFF  # the segments' state byte
        self._opacity = 255
        self._lift = 0.0
        self.segments: list[Segment] = []
        self._slots = np.empty(0, np.intp)  # the segments' slots in `buf`, for state / opacity at once
        self.points: list[Point] = []
        # What the segments are laid out for, so a gradient that only changes colors
        # recolors them instead of laying them out again.
        self._layout_key: object = _STALE
        self._plain = True  # laid out for one color (no gradient, or nothing to spread it over)
        self._vfracs: list[float] = []  # fraction of the length at each segment vertex
        self._colors: dict[tuple, tuple[list, list]] = {}  # stops -> (off, on) color per vertex, this layout

    def set_points(self, points: list[Point]) -> None:
        self.points = list(points)
        self._layout_key = _STALE
        self._build()

    def set_gradient(self, stops: list[tuple[float, Pair]]) -> None:
        """Color pairs at fractions of the length, ascending; the line blends between them."""
        if all(c == stops[0][1] for _, c in stops):
            self.set_pair(stops[0][1])  # one color after all: plain line
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
            n = len(self._vfracs)
            offs, ons = [self._pair[0]] * n, [self._pair[1]] * n
        else:
            k = tuple(stops)
            if k not in self._colors:
                pairs = [sample(stops, f) for f in self._vfracs]
                self._colors[k] = [p[0] for p in pairs], [p[1] for p in pairs]
            offs, ons = self._colors[k]
        for seg, a0, b0, a1, b1 in zip(self.segments, offs, offs[1:], ons, ons[1:]):
            seg.set_colors(a0, b0, a1, b1)

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
        if len(self.segments) != n_seg:
            while len(self.segments) < n_seg:
                seg = Segment(self.thickness, self._pair[0], self.canvas, self.group)
                seg.state, seg.opacity, seg.lifted = self._on, self._opacity, self._lift  # match the others
                self.segments.append(seg)
            while len(self.segments) > n_seg:
                self.segments.pop().delete()
            self._slots = np.array([seg.slot for seg in self.segments], np.intp)
        # round caps only at the real corners (pieces of one straight segment need none, and the
        # line's own ends stay square: they sit under a pin or junction dot, or on the cursor)
        corners = set(corner_idx)
        for k, (seg, a, b) in enumerate(zip(self.segments, verts, verts[1:])):
            seg.place(a, b, k in corners, k + 1 in corners)

    @property
    def color(self):
        return self._pair[0]

    @color.setter
    def color(self, value) -> None:
        """One color for the whole line, on or off (drops any gradient)."""
        self.set_pair((value, value))

    def set_pair(self, pair: Pair) -> None:
        """One (off, on) color pair for the whole line (drops any gradient)."""
        self._pair = pair
        if self._stops is not None:
            self._stops = None
            self._build()
            return
        off, on = pair
        for s in self.segments:
            s.set_colors(off, off, on, on)

    @property
    def state(self) -> int:
        return self._on

    @state.setter
    def state(self, value) -> None:
        """A SHOW_* byte, a Level or a bool."""
        v = value if type(value) is int else show(value)
        if v != self._on:
            self._on = v
            self._write_flag(0, v)

    @property
    def opacity(self) -> int:
        return self._opacity

    @opacity.setter
    def opacity(self, value: int) -> None:
        self._opacity = value
        self._write_flag(1, value)

    @property
    def lifted(self) -> bool:
        return bool(self._lift)

    @lifted.setter
    def lifted(self, on: bool) -> None:
        """Drawn shifted by the canvas's offset (see canvas.py)."""
        self._lift = 1.0 if on else 0.0
        if self._slots.size:
            self.buf.f["lift"][self._slots] = self._lift
            self.buf.mark_many(self._slots)

    def _write_flag(self, i: int, value: int) -> None:
        if self._slots.size:
            self.buf.f["flags"][self._slots, i] = value
            self.buf.mark_many(self._slots)

    def distance_to(self, wx: float, wy: float) -> float:
        return min((_segment_distance(wx, wy, a, b) for a, b in zip(self.points, self.points[1:])),
                   default=math.inf)

    def delete(self) -> None:
        for s in self.segments:
            s.delete()
        self.segments.clear()
        self._slots = np.empty(0, np.intp)


def _segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


class Box:
    """Sharp-edged box with a solid border, from pyglet shapes: for the HUD (menus,
    picker, prompts). Parts in the world are sdf_shapes.Rect.

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
    def __init__(self, part: Part, x: float, y: float, canvas: Canvas,
                 layers: Layers, text: SDFText, pin_labels: bool = True,
                 index: SpatialIndex | None = None) -> None:
        """`index`: where to register for hit testing; kept up to date as it moves."""
        self._init(part, x, y, canvas, layers, text, index)
        _make_part_shapes([self], pin_labels)

    @classmethod
    def many(cls, placed: list[tuple[Part, float, float]], canvas: Canvas, layers: Layers, text: SDFText,
             pin_labels: bool = True, index: SpatialIndex | None = None) -> list[PartView]:
        """A view for each (part, x, y): the same as making them one by one, in order, but
        their shapes are made all at once (undo, loading, pasting thousands of parts)."""
        views = []
        for part, x, y in placed:
            view = cls.__new__(cls)
            view._init(part, x, y, canvas, layers, text, index)
            views.append(view)
        _make_part_shapes(views, pin_labels)
        return views

    def _init(self, part: Part, x: float, y: float, canvas: Canvas, layers: Layers, text: SDFText,
              index: SpatialIndex | None) -> None:
        """Everything but the shapes (see _make_part_shapes)."""
        self.part = part
        self.seq = next(_seq)
        self.index = index
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
        self.title, self.title_size = title, title_size
        self.canvas, self.layers = canvas, layers
        self.pin_tags: list = []  # (background, SDF label) per pin, while shown (see set_pin_labels)
        self.opacity = 255
        self.lifted = False  # being dragged: drawn at the canvas's offset (see set_lifted)
        # Selection outline: a ring just outside the body, drawn under it and the pins.
        self.outline: Rect | None = None  # only while selected (see set_selected)
        self._last_state: tuple[bool, ...] | None = None
        self.pin_tints: list[Rgb | None] = [None] * len(part.pins)  # hue for each lit pin (paint.py)
        self.body_tint: Rgb | None = None  # hue for a lit body (switches, LEDs)
        # Made by _make_part_shapes:
        self.body: Rect
        self.kind_text: SDFLabel
        self.name: SDFLabel  # the user's label
        self.pin_dots: list[Dot]

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
            _make_pin_tags([(self, self.text.labels(self._tag_specs()))])

    def _tag_specs(self) -> list[tuple]:
        """SDFText.labels specs for the pin name tags' text, in place (see _tag_at)."""
        t = self.part.type
        color = (*T.LABEL_TEXT[:3], self.opacity)
        return [((t.ins if pin.is_input else t.outs)[pin.index], *self._tag_at(pin), T.PIN_LABEL_SIZE, color,
                 "right" if pin.is_input else "left") for pin in self.part.pins]

    def _tag_at(self, pin: Pin) -> Point:
        """Where a pin's tag text is anchored: inputs' to the left of the pin, outputs' right."""
        px, py = self.pin_pos(pin)
        off = T.PIN_RADIUS + T.PIN_TAG_GAP + T.PIN_TAG_PAD[0]
        return (px - off if pin.is_input else px + off), py

    def _place_pin_tags(self) -> None:
        for (bg, label), pin in zip(self.pin_tags, self.part.pins):
            label.move_to(*self._tag_at(pin))
            x, y, w, h = _tag_backing(label, pin)
            bg.width, bg.height = w, h
            bg.position = (x, y)

    @property
    def selected(self) -> bool:
        return self.outline is not None

    def set_selected(self, on: bool) -> None:
        if on and self.outline is None:
            o = T.SELECT_OUTSET
            self.outline = Rect(self.x - o, self.y - o, self.w + 2 * o, self.h + 2 * o, T.SELECT_THICKNESS,
                                (*T.SELECT, 0), T.SELECT, self.canvas, self.layers.selection)
            self.outline.lifted = self.lifted
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
        Touched.parts.add(self.part.uid)
        self.body.position = (x, y)
        if self.outline is not None:
            self.outline.position = (x - T.SELECT_OUTSET, y - T.SELECT_OUTSET)
        self.kind_text.move_to(x + self.w / 2, y + self.h / 2)
        self.name.move_to(*self.name_pos())
        for dot, pin in zip(self.pin_dots, self.part.pins):
            dot.position = self.pin_pos(pin)
        self._place_pin_tags()
        self._register()

    def name_pos(self) -> Point:
        if self.look.label == "left":
            return self.x - T.LABEL_GAP, self.y + self.h / 2
        if self.look.label == "right":
            return self.x + self.w + T.LABEL_GAP, self.y + self.h / 2
        return self.x + self.w / 2, self.y - T.LABEL_GAP - self.text.cap_height(T.LABEL_SIZE) / 2

    def refresh_name(self) -> None:
        """Show part.label (after it was edited)."""
        Touched.parts.add(self.part.uid)  # (labels don't change colors)
        self.name.set_text(self.part.label)

    def contains(self, wx: float, wy: float) -> bool:
        return self.x <= wx <= self.x + self.w and self.y <= wy <= self.y + self.h

    def distance_to(self, wx: float, wy: float) -> float:
        """From the body's edge (0 inside)."""
        dx = max(self.x - wx, 0.0, wx - self.x - self.w)
        dy = max(self.y - wy, 0.0, wy - self.y - self.h)
        return math.hypot(dx, dy)

    def pin_at(self, wx: float, wy: float, slop: float) -> Pin | None:
        r = T.PIN_RADIUS + slop
        for pin in self.part.pins:
            px, py = self.pin_pos(pin)
            if (px - wx) ** 2 + (py - wy) ** 2 <= r * r:
                return pin
        return None

    def set_lifted(self, on: bool) -> None:
        """While lifted, the part is drawn shifted by the canvas's offset and its x / y
        are where it was lifted from: dragging moves the offset, not the shapes. Put it
        down with move_to (after set_lifted(False)), or many at once with put_down."""
        lift([self], [], on)

    def _register(self) -> None:
        if self.index is not None:
            self.index.put_rect(self, *self._index_box())

    def _index_box(self) -> tuple[float, float, float, float]:
        """What the spatial index knows it by: the body, and the pins sticking out of its sides."""
        r = T.PIN_RADIUS
        return self.x - r, self.y, self.x + self.w + r, self.y + self.h

    def set_ghost(self, ghost: bool) -> None:
        """Semi-transparent while being carried around before placement."""
        a = self.opacity = T.GHOST_OPACITY if ghost else 255
        self.body.opacity = a
        self.kind_text.opacity = a
        self.name.opacity = a
        for bg, label in self.pin_tags:
            label.opacity = a
            bg.opacity = a  # (multiplies the backing's own alpha)
        for dot in self.pin_dots:
            dot.opacity = a

    # ---- state -> visuals --------------------------------------------------

    def sync(self) -> None:
        state = tuple(p.state for p in self.part.pins)
        if state == self._last_state:
            return
        self._last_state = state
        for dot, level in zip(self.pin_dots, state):
            dot.state = level
        if self.look.lit and state:  # body color follows the first pin (switches, LEDs)
            self.body.state = state[0]

    def set_tints(self, pins: list[Rgb | None], body: Rgb | None) -> None:
        if pins == self.pin_tints and body == self.body_tint:
            return
        self.pin_tints, self.body_tint = pins, body
        self._recolor()

    def _recolor(self) -> None:
        """Both colors of every pin (and of a lit body), per the tints."""
        for dot, tint in zip(self.pin_dots, self.pin_tints):
            dot.set_colors(T.PIN_OFF, with_hue(T.PIN_ON, tint))
        if self.look.lit:
            off, on = ([with_hue(c, self.body_tint) for c in theme_color(name)] for name in self.look.lit)
            self.body.set_colors(off[0], off[1], on[0], on[1])

    def delete(self) -> None:
        delete_views([self], [])


class WireView:
    """A wire drawn from its src pin, through user-placed bend points, to its dst pin.

    Bend points are layout data, so they live here and not in the sim. So is color:
    `color` is what the user picked (None: Default), `stops` the gradient that
    paint.py worked out from it and from what the wire connects to.
    """

    def __init__(self, wire: Wire, src: Point, bends: list[Point], dst: Point,
                 canvas: Canvas, layers: Layers, color: str | None = None,
                 index: SpatialIndex | None = None) -> None:
        self._init(wire, src, bends, dst, canvas, layers, color, index)
        _make_wire_shapes([self])

    @classmethod
    def many(cls, specs: list[tuple], canvas: Canvas, layers: Layers,
             index: SpatialIndex | None = None) -> list[WireView]:
        """A view for each (wire, src, bends, dst, color): like making them one by one, in
        order, with the shapes made all at once."""
        views = []
        for wire, src, bends, dst, color in specs:
            view = cls.__new__(cls)
            view._init(wire, src, bends, dst, canvas, layers, color, index)
            views.append(view)
        _make_wire_shapes(views)
        return views

    def _init(self, wire: Wire, src: Point, bends: list[Point], dst: Point, canvas: Canvas, layers: Layers,
              color: str | None, index: SpatialIndex | None) -> None:
        """Everything but the shapes (see _make_wire_shapes)."""
        self.wire = wire
        self.seq = next(_seq)
        self.index = index
        self.src, self.dst = src, dst
        self.bends = list(bends)
        self._color = color  # a T.WIRE_COLORS name; None = Default (inherit from the ends)
        self.stops: list[tuple[float, Pair]] = []  # empty: neutral, the classic grey / red
        self.canvas, self.layers = canvas, layers
        self.line = Polyline.__new__(Polyline)  # (its segments: _make_wire_shapes)
        self.line._init(T.WIRE_OFF, canvas, layers.wires, T.WIRE_THICKNESS)
        self.highlight: Polyline | None = None  # selection glow, only while selected
        # A dot on each end that attaches to another wire (a junction), like on schematics.
        # (In the wires layer, right after the line: a wire crossing the junction covers it.)
        self.dots: dict[str, WireDot] = {}
        self._last_state: tuple | None = None
        self.lifted = False  # see set_lifted

    @property
    def points(self) -> list[Point]:
        """Every vertex: src pin, bends..., dst pin. Segment k runs points[k] -> points[k+1]."""
        return [self.src, *self.bends, self.dst]

    @property
    def color(self) -> str | None:
        return self._color

    @color.setter
    def color(self, value: str | None) -> None:
        if value != getattr(self, "_color", object()):
            Touched.wire(self.wire.uid)
        self._color = value

    def set_ends(self, src: Point, dst: Point) -> None:
        if (src, dst) == (self.src, self.dst):
            return  # (re-attaching after a move often lands exactly where it was)
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
        if self.index is not None:
            self.index.put_polyline(self, self.points)
        Touched.wire(self.wire.uid)  # (its shape: colors of branches depend on where they attach)

    def set_lifted(self, on: bool) -> None:
        """Drawn shifted by the canvas's offset, like PartView.set_lifted: for wires that
        move rigidly with a dragged selection. Put it down with set_ends / set_bends,
        or put_down."""
        lift([], [self], on)


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
            self.highlight = Polyline(self.points, T.SELECT_WIRE, self.canvas, self.layers.wire_halo,
                                      thickness=T.WIRE_THICKNESS + 6)
            self.highlight.lifted = self.lifted
        elif not on and self.highlight is not None:
            self.highlight.delete()
            self.highlight = None

    def inside(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Is the whole wire within the world-space rectangle (x0, y0)-(x1, y1)?"""
        return all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in self.points)

    def sync(self, state: tuple) -> None:
        """`state` is the net's (value, conflict), from Circuit.wire_state."""
        if state == self._last_state:
            return
        self._last_state = state
        b = show(*state)  # 0 / 1: its colors; X, Z, a conflict: their patterns
        self.line.state = b
        for dot in self.dots.values():
            dot.state = b

    def _recolor(self) -> None:
        """Both colors of the line and dots: the gradient, else the classic grey / red.
        (X, Z and conflicts are patterns the state picks, whatever these are.)"""
        if not self.stops:
            pair = (T.WIRE_OFF, T.WIRE_ON)
            self.line.set_pair(pair)
            for dot in self.dots.values():
                dot.set_pair(*pair)
            return
        self.line.set_gradient(self.stops)
        for end, dot in self.dots.items():
            dot.set_pair(*sample(self.stops, 0.0 if end == "src" else 1.0))

    def set_stops(self, stops: list[tuple[float, Pair]]) -> None:
        if stops == self.stops:
            return
        self.stops = stops
        self._recolor()

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
        delete_views([], [self])


# ---- making views ------------------------------------------------------------------
# Views are made in batches (a batch of one for the constructors): per shape role, the
# slots they would have got one view at a time, then each field written as one array.
# Loading or undoing thousands of parts one constructor at a time was seconds.


@contextmanager
def paused_gc():
    """No cycle collection meanwhile: a big batch makes hundreds of thousands of objects,
    and Python's collector would walk the whole heap again and again as it did (half the
    time of undoing a 20k-part delete). Whatever became garbage is collected after."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def _make_part_shapes(views: list[PartView], pin_labels: bool) -> None:
    """Bodies, titles, user labels, pin dots, colors, placement: a new view's shapes."""
    if not views:
        return
    v0 = views[0]
    canvas, layers, text = v0.canvas, v0.layers, v0.text
    pins = [v.part.pins for v in views]
    # the pins' states, in one gather
    flat = [p for ps in pins for p in ps]
    on_flat = v0.part.pins[0]._store.states[[p.slot for p in flat]].tolist() if flat else []
    states, k = [], 0
    for ps in pins:
        states.append(tuple(on_flat[k:k + len(ps)]))
        k += len(ps)
    looks: dict[int, tuple] = {}  # what a look decides (anchor, body colors), worked out once per look
    # titles and user labels (a view's title, then its label: the one-at-a-time order),
    # and pin name tags if shown
    specs, tagged = [], []
    for v in views:
        look = looks.get(id(v.look))
        if look is None:
            look = looks[id(v.look)] = _look_data(v.look)
        specs.append((v.title, v.x + v.w / 2, v.y + v.h / 2, v.title_size, T.PART_TEXT, "center"))
        specs.append((v.part.label, *v.name_pos(), T.LABEL_SIZE, T.LABEL_TEXT, look[0]))
        if pin_labels and v.look.pin_labels:
            specs += v._tag_specs()
    labels = iter(text.labels(specs))
    for v, ps in zip(views, pins):
        v.kind_text, v.name = next(labels), next(labels)
        if pin_labels and v.look.pin_labels:
            tagged.append((v, [next(labels) for _ in ps]))
    _make_pin_tags(tagged)
    # bodies: the look's colors; lit ones (switches, LEDs) also on-colors, on per the first pin
    rows, colors, on = [], [], []
    for v, state in zip(views, states):
        _, row, lit = looks[id(v.look)]
        rows.append((v.x, v.y, v.w, v.h))
        colors.append(row)
        on.append(SHOW_BY_CODE[state[0]] if lit and state else SHOW_OFF)
    buf = canvas.buffer(RECT, layers.bodies)
    slots = buf.alloc_many(len(views))
    f = buf.f
    f["rect"][slots] = rows
    f["border"][slots] = T.PART_BORDER
    colors = np.array(colors, np.uint8)
    for k, name in enumerate(("fill", "fill_on", "edge", "edge_on")):
        f[name][slots] = colors[:, k]
    f["flags"][slots] = 0, 255, 0, 0
    f["flags"][slots, 0] = on
    f["lift"][slots] = 0.0
    buf.mark_many(slots)
    for v, body in zip(views, Rect.adopt(buf, slots.tolist())):
        v.body = body
    # pin dots
    centers = [v.pin_pos(pin) for v, ps in zip(views, pins) for pin in ps]
    buf = canvas.buffer(DOT, layers.pins)
    slots = buf.alloc_many(len(centers))
    if len(centers):
        f = buf.f
        f["center"][slots] = centers
        f["radius"][slots] = T.PIN_RADIUS
        f["color"][slots] = _rgba(T.PIN_OFF)
        f["color_on"][slots] = _rgba(with_hue(T.PIN_ON, None))
        f["flags"][slots] = 0, 255, 0, 0
        f["flags"][slots, 0] = SHOW_BY_CODE[np.array(on_flat, np.intp)]
        f["lift"][slots] = 0.0
        buf.mark_many(slots)
    dots = Dot.adopt(buf, slots.tolist())
    k = 0
    for v, ps, state in zip(views, pins, states):
        v.pin_dots = dots[k:k + len(ps)]
        k += len(ps)
        v._last_state = state
    _index_many([(v, [v._index_box()]) for v in views])
    Touched.part_many(v.part.uid for v in views)  # new: its own color (props) needs painting


def _look_data(look) -> tuple:
    """(user label anchor, body colors as (fill, fill_on, edge, edge_on) rgba, lit?) for a look."""
    # The user's label sits outside the body: left of IN switches, right of
    # OUT LEDs (so it reads like a pin name at the board edge), below gates.
    anchor = {"left": "right", "right": "left"}.get(look.label, "center")
    fill, edge = theme_color(look.body)
    if look.lit:  # (untinted: paint.py tints them)
        (fill, edge), (fill_on, edge_on) = (theme_color(name) for name in look.lit)
    else:
        fill_on, edge_on = fill, edge
    return anchor, (_rgba(fill), _rgba(fill_on), _rgba(edge), _rgba(edge_on)), bool(look.lit)


def _index_many(items: list[tuple]) -> None:
    """Register (view, boxes) in the views' spatial indexes, in bulk."""
    by_index: dict[int, tuple] = {}
    for v, boxes in items:
        if v.index is not None:
            by_index.setdefault(id(v.index), (v.index, []))[1].append((v, boxes))
    for index, members in by_index.values():
        index.put_many(members)


def _tag_backing(label: SDFLabel, pin: Pin) -> tuple[float, float, float, float]:
    """(x, y, width, height) of the dark backing behind a pin tag's text."""
    pad_x, pad_y = T.PIN_TAG_PAD
    h = label.cap_height + 2 * pad_y
    x = label.x - label.width - pad_x if pin.is_input else label.x - pad_x
    return x, label.y - h / 2, label.width + 2 * pad_x, h


def _make_pin_tags(made: list[tuple[PartView, list[SDFLabel]]]) -> None:
    """Pin name tags for these views, from their tag texts (made from _tag_specs):
    the backings, all at once, and the views' pin_tags."""
    rows, alpha, lift = [], [], []
    for v, labels in made:
        rows += [_tag_backing(label, pin) for label, pin in zip(labels, v.part.pins)]
        alpha += [v.opacity] * len(labels)
        lift += [1.0 if v.lifted else 0.0] * len(labels)
    if not rows:
        return
    v0 = made[0][0]
    buf = v0.canvas.buffer(RECT, v0.layers.tags)
    slots = buf.alloc_many(len(rows))
    f = buf.f
    f["rect"][slots] = rows
    f["border"][slots] = 0.0
    for name in ("fill", "fill_on", "edge", "edge_on"):
        f[name][slots] = _rgba(T.PIN_TAG_BG)
    f["flags"][slots] = 0, 255, 0, 0
    f["flags"][slots, 1] = alpha
    f["lift"][slots] = lift
    buf.mark_many(slots)
    backings = iter(Rect.adopt(buf, slots.tolist()))
    for v, labels in made:
        v.pin_tags = [(next(backings), label) for label in labels]
        if v.lifted:
            for label in labels:
                label.lifted = True


def _make_wire_shapes(views: list[WireView]) -> None:
    """The segments of a new wire's line (and its junction dots), placed and colored."""
    if not views:
        return
    v0 = views[0]
    buf = v0.line.buf
    a, b, radius, caps, kinds = [], [], [], [], []
    for v in views:
        v.dots = {end: None for end in ("src", "dst") if not isinstance(getattr(v.wire, end), Pin)}
        pts = v.points
        last = len(pts) - 2  # corners (round caps): the points between the ends
        for k in range(len(pts) - 1):
            a.append(pts[k])
            b.append(pts[k + 1])
            caps.append((255 if k >= 1 else 0, 255 if k + 1 <= last else 0))
        radius += [T.WIRE_THICKNESS / 2] * (len(pts) - 1)
        kinds += [Segment] * (len(pts) - 1)
        for end in v.dots:  # right after its line: a wire crossing the junction covers it
            p = getattr(v, end)
            a.append(p)
            b.append(p)
            caps.append((255, 255))
            radius.append(T.JUNCTION_RADIUS)
            kinds.append(WireDot)
    slots = buf.alloc_many(len(a))
    if len(a):
        f = buf.f
        f["a"][slots] = a
        f["b"][slots] = b
        f["radius"][slots] = radius
        f["ca"][slots] = f["cb"][slots] = _rgba(T.WIRE_OFF)  # neutral: paint.py gives gradients
        f["ca_on"][slots] = f["cb_on"][slots] = _rgba(T.WIRE_ON)
        f["flags"][slots] = 0, 255, 0, 0
        f["flags"][slots, 2:] = caps
        f["lift"][slots] = 0.0
        buf.mark_many(slots)
    k = 0
    slot_list = slots.tolist()
    for v in views:
        line, pts = v.line, v.points
        n = max(len(pts) - 1, 0)
        line.segments = Segment.adopt(buf, slot_list[k:k + n])
        line._slots = slots[k:k + n]
        k += n
        for end in v.dots:
            v.dots[end], = WireDot.adopt(buf, slot_list[k:k + 1])
            k += 1
        line.points = list(pts)
        line._pair = (T.WIRE_OFF, T.WIRE_ON)
        line._layout_key, line._plain, line._vfracs = None, True, [0.0] * len(pts)
        v._last_state = None  # (drawn off: the first sync shows what's really there)
    _index_many([(v, polyline_boxes(v.points)) for v in views])
    Touched.wire_many(v.wire.uid for v in views)


# ---- many at once ---------------------------------------------------------------
# Dropping a big selection moves every shape of every part and wire in it. One at a
# time that was over a second for 20k parts; these gather the shapes' slots per
# instance buffer and write each buffer once.


class _Slots:
    """Slots gathered per instance buffer. Shapes come in by role (every part's body,
    every pin dot, every label...): all shapes of one role share a buffer -- same
    canvas, same layer -- so it's looked up once per role, not per shape."""

    def __init__(self) -> None:
        self.by: dict[int, tuple] = {}

    def _add(self, buf, slots: np.ndarray) -> None:
        if slots.size:
            self.by.setdefault(id(buf), (buf, []))[1].append(slots)

    def shapes(self, shapes: list) -> None:
        """sdf_shapes instances: one slot each."""
        if shapes:
            self._add(shapes[0].buf, np.fromiter((s.slot for s in shapes), np.intp, len(shapes)))

    def labels(self, labels: list[SDFLabel]) -> None:
        if labels:
            self._add(labels[0].buf, np.concatenate([label.slots for label in labels]))

    def lines(self, lines: list[Polyline]) -> None:
        if lines:
            self._add(lines[0].buf, np.concatenate([line._slots for line in lines]))

    def parts(self, views: list[PartView]) -> _Slots:
        self.shapes([v.body for v in views])
        self.shapes([d for v in views for d in v.pin_dots])
        self.shapes([v.outline for v in views if v.outline is not None])
        self.shapes([bg for v in views for bg, _ in v.pin_tags])
        self.labels([v.kind_text for v in views if v.kind_text.slots.size])
        self.labels([v.name for v in views if v.name.slots.size])
        self.labels([label for v in views for _, label in v.pin_tags])
        return self

    def wires(self, views: list[WireView]) -> _Slots:
        self.lines([v.line for v in views])
        self.lines([v.highlight for v in views if v.highlight is not None])
        self.shapes([d for v in views for d in v.dots.values()])
        return self

    def __iter__(self):
        for buf, arrays in self.by.values():
            yield buf, (np.concatenate(arrays) if len(arrays) > 1 else arrays[0])


def lift(parts: list[PartView], wires: list[WireView], on: bool) -> None:
    """set_lifted for many parts and wires at once."""
    parts, wires = _lift_mirrors(parts, wires, on)
    for buf, slots in _Slots().parts(parts).wires(wires):
        buf.set_lift(slots, on)


def put_down(parts: list[PartView], wires: list[WireView], dx: float, dy: float) -> None:
    """Un-lift these and move them by (dx, dy): the parts as if by move_to, the wires
    whole (bends and junction ends too, no new layout). Dropping a dragged selection;
    every shape is written once. (A rigid move changes no colors: not reported to
    paint.py. The caller re-attaches wires stretched between these and the rest.)"""
    _lift_mirrors(parts, wires, False)
    if dx or dy:
        _move_part_mirrors(parts, dx, dy)
        _move_wire_mirrors(wires, dx, dy)
    for buf, slots in _Slots().parts(parts).wires(wires):
        buf.set_lift(slots, False)
        if dx or dy:
            buf.shift(slots, dx, dy)


def delete_views(parts: list[PartView], wires: list[WireView]) -> None:
    """delete() for many parts and wires at once: their shapes are freed per buffer."""
    Touched.parts.update(v.part.uid for v in parts)
    Touched.paint_parts.update(v.part.uid for v in parts)
    Touched.wires.update(v.wire.uid for v in wires)
    Touched.paint_wires.update(v.wire.uid for v in wires)
    by_index: dict[int, tuple] = {}
    for v in (*parts, *wires):
        if v.index is not None:
            by_index.setdefault(id(v.index), (v.index, []))[1].append(v)
    for index, members in by_index.values():
        index.remove_many(members)
    for buf, slots in _Slots().parts(parts).wires(wires):
        buf.free_many(slots)
    # the views are dead: they hold no slots any more (a second delete is a no-op)
    empty = np.empty(0, np.intp)
    for v in parts:
        v.body.slot = None
        for dot in v.pin_dots:
            dot.slot = None
        v.kind_text.slots = v.name.slots = empty
        v.outline, v.pin_tags = None, []
    for v in wires:
        for line in (v.line, v.highlight):
            if line is not None:
                line.segments.clear()
                line._slots = empty
        for dot in v.dots.values():
            dot.slot = None
        v.highlight = None


# What the views remember of their shapes (coordinates, lift), and the spatial index:
# the rest of the bulk operations, next to the one write per buffer.

def _lift_mirrors(parts: list[PartView], wires: list[WireView], on: bool) -> tuple[list, list]:
    """Mark them (un)lifted; returns the ones that weren't already."""
    parts = [v for v in parts if v.lifted != on]
    wires = [v for v in wires if v.lifted != on]
    value = 1.0 if on else 0.0
    for v in parts:
        v.lifted = on
        v.kind_text._lift = v.name._lift = value  # (what glyphs they get later start with)
        for _, label in v.pin_tags:
            label._lift = value
    for v in wires:
        v.lifted = on
        v.line._lift = value
        if v.highlight is not None:
            v.highlight._lift = value
    return parts, wires


def _move_part_mirrors(views: list[PartView], dx: float, dy: float) -> None:
    for v in views:
        v.x += dx
        v.y += dy
        for label in (v.kind_text, v.name):
            label.x += dx
            label.y += dy
        for _, label in v.pin_tags:
            label.x += dx
            label.y += dy
    _shift_indexed(views, dx, dy)
    Touched.parts.update(v.part.uid for v in views)


def _move_wire_mirrors(views: list[WireView], dx: float, dy: float) -> None:
    for v in views:
        v.src, v.dst = (v.src[0] + dx, v.src[1] + dy), (v.dst[0] + dx, v.dst[1] + dy)
        v.bends = [(x + dx, y + dy) for x, y in v.bends]
        for line in (v.line, v.highlight):
            if line is not None:
                line.points = [(x + dx, y + dy) for x, y in line.points]
    _shift_indexed(views, dx, dy)
    Touched.wires.update(v.wire.uid for v in views)


def _shift_indexed(views: list, dx: float, dy: float) -> None:
    by_index: dict[int, tuple] = {}
    for v in views:
        if v.index is not None:
            by_index.setdefault(id(v.index), (v.index, []))[1].append(v)
    for index, members in by_index.values():
        index.shift(members, dx, dy)


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
