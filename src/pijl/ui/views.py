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
from collections.abc import Mapping
from contextlib import contextmanager
from types import MappingProxyType

import numpy as np
import pyglet
from pyglet import shapes

from ..sim import Part, Pin, Wire
from . import theme as T
from .canvas import Canvas
from .paint import Pair, Rgb, sample, with_hue
from .sdf_shapes import (
    DOT,
    RECT,
    RECT_OUTLINE,
    SEGMENT,
    SEGMENT_HALO,
    SHOW_BY_CODE,
    SHOW_OFF,
    _rgba,
    show,
)
from .sdf_text import NO_SLOTS, SDFLabel, SDFText
from .spatial import SpatialIndex

Point = tuple[float, float]

_seq = (
    itertools.count()
)  # creation order of views: newer ones are drawn (and hit) on top


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
    # Rigid moves (put_down): (part uids, wire uids, dx, dy), where every one of them
    # only moved, exactly: what the undo history stores as a delta (see document.py)
    # instead of their data. take() drops them too; take_moves() first to keep them.
    moved: list[tuple[list[int], list[int], float, float]] = []

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
    def take_moves(cls) -> list[tuple[list[int], list[int], float, float]]:
        out, cls.moved = cls.moved, []
        return out

    @classmethod
    def take(cls) -> tuple[set[int], set[int], set[int], set[int]]:
        """(parts, wires, paint_parts, paint_wires), and start over."""
        cls.moved = []
        out = cls.parts, cls.wires, cls.paint_parts, cls.paint_wires
        cls.parts, cls.wires, cls.paint_parts, cls.paint_wires = (
            set(),
            set(),
            set(),
            set(),
        )
        return out


class Layers:
    """Draw order in the world (lower order draws first). The canvas sorts its instance
    buffers by these; the few pyglet shapes left (overlay) draw after all of them."""

    def __init__(self) -> None:
        self.wire_halo = pyglet.graphics.Group(
            order=-1
        )  # glow under selected / edited wires
        self.wires = pyglet.graphics.Group(order=0)
        self.selection = pyglet.graphics.Group(
            order=1
        )  # part outlines, just under the bodies
        self.bodies = pyglet.graphics.Group(order=2)
        self.pins = pyglet.graphics.Group(order=3)
        self.tags = pyglet.graphics.Group(
            order=4
        )  # pin name tag backgrounds: over wires and parts
        self.text_order = 5  # SDFText's layer
        self.overlay = pyglet.graphics.Group(order=6)


class _GradientLine(shapes.Line):
    """A Line whose two ends can have different colors (the GPU blends between them).
    For the HUD (menu swatches); wires use sdf_shapes.Segment."""

    def __init__(self, *args, **kwargs) -> None:
        self._rgba2: tuple[int, int, int] | None = (
            None  # end color; None: same as the start
        )
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
        self._vertex_list.colors[:] = (
            a + b + b + a + b + a
        )  # vertex order of shapes.Line: start, end, end, ...


_STALE = object()  # Polyline._layout_key when the points changed: lay out again
GRADIENT_STEPS = (
    8  # pieces per stretch between two gradient stops (OKLab isn't linear in RGB)
)


class Polyline:
    """Thick line through several points, rounded at the corners (the segments meeting
    there have round caps) so they have no gaps. One color, or a gradient along its
    length (set_gradient). Colors are (off, on) pairs; `state` picks which one shows
    (or a pattern: X, Z, a conflict; see sdf_shapes.show).

    Its segments are slots in a SEGMENT buffer (`_slots`), not objects: every wire on
    the board has one of these."""

    __slots__ = (
        "canvas",
        "group",
        "thickness",
        "buf",
        "_pair",
        "_stops",
        "_on",
        "source",
        "_opacity",
        "_lift",
        "_sel",
        "_slots",
        "points",
        "_layout_key",
        "_plain",
        "_vfracs",
        "_colors",
    )

    def __init__(
        self,
        points: list[Point],
        color,
        canvas: Canvas,
        group: pyglet.graphics.Group,
        thickness: float = T.WIRE_THICKNESS,
    ) -> None:
        self._init(color, canvas, group, thickness)
        self.set_points(points)

    def _init(
        self, color, canvas: Canvas, group: pyglet.graphics.Group, thickness: float
    ) -> None:
        """Everything but the points and segments."""
        self.canvas, self.group, self.thickness = canvas, group, thickness
        self.buf = canvas.buffer(SEGMENT, group)
        self._pair: Pair = (color, color)
        self._stops: list[tuple[float, Pair]] | None = (
            None  # gradient: (fraction of the length, pair)
        )
        self._on = SHOW_OFF  # the segments' state byte
        self.source = -1  # the circuit wire whose state the segments show, if any
        self._opacity = 255
        self._lift = 0.0
        self._sel = 0  # the segments' `sel` byte (see select_many)
        self._slots = NO_SLOTS  # the segments' slots in `buf`, in order along the line
        self.points: list[Point] = []
        # What the segments are laid out for, so a gradient that only changes colors
        # recolors them instead of laying them out again.
        self._layout_key: object = _STALE
        self._plain = (
            True  # laid out for one color (no gradient, or nothing to spread it over)
        )
        # Fraction of the length at each segment vertex (None when plain: then the
        # vertices are just the points), and per stops: rgba arrays per vertex (off, on)
        self._vfracs: list[float] | None = None
        self._colors: dict[tuple, tuple[np.ndarray, np.ndarray]] | None = None

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
        key = _stops_key(stops)
        if key != self._layout_key:
            self._lay_out()
            self._layout_key = key
        slots = self._slots
        if not slots.size:
            return
        if self._plain:
            self._color_plain()
            return
        k = tuple(stops)
        if self._colors is None:
            self._colors = {}
        if k not in self._colors:
            self._colors[k] = _gradient_colors(stops, self._vfracs)
        _write_gradient(self.buf, slots, *self._colors[k])

    def _color_plain(self) -> None:
        """Every segment in the one (off, on) pair."""
        if self._slots.size:
            _write_plain(self.buf, self._slots, self._pair)

    def _lay_out(self) -> None:
        """Place segments along the points (see _layout)."""
        self._colors = None
        verts, self._vfracs, corner_idx = _layout(self.points, self._stops)
        self._plain = self._vfracs is None
        n_seg = max(len(verts) - 1, 0)
        buf, slots = self.buf, self._slots
        if len(slots) < n_seg:  # new segments, set up like the others
            new = buf.alloc_many(n_seg - len(slots))
            _new_segments(buf, new, self.thickness, self._opacity, self._lift, self._sel)
            if self._on:
                buf.set_state(new, self._on)
            if self.source >= 0:
                buf.show_wires(new, self.source)
            slots = self._slots = np.concatenate((slots, new))
        elif len(slots) > n_seg:  # (the last ones go, last first)
            for s in slots[n_seg:].tolist()[::-1]:
                buf.free(s)
            slots = self._slots = slots[:n_seg] if n_seg else NO_SLOTS
        if n_seg:
            _write_layout(buf, slots, verts, corner_idx)

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
        self._color_plain()

    @property
    def state(self) -> int:
        return self._on

    @state.setter
    def state(self, value) -> None:
        """A SHOW_* byte, a Level or a bool."""
        v = value if type(value) is int else show(value)
        if v != self._on:
            self._on = v
            if self._slots.size:
                self.buf.set_state(self._slots, v)

    @property
    def opacity(self) -> int:
        return self._opacity

    @opacity.setter
    def opacity(self, value: int) -> None:
        self._opacity = value
        if self._slots.size:
            self.buf.f["flags"][self._slots, 1] = value
            self.buf.mark_many(self._slots)

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

    def distance_to(self, wx: float, wy: float) -> float:
        return min(
            (
                _segment_distance(wx, wy, a, b)
                for a, b in zip(self.points, self.points[1:])
            ),
            default=math.inf,
        )

    def delete(self) -> None:
        for s in self._slots.tolist():
            self.buf.free(s)
        self._slots = NO_SLOTS


def _layout(pts: list[Point], stops) -> tuple[list[Point], list[float] | None, list[int]]:
    """Where a line's segments go: (vertices, the fraction of the length at each one,
    which of them are real corners). One color (`stops` None) is a segment per pair of
    points, and the fractions are None: only the count matters. A gradient splits
    them further: at every stop, and in GRADIENT_STEPS pieces between stops, so each
    piece blends only a little."""
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.dist(a, b))
    total = cum[-1]
    if stops is None or total == 0:
        return pts, None, list(range(1, len(pts) - 1))
    fracs = [c / total for c in cum]
    cuts = set()
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
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
    return verts, vfracs, corner_idx


def _new_segments(buf, new: np.ndarray, thickness: float, opacity: int, lift: float, sel: int) -> None:
    """Fresh segments set up like a line's others (the layout writes the rest)."""
    f = buf.f
    f["radius"][new] = thickness / 2
    f["flags"][new] = 0, opacity, 0, 0
    f["lift"][new] = 1.0 if lift else 0.0
    if sel:
        f["sel"][new, 0] = sel
    buf.mark_many(new)


def _write_layout(buf, slots: np.ndarray, verts: list[Point], corner_idx: list[int]) -> None:
    # round caps only at the real corners (pieces of one straight segment need none, and the
    # line's own ends stay square: they sit under a pin or junction dot, or on the cursor)
    v = np.array(verts, np.float64)
    k = np.arange(len(slots))
    f = buf.f
    f["a"][slots] = v[:-1]
    f["b"][slots] = v[1:]
    f["flags"][slots, 2] = np.where(np.isin(k, corner_idx), 255, 0)
    f["flags"][slots, 3] = np.where(np.isin(k + 1, corner_idx), 255, 0)
    buf.mark_many(slots)


def _gradient_colors(stops, vfracs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """rgba per vertex, off and on, of a gradient laid out at these fractions."""
    pairs = [sample(stops, f) for f in vfracs]
    return (
        np.array([_rgba(p[0]) for p in pairs], np.uint8),
        np.array([_rgba(p[1]) for p in pairs], np.uint8),
    )


def _write_gradient(buf, slots: np.ndarray, offs: np.ndarray, ons: np.ndarray) -> None:
    f = buf.f
    f["ca"][slots], f["cb"][slots] = offs[:-1], offs[1:]
    f["ca_on"][slots], f["cb_on"][slots] = ons[:-1], ons[1:]
    buf.mark_many(slots)


def _write_plain(buf, slots: np.ndarray, pair: Pair) -> None:
    f = buf.f
    off, on = _rgba(pair[0]), _rgba(pair[1])
    f["ca"][slots] = f["cb"][slots] = off
    f["ca_on"][slots] = f["cb_on"][slots] = on
    buf.mark_many(slots)


def _stops_key(stops) -> tuple | None:
    """What a gradient's layout depends on: where the stops are and which neighbors
    differ, not the colors themselves (see _layout)."""
    if stops is None:
        return None
    return (
        tuple(f for f, _ in stops),
        tuple(c0 != c1 for (_, c0), (_, c1) in zip(stops, stops[1:])),
    )


# ---- shapes by slot -----------------------------------------------------------------
# Views keep their shapes as slots in instance buffers (see canvas.py), not as
# sdf_shapes objects: a part used to own a Rect, two Dots and two labels' worth of
# Python objects. These write one shape's fields the way those objects did.


def _set_opacity(buf, slot: int, value: int) -> None:
    buf.f["flags"][slot, 1] = value
    buf.mark(slot)


def _set_rect(buf, slot: int, rect: tuple) -> None:
    buf.f["rect"][slot] = rect
    buf.mark(slot)


def _set_dot_colors(buf, slot: int, off, on) -> None:
    f = buf.f
    f["color"][slot] = _rgba(off)
    f["color_on"][slot] = _rgba(on)
    buf.mark(slot)


def _place_wire_dot(buf, slot: int, xy: Point) -> None:
    """A junction dot (a zero-length segment, both ends capped) at xy."""
    f = buf.f
    f["a"][slot] = xy
    f["b"][slot] = xy
    f["flags"][slot, 2:] = (255, 255)
    buf.mark(slot)


def _set_wire_dot_pair(buf, slot: int, off, on) -> None:
    f = buf.f
    f["ca"][slot] = f["cb"][slot] = _rgba(off)
    f["ca_on"][slot] = f["cb_on"][slot] = _rgba(on)
    buf.mark(slot)


def _segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    t = (
        0.0
        if length_sq == 0
        else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    )
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


class Box:
    """Sharp-edged box with a solid border, from pyglet shapes: for the HUD (menus,
    picker, prompts). Parts in the world are sdf_shapes.Rect.

    Built from a fill rectangle plus four non-overlapping border strips.
    (pyglet's BorderedRectangle interpolates color between its inner and outer
    vertices, so its border smears into a gradient once you zoom in.)
    """

    def __init__(
        self,
        w: float,
        h: float,
        border: float,
        fill,
        border_color,
        batch: pyglet.graphics.Batch,
        group: pyglet.graphics.Group,
    ) -> None:
        self.w, self.h, self.b = w, h, border
        self.fill = shapes.Rectangle(
            0, 0, w - 2 * border, h - 2 * border, color=fill, batch=batch, group=group
        )
        # bottom, top, left, right (left/right sit between top and bottom)
        self.edges = [
            shapes.Rectangle(
                0, 0, w, border, color=border_color, batch=batch, group=group
            ),
            shapes.Rectangle(
                0, 0, w, border, color=border_color, batch=batch, group=group
            ),
            shapes.Rectangle(
                0,
                0,
                border,
                h - 2 * border,
                color=border_color,
                batch=batch,
                group=group,
            ),
            shapes.Rectangle(
                0,
                0,
                border,
                h - 2 * border,
                color=border_color,
                batch=batch,
                group=group,
            ),
        ]

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
        for edge, pos in zip(
            self.edges, ((x, y), (x, y + h - b), (x, y + b), (x + w - b, y + b))
        ):
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
    """A part on the board: where it is, and its shapes as slots in the canvas's
    instance buffers (body, pin dots, pin tag backings) plus its text labels. Slotted,
    with no object per shape: a big board has hundreds of thousands of these."""

    __slots__ = (
        "part",
        "seq",
        "index",
        "x",
        "y",
        "w",
        "h",
        "title",
        "title_size",
        "look",
        "canvas",
        "layers",
        "text",
        "opacity",
        "lifted",
        "_selected",
        "_tints",
        "body_tint",
        "body",
        "dots",
        "kind_text",
        "_name",
        "pin_tags",
    )

    def __init__(
        self,
        part: Part,
        x: float,
        y: float,
        canvas: Canvas,
        layers: Layers,
        text: SDFText,
        pin_labels: bool = True,
        index: SpatialIndex | None = None,
    ) -> None:
        """`index`: where to register for hit testing; kept up to date as it moves."""
        self._init(part, x, y, canvas, layers, text, index)
        _make_part_shapes([self], pin_labels)

    @classmethod
    def many(
        cls,
        placed: list[tuple[Part, float, float]],
        canvas: Canvas,
        layers: Layers,
        text: SDFText,
        pin_labels: bool = True,
        index: SpatialIndex | None = None,
    ) -> list[PartView]:
        """A view for each (part, x, y): the same as making them one by one, in order, but
        their shapes are made all at once (undo, loading, pasting thousands of parts)."""
        views = []
        for part, x, y in placed:
            view = cls.__new__(cls)
            view._init(part, x, y, canvas, layers, text, index)
            views.append(view)
        _make_part_shapes(views, pin_labels)
        return views

    def _init(
        self,
        part: Part,
        x: float,
        y: float,
        canvas: Canvas,
        layers: Layers,
        text: SDFText,
        index: SpatialIndex | None,
    ) -> None:
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
        # (tag backing slot, SDF label) per pin, while shown (see set_pin_labels)
        self.pin_tags: list | tuple = ()
        self.opacity = 255
        self.lifted = (
            False  # being dragged: drawn at the canvas's offset (see set_lifted)
        )
        # Selected: the body's `sel` flag, drawn as a ring around it (see select_many).
        self._selected = False
        self._tints: list[Rgb | None] | None = None  # see pin_tints
        self.body_tint: Rgb | None = None  # hue for a lit body (switches, LEDs)
        # Made by _make_part_shapes: slots in the bodies (RECT) and pins (DOT) buffers,
        # the title, and the user's label (made when it first has text: see `name`)
        self.body: int = -1
        self.dots: tuple[int, ...] = ()
        self.kind_text: SDFLabel
        self._name: SDFLabel | None = None

    @property
    def name(self) -> SDFLabel:
        """The label showing the user's name for the part (part.label)."""
        if self._name is None:  # (empty, as it would have been made: see refresh_name)
            self._name = SDFLabel(
                self.text,
                "",
                *self.name_pos(),
                T.LABEL_SIZE,
                (*T.LABEL_TEXT[:3], self.opacity),
                _label_anchor(self.look),
            )
            self._name.lifted = self.lifted
        return self._name

    @property
    def pin_tints(self) -> list[Rgb | None]:
        """Hue for each lit pin (paint.py)."""
        if self._tints is None:
            return [None] * (len(self.part.inputs) + len(self.part.outputs))
        return self._tints

    @property
    def pin_labels_shown(self) -> bool:
        return bool(self.pin_tags)

    def set_pin_labels(self, on: bool) -> None:
        """Show / hide the pin name tags: outside the body, next to each pin, on a dark
        backing so they read over wires and other parts. Only parts whose look asks."""
        on = on and self.look.pin_labels
        if on == self.pin_labels_shown:
            return
        if self.pin_tags:
            buf = self.canvas.buffer(RECT, self.layers.tags)
            for bg, label in self.pin_tags:
                buf.free(bg)
                label.delete()
        self.pin_tags = ()
        if on:
            _make_pin_tags([(self, self.text.labels(self._tag_specs()))])

    def _tag_specs(self) -> list[tuple]:
        """SDFText.labels specs for the pin name tags' text, in place (see _tag_at)."""
        t = self.part.type
        color = (*T.LABEL_TEXT[:3], self.opacity)
        return [
            (
                (t.ins if pin.is_input else t.outs)[pin.index],
                *self._tag_at(pin),
                T.PIN_LABEL_SIZE,
                color,
                "right" if pin.is_input else "left",
            )
            for pin in self.part.pins
        ]

    def _tag_at(self, pin: Pin) -> Point:
        """Where a pin's tag text is anchored: inputs' to the left of the pin, outputs' right."""
        px, py = self.pin_pos(pin)
        off = T.PIN_RADIUS + T.PIN_TAG_GAP + T.PIN_TAG_PAD[0]
        return (px - off if pin.is_input else px + off), py

    def _place_pin_tags(self) -> None:
        if not self.pin_tags:
            return
        buf = self.canvas.buffer(RECT, self.layers.tags)
        for (bg, label), pin in zip(self.pin_tags, self.part.pins):
            label.move_to(*self._tag_at(pin))
            _set_rect(buf, bg, _tag_backing(label, pin))

    @property
    def selected(self) -> bool:
        return self._selected

    def set_selected(self, on: bool) -> None:
        select_many([self], [], on)

    def intersects(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Does the body overlap the world-space rectangle (x0, y0)-(x1, y1)?"""
        return (
            self.x <= x1
            and x0 <= self.x + self.w
            and self.y <= y1
            and y0 <= self.y + self.h
        )

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
        bodies = self.canvas.buffer(RECT, self.layers.bodies)
        bodies.f["rect"][self.body, :2] = (x, y)
        bodies.mark(self.body)
        self.kind_text.move_to(x + self.w / 2, y + self.h / 2)
        if self._name is not None:
            self._name.move_to(*self.name_pos())
        if self.dots:
            buf = self.canvas.buffer(DOT, self.layers.pins)
            center = buf.f["center"]
            for dot, pin in zip(self.dots, self.part.pins):
                center[dot] = self.pin_pos(pin)
                buf.mark(dot)
        self._place_pin_tags()
        self._register()

    def name_pos(self) -> Point:
        if self.look.label == "left":
            return self.x - T.LABEL_GAP, self.y + self.h / 2
        if self.look.label == "right":
            return self.x + self.w + T.LABEL_GAP, self.y + self.h / 2
        return self.x + self.w / 2, self.y - T.LABEL_GAP - self.text.cap_height(
            T.LABEL_SIZE
        ) / 2

    def refresh_name(self) -> None:
        """Show part.label (after it was edited)."""
        Touched.parts.add(self.part.uid)  # (labels don't change colors)
        if self._name is not None or self.part.label:
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
        _set_opacity(self.canvas.buffer(RECT, self.layers.bodies), self.body, a)
        self.kind_text.opacity = a
        if self._name is not None:
            self._name.opacity = a
        if self.pin_tags:
            tags = self.canvas.buffer(RECT, self.layers.tags)
            for bg, label in self.pin_tags:
                label.opacity = a
                _set_opacity(tags, bg, a)  # (multiplies the backing's own alpha)
        if self.dots:
            pins = self.canvas.buffer(DOT, self.layers.pins)
            for dot in self.dots:
                _set_opacity(pins, dot, a)

    def set_tints(self, pins: list[Rgb | None], body: Rgb | None) -> None:
        if pins == self.pin_tints and body == self.body_tint:
            return
        self._tints = None if all(t is None for t in pins) else pins
        self.body_tint = body
        self._recolor()

    def _recolor(self) -> None:
        """Both colors of every pin (and of a lit body), per the tints."""
        if self.dots:
            buf = self.canvas.buffer(DOT, self.layers.pins)
            for dot, tint in zip(self.dots, self.pin_tints):
                _set_dot_colors(buf, dot, T.PIN_OFF, with_hue(T.PIN_ON, tint))
        if self.look.lit:
            off, on = (
                [with_hue(c, self.body_tint) for c in theme_color(name)]
                for name in self.look.lit
            )
            buf = self.canvas.buffer(RECT, self.layers.bodies)
            f = buf.f
            f["fill"][self.body], f["edge"][self.body] = _rgba(off[0]), _rgba(off[1])
            f["fill_on"][self.body], f["edge_on"][self.body] = _rgba(on[0]), _rgba(on[1])
            buf.mark(self.body)

    def delete(self) -> None:
        delete_views([self], [])


def _label_anchor(look) -> str:
    """Where the user's label hangs off its anchor point (see _look_data)."""
    return {"left": "right", "right": "left"}.get(look.label, "center")


SELECTED, LIFTED = 1, 2  # WireTable.flags


class WireTable:
    """Every wire view's data, as columns with a row per view: the ends, bends, colors,
    flags, and the slots of its shapes (line segments, junction dots) in the canvas's
    SEGMENT buffer. A WireView is just (wire, row, table). A big board has hundreds of
    thousands of wires; as objects (a view, its line, tuples and lists of points, an
    array of slots each) they were most of its memory, and moving many was a loop of
    small Python edits instead of a few array ones.

    Rows aren't reused (like circuit slots): a dead view's row stays dead, so a stale
    handle can't show another wire."""

    def __init__(self, canvas: Canvas, layers: Layers, index: SpatialIndex | None) -> None:
        self.canvas, self.layers, self.index = canvas, layers, index
        self._buf = None
        self.n = 0  # rows handed out
        cap = 1024
        self.xy = np.zeros((cap, 4))  # src x, y, dst x, y
        self.ints = np.zeros(cap, np.uint8)  # bits: which of those were given as ints
        self.seg = np.full(cap, -1, np.int32)  # the line's segment, when it has one
        self.dot = np.full((cap, 2), -1, np.int32)  # junction dot on src / dst end
        self.flags = np.zeros(cap, np.uint8)  # SELECTED, LIFTED
        self.opacity = np.full(cap, 255, np.uint8)
        self.seq = np.zeros(cap, np.int64)  # see _seq
        self.wslot = np.full(cap, -1, np.int32)  # the circuit wire's slot
        # Objects, None meaning the usual: segments when not exactly one (else see seg),
        # bend points (a tuple), color name, paint.py's stops, the line's own gradient
        # (None: one color) and its one (off, on) color pair (None: neutral).
        self.segs = np.full(cap, None, object)
        self.bends = np.full(cap, None, object)
        self.color = np.full(cap, None, object)
        self.stops = np.full(cap, None, object)
        self.lstops = np.full(cap, None, object)
        self.pair = np.full(cap, None, object)
        # row -> [layout key, fractions per vertex, colors per stops] (see _layout), for
        # the lines laid out for a gradient; the rest are a segment per pair of points
        self.grad: dict[int, list] = {}

    _COLS = (
        "xy", "ints", "seg", "dot", "flags", "opacity", "seq", "wslot",
        "segs", "bends", "color", "stops", "lstops", "pair",
    )  # fmt: skip

    @property
    def buf(self):
        if self._buf is None:
            self._buf = self.canvas.buffer(SEGMENT, self.layers.wires)
        return self._buf

    def new_rows(self, k: int) -> range:
        while self.n + k > len(self.seg):
            n = len(self.seg)
            for name in self._COLS:
                old = getattr(self, name)
                fill = {"seg": -1, "dot": -1, "opacity": 255, "wslot": -1}.get(name, 0)
                new = np.full((2 * n, *old.shape[1:]), None if old.dtype == object else fill, old.dtype)
                new[:n] = old
                setattr(self, name, new)
        rows = range(self.n, self.n + k)
        self.n += k
        return rows

    def forget(self, rows: np.ndarray) -> None:
        """Dead views' rows (their shapes are freed already): let go of what they hold."""
        self.seg[rows] = -1
        self.dot[rows] = -1
        self.flags[rows] = 0
        for col in (self.segs, self.bends, self.color, self.stops, self.lstops, self.pair):
            col[rows] = None
        for r in rows.tolist():
            self.grad.pop(r, None)

    # ---- one row -----------------------------------------------------------------

    def end(self, row: int, k: int) -> Point:
        """The src (k = 0) or dst (1) end, as it was given (ints stay ints)."""
        x, y = self.xy[row, 2 * k : 2 * k + 2].tolist()
        bits = int(self.ints[row]) >> (2 * k)
        if bits & 3:
            x, y = (int(x) if bits & 1 else x), (int(y) if bits & 2 else y)
        return x, y

    def set_end(self, row: int, k: int, p: Point) -> None:
        x, y = p
        self.xy[row, 2 * k], self.xy[row, 2 * k + 1] = x, y
        self.ints[row] = (int(self.ints[row]) & ~(3 << 2 * k)) | _int_bits(x, y, k)

    def points(self, row: int) -> list[Point]:
        b = self.bends[row]
        return [self.end(row, 0), *b, self.end(row, 1)] if b else [self.end(row, 0), self.end(row, 1)]

    def slots(self, row: int) -> np.ndarray:
        """The line's segments, in order along it."""
        s = int(self.seg[row])
        if s >= 0:
            return np.array([s], np.intp)
        segs = self.segs[row]
        return NO_SLOTS if segs is None else segs

    def _set_slots(self, row: int, slots: np.ndarray) -> None:
        if len(slots) == 1:
            self.seg[row], self.segs[row] = slots[0], None
        else:
            self.seg[row], self.segs[row] = -1, (slots if len(slots) else None)

    def segments(self, rows: np.ndarray) -> np.ndarray:
        """The segments of many lines (a line's in order, the lines in no set order)."""
        seg = self.seg[rows]
        one = seg[seg >= 0]
        more = [s for s in self.segs[rows[seg < 0]].tolist() if s is not None]
        return np.concatenate((one, *more)).astype(np.intp) if more else one.astype(np.intp)

    # ---- one row's line (what Polyline does for one line) ---------------------------

    def redraw(self, row: int) -> None:
        """Lay the line out again along its points, and color it."""
        self._build(row, stale=True)

    def _build(self, row: int, stale: bool = False) -> None:
        stops = self.lstops[row]
        key = _stops_key(stops)
        g = self.grad.get(row)
        if stale or key != (None if g is None else g[0]):
            verts, vfracs, corner_idx = _layout(self.points(row), stops)
            if key is None:
                self.grad.pop(row, None)
            else:
                self.grad[row] = g = [key, vfracs, None]
            self._place(row, verts, corner_idx)
        slots = self.slots(row)
        if not slots.size:
            return
        g = self.grad.get(row)
        if g is None or g[1] is None:  # one color
            _write_plain(self.buf, slots, self.pair[row] or _NEUTRAL)
            return
        if g[2] is None:
            g[2] = {}
        k = tuple(stops)
        if k not in g[2]:
            g[2][k] = _gradient_colors(stops, g[1])
        _write_gradient(self.buf, slots, *g[2][k])

    def _place(self, row: int, verts: list[Point], corner_idx: list[int]) -> None:
        n_seg = max(len(verts) - 1, 0)
        buf, slots = self.buf, self.slots(row)
        if len(slots) < n_seg:  # new segments, set up like the others
            new = buf.alloc_many(n_seg - len(slots))
            flags = int(self.flags[row])
            _new_segments(
                buf,
                new,
                T.WIRE_THICKNESS,
                int(self.opacity[row]),
                flags & LIFTED,
                255 if flags & SELECTED else 0,
            )
            buf.show_wires(new, int(self.wslot[row]))
            slots = np.concatenate((slots, new))
        elif len(slots) > n_seg:  # (the last ones go, last first)
            for s in slots[n_seg:].tolist()[::-1]:
                buf.free(s)
            slots = slots[:n_seg] if n_seg else NO_SLOTS
        self._set_slots(row, slots)
        if n_seg:
            _write_layout(buf, slots, verts, corner_idx)

    def set_pair(self, row: int, pair: Pair) -> None:
        """One (off, on) color pair for the whole line (drops any gradient)."""
        self.pair[row] = None if pair == _NEUTRAL else pair
        if self.lstops[row] is not None:
            self.lstops[row] = None
            self._build(row)
            return
        slots = self.slots(row)
        if slots.size:
            _write_plain(self.buf, slots, pair)

    def set_gradient(self, row: int, stops: list[tuple[float, Pair]]) -> None:
        """Color pairs at fractions of the length, ascending; the line blends between them."""
        if all(c == stops[0][1] for _, c in stops):
            self.set_pair(row, stops[0][1])  # one color after all: plain line
            return
        if stops != self.lstops[row]:
            self.lstops[row] = list(stops)
            self._build(row)


def _int_bits(x, y, k: int) -> int:
    return ((type(x) is int) | (type(y) is int) << 1) << 2 * k


def _fill_rows(views: list[WireView], specs: list[tuple]) -> None:
    """New views' rows (consecutive), from their (wire, src, bends, dst, color)."""
    if not views:
        return
    t, n = views[0].table, len(views)
    rows = slice(views[0].row, views[0].row + n)
    t.seq[rows] = list(itertools.islice(_seq, n))
    t.wslot[rows] = np.fromiter((w.slot for w, *_ in specs), np.int64, n)
    ends = [(*src, *dst) for _, src, _, dst, _ in specs]
    t.xy[rows] = np.fromiter(itertools.chain.from_iterable(ends), np.float64, 4 * n).reshape(n, 4)
    t.ints[rows] = [
        (type(a) is int) | (type(b) is int) << 1 | (type(c) is int) << 2 | (type(d) is int) << 3
        for a, b, c, d in ends
    ]
    for i, (_, _, bends, _, color) in enumerate(specs):
        if bends:
            t.bends[rows.start + i] = tuple(bends)
        if color is not None:  # a T.WIRE_COLORS name; None = Default (inherit from the ends)
            t.color[rows.start + i] = color


class WireView:
    """A wire drawn from its src pin, through user-placed bend points, to its dst pin.

    Bend points are layout data, so they live here and not in the sim. So is color:
    `color` is what the user picked (None: Default), `stops` the gradient that
    paint.py worked out from it and from what the wire connects to.

    A handle: what it knows is a row of its WireTable.
    """

    __slots__ = ("wire", "row", "table")

    def __init__(
        self,
        wire: Wire,
        src: Point,
        bends: list[Point],
        dst: Point,
        table: WireTable,
        color: str | None = None,
    ) -> None:
        self.wire, self.table, self.row = wire, table, table.new_rows(1)[0]
        spec = (wire, src, bends, dst, color)
        _fill_rows([self], [spec])
        _make_wire_shapes([self], [[src, *bends, dst]])

    @classmethod
    def many(cls, specs: list[tuple], table: WireTable) -> list[WireView]:
        """A view for each (wire, src, bends, dst, color): like making them one by one, in
        order, with the shapes made all at once."""
        views = []
        for spec, row in zip(specs, table.new_rows(len(specs))):
            view = cls.__new__(cls)
            view.wire, view.table, view.row = spec[0], table, row
            views.append(view)
        _fill_rows(views, specs)
        _make_wire_shapes(views, [[src, *bends, dst] for _, src, bends, dst, _ in specs])
        return views

    @property
    def seq(self) -> int:
        return int(self.table.seq[self.row])

    @property
    def index(self) -> SpatialIndex | None:
        return self.table.index

    @property
    def src(self) -> Point:
        return self.table.end(self.row, 0)

    @src.setter
    def src(self, p: Point) -> None:
        """(Just the number: set_bends or set_ends redraws.)"""
        self.table.set_end(self.row, 0, p)

    @property
    def dst(self) -> Point:
        return self.table.end(self.row, 1)

    @dst.setter
    def dst(self, p: Point) -> None:
        self.table.set_end(self.row, 1, p)

    @property
    def bends(self) -> tuple[Point, ...]:
        return self.table.bends[self.row] or ()

    @property
    def points(self) -> list[Point]:
        """Every vertex: src pin, bends..., dst pin. Segment k runs points[k] -> points[k+1]."""
        return self.table.points(self.row)

    @property
    def color(self) -> str | None:
        return self.table.color[self.row]

    @color.setter
    def color(self, value: str | None) -> None:
        if value != self.color:
            Touched.wire(self.wire.uid)
        self.table.color[self.row] = value

    @property
    def stops(self) -> list[tuple[float, Pair]] | tuple:
        """Empty: neutral, the classic grey / red."""
        stops = self.table.stops[self.row]
        return () if stops is None else stops

    @property
    def lifted(self) -> bool:
        return bool(self.table.flags[self.row] & LIFTED)

    @property
    def selected(self) -> bool:
        return bool(self.table.flags[self.row] & SELECTED)

    def set_ends(self, src: Point, dst: Point) -> None:
        t = self.table
        if t.xy[self.row].tolist() == [*src, *dst]:
            return  # (re-attaching after a move often lands exactly where it was)
        t.set_end(self.row, 0, src)
        t.set_end(self.row, 1, dst)
        self._redraw()

    def set_bends(self, bends: list[Point]) -> None:
        self.table.bends[self.row] = tuple(bends) or None
        self._redraw()

    def _redraw(self) -> None:
        t, row = self.table, self.row
        t.redraw(row)
        for k in (0, 1):
            dot = int(t.dot[row, k])
            if dot >= 0:
                _place_wire_dot(t.buf, dot, t.end(row, k))
        if t.index is not None:
            t.index.put_polyline(self, self.points)
        Touched.wire(
            self.wire.uid
        )  # (its shape: colors of branches depend on where they attach)

    def set_lifted(self, on: bool) -> None:
        """Drawn shifted by the canvas's offset, like PartView.set_lifted: for wires that
        move rigidly with a dragged selection. Put it down with set_ends / set_bends,
        or put_down."""
        lift([], [self], on)

    def set_ghost(self, ghost: bool) -> None:
        """Semi-transparent while being carried around before placement (paste)."""
        t, row = self.table, self.row
        a = T.GHOST_OPACITY if ghost else 255
        t.opacity[row] = a
        slots = t.slots(row)
        if slots.size:
            t.buf.f["flags"][slots, 1] = a
            t.buf.mark_many(slots)
        for dot in t.dot[row].tolist():
            if dot >= 0:
                _set_opacity(t.buf, dot, a)

    def set_selected(self, on: bool) -> None:
        select_many([], [self], on)

    def inside(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Is the whole wire within the world-space rectangle (x0, y0)-(x1, y1)?"""
        return all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in self.points)

    def _recolor(self) -> None:
        """Both colors of the line and dots: the gradient, else the classic grey / red.
        (X, Z and conflicts are patterns the state picks, whatever these are.)"""
        t, row = self.table, self.row
        dots = t.dot[row].tolist()
        stops = self.stops
        if not stops:
            t.set_pair(row, _NEUTRAL)
            for dot in dots:
                if dot >= 0:
                    _set_wire_dot_pair(t.buf, dot, *_NEUTRAL)
            return
        t.set_gradient(row, stops)
        for k, dot in enumerate(dots):
            if dot >= 0:
                _set_wire_dot_pair(t.buf, dot, *sample(stops, 1.0 if k else 0.0))

    def set_stops(self, stops: list[tuple[float, Pair]]) -> None:
        if stops == self.stops or not (stops or self.stops):
            return
        self.table.stops[self.row] = stops
        self._recolor()

    def color_at(self, p: Point) -> Pair | None:
        """The gradient at the spot nearest to `p` (where a branch attaches); None if neutral."""
        if not self.stops:
            return None
        pts = self.points
        total = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        return sample(self.stops, arc_length_at(pts, p) / total if total else 0.0)

    def distance_to(self, wx: float, wy: float) -> float:
        pts = self.points
        return min(
            (_segment_distance(wx, wy, a, b) for a, b in zip(pts, pts[1:])),
            default=math.inf,
        )

    def delete(self) -> None:
        delete_views([], [self])


_NEUTRAL: Pair = (T.WIRE_OFF, T.WIRE_ON)  # a wire's colors before paint.py's (one shared pair)


# ---- making views ------------------------------------------------------------------
# Views are made in batches (a batch of one for the constructors): per shape role, the
# slots they would have got one view at a time, then each field written as one array.
# Loading or undoing thousands of parts one constructor at a time was seconds.


FREEZE_AT = 50_000  # objects a batch must add for paused_gc to freeze them


@contextmanager
def paused_gc():
    """No cycle collection meanwhile: a big batch makes hundreds of thousands of objects,
    and Python's collector would walk the whole heap again and again as it did (half the
    time of undoing a 20k-part delete).

    A batch that grew the heap by FREEZE_AT objects or more ends with gc.freeze(): what
    it made (a whole board, a big paste) is moved where the collector never looks
    again. Otherwise every later collection would still walk it all, hundreds of ms at
    a million objects, landing on whatever runs next. The editor makes next to no
    cyclic garbage that grows with the board (deleted views are acyclic, freed by
    reference counting even when frozen), so what a freeze can strand for good is a
    few hundred objects at most: pyglet's per-call ctypes leftovers."""
    was = gc.isenabled()
    gc.disable()
    before = gc.get_count()[0]
    try:
        yield
    finally:
        if gc.get_count()[0] - before >= FREEZE_AT:
            gc.freeze()
        if was:
            gc.enable()


def _make_part_shapes(views: list[PartView], pin_labels: bool) -> None:
    """Bodies, titles, user labels, pin dots, colors, placement: a new view's shapes.
    Positions are worked out with arrays, per pin layout (views of one type share
    theirs), in the same arithmetic as pin_pos / name_pos."""
    if not views:
        return
    v0 = views[0]
    canvas, layers, text = v0.canvas, v0.layers, v0.text
    n = len(views)
    xs = np.fromiter((v.x for v in views), np.float64, n)
    ys = np.fromiter((v.y for v in views), np.float64, n)
    ws = np.fromiter((v.w for v in views), np.float64, n)
    hs = np.fromiter((v.h for v in views), np.float64, n)
    # pins: per view, how many and where they start in the flat list
    pins = [v.part.pins for v in views]
    flat = [p for ps in pins for p in ps]
    counts = np.fromiter((len(ps) for ps in pins), np.intp, n)
    starts = np.cumsum(counts) - counts
    pin_slots = np.fromiter((p.slot for p in flat), np.intp, len(flat))
    codes = flat[0]._store.states[pin_slots] if flat else np.empty(0, np.intp)
    # what each view's look decides, and the pin layout of each (look, pins, size)
    looks: dict[int, tuple] = {}
    group_of: dict[tuple, list[int]] = {}
    for i, v in enumerate(views):
        look = v.look
        if id(look) not in looks:
            looks[id(look)] = _look_data(look)
        part = v.part
        group_of.setdefault(
            (id(look), len(part.inputs), len(part.outputs), v.w, v.h), []
        ).append(i)
    is_out = np.zeros(len(flat), bool)
    pin_dy = np.zeros(len(flat))  # from the body's middle (see pin_pos)
    lit = np.zeros(n, bool)
    label_side = np.zeros(n, np.int8)  # the user label: -1 left, 1 right, 0 below
    for (look_id, n_in, n_out, _w, _h), members in group_of.items():
        idx = np.array(members, np.intp)
        anchor, _row, lit[idx] = looks[look_id]
        label_side[idx] = {"right": -1, "left": 1}.get(anchor, 0)
        if n_in + n_out:
            out = np.array([False] * n_in + [True] * n_out)
            dy = np.array(
                [((n_in - 1) / 2 - i) * T.PIN_SPACING for i in range(n_in)]
                + [((n_out - 1) / 2 - i) * T.PIN_SPACING for i in range(n_out)]
            )
            at = (starts[idx][:, None] + np.arange(n_in + n_out)).ravel()
            is_out[at] = np.tile(out, len(idx))
            pin_dy[at] = np.tile(dy, len(idx))
    # titles and user labels (a view's title, then its label: the one-at-a-time order),
    # and pin name tags if shown
    mid_x, mid_y = xs + ws / 2, ys + hs / 2
    name_x = np.where(
        label_side < 0,
        xs - T.LABEL_GAP,
        np.where(label_side > 0, (xs + ws) + T.LABEL_GAP, mid_x),
    )
    name_y = np.where(
        label_side == 0, (ys - T.LABEL_GAP) - text.cap_height(T.LABEL_SIZE) / 2, mid_y
    )
    specs, tagged = [], []
    for v, tx, ty, nx, ny in zip(
        views, mid_x.tolist(), mid_y.tolist(), name_x.tolist(), name_y.tolist()
    ):
        specs.append((v.title, tx, ty, v.title_size, T.PART_TEXT, "center"))
        if v.part.label:  # (none: no label until it gets one, see PartView.name)
            specs.append(
                (v.part.label, nx, ny, T.LABEL_SIZE, T.LABEL_TEXT, looks[id(v.look)][0])
            )
        if pin_labels and v.look.pin_labels:
            specs += v._tag_specs()
    labels = iter(text.labels(specs))
    for v, ps in zip(views, pins):
        v.kind_text = next(labels)
        if v.part.label:
            v._name = next(labels)
        if pin_labels and v.look.pin_labels:
            tagged.append((v, [next(labels) for _ in ps]))
    _make_pin_tags(tagged)
    # bodies: the look's colors; lit ones (switches, LEDs) also on-colors, on per the first pin
    has_pins = counts > 0
    on = np.full(n, SHOW_OFF, np.uint8)
    first_pin = np.full(n, -1, np.intp)
    shows = lit & has_pins
    on[shows] = SHOW_BY_CODE[codes[starts[shows]]]
    first_pin[shows] = pin_slots[starts[shows]]
    buf = canvas.buffer(RECT, layers.bodies)
    slots = buf.alloc_many(n)
    f = buf.f
    f["rect"][slots] = np.column_stack((xs, ys, ws, hs))
    f["border"][slots] = T.PART_BORDER
    colors = np.array([looks[id(v.look)][1] for v in views], np.uint8)
    for k, name in enumerate(("fill", "fill_on", "edge", "edge_on")):
        f[name][slots] = colors[:, k]
    f["flags"][slots] = 0, 255, 0, 0
    buf.set_state(slots, on)
    f["lift"][slots] = 0.0
    buf.mark_many(slots)
    buf.show_pins(slots, first_pin)  # lit bodies follow their first pin (switches, LEDs)
    for v, body in zip(views, slots.tolist()):
        v.body = body
    # pin dots
    buf = canvas.buffer(DOT, layers.pins)
    slots = buf.alloc_many(len(flat))
    if len(flat):
        owner = np.repeat(np.arange(n), counts)
        f = buf.f
        f["center"][slots] = np.column_stack(
            (
                np.where(is_out, xs[owner] + ws[owner], xs[owner]),
                mid_y[owner] + pin_dy,
            )
        )
        f["radius"][slots] = T.PIN_RADIUS
        f["color"][slots] = _rgba(T.PIN_OFF)
        f["color_on"][slots] = _rgba(with_hue(T.PIN_ON, None))
        f["flags"][slots] = 0, 255, 0, 0
        buf.set_state(slots, SHOW_BY_CODE[codes])
        f["lift"][slots] = 0.0
        buf.mark_many(slots)
        buf.show_pins(slots, pin_slots)
    dots = slots.tolist()
    for v, k, c in zip(views, starts.tolist(), counts.tolist()):
        v.dots = tuple(dots[k : k + c])
    # the spatial index: the body, and the pins sticking out of its sides (_index_box)
    r = T.PIN_RADIUS
    boxes = np.column_stack((xs - r, ys, (xs + ws) + r, ys + hs))
    by_index: dict[int, tuple] = {}
    for i, v in enumerate(views):
        if v.index is not None:
            by_index.setdefault(id(v.index), (v.index, []))[1].append(i)
    for index, members in by_index.values():
        index.put_boxes([views[i] for i in members], boxes[members])
    Touched.part_many(
        v.part.uid for v in views
    )  # new: its own color (props) needs painting


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
    return (
        anchor,
        (_rgba(fill), _rgba(fill_on), _rgba(edge), _rgba(edge_on)),
        bool(look.lit),
    )


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
    backings = iter(slots.tolist())
    for v, labels in made:
        v.pin_tags = [(next(backings), label) for label in labels]
        if v.lifted:
            for label in labels:
                label.lifted = True


def _make_wire_shapes(views: list[WireView], lines: list[list[Point]]) -> None:
    """The segments of a new wire's line (and its junction dots), placed and colored.
    Per wire, in order: its segments, then a dot on each end that's a junction (right
    after its line: a wire crossing the junction covers it)."""
    if not views:
        return
    t = views[0].table
    buf = t.buf
    n = len(views)
    ends = [  # the ends that attach to another wire (junctions) get a dot
        tuple(end for end in ("src", "dst") if not isinstance(getattr(v.wire, end), Pin))
        for v in views
    ]
    n_pts = np.fromiter((len(pts) for pts in lines), np.intp, n)
    n_seg = np.maximum(n_pts - 1, 0)
    dot_src = np.fromiter(("src" in e for e in ends), bool, n)
    dot_dst = np.fromiter(("dst" in e for e in ends), bool, n)
    rows = n_seg + dot_src + dot_dst
    first = np.cumsum(rows) - rows  # each wire's first row
    total = int(rows.sum())
    slots = buf.alloc_many(total)
    if total:
        p = np.array([xy for pts in lines for xy in pts], np.float64).reshape(-1, 2)
        first_pt = np.cumsum(n_pts) - n_pts
        a, b = np.empty((total, 2)), np.empty((total, 2))
        radius = np.empty(total)
        caps = np.full((total, 2), 255, np.uint8)
        # segments: point k -> k + 1, round caps at the corners (the points between the ends)
        wire = np.repeat(np.arange(n), n_seg)
        k = np.arange(len(wire)) - np.repeat(np.cumsum(n_seg) - n_seg, n_seg)
        at = first[wire] + k
        a[at] = p[first_pt[wire] + k]
        b[at] = p[first_pt[wire] + k + 1]
        radius[at] = T.WIRE_THICKNESS / 2
        caps[at, 0] = np.where(k >= 1, 255, 0)
        caps[at, 1] = np.where(k + 1 <= n_seg[wire] - 1, 255, 0)
        # junction dots: zero-length, capped both ends
        for has, row, point in (
            (dot_src, n_seg, first_pt),
            (dot_dst, n_seg + dot_src, first_pt + n_pts - 1),
        ):
            w = np.flatnonzero(has)
            a[first[w] + row[w]] = b[first[w] + row[w]] = p[point[w]]
            radius[first[w] + row[w]] = T.JUNCTION_RADIUS
        f = buf.f
        f["a"][slots] = a
        f["b"][slots] = b
        f["radius"][slots] = radius
        f["ca"][slots] = f["cb"][slots] = _rgba(
            T.WIRE_OFF
        )  # neutral: paint.py gives gradients
        f["ca_on"][slots] = f["cb_on"][slots] = _rgba(T.WIRE_ON)
        f["flags"][slots] = 0, 255, 0, 0
        f["flags"][slots, 2:] = caps
        f["lift"][slots] = 0.0
        buf.mark_many(slots)
        buf.show_wires(
            slots, np.repeat(np.fromiter((v.wire.slot for v in views), np.intp, n), rows)
        )
    rows = np.fromiter((v.row for v in views), np.intp, n)
    one = n_seg == 1
    t.seg[rows[one]] = slots[first[one]]
    for i in np.flatnonzero(n_seg > 1).tolist():
        t.segs[rows[i]] = slots[first[i] : first[i] + n_seg[i]]
    t.dot[rows[dot_src], 0] = slots[(first + n_seg)[dot_src]]
    t.dot[rows[dot_dst], 1] = slots[(first + n_seg + dot_src)[dot_dst]]
    if t.index is not None:
        t.index.put_polylines(list(zip(views, lines)))
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

    def slots(self, canvas: Canvas, kind, layer, slots: list[int]) -> None:
        """Shapes of one role, by slot. (Their buffer is looked up only if there are
        any: looking one up makes it.)"""
        if slots:
            self._add(canvas.buffer(kind, layer), np.array(slots, np.intp))

    def labels(self, labels: list[SDFLabel]) -> None:
        if labels:
            self._add(labels[0].buf, np.concatenate([label.slots for label in labels]))

    def lines(self, views: list[WireView]) -> None:
        """Their lines' segments."""
        if views:
            t = views[0].table
            self._add(t.buf, t.segments(_rows_of(views)))

    def parts(self, views: list[PartView]) -> _Slots:
        if not views:
            return self
        canvas, layers = views[0].canvas, views[0].layers
        self.slots(canvas, RECT, layers.bodies, [v.body for v in views])
        self.slots(canvas, DOT, layers.pins, [d for v in views for d in v.dots])
        self.slots(canvas, RECT, layers.tags, [bg for v in views for bg, _ in v.pin_tags])
        self.labels([v.kind_text for v in views if v.kind_text.slots.size])
        self.labels(
            [v._name for v in views if v._name is not None and v._name.slots.size]
        )
        self.labels([label for v in views for _, label in v.pin_tags])
        return self

    def wires(self, views: list[WireView]) -> _Slots:
        if views:
            t = views[0].table
            rows = _rows_of(views)
            self._add(t.buf, t.segments(rows))
            dots = t.dot[rows].ravel()
            self._add(t.buf, dots[dots >= 0])
        return self

    def __iter__(self):
        for buf, arrays in self.by.values():
            yield buf, (np.concatenate(arrays) if len(arrays) > 1 else arrays[0])


def lift(parts: list[PartView], wires: list[WireView], on: bool) -> None:
    """set_lifted for many parts and wires at once."""
    parts, wires = _lift_mirrors(parts, wires, on)
    for buf, slots in _Slots().parts(parts).wires(wires):
        buf.set_lift(slots, on)


def put_down(
    parts: list[PartView], wires: list[WireView], dx: float, dy: float
) -> None:
    """Un-lift these and move them by (dx, dy): the parts as if by move_to, the wires
    whole (bends and junction ends too, no new layout). Dropping a dragged selection;
    every shape is written once. (A rigid move changes no colors: not reported to
    paint.py. The caller re-attaches wires stretched between these and the rest.)"""
    _lift_mirrors(parts, wires, False)
    if dx or dy:
        moved = (_move_part_mirrors(parts, dx, dy), _move_wire_mirrors(wires, dx, dy))
        if moved[0] or moved[1]:
            Touched.moved.append((*moved, dx, dy))
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
    if wires:
        wires[0].table.forget(_rows_of(wires))
    # the views are dead: they hold no slots any more (a second delete is a no-op)
    for v in parts:
        v.body, v.dots = -1, ()
        v.kind_text.slots = NO_SLOTS
        if v._name is not None:
            v._name.slots = NO_SLOTS
        v.pin_tags = ()
        v._selected = False


def select_many(parts: list[PartView], wires: list[WireView], on: bool) -> None:
    """set_selected for many parts and wires at once. Selection is a flag on the shapes
    themselves (`sel`: the body, the line's segments), which the canvas draws again as
    outlines and halos (see canvas.Echo, sdf_shapes.RECT_OUTLINE / SEGMENT_HALO): one
    write per buffer, and nothing to keep in step when they move, lift or go away."""
    parts = [v for v in parts if v._selected != on]
    wires = _flag_wires(wires, SELECTED, on)
    if not (parts or wires):
        return
    value = 255 if on else 0
    for v in parts:
        v._selected = on
    t = wires[0].table if wires else None
    canvas, layers = (parts[0].canvas, parts[0].layers) if parts else (t.canvas, t.layers)
    canvas.echo(RECT_OUTLINE, layers.selection, RECT, layers.bodies)
    canvas.echo(SEGMENT_HALO, layers.wire_halo, SEGMENT, layers.wires)
    flagged = _Slots()
    flagged.slots(canvas, RECT, layers.bodies, [v.body for v in parts])
    flagged.lines(wires)
    for buf, slots in flagged:
        buf.f["sel"][slots, 0] = value
        buf.mark_many(slots)


# What the views remember of their shapes (coordinates, lift), and the spatial index:
# the rest of the bulk operations, next to the one write per buffer.


def _lift_mirrors(
    parts: list[PartView], wires: list[WireView], on: bool
) -> tuple[list, list]:
    """Mark them (un)lifted; returns the ones that weren't already."""
    parts = [v for v in parts if v.lifted != on]
    wires = _flag_wires(wires, LIFTED, on)
    value = 1.0 if on else 0.0
    for v in parts:
        v.lifted = on
        v.kind_text._lift = value  # (what glyphs they get later start with)
        if v._name is not None:
            v._name._lift = value
        for _, label in v.pin_tags:
            label._lift = value
    return parts, wires


def _rows_of(views: list[WireView]) -> np.ndarray:
    return np.fromiter((v.row for v in views), np.intp, len(views))


def _flag_wires(views: list[WireView], flag: int, on: bool) -> list[WireView]:
    """Set or clear a WireTable flag on these; returns the ones it changed."""
    if not views:
        return views
    t = views[0].table
    rows = _rows_of(views)
    had = (t.flags[rows] & flag) != 0
    if on:
        t.flags[rows] |= flag
    else:
        t.flags[rows] &= ~flag & 0xFF
    changed = had != on
    return views if changed.all() else [v for v, c in zip(views, changed.tolist()) if c]


def _exact(x, d: float) -> bool:
    """Does moving `x` by d and back give x again, bit for bit? (And is it a float: an
    int would come back from the undo history as a float. See document._compress.)"""
    return type(x) is float and (x + d) - d == x


def _move_part_mirrors(views: list[PartView], dx: float, dy: float) -> list[int]:
    """Move these views' coordinates by (dx, dy). Returns the uids of the ones that
    moved exactly (see _exact); the others are reported as changed (Touched)."""
    exact, other = [], []
    for v in views:
        x, y = v.x, v.y
        v.x, v.y = x + dx, y + dy
        (exact if _exact(x, dx) and _exact(y, dy) else other).append(v.part.uid)
        for label in (v.kind_text, v._name) if v._name is not None else (v.kind_text,):
            label.x += dx
            label.y += dy
        for _, label in v.pin_tags:
            label.x += dx
            label.y += dy
    _shift_indexed(views, dx, dy)
    Touched.parts.update(other)
    return exact


def _move_wire_mirrors(views: list[WireView], dx: float, dy: float) -> list[int]:
    """_move_part_mirrors for wires. A wire whose data has no coordinates (pin to pin,
    no bends: see document.wire_data) is the same after a move: not reported at all."""
    if not views:
        return []
    t = views[0].table
    rows = _rows_of(views)
    # which ends are in the data (junctions: see wire_data), and moved exactly
    junction = np.array(
        [(not isinstance(v.wire.src, Pin), not isinstance(v.wire.dst, Pin)) for v in views],
        bool,
    ).reshape(-1, 2)
    xy, ints = t.xy[rows], t.ints[rows]
    d = np.array([dx, dy, dx, dy])
    ok = ((xy + d) - d == xy) & ((ints[:, None] >> np.arange(4)) & 1 == 0)
    end_ok = ok[:, 0::2] & ok[:, 1::2]
    has = junction.any(axis=1)
    good = (end_ok | ~junction).all(axis=1)
    t.xy[rows] = xy + d
    lost = (0 if type(dx) is int else 0b0101) | (0 if type(dy) is int else 0b1010)
    if lost:
        t.ints[rows] = ints & (~lost & 0xFF)
    for i, b in enumerate(t.bends[rows].tolist()):
        if b is None:
            continue
        has[i] = True
        good[i] &= all(_exact(x, dx) and _exact(y, dy) for x, y in b)
        t.bends[rows[i]] = tuple((x + dx, y + dy) for x, y in b)
    exact = [v.wire.uid for v, h, g in zip(views, has.tolist(), good.tolist()) if h and g]
    Touched.wires.update(
        v.wire.uid for v, h, g in zip(views, has.tolist(), good.tolist()) if h and not g
    )
    if t.index is not None:
        t.index.shift(views, dx, dy)
    return exact


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
        t = (
            0.0
            if seg == 0
            else max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / (seg * seg)))
        )
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
        t = (
            0.0
            if length_sq == 0
            else max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / length_sq))
        )
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
