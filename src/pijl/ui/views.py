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
import operator
from array import array
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
from types import MappingProxyType

import numpy as np
import pyglet
from pyglet import shapes

from ..sim import Part, Pin, Wire
from . import theme as T
from .canvas import Canvas
from .paint import Pair, Rgb, color_pair, sample, with_hue
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
    boxes: set[int] = set()  # (boxes.py: no colors, no moves-as-deltas, few of them)
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
    def take_boxes(cls) -> set[int]:
        out, cls.boxes = cls.boxes, set()
        return out

    @classmethod
    def take_moves(cls) -> list[tuple[list[int], list[int], float, float]]:
        out, cls.moved = cls.moved, []
        return out

    @classmethod
    def take(cls) -> tuple[set[int], set[int], set[int], set[int]]:
        """(parts, wires, paint_parts, paint_wires), and start over (boxes too: take
        them first with take_boxes to keep them)."""
        cls.moved = []
        cls.boxes = set()
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
        self.boxes = pyglet.graphics.Group(order=-3)  # boxes (see boxes.py): behind it all
        self.box_text_order = -2  # their labels' SDFText layer
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


def _place_wire_dot(buf, slot: int, xy: Point, free: bool = False) -> None:
    """A junction dot (a zero-length segment, both ends capped) at xy -- or, for a
    free end, a small square (a segment as long as it is thick, uncapped)."""
    f = buf.f
    if free:
        x, y = xy
        h = T.FREE_END_HALF
        f["a"][slot], f["b"][slot] = (x - h, y), (x + h, y)
        f["radius"][slot] = h
        f["flags"][slot, 2:] = (0, 0)
    else:
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


class PartTable:
    """Every part view's data, as columns with a row per view (see WireTable, the same
    idea): position, size, flags, its body's slot, and which pins are its. Per pin, by
    the circuit's pin slot: whose view it is, its side and offset, its dot's slot -- so
    where every pin of a selection is, is one array expression (Editor.refresh_wires).
    The labels are still objects (SDFLabel), held in object columns.

    Rows aren't reused, as in WireTable."""

    def __init__(
        self, canvas: Canvas, layers: Layers, text: SDFText, index: SpatialIndex | None
    ) -> None:
        self.canvas, self.layers, self.text, self.index = canvas, layers, text, index
        self.n = 0
        cap = 1024
        # (_xyb, _intsb, ...: the same memory as Python arrays, see _buffered)
        self._xyb, self.xy = _buffered("d", 2, cap)
        self._intsb, self.ints = _buffered("B", 1, cap)  # bits: x / y were given as ints
        self._whb, self.wh = _buffered("i", 2, cap)  # (always whole: see _fill_part_rows)
        self.body = np.full(cap, -1, np.int32)
        self.flags = np.zeros(cap, np.uint8)  # SELECTED, LIFTED, TAGGED
        self.opacity = np.full(cap, 255, np.uint8)
        self.seq = np.zeros(cap, np.int64)
        self.view = np.full(cap, None, object)  # the handle (one per view: identity)
        # its pins' slots: pin0, pin0 + 1, ... npin of them; or pin0 = -1 and `pinslots`
        # (a part whose pins weren't made one after another: macros can be)
        self.pin0 = np.zeros(cap, np.int32)
        self.npin = np.zeros(cap, np.int32)
        # its title's glyphs in the text's buffer, the same way (a run: glyph0 and
        # nglyph; else listed in `glyphs`). No label object: the title is where the
        # body is (see PartView.title_at), its layout cached per text (sdf_text).
        self.glyph0 = np.zeros(cap, np.int32)
        self.nglyph = np.zeros(cap, np.int32)
        self.glyphs = np.full(cap, None, object)
        # objects, None meaning the usual: title text, the kind label, the user's label
        # (made on first use), pin tags ((backing slot, label) per pin, while shown),
        # pin tints, body tint, pin slots (see pin0)
        self.title = np.full(cap, None, object)
        self.name = np.full(cap, None, object)
        self.tints = np.full(cap, None, object)
        self.body_tint = np.full(cap, None, object)
        self.pinslots = np.full(cap, None, object)
        # by pin slot
        self.pin_row = np.full(cap, -1, np.int32)
        self.pin_out = np.zeros(cap, bool)
        # from the body's middle (see PartView.pin_pos)
        self._pin_dyb, self.pin_dy = _buffered("d", 1, cap)
        self.pin_dot = np.full(cap, -1, np.int32)
        # its name tag, while shown (TAGGED): the backing's slot (RECT, tags layer) and
        # the text's glyphs, as runs (see glyph0)
        self.tag_bg = np.full(cap, -1, np.int32)
        self.tag_g0 = np.zeros(cap, np.int32)
        self.tag_ng = np.zeros(cap, np.int32)
        self.tag_glyphs = np.full(cap, None, object)
        # its face's marks (SEGMENT, bodies layer), as runs like the title's glyphs, and
        # the color they're tinted with (props["color"], see set_face_color)
        self.face0 = np.zeros(cap, np.int32)
        self.nface = np.zeros(cap, np.int32)
        self.faces = np.full(cap, None, object)
        self.face_color = np.full(cap, None, object)
        # the rows whose face has marks the face() hook lights (see sync.py); face_gen
        # moves whenever that changes
        self.hooked: dict[int, None] = {}
        self.face_gen = 0

    _COLS = (
        "xy", "ints", "wh", "body", "flags", "opacity", "seq", "pin0", "npin",
        "glyph0", "nglyph", "view", "title", "name", "tints", "body_tint",
        "pinslots", "glyphs", "face0", "nface", "faces", "face_color",
    )  # fmt: skip
    _PIN_COLS = (
        "pin_row", "pin_out", "pin_dy", "pin_dot", "tag_bg", "tag_g0", "tag_ng", "tag_glyphs",
    )  # fmt: skip
    _FILL = {"body": -1, "opacity": 255, "pin_row": -1, "pin_dot": -1, "tag_bg": -1}
    _BUFFERED = {"xy": ("d", 2), "ints": ("B", 1), "wh": ("i", 2), "pin_dy": ("d", 1)}

    def _grow(self, names: tuple, n: int) -> None:
        for name in names:
            old = getattr(self, name)
            if name in self._BUFFERED:
                buf, view = _buffered(*self._BUFFERED[name], n, old)
                setattr(self, f"_{name}b", buf)
                setattr(self, name, view)
                continue
            fill = None if old.dtype == object else self._FILL.get(name, 0)
            new = np.full((n, *old.shape[1:]), fill, old.dtype)
            new[: len(old)] = old
            setattr(self, name, new)

    def new_rows(self, k: int) -> range:
        cap = len(self.seq)
        if self.n + k > cap:
            while self.n + k > cap:
                cap *= 2
            self._grow(self._COLS, cap)
        rows = range(self.n, self.n + k)
        self.n += k
        return rows

    def pin_room(self, top: int) -> None:
        """Pin columns long enough for pin slots below `top`."""
        cap = len(self.pin_row)
        if top > cap:
            while top > cap:
                cap *= 2
            self._grow(self._PIN_COLS, cap)

    def forget(self, rows: np.ndarray) -> None:
        """Dead views' rows (their shapes are freed already): let go of what they hold."""
        pins = self.pins_of(rows)
        self.pin_row[pins] = -1
        self.pin_dot[pins] = -1
        self._forget_tags(pins)
        self.body[rows] = -1
        self.flags[rows] = 0
        self.npin[rows] = self.pin0[rows] = 0
        self.nglyph[rows] = self.glyph0[rows] = 0
        self.nface[rows] = self.face0[rows] = 0
        for col in (
            self.view, self.title, self.name, self.tints, self.body_tint,
            self.pinslots, self.glyphs, self.faces, self.face_color,
        ):  # fmt: skip
            col[rows] = None
        if self.hooked:
            for row in rows.tolist():
                if row in self.hooked:
                    del self.hooked[row]
                    self.face_gen += 1

    # ---- one row -----------------------------------------------------------------

    def coord(self, row: int, k: int) -> float:
        """x (k = 0) or y (1), as it was given (an int stays an int)."""
        v = self._xyb[2 * row + k]
        return int(v) if self._intsb[row] >> k & 1 else v

    def pos(self, row: int) -> Point:
        """(x, y), as given."""
        xy = self._xyb
        x, y = xy[2 * row], xy[2 * row + 1]
        b = self._intsb[row]
        if b:
            x, y = (int(x) if b & 1 else x), (int(y) if b & 2 else y)
        return x, y

    def set_coord(self, row: int, k: int, v: float) -> None:
        self._xyb[2 * row + k] = v
        bit = 1 << k
        self._intsb[row] = (self._intsb[row] & ~bit) | (bit if type(v) is int else 0)

    def pin_slots(self, row: int) -> np.ndarray:
        p0 = int(self.pin0[row])
        if p0 >= 0:
            return np.arange(p0, p0 + int(self.npin[row]))
        return self.pinslots[row]

    def pins_of(self, rows: np.ndarray) -> np.ndarray:
        """The pin slots of many views (each view's in order, the views in no set order)."""
        return _runs(self.pin0[rows], self.npin[rows], self.pinslots, rows)

    def title_glyphs(self, row: int) -> np.ndarray:
        g0 = int(self.glyph0[row])
        if g0 >= 0:
            return np.arange(g0, g0 + int(self.nglyph[row]))
        return self.glyphs[row]

    def titles_of(self, rows: np.ndarray) -> np.ndarray:
        """The title glyph slots of many views (see pins_of)."""
        return _runs(self.glyph0[rows], self.nglyph[rows], self.glyphs, rows)

    def face_slots(self, row: int) -> np.ndarray:
        f0 = int(self.face0[row])
        if f0 >= 0:
            return np.arange(f0, f0 + int(self.nface[row]))
        return self.faces[row]

    def faces_of(self, rows: np.ndarray) -> np.ndarray:
        """The face mark slots of many views (see pins_of)."""
        return _runs(self.face0[rows], self.nface[rows], self.faces, rows)

    def tags_of(self, pins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The name tags of these pins: (backing slots, glyph slots), of those shown."""
        bg = self.tag_bg[pins]
        shown = pins[bg >= 0]
        return bg[bg >= 0], _runs(self.tag_g0[shown], self.tag_ng[shown], self.tag_glyphs, shown)

    def _forget_tags(self, pins: np.ndarray) -> None:
        self.tag_bg[pins] = -1
        self.tag_g0[pins] = self.tag_ng[pins] = 0
        self.tag_glyphs[pins] = None

    def tagged(self) -> list[PartView]:
        """The views showing their pin name tags, oldest first."""
        rows = np.flatnonzero(self.flags[: self.n] & TAGGED)
        return self.view[rows].tolist()

    def _set_runs(self, rows: np.ndarray, slots: np.ndarray, counts: np.ndarray, first, n, listed) -> None:
        """Rows `rows` (any index array) get the next counts[i] of `slots` each: as a
        run (first, n) where they're consecutive, else listed."""
        k = len(counts)
        starts = np.cumsum(counts) - counts
        head = np.zeros(k, np.intp)
        has = counts > 0
        head[has] = slots[starts[has]]
        # each slot minus where it would be in a run from its row's first: 0 in a run
        off = slots - np.repeat(head - starts, counts) - np.arange(len(slots))
        run = np.ones(k, bool)
        run[np.repeat(np.arange(k), counts)[off != 0]] = False
        first[rows] = np.where(run, head, -1)
        n[rows] = counts
        for i in np.flatnonzero(~run).tolist():
            listed[rows[i]] = slots[starts[i] : starts[i] + counts[i]]

    def pin_xy(self, pins: np.ndarray) -> np.ndarray:
        """Where these pins are (n x 2): PartView.pin_pos for each, in the same arithmetic."""
        row = self.pin_row[pins]
        x, y = self.xy[row, 0], self.xy[row, 1]
        w, h = self.wh[row, 0], self.wh[row, 1]
        px = np.where(self.pin_out[pins], x + w, x)
        py = y + h / 2 + self.pin_dy[pins]
        return np.column_stack((px, py))


def _runs(first: np.ndarray, n: np.ndarray, listed: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Slots held as runs (first, n; first -1: in `listed` instead), for many rows."""
    run = first >= 0
    first, n = first[run].astype(np.intp), n[run].astype(np.intp)
    out = np.repeat(first - (np.cumsum(n) - n), n) + np.arange(int(n.sum()))
    more = [listed[r] for r in rows[~run].tolist()]
    return np.concatenate((out, *more)) if more else out


class PartView:
    """A part on the board: where it is, and its shapes as slots in the canvas's
    instance buffers (body, pin dots, pin tag backings) plus its text labels.

    A handle: what it knows is a row of its PartTable (the part itself aside)."""

    __slots__ = ("part", "row", "table")

    def __init__(
        self, part: Part, x: float, y: float, table: PartTable, pin_labels: bool = True
    ) -> None:
        self.part, self.table, self.row = part, table, table.new_rows(1)[0]
        table.view[self.row] = self
        _fill_part_rows([self], [(part, x, y)])
        _make_part_shapes([self], pin_labels)

    @classmethod
    def many(
        cls,
        placed: list[tuple[Part, float, float]],
        table: PartTable,
        pin_labels: bool = True,
    ) -> list[PartView]:
        """A view for each (part, x, y): the same as making them one by one, in order, but
        their shapes are made all at once (undo, loading, pasting thousands of parts)."""
        views = []
        for (part, _, _), row in zip(placed, table.new_rows(len(placed))):
            view = cls.__new__(cls)
            view.part, view.table, view.row = part, table, row
            views.append(view)
        if views:
            table.view[views[0].row : views[0].row + len(views)] = views
        _fill_part_rows(views, placed)
        _make_part_shapes(views, pin_labels)
        return views

    # ---- what's in the row ----------------------------------------------------

    @property
    def x(self) -> float:
        return self.table.coord(self.row, 0)

    @x.setter
    def x(self, v: float) -> None:
        self.table.set_coord(self.row, 0, v)

    @property
    def y(self) -> float:
        return self.table.coord(self.row, 1)

    @y.setter
    def y(self, v: float) -> None:
        self.table.set_coord(self.row, 1, v)

    @property
    def w(self) -> int:
        return self.table._whb[2 * self.row]

    @property
    def h(self) -> int:
        return self.table._whb[2 * self.row + 1]

    @property
    def seq(self) -> int:
        return int(self.table.seq[self.row])

    @property
    def index(self) -> SpatialIndex | None:
        return self.table.index

    @property
    def canvas(self) -> Canvas:
        return self.table.canvas

    @property
    def layers(self) -> Layers:
        return self.table.layers

    @property
    def text(self) -> SDFText:
        return self.table.text

    @property
    def look(self):
        return self.part.type.look

    @property
    def title(self) -> str:
        return self.table.title[self.row]

    @property
    def title_size(self) -> float:
        return T.IO_TITLE_SIZE if self.look.narrow else T.TITLE_SIZE

    @property
    def opacity(self) -> int:
        return int(self.table.opacity[self.row])

    @property
    def lifted(self) -> bool:
        """Being dragged: drawn at the canvas's offset (see set_lifted)."""
        return bool(self.table.flags[self.row] & LIFTED)

    @property
    def selected(self) -> bool:
        """The body's `sel` flag, drawn as a ring around it (see select_many)."""
        return bool(self.table.flags[self.row] & SELECTED)

    @property
    def body(self) -> int:
        """Its slot in the bodies (RECT) buffer."""
        return int(self.table.body[self.row])

    @property
    def dots(self) -> tuple[int, ...]:
        """Its pins' slots in the pins (DOT) buffer, in part.pins order."""
        t = self.table
        return tuple(t.pin_dot[t.pin_slots(self.row)].tolist())

    def title_at(self) -> Point:
        """Where the title is anchored: the body's middle."""
        x, y = self.table.pos(self.row)
        return x + self.w / 2, y + self.h / 2

    def _place_title(self) -> None:
        glyphs = self.table.title_glyphs(self.row)
        if glyphs.size:
            self.text.place_at(glyphs, self.title, *self.title_at(), self.title_size, "center")

    @property
    def _name(self) -> SDFLabel | None:
        return self.table.name[self.row]

    @property
    def body_tint(self) -> Rgb | None:
        """Hue for a lit body (switches, LEDs)."""
        return self.table.body_tint[self.row]

    # ---- labels -------------------------------------------------------------

    @property
    def name(self) -> SDFLabel:
        """The label showing the user's name for the part (part.label)."""
        t, row = self.table, self.row
        if t.name[row] is None:  # (empty, as it would have been made: see refresh_name)
            label = t.name[row] = SDFLabel(
                t.text,
                "",
                *self.name_pos(),
                T.LABEL_SIZE,
                (*T.LABEL_TEXT[:3], self.opacity),
                _label_anchor(self.look),
            )
            label.lifted = self.lifted
        return t.name[row]

    @property
    def pin_tints(self) -> list[Rgb | None]:
        """Hue for each lit pin (paint.py)."""
        tints = self.table.tints[self.row]
        if tints is None:
            return [None] * (len(self.part.inputs) + len(self.part.outputs))
        return tints

    @property
    def pin_labels_shown(self) -> bool:
        return bool(self.table.flags[self.row] & TAGGED)

    def set_pin_labels(self, on: bool) -> None:
        """Show / hide the pin name tags: outside the body, next to each pin, on a dark
        backing so they read over wires and other parts. Only parts whose look asks."""
        set_pin_labels([self], on)

    def _tag_specs(self) -> list[tuple]:
        """SDFText.labels specs for the pin name tags' text, in place (see _tag_at)."""
        lay = self.part.layout  # (its own pins: see PartType.layout)
        color = (*T.LABEL_TEXT[:3], self.opacity)
        return [
            (
                (lay.ins if pin.is_input else lay.outs)[pin.index],
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
        if not self.pin_labels_shown:
            return
        t, buf = self.table, self.canvas.buffer(RECT, self.layers.tags)
        for spec, pin in zip(self._tag_specs(), self.part.pins):
            text, x, y, size, _, anchor = spec
            p = np.array([pin.slot])
            _, glyphs = t.tags_of(p)
            if glyphs.size:
                self.text.place_at(glyphs, text, x, y, size, anchor)
            _set_rect(buf, int(t.tag_bg[pin.slot]), _tag_backing(self.text, spec, pin))

    def set_selected(self, on: bool) -> None:
        select_many([self], [], on)

    def intersects(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Does the body overlap the world-space rectangle (x0, y0)-(x1, y1)?"""
        x, y = self.x, self.y
        return x <= x1 and x0 <= x + self.w and y <= y1 and y0 <= y + self.h

    # ---- geometry --------------------------------------------------------

    def pin_pos(self, pin: Pin) -> Point:
        t, row = self.table, self.row
        x, y = t.pos(row)
        wh = t._whb
        px = x if pin.is_input else x + wh[2 * row]
        # index 0 at the top, pins centered vertically: pin_dy is
        # ((n - 1) / 2 - pin.index) * T.PIN_SPACING, n the pins on its side
        # (PartTable.pin_xy: the same, in bulk)
        return px, y + wh[2 * row + 1] / 2 + t._pin_dyb[pin.slot]

    def move_to(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        Touched.parts.add(self.part.uid)
        bodies = self.canvas.buffer(RECT, self.layers.bodies)
        body = self.body
        bodies.f["rect"][body, :2] = (x, y)
        bodies.mark(body)
        self._place_title()
        if self._name is not None:
            self._name.move_to(*self.name_pos())
        faces = self.table.face_slots(self.row)
        if faces.size:
            layout = _face_layout(self.part.layout, self.part.type.look)
            buf = self.canvas.buffer(SEGMENT, self.layers.bodies)
            buf.f["a"][faces] = layout.a + (x, y)
            buf.f["b"][faces] = layout.b + (x, y)
            buf.mark_many(faces)
        dots = self.dots
        if dots:
            buf = self.canvas.buffer(DOT, self.layers.pins)
            center = buf.f["center"]
            for dot, pin in zip(dots, self.part.pins):
                center[dot] = self.pin_pos(pin)
                buf.mark(dot)
        self._place_pin_tags()
        self._register()

    def name_pos(self) -> Point:
        x, y, w, h = self.x, self.y, self.w, self.h
        if self.look.label == "left":
            return x - T.LABEL_GAP, y + h / 2
        if self.look.label == "right":
            return x + w + T.LABEL_GAP, y + h / 2
        return x + w / 2, y - T.LABEL_GAP - self.text.cap_height(T.LABEL_SIZE) / 2

    def refresh_name(self) -> None:
        """Show part.label (after it was edited)."""
        Touched.parts.add(self.part.uid)  # (labels don't change colors)
        if self._name is not None or self.part.label:
            self.name.set_text(self.part.label)

    def contains(self, wx: float, wy: float) -> bool:
        x, y = self.x, self.y
        return x <= wx <= x + self.w and y <= wy <= y + self.h

    def distance_to(self, wx: float, wy: float) -> float:
        """From the body's edge (0 inside)."""
        x, y = self.x, self.y
        dx = max(x - wx, 0.0, wx - x - self.w)
        dy = max(y - wy, 0.0, wy - y - self.h)
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
        x, y = self.x, self.y
        return x - r, y, x + self.w + r, y + self.h

    def set_ghost(self, ghost: bool) -> None:
        """Semi-transparent while being carried around before placement."""
        a = T.GHOST_OPACITY if ghost else 255
        self.table.opacity[self.row] = a
        _set_opacity(self.canvas.buffer(RECT, self.layers.bodies), self.body, a)
        glyphs = self.table.title_glyphs(self.row)
        if glyphs.size:
            buf = self.text.buf
            buf.f["color"][glyphs, 3] = a
            buf.mark_many(glyphs)
        if self._name is not None:
            self._name.opacity = a
        faces = self.table.face_slots(self.row)
        if faces.size:
            buf = self.canvas.buffer(SEGMENT, self.layers.bodies)
            buf.f["flags"][faces, 1] = a
            buf.mark_many(faces)
        if self.pin_labels_shown:
            bgs, glyphs = self.table.tags_of(self.table.pin_slots(self.row))
            if glyphs.size:
                buf = self.text.buf
                buf.f["color"][glyphs, 3] = a
                buf.mark_many(glyphs)
            tags = self.canvas.buffer(RECT, self.layers.tags)
            for bg in bgs.tolist():
                _set_opacity(tags, bg, a)  # (multiplies the backing's own alpha)
        dots = self.dots
        if dots:
            pins = self.canvas.buffer(DOT, self.layers.pins)
            for dot in dots:
                _set_opacity(pins, dot, a)

    def set_tints(self, pins: list[Rgb | None], body: Rgb | None) -> None:
        if pins == self.pin_tints and body == self.body_tint:
            return
        t, row = self.table, self.row
        t.tints[row] = None if all(c is None for c in pins) else pins
        t.body_tint[row] = body
        self._recolor()

    def set_face_color(self, color: str | None) -> None:
        """Tint the face's marks with a palette color (props["color"]; None: as the
        look says)."""
        t, row = self.table, self.row
        if t.face_color[row] == color:
            return
        t.face_color[row] = color
        faces = t.face_slots(row)
        if faces.size:
            pair = color_pair(color)
            off, on = _face_rgba(self.look.face_colors, pair[1] if pair else None)
            buf = self.canvas.buffer(SEGMENT, self.layers.bodies)
            f = buf.f
            f["ca"][faces] = f["cb"][faces] = off
            f["ca_on"][faces] = f["cb_on"][faces] = on
            buf.mark_many(faces)

    def _recolor(self) -> None:
        """Both colors of every pin (and of a lit body), per the tints."""
        dots = self.dots
        if dots:
            buf = self.canvas.buffer(DOT, self.layers.pins)
            for dot, tint in zip(dots, self.pin_tints):
                _set_dot_colors(buf, dot, T.PIN_OFF, with_hue(T.PIN_ON, tint))
        if self.look.lit:
            off, on = (
                [with_hue(c, self.body_tint) for c in theme_color(name)]
                for name in self.look.lit
            )
            buf = self.canvas.buffer(RECT, self.layers.bodies)
            f = buf.f
            body = self.body
            f["fill"][body], f["edge"][body] = _rgba(off[0]), _rgba(off[1])
            f["fill_on"][body], f["edge_on"][body] = _rgba(on[0]), _rgba(on[1])
            buf.mark(body)

    def delete(self) -> None:
        delete_views([self], [])


def _fill_part_rows(views: list[PartView], placed: list[tuple]) -> None:
    """New views' rows (consecutive), from their (part, x, y): everything but the
    shapes (see _make_part_shapes)."""
    if not views:
        return
    t, n = views[0].table, len(views)
    rows = slice(views[0].row, views[0].row + n)
    text = t.text
    t.seq[rows] = list(itertools.islice(_seq, n))
    t.xy[rows] = np.fromiter(
        itertools.chain.from_iterable((x, y) for _, x, y in placed), np.float64, 2 * n
    ).reshape(n, 2)
    t.ints[rows] = [(type(x) is int) | (type(y) is int) << 1 for _, x, y in placed]
    by_type: dict[tuple, tuple] = {}  # (type, pins a side) -> (w, h, title)
    wh, titles = [], []
    for part, _, _ in placed:
        t_ = part.type
        lay = part.layout  # (its own pins: see PartType.layout)
        ins, outs = lay.ins, lay.outs
        cells = _cells(lay, t_.look)
        key = (id(t_), t_.kind, len(ins), len(outs), cells)
        made = by_type.get(key)
        if made is None:
            look = part.type.look
            title = (part.type.title or part.kind) if look.titled and not cells else ""
            if look.size:  # (the registry checked it's on the grid)
                w, h = look.size
            else:
                w = T.IO_WIDTH if look.narrow else T.PART_WIDTH
                if not look.narrow and title:  # long titles (macro names) widen the body
                    need = text.measure(title, T.TITLE_SIZE) + 2 * T.TITLE_PAD
                    w = max(w, math.ceil(need / (2 * T.GRID)) * 2 * T.GRID)  # (grid steps)
                h = (max(len(ins), len(outs), cells, 1) + 1) * T.PIN_SPACING  # (see theme.py)
            made = by_type[key] = (w, h, title)
        wh.append(made[:2])
        titles.append(made[2])
    t.wh[rows] = wh
    t.title[rows] = titles
    # pins: a run of slots each, as plain parts' are made (pin0, npin); else listed
    # (macros' pins can be made apart)
    slots, counts = placed[0][0].circuit.pin_slots_of([part for part, _, _ in placed])
    t._set_runs(np.arange(rows.start, rows.stop), slots, counts, t.pin0, t.npin, t.pinslots)
    t.pin_room(int(slots.max()) + 1 if len(slots) else 0)


def _label_anchor(look) -> str:
    """Where the user's label hangs off its anchor point (see _look_data)."""
    return {"left": "right", "right": "left"}.get(look.label, "center")


SELECTED, LIFTED, TAGGED = 1, 2, 4  # WireTable / PartTable flags (TAGGED: parts)


def _buffered(code: str, width: int, n: int, old: np.ndarray | None = None):
    """A column whose memory is a Python array ("d" float64, "i" int32, "B" uint8):
    (the array, a numpy view of it, n x width). Whole-column work goes through the
    view; one item at a time, indexing the array is several times quicker than numpy
    (the per-part reads: positions, pin positions)."""
    buf = bytearray(n) if code == "B" else array(code, [0]) * (n * width)
    view = np.frombuffer(buf, {"d": np.float64, "i": np.int32, "B": np.uint8}[code])
    if width > 1:
        view = view.reshape(n, width)
    if old is not None:
        view[: len(old)] = old
    return buf, view


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
        # src x, y, dst x, y; bits: which of those were given as ints (_xyb, _intsb:
        # the same memory as Python arrays, see _buffered)
        self._xyb, self.xy = _buffered("d", 4, cap)
        self._intsb, self.ints = _buffered("B", 1, cap)
        self.seg = np.full(cap, -1, np.int32)  # the line's segment, when it has one
        self.dot = np.full((cap, 2), -1, np.int32)  # junction dot on src / dst end
        self.flags = np.zeros(cap, np.uint8)  # SELECTED, LIFTED
        self.opacity = np.full(cap, 255, np.uint8)
        self.seq = np.zeros(cap, np.int64)  # see _seq
        self.wslot = np.full(cap, -1, np.int32)  # the circuit wire's slot
        self.thick = np.full(cap, T.WIRE_THICKNESS, np.float32)  # its line (a bus: thicker)
        # Objects, None meaning the usual: segments when not exactly one (else see seg),
        # bend points (a tuple), color name, paint.py's stops, the line's own gradient
        # (None: one color) and its one (off, on) color pair (None: neutral).
        self.segs = np.full(cap, None, object)
        self.bends = np.full(cap, None, object)
        self.color = np.full(cap, None, object)
        self.stops = np.full(cap, None, object)
        self.lstops = np.full(cap, None, object)
        self.pair = np.full(cap, None, object)
        self.row_of = np.full(cap, -1, np.int32)  # by circuit wire slot: its view's row
        # row -> [layout key, fractions per vertex, colors per stops] (see _layout), for
        # the lines laid out for a gradient; the rest are a segment per pair of points
        self.grad: dict[int, list] = {}

    _COLS = (
        "xy", "ints", "seg", "dot", "flags", "opacity", "seq", "wslot", "thick",
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
                if name in ("xy", "ints"):
                    buf, view = _buffered("d" if name == "xy" else "B", old.shape[1] if old.ndim > 1 else 1, 2 * n, old)
                    setattr(self, f"_{name}b", buf)
                    setattr(self, name, view)
                    continue
                fill = {"seg": -1, "dot": -1, "opacity": 255, "wslot": -1, "thick": T.WIRE_THICKNESS}.get(name, 0)
                new = np.full((2 * n, *old.shape[1:]), None if old.dtype == object else fill, old.dtype)
                new[:n] = old
                setattr(self, name, new)
        rows = range(self.n, self.n + k)
        self.n += k
        return rows

    def forget(self, rows: np.ndarray) -> None:
        """Dead views' rows (their shapes are freed already): let go of what they hold."""
        wslot = self.wslot[rows]
        mine = self.row_of[wslot] == rows  # (unless a newer view has its wire already)
        self.row_of[wslot[mine]] = -1
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
        i = 4 * row + 2 * k
        xy = self._xyb
        x, y = xy[i], xy[i + 1]
        bits = self._intsb[row] >> (2 * k)
        if bits & 3:
            x, y = (int(x) if bits & 1 else x), (int(y) if bits & 2 else y)
        return x, y

    def set_end(self, row: int, k: int, p: Point) -> None:
        x, y = p
        i = 4 * row + 2 * k
        self._xyb[i], self._xyb[i + 1] = x, y
        self._intsb[row] = (self._intsb[row] & ~(3 << 2 * k)) | _int_bits(x, y, k)

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
                float(self.thick[row]),
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
    wslot = np.fromiter((w.slot for w, *_ in specs), np.intp, n)
    t.wslot[rows] = wslot
    if wslot.max() >= len(t.row_of):
        grown = np.full(2 * int(wslot.max()) + 2, -1, np.int32)
        grown[: len(t.row_of)] = t.row_of
        t.row_of = grown
    t.row_of[wslot] = np.arange(rows.start, rows.stop)
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
        i = 4 * self.row
        if t._xyb[i : i + 4].tolist() == [*src, *dst]:
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
        w = self.wire
        for k, end in enumerate(w.ends):
            dot = int(t.dot[row, k])
            if dot >= 0:
                _place_wire_dot(t.buf, dot, t.end(row, k), end is w)
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
    t = views[0].table
    canvas, layers, text = t.canvas, t.layers, t.text
    n = len(views)
    rows = _rows_of(views)
    xs, ys = t.xy[rows, 0], t.xy[rows, 1]
    ws, hs = t.wh[rows, 0].astype(np.float64), t.wh[rows, 1].astype(np.float64)
    # pins: per view, how many and where they start in the flat list
    circuit = views[0].part.circuit
    pin_slots, counts = circuit.pin_slots_of([v.part for v in views])
    starts = np.cumsum(counts) - counts
    codes = circuit.pin_codes(pin_slots)
    # what each view's look decides, and the pin layout of each (look, pins, size)
    looks: dict[int, tuple] = {}
    group_of: dict[tuple, list[int]] = {}
    for i, (v, w, h) in enumerate(zip(views, ws.tolist(), hs.tolist())):
        look = v.look
        if id(look) not in looks:
            looks[id(look)] = _look_data(look)
        lay = v.part.layout
        group_of.setdefault((id(look), len(lay.ins), len(lay.outs), w, h), []).append(i)
    is_out = np.zeros(len(pin_slots), bool)
    pin_dy = np.zeros(len(pin_slots))  # from the body's middle (see pin_pos)
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
    # (before anything asks pin_pos: the pin name tags below do)
    t.pin_row[pin_slots] = np.repeat(rows, counts)
    t.pin_out[pin_slots] = is_out
    t.pin_dy[pin_slots] = pin_dy
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
    glyphs, per_spec = text.place(specs)
    first = np.cumsum(per_spec) - per_spec  # where each spec's glyphs start in `glyphs`
    titles, tag_specs = [], []  # spec indices
    i = 0
    for v, k in zip(views, counts.tolist()):
        titles.append(i)
        i += 1
        if v.part.label:
            k = first[i]
            t.name[v.row] = text.wrap(specs[i], glyphs[k : k + per_spec[i]])
            i += 1
        if pin_labels and v.look.pin_labels:
            tagged.append(v)
            tag_specs += range(i, i + k)
            i += k
    t._set_runs(rows, *_pick(glyphs, first, per_spec, titles), t.glyph0, t.nglyph, t.glyphs)
    if tagged:
        _store_tags(tagged, [specs[i] for i in tag_specs], *_pick(glyphs, first, per_spec, tag_specs))
    # bodies: the look's colors; lit ones (switches, LEDs) also on-colors, on per the first pin
    has_pins = counts > 0
    on = np.full(n, SHOW_OFF, np.uint8)
    first_pin = np.full(n, -1, np.intp)
    # (one showing bit cells doesn't: they say it all, and a lit body would hide them)
    celled = np.fromiter((_cells(v.part.layout, v.part.type.look) > 0 for v in views), bool, n)
    shows = lit & has_pins & ~celled
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
    t.body[rows] = slots
    # pin dots
    buf = canvas.buffer(DOT, layers.pins)
    slots = buf.alloc_many(len(pin_slots))
    if len(pin_slots):
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
    t.pin_dot[pin_slots] = slots
    _make_faces(views, rows, xs, ys, pin_slots, starts)
    # the spatial index: the body, and the pins sticking out of its sides (_index_box)
    if t.index is not None:
        r = T.PIN_RADIUS
        t.index.put_boxes(views, np.column_stack((xs - r, ys, (xs + ws) + r, ys + hs)))
    Touched.part_many(
        v.part.uid for v in views
    )  # new: its own color (props) needs painting


class _FaceLayout:
    """A face's marks, worked out once per (pins, look): ends from the body's corner
    (k x 2 each; a dot's b is its a), radii, the pin each follows (its index in
    part.pins, -1 for none) and the indices of the marks face() lights. Bit cells
    (Look.cells) are marks too, after the look's own: one per lane of the cells pin,
    following just that lane (`lane`: -1 for the look's own marks)."""

    def __init__(self, lay, look) -> None:
        marks = look.face
        names = list(lay.ins) + list(lay.outs)
        a = [m.a for m in marks]
        b = [m.a if m.b is None else m.b for m in marks]
        radius = [m.radius for m in marks]
        pin = [-1 if m.pin is None else names.index(m.pin) for m in marks]
        lane = [-1] * len(marks)
        n = _cells(lay, look)
        if n:
            k = names.index(look.cells)
            cx = (look.size[0] if look.size else T.IO_WIDTH if look.narrow else T.PART_WIDTH) / 2
            h = T.CELL_HALF
            for i in range(n):  # (lane 0 at the top: in pin order, as a SPLIT's pins are)
                y = (n - i) * T.PIN_SPACING
                a.append((cx - h, y))
                b.append((cx + h, y))
                radius.append(T.CELL_RADIUS)
                pin.append(k)
                lane.append(i)
        self.k = len(a)
        self.a = np.array(a, np.float64).reshape(-1, 2)
        self.b = np.array(b, np.float64).reshape(-1, 2)
        self.radius = np.array(radius, np.float32)
        self.pin = np.array(pin, np.intp)
        self.lane = np.array(lane, np.intp)
        self.hooked = np.flatnonzero(self.pin < 0)
        self.colors = look.face_colors


@lru_cache(maxsize=256)
def _face_layout(lay, look) -> _FaceLayout:
    return _FaceLayout(lay, look)


def _cells(lay, look) -> int:
    """How many bit cells a part with this layout and look shows (Look.cells): its
    cells pin's lanes, if that's a bus; else none."""
    if look.cells is None:
        return 0
    names = lay.ins + lay.outs
    if look.cells not in names:
        return 0
    w = lay.widths[names.index(look.cells)]
    return w if w > 1 else 0


def cell_at(view, wx: float, wy: float) -> int | None:
    """The bit cell (its lane) of a part view at world point (wx, wy), if any."""
    lay, look = view.part.layout, view.part.type.look
    n = _cells(lay, look)
    if not n:
        return None
    layout = _face_layout(lay, look)
    first = layout.k - n
    for i in range(n):
        (ax, ay), (bx, _) = layout.a[first + i], layout.b[first + i]
        r = T.CELL_RADIUS + 1
        if view.x + ax - r <= wx <= view.x + bx + r and abs(wy - (view.y + ay)) <= r:
            return i
    return None


def _face_rgba(names: tuple[str, str], tint: Rgb | None) -> tuple[tuple, tuple]:
    """A face's (off, on) mark colors, from theme names, tinted with `tint`'s hue."""

    def rgb(name, fallback):
        value = getattr(T, name, None)
        if isinstance(value, tuple) and value and isinstance(value[0], tuple):
            value = value[0]  # (a (fill, edge) pair: its fill)
        return value if isinstance(value, tuple) and len(value) in (3, 4) else fallback

    off, on = rgb(names[0], T.FACE_OFF), rgb(names[1], T.FACE_ON)
    return _rgba(with_hue(off, tint)), _rgba(with_hue(on, tint))


def _make_faces(views: list[PartView], rows: np.ndarray, xs, ys, pin_slots, starts) -> None:
    """The face marks of the new views whose look has any: one SEGMENT each in the
    bodies layer (over every body, under the pins). Marks that follow a pin show its
    state like a pin dot does (sync.py); the rest wait for face() (also sync.py)."""
    groups: dict[_FaceLayout, list[int]] = {}
    for i, v in enumerate(views):
        look = v.part.type.look
        if look.face or look.cells is not None:
            lay = v.part.layout
            if look.face or _cells(lay, look):
                groups.setdefault(_face_layout(lay, look), []).append(i)
    if not groups:
        return
    t = views[0].table
    buf = t.canvas.buffer(SEGMENT, t.layers.bodies)
    for layout, members in groups.items():
        idx = np.array(members, np.intp)
        n, k = len(idx), layout.k
        slots = buf.alloc_many(n * k)
        corner = np.column_stack((xs[idx], ys[idx]))[:, None, :]
        f = buf.f
        f["a"][slots] = (corner + layout.a).reshape(-1, 2)
        f["b"][slots] = (corner + layout.b).reshape(-1, 2)
        f["radius"][slots] = np.tile(layout.radius, n)
        off, on = _face_rgba(layout.colors, None)
        f["ca"][slots] = f["cb"][slots] = off
        f["ca_on"][slots] = f["cb_on"][slots] = on
        f["flags"][slots] = 0, 255, 255, 255  # (round caps at both ends)
        f["lift"][slots] = 0.0
        f["sel"][slots] = 0
        buf.set_state(slots, SHOW_OFF)
        buf.mark_many(slots)
        grid = slots.reshape(n, k)
        pinned = (layout.pin >= 0) & (layout.lane < 0)
        if pinned.any():
            src = pin_slots[starts[idx][:, None] + layout.pin[pinned]]
            buf.show_pins(grid[:, pinned].ravel(), src.ravel())
        cells = layout.lane >= 0
        if cells.any():  # (pin_slots are the pins' heads: a lane is head + lane)
            src = pin_slots[starts[idx][:, None] + layout.pin[cells]] + layout.lane[cells]
            buf.show_lanes(grid[:, cells].ravel(), src.ravel())
        t._set_runs(rows[idx], slots, np.full(n, k, np.intp), t.face0, t.nface, t.faces)
        if layout.hooked.size:
            t.hooked.update(dict.fromkeys(rows[idx].tolist()))
            t.face_gen += 1


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


def _tag_backing(text: SDFText, spec: tuple, pin: Pin) -> tuple[float, float, float, float]:
    """(x, y, width, height) of the dark backing behind a pin tag's text (`spec`: see
    PartView._tag_specs), as an SDFLabel of it would measure."""
    pad_x, pad_y = T.PIN_TAG_PAD
    words, x, y, size = spec[:4]
    width = text.width(words, size)
    h = text.cap_height(size) + 2 * pad_y
    left = x - width - pad_x if pin.is_input else x - pad_x
    return left, y - h / 2, width + 2 * pad_x, h


def _pick(glyphs: np.ndarray, first: np.ndarray, counts: np.ndarray, which: list[int]):
    """The glyphs of some of SDFText.place's specs: (their slots, how many each)."""
    which = np.array(which, np.intp)
    n = counts[which]
    at = np.repeat(first[which] - (np.cumsum(n) - n), n) + np.arange(int(n.sum()))
    return glyphs[at] if len(at) else np.empty(0, np.intp), n


def set_pin_labels(views: list[PartView], on: bool) -> None:
    """set_pin_labels for many views at once (only those whose look shows tags, and
    whose tags aren't that way already)."""
    if not views:
        return
    t = views[0].table
    rows = _rows_of(views)
    shown = (t.flags[rows] & TAGGED) != 0
    want = np.array([on and v.look.pin_labels for v in views], bool)
    change = [v for v, c in zip(views, (shown != want).tolist()) if c]
    if not change:
        return
    if not on:
        _drop_tags(change)
        return
    specs = [spec for v in change for spec in v._tag_specs()]
    glyphs, counts = t.text.place(specs)
    _store_tags(change, specs, glyphs, counts)


def _store_tags(views: list[PartView], specs: list[tuple], glyphs: np.ndarray, counts: np.ndarray) -> None:
    """Name tags for every pin of these views, whose text is placed already (specs
    from _tag_specs, in order; glyphs and counts from SDFText.place): their backings,
    all at once, and the table's tag columns."""
    t = views[0].table
    pins = [p for v in views for p in v.part.pins]
    rects, alpha, lift = [], [], []
    for v in views:
        k = len(v.part.pins)
        alpha += [v.opacity] * k
        lift += [1.0 if v.lifted else 0.0] * k
    rects = [_tag_backing(t.text, spec, pin) for spec, pin in zip(specs, pins)]
    if not rects:
        return
    buf = t.canvas.buffer(RECT, t.layers.tags)
    slots = buf.alloc_many(len(rects))
    f = buf.f
    f["rect"][slots] = rects
    f["border"][slots] = 0.0
    for name in ("fill", "fill_on", "edge", "edge_on"):
        f[name][slots] = _rgba(T.PIN_TAG_BG)
    f["flags"][slots] = 0, 255, 0, 0
    f["flags"][slots, 1] = alpha
    f["lift"][slots] = lift
    buf.mark_many(slots)
    pin_slots = np.fromiter((p.slot for p in pins), np.intp, len(pins))
    t.tag_bg[pin_slots] = slots
    t._set_runs(pin_slots, glyphs, counts, t.tag_g0, t.tag_ng, t.tag_glyphs)
    t.flags[_rows_of([v for v in views if v.part.pins])] |= TAGGED  # (no pins: no tags)
    lifted = [v for v in views if v.lifted]
    if lifted:  # (placed unlifted)
        _, up = t.tags_of(t.pins_of(_rows_of(lifted)))
        if up.size:
            t.text.buf.f["lift"][up] = 1.0
            t.text.buf.mark_many(up)


def _drop_tags(views: list[PartView]) -> None:
    """Take these views' name tags away."""
    t = views[0].table
    rows = _rows_of(views)
    pins = t.pins_of(rows)
    bgs, glyphs = t.tags_of(pins)
    t.canvas.buffer(RECT, t.layers.tags).free_many(bgs.astype(np.intp))
    t.text.buf.free_many(glyphs.astype(np.intp))
    t._forget_tags(pins)
    t.flags[rows] &= ~TAGGED & 0xFF


def _make_wire_shapes(views: list[WireView], lines: list[list[Point]]) -> None:
    """The segments of a new wire's line (and its junction dots), placed and colored.
    Per wire, in order: its segments, then a dot on each end that's a junction (right
    after its line: a wire crossing the junction covers it), or a square on each end
    that's free."""
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
    free_src = np.fromiter((v.wire.src is v.wire for v in views), bool, n)
    free_dst = np.fromiter((v.wire.dst is v.wire for v in views), bool, n)
    bus = np.fromiter((v.wire.width > 1 for v in views), bool, n)
    t.thick[np.fromiter((v.row for v in views), np.intp, n)] = np.where(
        bus, T.BUS_THICKNESS, T.WIRE_THICKNESS
    )
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
        radius[at] = np.where(bus[wire], T.BUS_THICKNESS, T.WIRE_THICKNESS) / 2
        caps[at, 0] = np.where(k >= 1, 255, 0)
        caps[at, 1] = np.where(k + 1 <= n_seg[wire] - 1, 255, 0)
        # junction dots: zero-length, capped both ends
        for has, free, row, point in (
            (dot_src, free_src, n_seg, first_pt),
            (dot_dst, free_dst, n_seg + dot_src, first_pt + n_pts - 1),
        ):
            w = np.flatnonzero(has)
            a[first[w] + row[w]] = b[first[w] + row[w]] = p[point[w]]
            radius[first[w] + row[w]] = np.where(bus[w], T.BUS_JUNCTION_RADIUS, T.JUNCTION_RADIUS)
            # free ends: squares, a segment as long as it's thick and uncapped
            sq = first[w] + row[w]
            sq = sq[free[w]]
            h = T.FREE_END_HALF
            a[sq, 0] -= h
            b[sq, 0] += h
            radius[sq] = h
            caps[sq] = 0
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

    def lines(self, views: list[WireView], rows=None) -> None:
        """Their lines' segments."""
        if views:
            t = views[0].table
            self._add(t.buf, t.segments(_rows_of(views) if rows is None else rows))

    def bodies(self, views: list[PartView], rows=None) -> None:
        if views:
            t = views[0].table
            body = t.body[_rows_of(views) if rows is None else rows]
            self.array(t.canvas, RECT, t.layers.bodies, body[body >= 0])

    def array(self, canvas: Canvas, kind, layer, slots: np.ndarray) -> None:
        """slots(), from an array."""
        if slots.size:
            self._add(canvas.buffer(kind, layer), slots.astype(np.intp))

    def parts(self, views: list[PartView], rows=None) -> _Slots:
        if not views:
            return self
        t = views[0].table
        canvas, layers = t.canvas, t.layers
        rows = _rows_of(views) if rows is None else rows
        self.bodies(views, rows)
        self.array(canvas, SEGMENT, layers.bodies, t.faces_of(rows))
        dots = t.pin_dot[t.pins_of(rows)]
        self.array(canvas, DOT, layers.pins, dots[dots >= 0])
        tagged = rows[(t.flags[rows] & TAGGED) != 0]
        if tagged.size:
            bgs, glyphs = t.tags_of(t.pins_of(tagged))
            self.array(canvas, RECT, layers.tags, bgs)
        self._add(t.text.buf, t.titles_of(rows))
        self.labels([k for k in t.name[rows].tolist() if k is not None and k.slots.size])
        if tagged.size:
            self._add(t.text.buf, glyphs.astype(np.intp))
        return self

    def wires(self, views: list[WireView], rows=None) -> _Slots:
        if views:
            t = views[0].table
            rows = _rows_of(views) if rows is None else rows
            self._add(t.buf, t.segments(rows))
            dots = t.dot[rows].ravel()
            self._add(t.buf, dots[dots >= 0])
        return self

    def __iter__(self):
        for buf, arrays in self.by.values():
            yield buf, (np.concatenate(arrays) if len(arrays) > 1 else arrays[0])


def lift(parts: list[PartView], wires: list[WireView], on: bool) -> None:
    """set_lifted for many parts and wires at once."""
    parts, prows, wires, wrows = _lift_mirrors(parts, wires, on)
    for buf, slots in _Slots().parts(parts, prows).wires(wires, wrows):
        buf.set_lift(slots, on)


def put_down(
    parts: list[PartView], wires: list[WireView], dx: float, dy: float
) -> None:
    """Un-lift these and move them by (dx, dy): the parts as if by move_to, the wires
    whole (bends and junction ends too, no new layout). Dropping a dragged selection;
    every shape is written once. (A rigid move changes no colors: not reported to
    paint.py. The caller re-attaches wires stretched between these and the rest.)"""
    prows, wrows = _rows_of(parts), _rows_of(wires)
    _lift_mirrors(parts, wires, False, prows, wrows)
    if dx or dy:
        moved = (
            _move_part_mirrors(parts, dx, dy, prows),
            _move_wire_mirrors(wires, dx, dy, wrows),
        )
        if moved[0] or moved[1]:
            Touched.moved.append((*moved, dx, dy))
    for buf, slots in _Slots().parts(parts, prows).wires(wires, wrows):
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
    if parts:
        # the views are dead: they hold no slots any more (a second delete is a no-op)
        t = parts[0].table
        rows = _rows_of(parts)
        for label in t.name[rows].tolist():
            if label is not None:  # (whoever still has it, has an empty label)
                label.slots = NO_SLOTS
        t.forget(rows)


def select_many(parts: list[PartView], wires: list[WireView], on: bool) -> None:
    """set_selected for many parts and wires at once. Selection is a flag on the shapes
    themselves (`sel`: the body, the line's segments), which the canvas draws again as
    outlines and halos (see canvas.Echo, sdf_shapes.RECT_OUTLINE / SEGMENT_HALO): one
    write per buffer, and nothing to keep in step when they move, lift or go away."""
    parts, prows = _flag(parts, SELECTED, on)
    wires, wrows = _flag(wires, SELECTED, on)
    if not (parts or wires):
        return
    value = 255 if on else 0
    t = (parts or wires)[0].table
    canvas, layers = t.canvas, t.layers
    canvas.echo(RECT_OUTLINE, layers.selection, RECT, layers.bodies)
    canvas.echo(SEGMENT_HALO, layers.wire_halo, SEGMENT, layers.wires)
    flagged = _Slots()
    flagged.bodies(parts, prows)
    flagged.lines(wires, wrows)
    for buf, slots in flagged:
        buf.f["sel"][slots, 0] = value
        buf.mark_many(slots)


# What the views remember of their shapes (coordinates, lift), and the spatial index:
# the rest of the bulk operations, next to the one write per buffer.


def _lift_mirrors(
    parts: list[PartView], wires: list[WireView], on: bool, prows=None, wrows=None
) -> tuple[list, np.ndarray, list, np.ndarray]:
    """Mark them (un)lifted; returns the ones that weren't already, and their rows."""
    parts, prows = _flag(parts, LIFTED, on, prows)
    wires, wrows = _flag(wires, LIFTED, on, wrows)
    value = 1.0 if on else 0.0
    for label in _labels(parts, prows):
        label._lift = value  # (what glyphs they get later start with)
    return parts, prows, wires, wrows


def _labels(views: list[PartView], rows=None) -> list[SDFLabel]:
    """Every label object of these views: their user labels. (Titles and pin tags
    have none: see PartTable.glyph0, tag_bg.)"""
    if not views:
        return []
    t = views[0].table
    rows = _rows_of(views) if rows is None else rows
    out = [k for k in t.name[rows].tolist() if k is not None]
    return out


def _rows_of(views: list) -> np.ndarray:
    return np.fromiter(map(_ROW, views), np.intp, len(views))


_ROW = operator.attrgetter("row")


def _flag(views: list, flag: int, on: bool, rows=None) -> tuple[list, np.ndarray]:
    """Set or clear a flag (SELECTED, LIFTED) of these views in their table; returns
    the ones it changed, and their rows. (`rows`: the views', if the caller has them.)"""
    if not views:
        return views, NO_SLOTS
    t = views[0].table
    rows = _rows_of(views) if rows is None else rows
    had = (t.flags[rows] & flag) != 0
    if on:
        t.flags[rows] |= flag
    else:
        t.flags[rows] &= ~flag & 0xFF
    changed = had != on
    if changed.all():
        return views, rows
    return [v for v, c in zip(views, changed.tolist()) if c], rows[changed]


def _exact(x, d: float) -> bool:
    """Does moving `x` by d and back give x again, bit for bit? (And is it a float: an
    int would come back from the undo history as a float. See document._compress.)"""
    return type(x) is float and (x + d) - d == x


def _move_part_mirrors(
    views: list[PartView], dx: float, dy: float, rows=None
) -> list[int]:
    """Move these views' coordinates by (dx, dy). Returns the uids of the ones that
    moved exactly (see _exact); the others are reported as changed (Touched)."""
    if not views:
        return []
    t = views[0].table
    rows = _rows_of(views) if rows is None else rows
    xy, ints = t.xy[rows], t.ints[rows]
    d = np.array([dx, dy])
    ok = (((xy + d) - d == xy) & ((ints[:, None] >> np.arange(2)) & 1 == 0)).all(axis=1)
    t.xy[rows] = xy + d
    lost = (0 if type(dx) is int else 1) | (0 if type(dy) is int else 2)
    if lost:
        t.ints[rows] = ints & (~lost & 0xFF)
    for label in _labels(views, rows):
        label.x += dx
        label.y += dy
    if t.index is not None:
        t.index.shift(views, dx, dy)
    good = ok.tolist()
    Touched.parts.update(v.part.uid for v, g in zip(views, good) if not g)
    return [v.part.uid for v, g in zip(views, good) if g]


def _move_wire_mirrors(
    views: list[WireView], dx: float, dy: float, rows=None
) -> list[int]:
    """_move_part_mirrors for wires. A wire whose data has no coordinates (pin to pin,
    no bends: see document.wire_data) is the same after a move: not reported at all."""
    if not views:
        return []
    t = views[0].table
    rows = _rows_of(views) if rows is None else rows
    # which ends are in the data (junctions, see wire_data: the ends with a dot), and
    # moved exactly
    junction = t.dot[rows] >= 0
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
