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


class Polyline:
    """Thick line through several points, with round joints so corners have no gaps."""

    def __init__(self, points: list[Point], color, batch: pyglet.graphics.Batch,
                 group: pyglet.graphics.Group, thickness: float = T.WIRE_THICKNESS) -> None:
        self.batch, self.group, self.thickness = batch, group, thickness
        self._color = color
        self._opacity: int | None = None  # None: whatever alpha the color carries
        self.segments: list[shapes.Line] = []
        self.joints: list[shapes.Circle] = []
        self.points: list[Point] = []
        self.set_points(points)

    def set_points(self, points: list[Point]) -> None:
        n_seg = max(len(points) - 1, 0)
        n_joint = max(len(points) - 2, 0)
        while len(self.segments) < n_seg:
            self.segments.append(self._styled(shapes.Line(0, 0, 0, 0, thickness=self.thickness, color=self._color,
                                                          batch=self.batch, group=self.group)))
        while len(self.segments) > n_seg:
            self.segments.pop().delete()
        while len(self.joints) < n_joint:
            self.joints.append(self._styled(shapes.Circle(0, 0, self.thickness / 2, segments=T.JOINT_SEGMENTS,
                                                          color=self._color, batch=self.batch, group=self.group)))
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

    @property
    def opacity(self) -> int:
        return 255 if self._opacity is None else self._opacity

    @opacity.setter
    def opacity(self, value: int) -> None:
        self._opacity = value
        for shape in self.segments + self.joints:
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
        self.pin_dots = [shapes.Circle(0, 0, T.PIN_RADIUS, segments=T.PIN_SEGMENTS, color=T.PIN_OFF,
                                       batch=batch, group=layers.pins)
                         for _ in part.pins]
        # Selection outline: a ring just outside the body, drawn under it and the pins.
        o = T.SELECT_OUTSET
        self.outline = shapes.Box(0, 0, self.w + 2 * o, self.h + 2 * o, thickness=T.SELECT_THICKNESS,
                                  color=T.SELECT, batch=batch, group=layers.selection)
        self.outline.visible = False
        self._last_state: tuple[bool, ...] | None = None
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
        return self.outline.visible

    def set_selected(self, on: bool) -> None:
        self.outline.visible = on

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
        for dot, on in zip(self.pin_dots, state):
            dot.color = T.PIN_ON if on else T.PIN_OFF
        if self.look.lit and state:  # body color follows the first pin (switches, LEDs)
            self._set_body(theme_color(self.look.lit[1] if state[0] else self.look.lit[0]))

    def _set_body(self, colors: tuple[tuple[int, int, int], tuple[int, int, int]]) -> None:
        self.body.color, self.body.border_color = colors

    def delete(self) -> None:
        self.body.delete()
        self.outline.delete()
        self.kind_text.delete()
        self.name.delete()
        for dot in self.pin_dots:
            dot.delete()
        for bg, label in self.pin_tags:
            bg.delete()
            label.delete()


class WireView:
    """A wire drawn from its src pin, through user-placed bend points, to its dst pin.

    Bend points are layout data, so they live here and not in the sim.
    """

    def __init__(self, wire: Wire, src: Point, bends: list[Point], dst: Point,
                 batch: pyglet.graphics.Batch, layers: Layers, color: str | None = None) -> None:
        self.wire = wire
        self.src, self.dst = src, dst
        self.bends = list(bends)
        self.color = color  # a T.WIRE_COLORS name; None = the default look
        self.batch, self.layers = batch, layers
        self.line = Polyline(self.points, T.WIRE_OFF, batch, layers.wires)
        self.highlight: Polyline | None = None  # selection glow, only while selected
        # A dot on each end that attaches to another wire (a junction), like on schematics.
        self.dots = {end: shapes.Circle(0, 0, T.JUNCTION_RADIUS, segments=T.PIN_SEGMENTS, color=T.WIRE_OFF,
                                        batch=batch, group=layers.pins)
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
        off_color, on_color = T.WIRE_COLORS.get(self.color, T.WIRE_COLORS[None])  # unknown name: default
        color = T.WIRE_CONFLICT if conflict else on_color if on else off_color
        self.line.color = color
        for dot in self.dots.values():
            dot.color = color

    def set_color(self, color: str | None) -> None:
        self.color = color
        state, self._last_state = self._last_state, None
        self.sync(state or (False, False))

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
