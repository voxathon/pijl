"""Editing one wire's bend points and junctions.

While a wire is being edited it glows, every bend point gets a square handle,
every segment gets a small round "+" handle at its midpoint, and every junction
gets a filled round handle -- the wire's own junction ends as well as the ends
of branches hanging off it:

  hold + drag a square       move that bend point (Ctrl snaps to the grid)
  hold + drag a "+"          insert a bend point there and drag it
  hold + drag on the wire    same, split at the spot you grabbed
  right-click a square       remove that bend point
  hold + drag a round one    slide that junction along the wire it sits on (its "rail")
  Alt + drag                 grab a junction even where it sits under a bend square

Branch junctions are anchored to the edited wire as (segment, fraction), not as
points: moving bends carries them along (one sitting on a corner stays on that
corner), and inserting / removing bends remaps the anchors. Re-projecting onto the
nearest point instead made a corner junction jump to whichever segment ended up
closest.

Handles are world objects (they sit on the wire) but are sized and hit-tested
in *screen* pixels, so they stay equally easy to grab at every zoom level.
The session only knows about its WireView and the wires its junctions touch;
the editor owns the mode switch.
"""

from __future__ import annotations

import math

import pyglet
from pyglet import shapes

from . import theme as T
from .camera import Camera
from .views import Layers, Point, Polyline, WireView, project_onto

BEND_PX = 10          # square side
ADD_RADIUS_PX = 4.5
JUNCTION_RADIUS_PX = 6
BORDER_PX = 1.5
HIT_PX = 8            # grab distance for any handle
MIN_ADD_SEGMENT_PX = 28  # hide "+" on segments too short to hold one comfortably

Target = tuple[str, int]    # ("bend", bend index), ("add", segment index) or ("junction", index)
End = str                   # "src" or "dst"
Anchor = tuple[int, float]  # segment k of the edited wire, fraction t along it


class _Handle:
    def __init__(self, kind: str, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group) -> None:
        self.kind = kind
        if kind == "bend":
            self.outer = shapes.Rectangle(0, 0, 1, 1, color=T.HANDLE_BORDER, batch=batch, group=group)
            self.inner = shapes.Rectangle(0, 0, 1, 1, color=T.HANDLE_FILL, batch=batch, group=group)
        elif kind == "junction":
            self.outer = shapes.Circle(0, 0, 1, segments=24, color=T.HANDLE_BORDER, batch=batch, group=group)
            self.inner = shapes.Circle(0, 0, 1, segments=24, color=T.JUNCTION_HANDLE_FILL,
                                       batch=batch, group=group)
        else:
            self.outer = shapes.Circle(0, 0, 1, segments=24, color=T.ADD_HANDLE_BORDER, batch=batch, group=group)
            self.inner = shapes.Circle(0, 0, 1, segments=24, color=T.ADD_HANDLE_FILL, batch=batch, group=group)
        self.pos: Point = (0.0, 0.0)

    def place(self, pos: Point, zoom: float, hovered: bool, visible: bool = True) -> None:
        self.pos = pos
        x, y = pos
        b = BORDER_PX / zoom
        if self.kind == "bend":
            half = BEND_PX / 2 / zoom
            self.outer.position, self.outer.width, self.outer.height = (x - half, y - half), 2 * half, 2 * half
            self.inner.position = (x - half + b, y - half + b)
            self.inner.width = self.inner.height = 2 * (half - b)
            self.inner.color = T.HANDLE_HOVER if hovered else T.HANDLE_FILL
        elif self.kind == "junction":
            r = JUNCTION_RADIUS_PX / zoom
            self.outer.position, self.outer.radius = pos, r
            self.inner.position, self.inner.radius = pos, r - b
            self.inner.color = T.HANDLE_HOVER if hovered else T.JUNCTION_HANDLE_FILL
        else:
            r = ADD_RADIUS_PX / zoom
            self.outer.position, self.outer.radius = pos, r
            self.inner.position, self.inner.radius = pos, r - b
            self.outer.color = T.HANDLE_HOVER if hovered else T.ADD_HANDLE_BORDER
        self.outer.visible = self.inner.visible = visible

    @property
    def visible(self) -> bool:
        return self.outer.visible

    def delete(self) -> None:
        self.outer.delete()
        self.inner.delete()


class WireEditSession:
    def __init__(self, view: WireView, camera: Camera, batch: pyglet.graphics.Batch, layers: Layers,
                 parents: dict[End, WireView], branches: list[tuple[WireView, End]]) -> None:
        """`parents`: the wire each of this wire's junction ends sits on.
        `branches`: the (wire, end) pairs whose junction sits on this wire."""
        self.view = view
        self.camera = camera
        self.batch = batch
        self.group = layers.overlay
        self.original = list(view.bends)
        self.original_ends = view.src, view.dst
        # One per junction handle: (the wire whose end it is, which end, the rail it slides on).
        self.junctions: list[tuple[WireView, End, WireView]] = [
            *((view, end, rail) for end, rail in parents.items()),
            *((b, end, view) for b, end in branches)]
        self.branch_original = [(b, b.src, b.dst) for b, _ in branches]
        self.anchors: dict[int, Anchor] = {j: _anchor_of(view.points, getattr(w, end))
                                          for j, (w, end, rail) in enumerate(self.junctions) if rail is view}
        self.halo = Polyline(view.points, T.WIRE_HALO, batch, layers.wire_halo,
                             thickness=T.WIRE_THICKNESS + 6)
        self.bend_handles: list[_Handle] = []
        self.add_handles: list[_Handle] = []
        self.junction_handles: list[_Handle] = []
        self.dragging: Target | None = None
        self.grab = (0.0, 0.0)
        self.hover: Target | None = None
        self.refresh()

    # ---- display -----------------------------------------------------------

    def refresh(self) -> None:
        """Re-place handles and glow (after the wire changed, or the camera did)."""
        pts = self.view.points
        zoom = self.camera.zoom
        self._resize(self.bend_handles, len(self.view.bends), "bend")
        self._resize(self.add_handles, len(pts) - 1, "add")
        self._resize(self.junction_handles, len(self.junctions), "junction")
        for i, (h, p) in enumerate(zip(self.bend_handles, self.view.bends)):
            h.place(p, zoom, self._lit(("bend", i)))
        for k, h in enumerate(self.add_handles):
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            long_enough = math.hypot(bx - ax, by - ay) * zoom >= MIN_ADD_SEGMENT_PX
            h.place(((ax + bx) / 2, (ay + by) / 2), zoom, self.hover == ("add", k),
                    visible=long_enough and self.dragging is None)
        for j, (h, (w, end, _)) in enumerate(zip(self.junction_handles, self.junctions)):
            h.place(getattr(w, end), zoom, self._lit(("junction", j)))
        self.halo.set_points(pts)

    def _lit(self, target: Target) -> bool:
        return self.hover == target or self.dragging == target

    def _resize(self, handles: list[_Handle], n: int, kind: str) -> None:
        while len(handles) < n:
            handles.append(_Handle(kind, self.batch, self.group))
        while len(handles) > n:
            handles.pop().delete()

    # ---- hit testing (screen pixels) ------------------------------------------

    def target_at(self, sx: float, sy: float, prefer_junction: bool = False) -> Target | None:
        """Bend handles win over junctions (for a branch on a corner, moving the corner is
        the likelier intent, and the junction follows it); junctions win over "+" handles.
        `prefer_junction` (Alt held) puts junctions first."""
        order = [("bend", self.bend_handles), ("junction", self.junction_handles), ("add", self.add_handles)]
        if prefer_junction:
            order[0], order[1] = order[1], order[0]
        for kind, handles in order:
            for i, h in enumerate(handles):
                if not h.visible:
                    continue
                hx, hy = self.camera.world_to_screen(*h.pos)
                if math.hypot(hx - sx, hy - sy) <= HIT_PX:
                    return kind, i
        return None

    def segment_at(self, wx: float, wy: float) -> tuple[int, Point] | None:
        """Nearest segment within grabbing distance, and the grabbed point on it."""
        limit = (T.WIRE_THICKNESS / 2) + T.HIT_SLOP_PX / self.camera.zoom
        best = None
        pts = self.view.points
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            p = _project(wx, wy, a, b)
            d = math.hypot(p[0] - wx, p[1] - wy)
            if d <= limit and (best is None or d < best[0]):
                best = (d, k, p)
        return None if best is None else (best[1], best[2])

    def set_hover(self, target: Target | None) -> bool:
        """Returns True if the hover target changed."""
        if target == self.hover:
            return False
        self.hover = target
        self.refresh()
        return True

    # ---- edits -----------------------------------------------------------------

    def insert(self, segment: int, point: Point) -> int:
        """Split `segment` at `point`; returns the new bend's index."""
        a, b = self.view.points[segment:segment + 2]
        seg = math.dist(a, b)
        s = 0.0 if seg == 0 else math.dist(a, point) / seg  # where along the old segment
        for j, (k, t) in self.anchors.items():
            if k > segment:
                self.anchors[j] = k + 1, t
            elif k == segment:  # the split point itself stays put, so t == s maps either way
                self.anchors[j] = (k, t / s) if t < s else (k + 1, 1.0 if s >= 1 else (t - s) / (1 - s))
        bends = list(self.view.bends)
        bends.insert(segment, point)  # segment k ends at points[k+1] == bends[k]
        self.view.set_bends(bends)
        self._follow_anchors()
        self.refresh()
        return segment

    def add_handle_pos(self, segment: int) -> Point:
        return self.add_handles[segment].pos

    def remove(self, index: int) -> None:
        # Segments `index` and `index + 1` merge; each anchor on them keeps its share of the path.
        pts = self.view.points
        l1, l2 = math.dist(pts[index], pts[index + 1]), math.dist(pts[index + 1], pts[index + 2])
        for j, (k, t) in self.anchors.items():
            if k > index + 1:
                self.anchors[j] = k - 1, t
            elif k in (index, index + 1):
                along = t * l1 if k == index else l1 + t * l2
                self.anchors[j] = index, (along / (l1 + l2) if l1 + l2 else 0.0)
        bends = list(self.view.bends)
        del bends[index]
        self.view.set_bends(bends)
        self._follow_anchors()
        self.hover = None
        self.refresh()

    def begin_drag(self, target: Target, wx: float, wy: float) -> None:
        kind, i = target
        if kind == "bend":
            px, py = self.view.bends[i]
        else:
            w, end, _ = self.junctions[i]
            px, py = getattr(w, end)
        self.dragging = target
        self.grab = (px - wx, py - wy)  # grab where you clicked; no jump to the cursor
        self.refresh()

    def drag_to(self, wx: float, wy: float, snap) -> None:
        kind, i = self.dragging
        p = snap(wx + self.grab[0], wy + self.grab[1])
        if kind == "bend":
            bends = list(self.view.bends)
            bends[i] = p
            self.view.set_bends(bends)
        else:  # a junction never leaves its rail; with Ctrl, the grid point gets projected onto it
            w, end, rail = self.junctions[i]
            q = project_onto(rail.points, p)
            _set_end(w, end, q)
            if rail is self.view:
                self.anchors[i] = _anchor_of(rail.points, q)
        self._follow_anchors()
        self.refresh()

    def end_drag(self) -> None:
        self.dragging = None
        self.refresh()

    def _follow_anchors(self) -> None:
        """Put every branch junction back where its anchor says, on the (changed) wire."""
        pts = self.view.points
        for j, (k, t) in self.anchors.items():
            w, end, _ = self.junctions[j]
            _set_end(w, end, _at(pts, k, t))

    def revert(self) -> None:
        self.view.set_ends(*self.original_ends)
        self.view.set_bends(self.original)
        for b, src, dst in self.branch_original:
            b.set_ends(src, dst)

    def close(self) -> None:
        self.halo.delete()
        for h in self.bend_handles + self.add_handles + self.junction_handles:
            h.delete()
        self.bend_handles.clear()
        self.add_handles.clear()
        self.junction_handles.clear()


def _set_end(view: WireView, end: End, p: Point) -> None:
    view.set_ends(*((p, view.dst) if end == "src" else (view.src, p)))


def _anchor_of(points: list[Point], p: Point) -> Anchor:
    """The (segment, fraction) of the spot on the polyline nearest to `p`. A point on a
    vertex gets an exact 0 or 1, so it stays bit for bit on that vertex (see _at)."""
    best, best_d = (0, 0.0), math.inf
    for k, (a, b) in enumerate(zip(points, points[1:])):
        if math.dist(a, p) < 1e-9:
            return k, 0.0
        if math.dist(b, p) < 1e-9:
            return k, 1.0
        q = _project(*p, a, b)
        d = math.dist(q, p)
        if d < best_d:
            seg = math.dist(a, b)
            best, best_d = (k, 0.0 if seg == 0 else math.dist(a, q) / seg), d
    return best


def _at(points: list[Point], k: int, t: float) -> Point:
    a, b = points[k], points[k + 1]
    if t <= 0:
        return a
    if t >= 1:
        return b
    return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])


def _project(px: float, py: float, a: Point, b: Point) -> Point:
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return x1 + t * dx, y1 + t * dy
