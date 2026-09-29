"""Editing one wire's bend points.

While a wire is being edited it glows, every bend point gets a square handle
and every segment gets a small round "+" handle at its midpoint:

  hold + drag a square       move that bend point (Ctrl snaps to the grid)
  hold + drag a "+"          insert a bend point there and drag it
  hold + drag on the wire    same, split at the spot you grabbed
  right-click a square       remove that bend point

Handles are world objects (they sit on the wire) but are sized and hit-tested
in *screen* pixels, so they stay equally easy to grab at every zoom level.
The session only knows about its WireView; the editor owns the mode switch.
"""

from __future__ import annotations

import math

import pyglet
from pyglet import shapes

from . import theme as T
from .camera import Camera
from .views import Layers, Point, Polyline, WireView

BEND_PX = 10          # square side
ADD_RADIUS_PX = 4.5
BORDER_PX = 1.5
HIT_PX = 8            # grab distance for any handle
MIN_ADD_SEGMENT_PX = 28  # hide "+" on segments too short to hold one comfortably

Target = tuple[str, int]  # ("bend", bend index) or ("add", segment index)


class _Handle:
    def __init__(self, kind: str, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group) -> None:
        self.kind = kind
        if kind == "bend":
            self.outer = shapes.Rectangle(0, 0, 1, 1, color=T.HANDLE_BORDER, batch=batch, group=group)
            self.inner = shapes.Rectangle(0, 0, 1, 1, color=T.HANDLE_FILL, batch=batch, group=group)
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
    def __init__(self, view: WireView, camera: Camera,
                 batch: pyglet.graphics.Batch, layers: Layers) -> None:
        self.view = view
        self.camera = camera
        self.batch = batch
        self.group = layers.overlay
        self.original = list(view.bends)
        self.halo = Polyline(view.points, T.WIRE_HALO, batch, layers.wire_halo,
                             thickness=T.WIRE_THICKNESS + 6)
        self.bend_handles: list[_Handle] = []
        self.add_handles: list[_Handle] = []
        self.dragging: int | None = None
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
        for i, (h, p) in enumerate(zip(self.bend_handles, self.view.bends)):
            h.place(p, zoom, self.hover == ("bend", i) or self.dragging == i)
        for k, h in enumerate(self.add_handles):
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            long_enough = math.hypot(bx - ax, by - ay) * zoom >= MIN_ADD_SEGMENT_PX
            h.place(((ax + bx) / 2, (ay + by) / 2), zoom, self.hover == ("add", k),
                    visible=long_enough and self.dragging is None)
        self.halo.set_points(pts)

    def _resize(self, handles: list[_Handle], n: int, kind: str) -> None:
        while len(handles) < n:
            handles.append(_Handle(kind, self.batch, self.group))
        while len(handles) > n:
            handles.pop().delete()

    # ---- hit testing (screen pixels) ------------------------------------------

    def target_at(self, sx: float, sy: float) -> Target | None:
        """Bend handles win over "+" handles (they're what you most likely meant)."""
        for kind, handles in (("bend", self.bend_handles), ("add", self.add_handles)):
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
        bends = list(self.view.bends)
        bends.insert(segment, point)  # segment k ends at points[k+1] == bends[k]
        self.view.set_bends(bends)
        self.refresh()
        return segment

    def add_handle_pos(self, segment: int) -> Point:
        return self.add_handles[segment].pos

    def remove(self, index: int) -> None:
        bends = list(self.view.bends)
        del bends[index]
        self.view.set_bends(bends)
        self.hover = None
        self.refresh()

    def begin_drag(self, index: int, wx: float, wy: float) -> None:
        bx, by = self.view.bends[index]
        self.dragging = index
        self.grab = (bx - wx, by - wy)  # grab where you clicked; no jump to the cursor
        self.refresh()

    def drag_to(self, wx: float, wy: float, snap) -> None:
        bends = list(self.view.bends)
        bends[self.dragging] = snap(wx + self.grab[0], wy + self.grab[1])
        self.view.set_bends(bends)
        self.refresh()

    def end_drag(self) -> None:
        self.dragging = None
        self.refresh()

    def revert(self) -> None:
        self.view.set_bends(self.original)

    def close(self) -> None:
        self.halo.delete()
        for h in self.bend_handles + self.add_handles:
            h.delete()
        self.bend_handles.clear()
        self.add_handles.clear()


def _project(px: float, py: float, a: Point, b: Point) -> Point:
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return x1 + t * dx, y1 + t * dy
