"""Boxes: labeled, colored rectangles drawn behind everything else on the board.

They're the editor's alone: the circuit never hears of them. A box holds nothing --
what's "in" it is whatever lies inside its rectangle right now (see Editor.box_contents)
-- so clicking one selects that, and dragging it by its header carries it along.

A box is a few shapes: its body (a faint fill with a border), a header strip along the
top (what you grab to move it) and its label in that strip. Boards have a handful of
them, so each is its own little object, unlike parts and wires (see views.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..snapshot import BoxData
from . import theme as T
from .sdf_shapes import Rect

if TYPE_CHECKING:
    from .canvas import Canvas
    from .sdf_text import SDFText
    from .views import Layers

Point = tuple[float, float]

HEADER = 16  # header strip height, world units
BORDER = 1.5
PAD = 12  # around what a wrapped selection's box goes
MIN_SIZE = 2 * HEADER  # smallest width / height a box can be resized to
NEUTRAL = (150, 150, 170)  # a box with no color of its own
FILL_ALPHA, HEADER_ALPHA, BORDER_ALPHA = 22, 70, 170
GHOST_OPACITY = 120  # (a pasted box being carried)


def rgb(color: str | None) -> tuple[int, int, int]:
    """A box color name's on-color (names as in T.WIRE_COLORS; others: neutral)."""
    pair = T.WIRE_COLORS.get(color) if color else None
    return pair[1] if pair else NEUTRAL


class BoxView:
    """One box on the board. `uid` is stable (undo, saves); x, y is its bottom left."""

    def __init__(
        self,
        uid: int,
        data: BoxData,
        canvas: Canvas,
        layers: Layers,
        text: SDFText,
    ) -> None:
        self.uid = uid
        self.label, self.x, self.y, self.w, self.h, self.color = data
        self.selected = False
        self.ghost = False
        self.body = Rect(0, 0, 1, 1, BORDER, (0, 0, 0, 0), (0, 0, 0, 0), canvas, layers.boxes)
        self.header = Rect(0, 0, 1, 1, 0, (0, 0, 0, 0), (0, 0, 0, 0), canvas, layers.boxes)
        self.text = text
        self.name = text.label(self.label, 0, 0, T.LABEL_SIZE, (*T.LABEL_TEXT[:3], 230), "left")
        self._paint()
        self._place()

    # ---- data ----------------------------------------------------------------

    @property
    def data(self) -> BoxData:
        return self.label, self.x, self.y, self.w, self.h, self.color

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1)"""
        return self.x, self.y, self.x + self.w, self.y + self.h

    def set_data(self, data: BoxData) -> None:
        label, x, y, w, h, color = data
        if label != self.label:
            self.set_label(label)
        if color != self.color:
            self.set_color(color)
        if (x, y, w, h) != (self.x, self.y, self.w, self.h):
            self.set_rect(x, y, w, h)

    def set_rect(self, x: float, y: float, w: float, h: float) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self._place()

    def move_by(self, dx: float, dy: float) -> None:
        self.set_rect(self.x + dx, self.y + dy, self.w, self.h)

    def set_label(self, label: str) -> None:
        self.label = label
        self.name.set_text(label)
        self._place()

    def set_color(self, color: str | None) -> None:
        self.color = color
        self._paint()

    # ---- looks ---------------------------------------------------------------

    def set_selected(self, on: bool) -> None:
        if on != self.selected:
            self.selected = on
            self._paint()

    def set_ghost(self, on: bool) -> None:
        self.ghost = on
        for shape in (self.body, self.header):
            shape.opacity = GHOST_OPACITY if on else 255
        self.name.opacity = GHOST_OPACITY if on else 230

    def set_lifted(self, on: bool) -> None:
        self.body.lifted = self.header.lifted = on
        self.name.lifted = on

    def _paint(self) -> None:
        c = rgb(self.color)
        edge = (*T.SELECT, 255) if self.selected else (*c, BORDER_ALPHA)
        self.body.set_colors((*c, FILL_ALPHA), edge)
        self.header.set_colors((*c, HEADER_ALPHA), (*c, HEADER_ALPHA))

    def name_pos(self) -> Point:
        return self.x + HEADER / 2, self.y + self.h - HEADER / 2

    def _place(self) -> None:
        x, y, w, h = self.x, self.y, self.w, self.h
        self.body.position = (x, y)
        self.body.width, self.body.height = w, h
        hh = min(HEADER, h)
        self.header.position = (x, y + h - hh)
        self.header.width, self.header.height = w, hh
        self.name.move_to(*self.name_pos())

    def delete(self) -> None:
        self.body.delete()
        self.header.delete()
        self.name.delete()

    # ---- hit testing (world units) -------------------------------------------

    def contains(self, wx: float, wy: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= wx <= x1 and y0 <= wy <= y1

    def in_header(self, wx: float, wy: float) -> bool:
        return self.contains(wx, wy) and wy >= self.y + self.h - HEADER

    def edges_at(self, wx: float, wy: float, slop: float) -> tuple[int, int] | None:
        """Which edges a press here grabs, for resizing: (-1 left / 1 right / 0,
        -1 bottom / 1 top / 0); None if it's not near an edge."""
        x0, y0, x1, y1 = self.rect
        if not (x0 - slop <= wx <= x1 + slop and y0 - slop <= wy <= y1 + slop):
            return None
        sx = -1 if abs(wx - x0) <= slop else 1 if abs(wx - x1) <= slop else 0
        sy = -1 if abs(wy - y0) <= slop else 1 if abs(wy - y1) <= slop else 0
        return (sx, sy) if sx or sy else None


def resized(
    rect: tuple[float, float, float, float], edges: tuple[int, int], to: Point
) -> tuple[float, float, float, float]:
    """`rect` (x, y, w, h) with the grabbed edges moved to `to`, no smaller than MIN_SIZE
    (the opposite edges stay put)."""
    x, y, w, h = rect
    sx, sy = edges
    x0, y0, x1, y1 = x, y, x + w, y + h
    if sx < 0:
        x0 = min(to[0], x1 - MIN_SIZE)
    elif sx > 0:
        x1 = max(to[0], x0 + MIN_SIZE)
    if sy < 0:
        y0 = min(to[1], y1 - MIN_SIZE)
    elif sy > 0:
        y1 = max(to[1], y0 + MIN_SIZE)
    return x0, y0, x1 - x0, y1 - y0
