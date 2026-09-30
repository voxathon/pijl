"""Ctrl+D: duplicate the selection into a growing block.

Each press doubles the block, alternating right and down, so repeated presses
build power-of-two arrays: 1 -> 2x1 -> 2x2 -> 4x2 -> 4x4 ... The block is a
lattice of cells, each a copy of the original selection (the "unit") plus the
wires running inside it. Wires to anything outside the unit aren't copied.

Cells are spaced by the unit's own size: a bigger part has more pins, so it gets
more room for wiring. Both the size and the gap are whole grid cells, so a unit on
the grid stays on it. While the block is still selected, Ctrl+scroll widens or
narrows the gap along the axis of the last doubling (Ctrl+Shift+scroll: the other
axis) and every cell slides into place.

Changing the selection, or moving it, ends the pattern: the next Ctrl+D starts
over with whatever is selected then as the unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..snapshot import Snapshot
from . import theme as T
from .views import PartView, Point, WireView

RIGHT, DOWN = 0, 1
MIN_GAP = (
    1  # grid cells: pins sit on part edges, so touching parts would short visually
)


@dataclass
class Cell:
    """One copy of the unit, with where each piece sits in the unit (cell 0, 0)."""

    parts: list[tuple[PartView, float, float]] = field(default_factory=list)
    wires: list[tuple[WireView, list[Point], Point, Point]] = field(
        default_factory=list
    )

    @classmethod
    def of(cls, parts: list[PartView], wires: list[WireView]) -> Cell:
        return cls(
            [(v, v.x, v.y) for v in parts],
            [(w, list(w.bends), w.src, w.dst) for w in wires],
        )


class Tiling:
    def __init__(
        self, unit: Snapshot, parts: list[PartView], wires: list[WireView]
    ) -> None:
        self.unit = unit
        xs = [x for v in parts for x in (v.x, v.x + v.w)] + [
            p[0] for w in wires for p in w.points
        ]
        ys = [y for v in parts for y in (v.y, v.y + v.h)] + [
            p[1] for w in wires for p in w.points
        ]
        self.size = (_cells(max(xs) - min(xs)), _cells(max(ys) - min(ys)))
        self.gap = [max(MIN_GAP, s) for s in self.size]  # default: the unit's own size
        self.cells: dict[tuple[int, int], Cell] = {(0, 0): Cell.of(parts, wires)}
        self.cols = self.rows = 1
        self.last: int | None = None  # axis of the last doubling
        self.signature = None  # the selection right after our last change (see Editor)
        self.adjusting = (
            False  # the last history entry is a spacing change (merge into it)
        )

    def next_axis(self) -> int:
        return RIGHT if self.last != RIGHT else DOWN

    def grow(self, axis: int, make_cell) -> None:
        """Double along `axis`; make_cell() returns a fresh Cell at the unit's position."""
        shift = (self.cols, 0) if axis == RIGHT else (0, self.rows)
        for i, j in list(self.cells):
            self.cells[i + shift[0], j + shift[1]] = make_cell()
        if axis == RIGHT:
            self.cols *= 2
        else:
            self.rows *= 2
        self.last = axis

    def adjust(self, axis: int, notches: int) -> bool:
        """Change the gap along `axis`; False if it's already at the minimum."""
        new = max(MIN_GAP, self.gap[axis] + notches)
        changed = new != self.gap[axis]
        self.gap[axis] = new
        return changed

    def offset(self, i: int, j: int) -> Point:
        """Where cell (i, j) sits relative to the unit. Down is -y (world y points up)."""
        return (
            i * (self.size[RIGHT] + self.gap[RIGHT]) * T.GRID,
            -j * (self.size[DOWN] + self.gap[DOWN]) * T.GRID,
        )

    def layout(self) -> list[WireView]:
        """Move every cell into place; returns the wires moved (their ends need re-attaching)."""
        moved = []
        for (i, j), cell in self.cells.items():
            dx, dy = self.offset(i, j)
            for view, x, y in cell.parts:
                view.move_to(x + dx, y + dy)
            for view, bends, (sx, sy), (tx, ty) in cell.wires:
                view.src, view.dst = (
                    (sx + dx, sy + dy),
                    (tx + dx, ty + dy),
                )  # junction ends ride along
                view.set_bends([(bx + dx, by + dy) for bx, by in bends])
                moved.append(view)
        return moved

    def all_parts(self) -> list[PartView]:
        return [v for c in self.cells.values() for v, _, _ in c.parts]

    def all_wires(self) -> list[WireView]:
        return [w for c in self.cells.values() for w, *_ in c.wires]


def _cells(length: float) -> int:
    return max(1, math.ceil(length / T.GRID - 1e-6))
