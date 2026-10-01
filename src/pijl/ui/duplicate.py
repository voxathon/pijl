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
from collections.abc import Callable
from dataclasses import dataclass

from ..snapshot import EndRef, Snapshot
from . import theme as T
from .views import PartView, Point, WireView

RIGHT, DOWN = 0, 1
MIN_GAP = (
    1  # grid cells: pins sit on part edges, so touching parts would short visually
)


@dataclass
class Cell:
    """One copy of the unit: its views, and where it sits relative to the unit (cell
    0, 0). Moving it to another offset moves every view by the difference."""

    parts: list[PartView]
    wires: list[WireView]
    at: Point = (0, 0)


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
        self.cells: dict[tuple[int, int], Cell] = {(0, 0): Cell(parts, wires)}
        self.cols = self.rows = 1
        self.last: int | None = None  # axis of the last doubling
        self.signature = None  # the selection right after our last change (see Editor)
        self.adjusting = (
            False  # the last history entry is a spacing change (merge into it)
        )

    def next_axis(self) -> int:
        return RIGHT if self.last != RIGHT else DOWN

    def grow(self, axis: int, make_cells: Callable[[list[Point]], list[Cell]]) -> None:
        """Double along `axis`. make_cells(offsets) returns a fresh Cell already in place
        at each of those offsets from the unit, all made at once (see tiled()): the
        cells already there stay where they are."""
        shift = (self.cols, 0) if axis == RIGHT else (0, self.rows)
        new = [(i + shift[0], j + shift[1]) for i, j in self.cells]
        for ij, cell in zip(
            new, make_cells([self.offset(*ij) for ij in new]), strict=True
        ):
            self.cells[ij] = cell
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
            at = self.offset(i, j)
            dx, dy = at[0] - cell.at[0], at[1] - cell.at[1]
            if not (dx or dy):
                continue
            cell.at = at
            for view in cell.parts:
                view.move_to(view.x + dx, view.y + dy)
            for view in cell.wires:
                (sx, sy), (tx, ty) = view.src, view.dst
                view.src, view.dst = (
                    (sx + dx, sy + dy),
                    (tx + dx, ty + dy),
                )  # junction ends ride along
                view.set_bends([(bx + dx, by + dy) for bx, by in view.bends])
                moved.append(view)
        return moved

    def all_parts(self) -> list[PartView]:
        return [v for c in self.cells.values() for v in c.parts]

    def all_wires(self) -> list[WireView]:
        return [w for c in self.cells.values() for w in c.wires]


def tiled(unit: Snapshot, offsets: list[Point]) -> tuple[Snapshot, int]:
    """One snapshot holding a copy of `unit` at each offset, for instantiating all of
    them in one go. Copy k's uids are k * stride + the unit's uid, so its parts and
    wires stay together and in the unit's order. Returns it and the stride."""
    stride = max([*unit.parts, *unit.wires], default=0) + 1

    def ref(r: EndRef, base: int) -> EndRef:
        return (r[0], base + r[1], *r[2:])

    def pt(p: Point | None, dx: float, dy: float) -> Point | None:
        return None if p is None else (p[0] + dx, p[1] + dy)

    parts, wires, colors = {}, {}, {}
    for k, (dx, dy) in enumerate(offsets):
        base = k * stride
        for uid, (kind, label, x, y, props) in unit.parts.items():
            parts[base + uid] = (kind, label, x + dx, y + dy, props)
        for uid, (src, dst, bends, src_pt, dst_pt) in unit.wires.items():
            wires[base + uid] = (
                ref(src, base),
                ref(dst, base),
                tuple((bx + dx, by + dy) for bx, by in bends),
                pt(src_pt, dx, dy),
                pt(dst_pt, dx, dy),
            )
        for uid, color in unit.wire_colors.items():
            colors[base + uid] = color
    return Snapshot(parts, wires, colors), stride


def _cells(length: float) -> int:
    return max(1, math.ceil(length / T.GRID - 1e-6))
