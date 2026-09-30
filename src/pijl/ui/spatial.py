"""Finding what's near a point without looking at every view in Python: hit testing
on every mouse move, box selection.

Objects register boxes: a part its body, a wire one box per piece of its line (long
segments are cut into pieces no longer than PIECE, so a diagonal doesn't claim its
whole bounding box). The boxes live in flat numpy arrays and a query compares them
all at once: a fraction of a millisecond even at 100k parts, and moving many
objects together (dropping a dragged selection) is one array add, not a re-sort
of each object into grid cells. Results are candidates: the caller still does the
exact test.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable

import numpy as np

Point = tuple[float, float]

PIECE = 80.0  # world units: longest piece of a line that gets a single box


class SpatialIndex:
    def __init__(self, capacity: int = 1024) -> None:
        self._cols = np.full((4, capacity), np.inf)  # x0, y0, x1, y1 per box; free rows never match
        self._cols[2:] = -np.inf
        self._owner: list[Hashable | None] = [None] * capacity
        self._free: list[int] = []
        self._end = 0  # rows below this were handed out at some point
        self.where: dict[Hashable, list[int]] = {}  # object -> its boxes' rows

    def __len__(self) -> int:
        return len(self.where)

    # ---- registering -------------------------------------------------------

    def put_rect(self, obj: Hashable, x0: float, y0: float, x1: float, y1: float) -> None:
        self._put(obj, [(x0, y0, x1, y1)])

    def put_polyline(self, obj: Hashable, points: list[Point]) -> None:
        self._put(obj, polyline_boxes(points))

    def put_many(self, items: list[tuple[Hashable, list[tuple]]]) -> None:
        """(object, its boxes) for many objects at once (new ones: in bulk)."""
        new = []
        for obj, boxes in items:
            if obj in self.where:
                self._put(obj, boxes)
            else:
                new.append((obj, boxes))
        n = sum(len(boxes) for _, boxes in new)
        if not n:
            return
        free = self._free
        rows = [free.pop() for _ in range(min(n, len(free)))]
        rest = n - len(rows)
        while self._end + rest > self._cols.shape[1]:
            self._grow()
        rows += range(self._end, self._end + rest)
        self._end += rest
        k = 0
        owner, where = self._owner, self.where
        for obj, boxes in new:
            mine = where[obj] = rows[k:k + len(boxes)]
            for r in mine:
                owner[r] = obj
            k += len(boxes)
        self._cols[:, rows] = np.array([box for _, boxes in new for box in boxes], np.float64).T

    def _put(self, obj: Hashable, boxes: list[tuple]) -> None:
        rows = self.where.get(obj)
        if rows is None:
            rows = self.where[obj] = []
        while len(rows) < len(boxes):
            rows.append(self._alloc(obj))
        while len(rows) > len(boxes):
            self._release(rows.pop())
        cols = self._cols
        for row, (x0, y0, x1, y1) in zip(rows, boxes):
            cols[0, row], cols[1, row], cols[2, row], cols[3, row] = x0, y0, x1, y1

    def _alloc(self, obj: Hashable) -> int:
        if self._free:
            row = self._free.pop()
        else:
            if self._end == self._cols.shape[1]:
                self._grow()
            row = self._end
            self._end += 1
        self._owner[row] = obj
        return row

    def _grow(self) -> None:
        n = self._cols.shape[1]
        grown = np.full((4, 2 * n), np.inf)
        grown[2:] = -np.inf
        grown[:, :n] = self._cols
        self._cols = grown
        self._owner.extend([None] * n)

    def _release(self, row: int) -> None:
        self._cols[:2, row], self._cols[2:, row] = np.inf, -np.inf
        self._owner[row] = None
        self._free.append(row)

    def remove(self, obj: Hashable) -> None:
        for row in self.where.pop(obj, ()):
            self._release(row)

    def remove_many(self, objs: Iterable[Hashable]) -> None:
        pop = self.where.pop
        rows = [r for obj in objs for r in pop(obj, ())]
        if rows:
            owner = self._owner
            for r in rows:
                owner[r] = None
            self._free.extend(rows)
            rows = np.array(rows, np.intp)
            self._cols[:2, rows], self._cols[2:, rows] = np.inf, -np.inf

    def shift(self, objs: Iterable[Hashable], dx: float, dy: float) -> None:
        """Move these objects' boxes by (dx, dy), all at once."""
        where = self.where
        rows = [r for obj in objs for r in where.get(obj, ())]
        if rows:
            rows = np.array(rows, np.intp)
            self._cols[0::2, rows] += dx
            self._cols[1::2, rows] += dy

    def clear(self) -> None:
        self.__init__()

    # ---- asking --------------------------------------------------------------

    def query(self, x0: float, y0: float, x1: float, y1: float) -> set:
        """Everything with a box overlapping the rectangle."""
        n = self._end
        c = self._cols
        hit = np.flatnonzero((c[0, :n] <= x1) & (c[2, :n] >= x0) & (c[1, :n] <= y1) & (c[3, :n] >= y0))
        owner = self._owner
        return {owner[i] for i in hit.tolist()}

    def near(self, x: float, y: float, r: float) -> set:
        return self.query(x - r, y - r, x + r, y + r)


def polyline_boxes(points: list[Point]) -> list[tuple]:
    """A line's boxes: one per piece (see PIECE)."""
    boxes = []
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        n = max(1, math.ceil(max(abs(bx - ax), abs(by - ay)) / PIECE))
        if n == 1:
            boxes.append((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)))
            continue
        for k in range(n):
            px, py = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            qx, qy = ax + (bx - ax) * (k + 1) / n, ay + (by - ay) * (k + 1) / n
            boxes.append((min(px, qx), min(py, qy), max(px, qx), max(py, qy)))
    if len(points) == 1:
        (x, y), = points
        boxes.append((x, y, x, y))
    return boxes


def ordered(views: Iterable, newest_first: bool = False) -> list:
    """Candidates back in creation order (views carry a `seq`), for "topmost wins"."""
    return sorted(views, key=lambda v: v.seq, reverse=newest_first)
