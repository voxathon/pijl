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
        self._cols = np.full(
            (4, capacity), np.inf
        )  # x0, y0, x1, y1 per box; free rows never match
        self._cols[2:] = -np.inf
        self._owner = np.full(capacity, None, object)  # per row: whose box it is
        self._free: list[int] = []
        self._end = 0  # rows below this were handed out at some point
        self.where: dict[Hashable, list[int]] = {}  # object -> its boxes' rows

    def __len__(self) -> int:
        return len(self.where)

    # ---- registering -------------------------------------------------------

    def put_rect(
        self, obj: Hashable, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        self._put(obj, [(x0, y0, x1, y1)])

    def put_polyline(self, obj: Hashable, points: list[Point]) -> None:
        self._put(obj, polyline_boxes(points))

    def put_many(self, items: list[tuple[Hashable, list[tuple]]]) -> None:
        """(object, its boxes) for many objects at once."""
        if not items:
            return
        boxes = [box for _, bs in items for box in bs]
        counts = np.fromiter((len(bs) for _, bs in items), np.intp, len(items))
        self._put_arrays(
            [obj for obj, _ in items],
            np.array(boxes, np.float64).reshape(-1, 4),
            counts,
        )

    def put_boxes(self, objs: list, boxes: np.ndarray) -> None:
        """One box each (n x 4: x0, y0, x1, y1) for many objects at once."""
        if objs:
            self._put_arrays(objs, boxes, np.ones(len(objs), np.intp))

    def put_polylines(self, items: list[tuple[Hashable, list[Point]]]) -> None:
        """put_polyline for many objects at once: (object, its points)."""
        if items:
            boxes, counts = polylines_boxes([points for _, points in items])
            self._put_arrays([obj for obj, _ in items], boxes, counts)

    def _put_arrays(self, objs: list, boxes: np.ndarray, counts: np.ndarray) -> None:
        """objs[i] gets the next counts[i] rows of `boxes` (n x 4), replacing any it had."""
        self.remove_many([obj for obj in objs if obj in self.where])
        n = len(boxes)
        free = self._free
        take = min(n, len(free))
        rows = np.empty(n, np.intp)
        if take:  # (from the end, last first: as popping them one at a time would)
            rows[:take] = free[len(free) - take :][::-1]
            del free[len(free) - take :]
        rest = n - take
        while self._end + rest > self._cols.shape[1]:
            self._grow()
        rows[take:] = np.arange(self._end, self._end + rest)
        self._end += rest
        owners = np.empty(len(objs), object)
        for i, obj in enumerate(objs):  # (owners[:] = objs would unpack tuples)
            owners[i] = obj
        self._owner[rows] = np.repeat(owners, counts)
        self._cols[:, rows] = boxes.T
        row_list = rows.tolist()
        k, where = 0, self.where
        for obj, c in zip(objs, counts.tolist()):
            where[obj] = row_list[k : k + c]
            k += c

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
        owner = np.full(2 * n, None, object)
        owner[:n] = self._owner
        self._owner = owner

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
            self._free.extend(rows)
            rows = np.array(rows, np.intp)
            self._owner[rows] = None
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
        hit = np.flatnonzero(
            (c[0, :n] <= x1) & (c[2, :n] >= x0) & (c[1, :n] <= y1) & (c[3, :n] >= y0)
        )
        return set(self._owner[hit].tolist())

    def bounds(self) -> tuple[float, float, float, float] | None:
        """The box around everything (x0, y0, x1, y1), or None if empty."""
        c = self._cols[:, : self._end]
        if not len(self.where):
            return None
        return (
            float(c[0].min()),
            float(c[1].min()),
            float(c[2].max()),
            float(c[3].max()),
        )

    def near(self, x: float, y: float, r: float) -> set:
        return self.query(x - r, y - r, x + r, y + r)


def polyline_boxes(points: list[Point]) -> list[tuple]:
    """A line's boxes (see polylines_boxes; for one line, plain Python is quicker)."""
    if len(points) == 1:
        points = [points[0], points[0]]
    boxes = []
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        n = max(1, math.ceil(min(abs(dx), abs(dy)) / PIECE))
        for k in range(n):
            px, py = ax + dx * k / n, ay + dy * k / n
            qx, qy = ax + dx * (k + 1) / n, ay + dy * (k + 1) / n
            boxes.append((min(px, qx), min(py, qy), max(px, qx), max(py, qy)))
    return boxes


def polylines_boxes(lines: list[list[Point]]) -> tuple[np.ndarray, np.ndarray]:
    """The boxes of many lines at once: all of them, line by line (n x 4: x0, y0, x1,
    y1), and how many each line has. A segment is cut into pieces until each piece's
    box is at most PIECE across its narrow side -- that's how far a box can stray from
    the line -- so a diagonal is boxed tightly and a straight run of any length is one
    box. A single point is a box of its own."""
    lines = [pts if len(pts) != 1 else [pts[0], pts[0]] for pts in lines]
    sizes = np.fromiter((len(pts) for pts in lines), np.intp, len(lines))
    if not sizes.sum():
        return np.empty((0, 4)), np.zeros(len(lines), np.intp)
    p = np.array([xy for pts in lines for xy in pts], np.float64)
    line_of = np.repeat(np.arange(len(lines)), sizes)
    seg = np.flatnonzero(line_of[:-1] == line_of[1:])  # point i -> i + 1
    a, d = p[seg], p[seg + 1] - p[seg]
    pieces = np.maximum(1, np.ceil(np.abs(d).min(axis=1) / PIECE)).astype(np.intp)
    which = np.repeat(np.arange(len(seg)), pieces)
    k = (np.arange(len(which)) - (np.cumsum(pieces) - pieces)[which])[:, None]
    n = pieces[which][:, None]
    lo = a[which] + d[which] * (k / n)
    hi = a[which] + d[which] * ((k + 1) / n)
    boxes = np.hstack((np.minimum(lo, hi), np.maximum(lo, hi)))
    counts = np.bincount(line_of[seg], weights=pieces, minlength=len(lines))
    return boxes, counts.astype(np.intp)


def ordered(views: Iterable, newest_first: bool = False) -> list:
    """Candidates back in creation order (views carry a `seq`), for "topmost wins"."""
    return sorted(views, key=lambda v: v.seq, reverse=newest_first)
