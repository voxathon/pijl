"""A uniform grid over the world, for finding what's near a point without looking at
everything: hit testing on every mouse move, box selection.

Objects register the cells they cover (a part's box, a wire's line) and re-register
when they move; a query collects whatever is registered in the cells it overlaps.
Results are candidates: the caller still does the exact test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Hashable

Point = tuple[float, float]
Key = tuple[int, int]

CELL = 80.0  # world units: a typical part covers a few cells, a pin grid step is 1/8 of one


class SpatialHash:
    def __init__(self, cell: float = CELL) -> None:
        self.cell = cell
        self.cells: dict[Key, set] = {}
        self.where: dict[Hashable, frozenset[Key]] = {}  # object -> the cells it's in

    def __len__(self) -> int:
        return len(self.where)

    def _span(self, x0: float, y0: float, x1: float, y1: float) -> tuple[range, range]:
        c = self.cell
        return range(math.floor(x0 / c), math.floor(x1 / c) + 1), range(math.floor(y0 / c), math.floor(y1 / c) + 1)

    def put_rect(self, obj: Hashable, x0: float, y0: float, x1: float, y1: float) -> None:
        xs, ys = self._span(x0, y0, x1, y1)
        self._put(obj, frozenset((i, j) for i in xs for j in ys))

    def put_polyline(self, obj: Hashable, points: list[Point]) -> None:
        """Every cell the line passes through. Long segments are walked in pieces no
        longer than a cell, so a diagonal doesn't claim its whole bounding box."""
        keys: set[Key] = set()
        c = self.cell
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            n = max(1, math.ceil(max(abs(bx - ax), abs(by - ay)) / c))
            for k in range(n):
                px, py = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
                qx, qy = ax + (bx - ax) * (k + 1) / n, ay + (by - ay) * (k + 1) / n
                xs, ys = self._span(min(px, qx), min(py, qy), max(px, qx), max(py, qy))
                keys.update((i, j) for i in xs for j in ys)
        if len(points) == 1:
            keys.add((math.floor(points[0][0] / c), math.floor(points[0][1] / c)))
        self._put(obj, frozenset(keys))

    def _put(self, obj: Hashable, keys: frozenset[Key]) -> None:
        old = self.where.get(obj)
        if old == keys:
            return  # moved within the same cells: the common case while dragging
        if old:
            for key in old - keys:
                self._drop(obj, key)
        for key in keys - old if old else keys:
            self.cells.setdefault(key, set()).add(obj)
        self.where[obj] = keys

    def _drop(self, obj: Hashable, key: Key) -> None:
        bucket = self.cells[key]
        bucket.discard(obj)
        if not bucket:
            del self.cells[key]

    def remove(self, obj: Hashable) -> None:
        for key in self.where.pop(obj, ()):
            self._drop(obj, key)

    def query(self, x0: float, y0: float, x1: float, y1: float) -> set:
        """Everything registered in a cell the rectangle overlaps."""
        xs, ys = self._span(x0, y0, x1, y1)
        if len(xs) * len(ys) > len(self.cells):
            # A huge box (zoomed far out): walking the occupied cells is cheaper.
            return {obj for (i, j), bucket in self.cells.items()
                    if i in xs and j in ys for obj in bucket}
        found: set = set()
        for i in xs:
            for j in ys:
                bucket = self.cells.get((i, j))
                if bucket:
                    found |= bucket
        return found

    def near(self, x: float, y: float, r: float) -> set:
        return self.query(x - r, y - r, x + r, y + r)

    def clear(self) -> None:
        self.cells.clear()
        self.where.clear()


def ordered(views: Iterable, newest_first: bool = False) -> list:
    """Candidates back in creation order (views carry a `seq`), for "topmost wins"."""
    return sorted(views, key=lambda v: v.seq, reverse=newest_first)
