"""The part library: which parts the picker offers, and how they're grouped.

Pure data, no pyglet, so it can be tested (and later serialized) on its own.
Parts live in user-made *collections*, or loose (in none). Every part appears
exactly once: moving it somewhere takes it out of wherever it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NEW_NAME = "NEW COLLECTION"


@dataclass(eq=False)
class Collection:
    name: str
    parts: list[str] = field(default_factory=list)
    builtin: bool = False  # the defaults; can be renamed and emptied, not deleted
    open: bool = True      # expanded in the picker


class Library:
    def __init__(self, parts: list[tuple[str, str]]) -> None:
        """`parts`: (kind, category) pairs, in order. Each category becomes a builtin
        collection (in order of first appearance); parts without one are loose."""
        by_name: dict[str, Collection] = {}
        self.loose: list[str] = []
        for kind, category in parts:
            if not category:
                self.loose.append(kind)
                continue
            if category not in by_name:
                by_name[category] = Collection(category, builtin=True)
            by_name[category].parts.append(kind)
        self.collections = list(by_name.values())

    def where(self, part: str) -> Collection | None:
        """The collection holding `part`, or None if it's loose."""
        return next((c for c in self.collections if part in c.parts), None)

    def _list(self, dest: Collection | None) -> list[str]:
        return self.loose if dest is None else dest.parts

    def move_part(self, part: str, dest: Collection | None, index: int | None = None) -> None:
        """Put `part` into `dest` (None = loose) so it ends up at position `index`
        (None = at the end). Positions count the list without the part in it."""
        self._list(self.where(part)).remove(part)
        target = self._list(dest)
        target.insert(len(target) if index is None else index, part)

    def new_collection(self, index: int | None = None) -> Collection:
        names = {c.name for c in self.collections}
        n = 1
        name = NEW_NAME
        while name in names:
            n += 1
            name = f"{NEW_NAME} {n}"
        c = Collection(name)
        self.collections.insert(len(self.collections) if index is None else index, c)
        return c

    def delete_collection(self, c: Collection) -> None:
        """Remove a (non-builtin) collection; its parts become loose."""
        if c.builtin:
            return
        self.collections.remove(c)
        self.loose.extend(c.parts)

    def move_collection(self, c: Collection, index: int) -> None:
        """Move `c` so it ends up at position `index` (counted without `c`)."""
        self.collections.remove(c)
        self.collections.insert(index, c)

    def rename(self, c: Collection, name: str) -> None:
        name = name.strip()
        if name:  # an empty name keeps the old one
            c.name = name
