"""The part library: which parts the picker offers, and how they're grouped.

Pure data, no pyglet, so it can be tested and saved on its own (library.json).
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
        """`parts`: (entry, category) pairs, in order. Each category becomes a builtin
        collection (in order of first appearance); parts without one are loose."""
        self.collections: list[Collection] = []
        self.loose: list[str] = []
        self.sync(parts)

    def sync(self, parts: list[tuple[str, str]]) -> None:
        """Match what actually exists: entries not in `parts` go away, missing ones are
        added to the collection named after their category (made if needed, as a
        builtin) or loose. Everything already placed stays where the user put it."""
        known = {entry for entry, _ in parts}
        for c in self.collections:
            c.parts = [p for p in c.parts if p in known]
        self.loose = [p for p in self.loose if p in known]
        placed = {p for c in self.collections for p in c.parts} | set(self.loose)
        for entry, category in parts:
            if entry in placed:
                continue
            placed.add(entry)
            if not category:
                self.loose.append(entry)
                continue
            c = next((c for c in self.collections if c.name == category), None)
            if c is None:
                c = Collection(category, builtin=True)
                self.collections.append(c)
            c.parts.append(entry)

    # ---- saving -------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"pijl": 1,
                "collections": [{"name": c.name, "parts": list(c.parts), "builtin": c.builtin, "open": c.open}
                                for c in self.collections],
                "loose": list(self.loose)}

    @classmethod
    def from_dict(cls, data: dict, parts: list[tuple[str, str]]) -> Library:
        """The saved arrangement, then synced to `parts` (see sync). Anything
        malformed in `data` is skipped rather than refused: it's only a layout."""
        lib = cls([])
        seen: set[str] = set()

        def entries(value) -> list[str]:
            out = []
            for e in value if isinstance(value, list) else []:
                if isinstance(e, str) and e not in seen:
                    seen.add(e)
                    out.append(e)
            return out

        names: set[str] = set()
        for d in data.get("collections", []) if isinstance(data, dict) else []:
            if not isinstance(d, dict) or not isinstance(d.get("name"), str) or d["name"] in names:
                continue
            names.add(d["name"])
            lib.collections.append(Collection(d["name"], entries(d.get("parts")), builtin=d.get("builtin") is True,
                                              open=d.get("open") is not False))
        lib.loose = entries(data.get("loose")) if isinstance(data, dict) else []
        lib.sync(parts)
        return lib

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
