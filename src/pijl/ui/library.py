"""The part library: which parts the picker offers, and how they're grouped.

Pure data, no pyglet, so it can be tested and saved on its own (library.json).
Parts live in *collections*, or loose (in none). Every part appears exactly once:
moving it somewhere takes it out of wherever it was. Collections are only
grouping: deleting one never deletes a part (the editor deletes macro files
itself, see Editor._delete_collection).

LibraryHistory is the arrangement's undo timeline, as snapshots of to_dict().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

NEW_NAME = "NEW COLLECTION"


@dataclass(eq=False)
class Collection:
    name: str
    parts: list[str] = field(default_factory=list)
    builtin: bool = False  # made by sync for a category (not by the user); nothing more
    open: bool = True  # expanded in the picker


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
        return {
            "pijl": 1,
            "collections": [
                {
                    "name": c.name,
                    "parts": list(c.parts),
                    "builtin": c.builtin,
                    "open": c.open,
                }
                for c in self.collections
            ],
            "loose": list(self.loose),
        }

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
            if (
                not isinstance(d, dict)
                or not isinstance(d.get("name"), str)
                or d["name"] in names
            ):
                continue
            names.add(d["name"])
            lib.collections.append(
                Collection(
                    d["name"],
                    entries(d.get("parts")),
                    builtin=d.get("builtin") is True,
                    open=d.get("open") is not False,
                )
            )
        lib.loose = entries(data.get("loose")) if isinstance(data, dict) else []
        lib.sync(parts)
        return lib

    def restore(self, data: dict, parts: list[tuple[str, str]]) -> None:
        """Become the arrangement `data` (a to_dict(), synced to `parts`), in place.
        Collections keep their object and expanded state where one of the same name
        is still around, so the picker doesn't redraw what didn't change."""
        new = Library.from_dict(data, parts)
        old = {c.name: c for c in self.collections}
        for i, c in enumerate(new.collections):
            if (same := old.pop(c.name, None)) is not None:
                same.parts, same.builtin = c.parts, c.builtin
                new.collections[i] = same
        self.collections, self.loose = new.collections, new.loose

    def where(self, part: str) -> Collection | None:
        """The collection holding `part`, or None if it's loose."""
        return next((c for c in self.collections if part in c.parts), None)

    def _list(self, dest: Collection | None) -> list[str]:
        return self.loose if dest is None else dest.parts

    def move_part(
        self, part: str, dest: Collection | None, index: int | None = None
    ) -> None:
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
        """Remove a collection; its parts become loose."""
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


def layout(data: dict) -> dict:
    """A to_dict() without what isn't worth an undo step (which collections are expanded)."""
    return {
        **data,
        "collections": [
            {k: v for k, v in c.items() if k != "open"} for c in data["collections"]
        ],
    }


@dataclass
class Step:
    """One undo step of the library: the arrangement before and after (to_dict()s),
    the macros it moved to the trash: [id, where the file is now in the trash], and
    the ones it renamed: (id, old title, new title)."""

    stamp: int  # when it happened, on the same clock as the board's history
    before: dict
    after: dict
    trashed: list[list] = field(default_factory=list)
    retitled: list[tuple[str, str, str]] = field(default_factory=list)


class LibraryHistory:
    """Undo/redo of the arrangement. `current` is the last recorded one; anything that
    changes the library without being a step (a macro saved, one deleted for good)
    moves it with rebase(), so the next record() only sees what the user did."""

    def __init__(
        self, current: dict, stamp: Callable[[], int], limit: int = 500
    ) -> None:
        self.current = current
        self.stamp = stamp
        self.limit = limit
        self.undo_stack: list[Step] = []
        self.redo_stack: list[Step] = []

    def record(
        self,
        now: dict,
        trashed: list[list] | None = None,
        retitled: list[tuple[str, str, str]] | None = None,
    ) -> bool:
        """A new step from `current` to `now`; no-op (False) if nothing changed."""
        if not trashed and not retitled and layout(now) == layout(self.current):
            self.current = now  # (expanded / collapsed: kept, but not a step)
            return False
        self.undo_stack.append(
            Step(self.stamp(), self.current, now, trashed or [], retitled or [])
        )
        del self.undo_stack[: -self.limit]
        self.redo_stack.clear()
        self.current = now
        return True

    def rebase(self, now: dict) -> None:
        self.current = now

    def undo(self) -> Step | None:
        if not self.undo_stack:
            return None
        step = self.undo_stack.pop()
        self.redo_stack.append(step)
        return step

    def redo(self) -> Step | None:
        if not self.redo_stack:
            return None
        step = self.redo_stack.pop()
        self.undo_stack.append(step)
        return step
