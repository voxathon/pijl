"""Macros as part types: what lets a saved board be placed on another board.

A MacroType is built from a macro's saved board (its body). Its pins are the
body's IN / OUT ports, top to bottom by position (ties: left to right, then
uid). A pin is *identified* by its port's uid (that's what save files use, so
moving ports around inside a macro never rewires the boards using it); its
index only says where it's drawn. A port's label is the pin's name; unlabeled
ones are numbered by position.

The circuit never evaluates a macro: it expands the body into hidden parts and
joins the nets through the ports (see Circuit._expand). So a macro costs exactly
what its gates cost and adds no delay.

MacroBook loads definitions on demand and caches them. A macro that (directly or
through others) contains itself can't load; it shows up as a missing kind.
Catalog is what the circuit and the file format look kinds up in: part scripts
plus macros.
"""

from __future__ import annotations

from typing import Callable

from .parts import Look, PartType, Registry
from .parts.registry import layout_cached
from .snapshot import MACRO, Snapshot


class CycleError(KeyError):
    """A macro that contains itself."""


class MacroType(PartType):
    def __init__(
        self, name: str, body: Snapshot, catalog: Catalog, title: str | None = None
    ) -> None:
        self.name = name  # its id (see storage.py): what boards refer to it by
        self.kind = MACRO + name
        self.title = title or name  # what it's called
        self.body = body
        self.look = Look(
            label="below", swatch="MACRO_SWATCH", body="MACRO_BODY", pin_labels=True
        )
        # Every kind inside must exist (KeyError / CycleError otherwise): a body that
        # can't be built fully isn't a macro that can be placed.
        types = {uid: catalog.get(d[0]) for uid, d in body.parts.items()}
        self.in_ids = _ports(body, types, "in")
        self.out_ids = _ports(body, types, "out")
        self.ins = _names(body, self.in_ids)
        self.outs = _names(body, self.out_ids)
        # each pin as wide as its port (names may repeat, so not by `widths`)
        seen: dict = {}
        self._widths = tuple(
            _port_width(body, types[uid], uid, seen) for uid in self.in_ids + self.out_ids
        )
        # the macros used directly in the body
        self.uses = frozenset(
            d[0][len(MACRO) :] for d in body.parts.values() if d[0].startswith(MACRO)
        )


    def layout(self, props) -> dict:
        return {"ins": self.ins, "outs": self.outs, "widths": self._widths, "joins": ()}


def _port_width(body: Snapshot, t: PartType, uid: int, seen: dict) -> int:
    """How many lanes a port's one pin has."""
    lay = layout_cached(t, body.parts[uid][4], seen)
    return lay.widths[0] if lay is not None and lay.widths else 1


def _ports(body: Snapshot, types: dict[int, PartType], side: str) -> tuple[int, ...]:
    """Uids of the body's ports on one side, top to bottom (world y points up)."""
    found = [(uid, d) for uid, d in body.parts.items() if types[uid].port == side]
    found.sort(key=lambda item: (-item[1][3], item[1][2], item[0]))
    return tuple(uid for uid, _ in found)


def _names(body: Snapshot, ids: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(body.parts[uid][1] or str(i + 1) for i, uid in enumerate(ids))


class MacroBook:
    """Macro definitions by name (id), loaded when first needed. `load(name)` gives a
    macro's body Snapshot, or raises (KeyError / OSError / ValueError) if it can't;
    `title_of(name)` what it's called."""

    def __init__(
        self,
        catalog: Catalog,
        load: Callable[[str], Snapshot],
        title_of: Callable[[str], str] | None = None,
        uses_of: Callable[[str], frozenset[str]] | None = None,
    ) -> None:
        self.catalog = catalog
        self.load = load
        self.title_of = title_of or (lambda name: name)
        # the macros a macro uses directly, read cheaply (see MacroStore.uses);
        # without it, contains() loads every definition it walks through
        self.uses_of = uses_of
        self._types: dict[str, MacroType] = {}
        self._loading: list[
            str
        ] = []  # names being loaded right now (a repeat means a cycle)

    def get(self, name: str) -> MacroType:
        t = self._types.get(name)
        if t is not None:
            return t
        if name in self._loading:
            raise CycleError(
                f"{name} contains itself ({' -> '.join(self._loading[self._loading.index(name) :])} -> {name})"
            )
        self._loading.append(name)
        try:  # both steps can need other macros (nested ones), so both count as loading
            try:
                body = self.load(name)
            except CycleError:
                raise
            except (KeyError, OSError, ValueError) as e:
                raise KeyError(
                    f"macro {name!r}: {e.args[0] if isinstance(e, KeyError) and e.args else e}"
                ) from None
            t = MacroType(name, body, self.catalog, self.title_of(name))
        finally:
            self._loading.pop()
        self._types[name] = t
        return t

    def forget(self) -> None:
        """Definitions changed on disk (a macro was saved): load them again when needed.
        Parts already on the board keep the definition they were made from."""
        self._types.clear()

    def contains(self, outer: str, inner: str) -> bool:
        """Does macro `outer` have `inner` inside it, at any depth? (Unloadable: no.)"""
        seen, todo = set(), [outer]
        while todo:
            name = todo.pop()
            try:
                uses = self.uses_of(name) if self.uses_of else self.get(name).uses
            except KeyError:
                continue
            if inner in uses:
                return True
            todo += [u for u in uses if u not in seen]
            seen.update(uses)
        return False


class Catalog:
    """Every kind of part that exists: the part scripts' types plus macros ("macro:<name>")."""

    def __init__(
        self,
        registry: Registry,
        load: Callable[[str], Snapshot],
        title_of: Callable[[str], str] | None = None,
        uses_of: Callable[[str], frozenset[str]] | None = None,
    ) -> None:
        self.registry = registry
        self.book = MacroBook(self, load, title_of, uses_of)

    def get(self, kind: str) -> PartType:
        if kind.startswith(MACRO):
            return self.book.get(kind[len(MACRO) :])
        return self.registry.get(kind)

    def __contains__(self, kind: str) -> bool:
        try:
            self.get(kind)
        except KeyError:
            return False
        return True
