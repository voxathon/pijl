"""Rows mods add to the editor's right-click menus.

    @after_import("pijl.ui.menus")
    def _(menus):
        @menus.items("part")
        def rows(editor, parts):
            return [MenuItem("Do a thing", lambda: ...)]

What each kind of menu passes as the target:

    "part"   the parts the menu acts on (a list of PartViews: one, or a selection of
             parts all of one kind)
    "wire"   the WireView clicked
    "box"    the BoxView (its header's menu, and the board menu inside it)
    "board"  the world point (x, y) clicked, on empty board

Mod rows go in above the menu's Delete rows. A function that raises is logged and
reported, and the menu opens without its rows. Rows are asked for each time a menu
opens, so they can depend on what's there.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger("pijl.ui")

KINDS = ("part", "wire", "box", "board")
PROVIDERS: dict[str, list[Callable[[Any, Any], list]]] = {k: [] for k in KINDS}


def items(kind: str):
    """Register a function (editor, target) -> list of MenuItems for one kind of menu."""
    if kind not in PROVIDERS:
        raise ValueError(f"no {kind!r} menu: one of {', '.join(KINDS)}")

    def register(fn):
        PROVIDERS[kind].append(fn)
        return fn

    return register


def extra(kind: str, editor, target) -> list:
    """Every registered row for this menu, in the order they were registered."""
    out = []
    for fn in PROVIDERS[kind]:
        try:
            out += list(fn(editor, target) or ())
        except Exception as e:
            name = getattr(fn, "__module__", "?")
            log.exception("menu rows from %s", name)
            editor._report(f"{name}: menu rows failed: {type(e).__name__}: {e}")
    return out
