"""The set of selected parts and wires, kept in sync with their highlight visuals.

Views own their highlight (PartView.outline, WireView.highlight); this class
only decides which views are selected and flips them on and off.
"""

from __future__ import annotations

from collections.abc import Iterable

from .views import PartView, WireView

View = PartView | WireView


class Selection:
    def __init__(self) -> None:
        self.parts: set[PartView] = set()
        self.wires: set[WireView] = set()

    def __bool__(self) -> bool:
        return bool(self.parts or self.wires)

    def __contains__(self, view: View) -> bool:
        return view in self.parts or view in self.wires

    def _bucket(self, view: View) -> set:
        return self.parts if isinstance(view, PartView) else self.wires

    def add(self, view: View) -> None:
        self._bucket(view).add(view)
        view.set_selected(True)

    def discard(self, view: View) -> None:
        """Deselect; safe to call for views that aren't selected (e.g. on delete)."""
        bucket = self._bucket(view)
        if view in bucket:
            bucket.discard(view)
            view.set_selected(False)

    def toggle(self, view: View) -> None:
        if view in self:
            self.discard(view)
        else:
            self.add(view)

    def set(self, parts: Iterable[PartView] = (), wires: Iterable[WireView] = ()) -> None:
        """Replace the selection, touching only views whose state actually changes
        (cheap to call on every mouse move while box-selecting)."""
        parts, wires = set(parts), set(wires)
        for old, new in ((self.parts, parts), (self.wires, wires)):
            for view in old - new:
                view.set_selected(False)
            for view in new - old:
                view.set_selected(True)
        self.parts, self.wires = parts, wires

    def clear(self) -> None:
        self.set()
