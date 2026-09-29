"""The set of selected chips and wires, kept in sync with their highlight visuals.

Views own their highlight (ChipView.outline, WireView.highlight); this class
only decides which views are selected and flips them on and off.
"""

from __future__ import annotations

from collections.abc import Iterable

from .views import ChipView, WireView

View = ChipView | WireView


class Selection:
    def __init__(self) -> None:
        self.chips: set[ChipView] = set()
        self.wires: set[WireView] = set()

    def __bool__(self) -> bool:
        return bool(self.chips or self.wires)

    def __contains__(self, view: View) -> bool:
        return view in self.chips or view in self.wires

    def _bucket(self, view: View) -> set:
        return self.chips if isinstance(view, ChipView) else self.wires

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

    def set(self, chips: Iterable[ChipView] = (), wires: Iterable[WireView] = ()) -> None:
        """Replace the selection, touching only views whose state actually changes
        (cheap to call on every mouse move while box-selecting)."""
        chips, wires = set(chips), set(wires)
        for old, new in ((self.chips, chips), (self.wires, wires)):
            for view in old - new:
                view.set_selected(False)
            for view in new - old:
                view.set_selected(True)
        self.chips, self.wires = chips, wires

    def clear(self) -> None:
        self.set()
