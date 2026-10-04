"""The set of selected parts, wires and boxes, kept in sync with their highlight visuals.

Views show it themselves (a flag on their shapes, see views.select_many); this
class only decides which views are selected and flips them on and off.
"""

from __future__ import annotations

from collections.abc import Iterable

from .boxes import BoxView
from .views import PartView, WireView, select_many

View = PartView | WireView | BoxView


class Selection:
    def __init__(self) -> None:
        self.parts: set[PartView] = set()
        self.wires: set[WireView] = set()
        self.boxes: set[BoxView] = set()

    def __bool__(self) -> bool:
        return bool(self.parts or self.wires or self.boxes)

    def __contains__(self, view: View) -> bool:
        return view in self.parts or view in self.wires or view in self.boxes

    def _bucket(self, view: View) -> set:
        if isinstance(view, BoxView):
            return self.boxes
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

    def set(
        self,
        parts: Iterable[PartView] = (),
        wires: Iterable[WireView] = (),
        boxes: Iterable[BoxView] = (),
    ) -> None:
        """Replace the selection, touching only views whose state actually changes
        (cheap to call on every mouse move while box-selecting)."""
        parts, wires, boxes = set(parts), set(wires), set(boxes)
        select_many(list(self.parts - parts), list(self.wires - wires), False)
        select_many(list(parts - self.parts), list(wires - self.wires), True)
        for box in self.boxes ^ boxes:
            box.set_selected(box in boxes)
        self.parts, self.wires, self.boxes = parts, wires, boxes

    def clear(self) -> None:
        self.set()
