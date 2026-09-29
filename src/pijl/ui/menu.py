"""Right-click context menu: a small screen-space popup of clickable items.

Lives in the HUD batch (identity view matrix), so it ignores the camera. It
opens down-right from the cursor and flips left/up when that would run off
the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pyglet
from pyglet import shapes

from . import theme as T
from .views import Box

ITEM_W, ITEM_H, PAD = 140, 26, 3


@dataclass
class MenuItem:
    text: str
    action: Callable[[], None]
    danger: bool = False  # drawn in red (e.g. Delete)


class ContextMenu:
    def __init__(self, batch: pyglet.graphics.Batch) -> None:
        self.batch = batch
        self.bg_group = pyglet.graphics.Group(order=10)  # above the toolbar
        self.row_group = pyglet.graphics.Group(order=11)
        self.text_group = pyglet.graphics.Group(order=12)
        self.items: list[MenuItem] = []
        self.rows: list[tuple[shapes.Rectangle, pyglet.text.Label]] = []
        self.panel: Box | None = None
        self.hovered: int | None = None

    @property
    def visible(self) -> bool:
        return self.panel is not None

    def open(self, sx: float, sy: float, items: list[MenuItem], win_w: int, win_h: int) -> None:
        self.close()
        self.items = items
        w = ITEM_W + 2 * PAD
        h = len(items) * ITEM_H + 2 * PAD
        # top-left corner at the cursor, flipped to stay on screen
        left = sx if sx + w <= win_w else sx - w
        top = sy if sy - h >= 0 else sy + h
        left = max(0, min(left, win_w - w))
        top = max(h, min(top, win_h))

        self.panel = Box(w, h, 1, *T.MENU_PANEL, self.batch, self.bg_group)
        self.panel.position = (left, top - h)
        for i, item in enumerate(items):
            y = top - PAD - (i + 1) * ITEM_H
            row = shapes.Rectangle(left + PAD, y, ITEM_W, ITEM_H, color=T.MENU_PANEL[0],
                                   batch=self.batch, group=self.row_group)
            label = pyglet.text.Label(item.text, font_name="Consolas", font_size=11,
                                      color=T.MENU_DANGER if item.danger else T.PART_TEXT,
                                      x=left + PAD + 10, y=y + ITEM_H / 2, anchor_y="center",
                                      batch=self.batch, group=self.text_group)
            self.rows.append((row, label))
        self.hovered = None

    def close(self) -> None:
        """Hide the menu. Its items stay remembered for activate_last()."""
        if self.panel is not None:
            self.panel.delete()
            self.panel = None
        for row, label in self.rows:
            row.delete()
            label.delete()
        self.rows.clear()
        self.hovered = None

    def contains(self, sx: float, sy: float) -> bool:
        return self.panel is not None and self.panel.contains(sx, sy)

    def item_at(self, sx: float, sy: float) -> int | None:
        return next((i for i, (r, _) in enumerate(self.rows)
                     if r.x <= sx <= r.x + r.width and r.y <= sy <= r.y + r.height), None)

    def hover(self, sx: float, sy: float) -> None:
        idx = self.item_at(sx, sy)
        if idx == self.hovered:
            return
        self.hovered = idx
        for i, (row, _) in enumerate(self.rows):
            row.color = T.MENU_HOVER if i == idx else T.MENU_PANEL[0]

    def activate_last(self, index: int) -> None:
        """Run an item of the most recently opened menu. Called after close(), so the
        action is free to open another menu or switch modes."""
        self.items[index].action()
