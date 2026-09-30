"""Right-click context menu: a small screen-space popup of clickable items.

Lives in the HUD batch (identity view matrix), so it ignores the camera. It
opens down-right from the cursor and flips left/up when that would run off
the window.

An item with a `submenu` shows an arrow; hovering it opens the submenu beside
it (right, or left if there's no room), with its first row level with the
item. Submenus nest to any depth. Hovering another item of a parent closes the
open submenu; leaving the menu altogether keeps it, so a diagonal mouse path
from an item into its submenu doesn't lose it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import colorsys

import pyglet
from pyglet import shapes

from . import theme as T
from .views import Box, _GradientLine

S = T.UI_SCALE
ITEM_W, ITEM_H, PAD = 140 * S, 26 * S, 3 * S
SWATCH = 12 * S   # color square before an item's text
ARROW = 5 * S     # half-height of the submenu arrow
OVERLAP = 2 * S   # a submenu slightly overlaps its parent, so there's no gap to cross
RAINBOW = "rainbow"  # a swatch that runs through every hue (Default: "any color")
RAINBOW_PIECES = 6  # blended hue to hue, so it reads as one smooth sweep


@dataclass
class MenuItem:
    text: str
    action: Callable[[], None] | None = None
    danger: bool = False  # drawn in red (e.g. Delete)
    submenu: list[MenuItem] | None = None  # hovering opens it; the item has no action of its own
    swatch: tuple[int, int, int] | str | None = None  # a color square before the text (or RAINBOW)
    checked: bool = False  # the current choice: a border around the swatch, or a dot if none

    def __post_init__(self) -> None:
        assert (self.action is None) != (self.submenu is None), "an item has an action or a submenu"


@dataclass
class _Panel:
    items: list[MenuItem]
    box: Box
    rows: list[shapes.Rectangle]
    shapes: list = field(default_factory=list)  # labels, swatches, arrows: deleted with the panel
    hovered: int | None = None

    def item_at(self, sx: float, sy: float) -> int | None:
        return next((i for i, r in enumerate(self.rows)
                     if r.x <= sx <= r.x + r.width and r.y <= sy <= r.y + r.height), None)

    def set_hover(self, idx: int | None) -> None:
        self.hovered = idx
        for i, row in enumerate(self.rows):
            row.color = T.MENU_HOVER if i == idx else T.MENU_PANEL[0]

    def delete(self) -> None:
        self.box.delete()
        for s in (*self.rows, *self.shapes):
            s.delete()


class ContextMenu:
    def __init__(self, batch: pyglet.graphics.Batch) -> None:
        self.batch = batch
        self.panels: list[_Panel] = []  # the root menu, then each open submenu
        self.win = (0, 0)

    @property
    def visible(self) -> bool:
        return bool(self.panels)

    def open(self, sx: float, sy: float, items: list[MenuItem], win_w: int, win_h: int) -> None:
        self.close()
        self.win = (win_w, win_h)
        self._push(items, sx, sy, flip_x=sx)

    def _push(self, items: list[MenuItem], sx: float, sy: float, flip_x: float) -> None:
        """Open a panel with its top-left at (sx, sy); if it doesn't fit on the right, its
        top-right goes at `flip_x` instead."""
        depth = len(self.panels)
        # groups per depth, so a submenu draws over its parent
        bg, row_g, fg = (pyglet.graphics.Group(order=10 + 3 * depth + k) for k in range(3))
        win_w, win_h = self.win
        labels = [pyglet.text.Label(item.text, font_name="Consolas", font_size=11 * S,
                                    color=T.MENU_DANGER if item.danger else T.PART_TEXT, anchor_y="center",
                                    batch=self.batch, group=fg) for item in items]
        text_x = 10 * S + (SWATCH + 8 * S if any(i.swatch or i.checked for i in items) else 0)
        arrow_w = 4 * ARROW if any(i.submenu for i in items) else 0
        item_w = max([ITEM_W, *(text_x + label.content_width + 10 * S + arrow_w for label in labels)])
        w = item_w + 2 * PAD
        h = len(items) * ITEM_H + 2 * PAD
        left = sx if sx + w <= win_w else flip_x - w
        top = sy if sy - h >= 0 else sy + h
        left = max(0, min(left, win_w - w))
        top = max(h, min(top, win_h))

        box = Box(w, h, max(1, round(S / 2)), *T.MENU_PANEL, self.batch, bg)
        box.position = (left, top - h)
        panel = _Panel(items, box, [], list(labels))
        for i, (item, label) in enumerate(zip(items, labels)):
            y = top - PAD - (i + 1) * ITEM_H
            cy = y + ITEM_H / 2
            panel.rows.append(shapes.Rectangle(left + PAD, y, item_w, ITEM_H, color=T.MENU_PANEL[0],
                                               batch=self.batch, group=row_g))
            label.position = (left + PAD + text_x, cy, 0)
            if item.swatch:
                sx0 = left + PAD + 10 * S
                if item.checked:  # a ring around the current color
                    b = 2 * S
                    panel.shapes.append(shapes.Rectangle(sx0 - b, cy - SWATCH / 2 - b, SWATCH + 2 * b,
                                                         SWATCH + 2 * b, color=T.PART_TEXT[:3],
                                                         batch=self.batch, group=fg))
                if item.swatch == RAINBOW:
                    piece = SWATCH / RAINBOW_PIECES
                    hue = [tuple(round(c * 255) for c in colorsys.hsv_to_rgb(k / RAINBOW_PIECES, 0.75, 0.95))
                           for k in range(RAINBOW_PIECES + 1)]
                    for k in range(RAINBOW_PIECES):
                        line = _GradientLine(sx0 + k * piece, cy, sx0 + (k + 1) * piece, cy, thickness=SWATCH,
                                             batch=self.batch, group=fg)
                        line.set_colors(hue[k], hue[k + 1])
                        panel.shapes.append(line)
                else:
                    panel.shapes.append(shapes.Rectangle(sx0, cy - SWATCH / 2, SWATCH, SWATCH, color=item.swatch,
                                                         batch=self.batch, group=fg))
            elif item.checked:  # no swatch: a dot marks the current choice
                panel.shapes.append(shapes.Circle(left + PAD + 10 * S + SWATCH / 2, cy, SWATCH / 4,
                                                  color=T.PART_TEXT[:3], batch=self.batch, group=fg))
            if item.submenu:
                ax = left + PAD + item_w - 10 * S
                panel.shapes.append(shapes.Triangle(ax - 1.5 * ARROW, cy - ARROW, ax - 1.5 * ARROW, cy + ARROW,
                                                    ax, cy, color=T.PART_TEXT[:3], batch=self.batch, group=fg))
        self.panels.append(panel)

    def close(self) -> None:
        for p in self.panels:
            p.delete()
        self.panels.clear()

    def _truncate(self, n: int) -> None:
        """Keep only the first n panels."""
        while len(self.panels) > n:
            self.panels.pop().delete()

    def _panel_at(self, sx: float, sy: float) -> int | None:
        """The topmost (deepest) panel under the point."""
        return next((d for d in range(len(self.panels) - 1, -1, -1) if self.panels[d].box.contains(sx, sy)), None)

    def contains(self, sx: float, sy: float) -> bool:
        return self._panel_at(sx, sy) is not None

    def rect_at(self, sx: float, sy: float) -> tuple[float, float, float, float] | None:
        """(x, y, w, h) of the topmost panel under the point, if any."""
        d = self._panel_at(sx, sy)
        if d is None:
            return None
        box = self.panels[d].box
        return box.x, box.y, box.w, box.h

    def item_at(self, sx: float, sy: float) -> MenuItem | None:
        d = self._panel_at(sx, sy)
        if d is None:
            return None
        i = self.panels[d].item_at(sx, sy)
        return None if i is None else self.panels[d].items[i]

    def hover(self, sx: float, sy: float) -> None:
        d = self._panel_at(sx, sy)
        if d is None:
            return  # outside: leave everything as is (see the module docstring)
        panel = self.panels[d]
        i = panel.item_at(sx, sy)
        if i is None or (i == panel.hovered and len(self.panels) > d + 1):
            return  # on the border / padding, or still on the item whose submenu is open
        self._truncate(d + 1)
        panel.set_hover(i)
        item = panel.items[i]
        if item.submenu:
            row = panel.rows[i]
            self._push(item.submenu, panel.box.x + panel.box.w - OVERLAP, row.y + row.height + PAD,
                       flip_x=panel.box.x + OVERLAP)
