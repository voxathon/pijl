"""Screen-space chip palette along the bottom of the window.

Drawn with an identity view matrix, so it ignores the camera.
"""

from __future__ import annotations

import pyglet
from pyglet import shapes

from . import theme as T

BUTTON_W, BUTTON_H, GAP, MARGIN = 64, 32, 6, 8


class Toolbar:
    def __init__(self, kinds: list[str], batch: pyglet.graphics.Batch) -> None:
        bg = pyglet.graphics.Group(order=0)
        fg = pyglet.graphics.Group(order=1)
        self.buttons: list[tuple[str, shapes.RoundedRectangle, pyglet.text.Label]] = []
        for i, kind in enumerate(kinds):
            x = MARGIN + i * (BUTTON_W + GAP)
            rect = shapes.RoundedRectangle(x, MARGIN, BUTTON_W, BUTTON_H, radius=5,
                                           color=T.TOOLBAR_BG, batch=batch, group=bg)
            label = pyglet.text.Label(kind, font_name="Consolas", font_size=11, color=T.CHIP_TEXT,
                                      x=x + BUTTON_W / 2, y=MARGIN + BUTTON_H / 2,
                                      anchor_x="center", anchor_y="center", batch=batch, group=fg)
            self.buttons.append((kind, rect, label))
        self.hovered: str | None = None

    def button_at(self, sx: float, sy: float) -> str | None:
        for kind, rect, _ in self.buttons:
            if rect.x <= sx <= rect.x + rect.width and rect.y <= sy <= rect.y + rect.height:
                return kind
        return None

    def set_hover(self, kind: str | None) -> None:
        if kind == self.hovered:
            return
        self.hovered = kind
        for k, rect, _ in self.buttons:
            rect.color = T.TOOLBAR_HOVER if k == kind else T.TOOLBAR_BG
