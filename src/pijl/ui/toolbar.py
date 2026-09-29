"""Screen-space chip palette along the bottom of the window.

Drawn with an identity view matrix, so it ignores the camera.
"""

from __future__ import annotations

import pyglet
from . import theme as T
from .views import Box

BUTTON_W, BUTTON_H, GAP, MARGIN = 64, 32, 6, 8


class Toolbar:
    def __init__(self, kinds: list[str], batch: pyglet.graphics.Batch) -> None:
        bg = pyglet.graphics.Group(order=0)
        fg = pyglet.graphics.Group(order=1)
        self.buttons: list[tuple[str, Box, pyglet.text.Label]] = []
        for i, kind in enumerate(kinds):
            x = MARGIN + i * (BUTTON_W + GAP)
            rect = Box(BUTTON_W, BUTTON_H, T.TOOLBAR_BORDER, *T.TOOLBAR_BG, batch, bg)
            rect.position = (x, MARGIN)
            label = pyglet.text.Label(kind, font_name="Consolas", font_size=11, color=T.CHIP_TEXT,
                                      x=x + BUTTON_W / 2, y=MARGIN + BUTTON_H / 2,
                                      anchor_x="center", anchor_y="center", batch=batch, group=fg)
            self.buttons.append((kind, rect, label))
        self.hovered: str | None = None

    def button_at(self, sx: float, sy: float) -> str | None:
        for kind, rect, _ in self.buttons:
            if rect.contains(sx, sy):
                return kind
        return None

    def set_hover(self, kind: str | None) -> None:
        if kind == self.hovered:
            return
        self.hovered = kind
        for k, rect, _ in self.buttons:
            rect.color, rect.border_color = T.TOOLBAR_HOVER if k == kind else T.TOOLBAR_BG
