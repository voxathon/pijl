"""The bar along the bottom of the window: what's open, a few runtime numbers, and
a cogwheel that opens the settings menu (projects, controls; see Editor._cog_menu).

Screen space (HUD batch). It runs from the picker's edge to the right edge of the
window, following the picker as it slides in and out. This module only lays out,
draws and hit-tests; the editor feeds it the text (set_doc, set_stats).
"""

from __future__ import annotations

import pyglet
from pyglet import shapes

from . import theme as T

S = T.UI_SCALE
BAR_H = round(22 * S)
PAD = 8 * S
COG = BAR_H           # the cog button is a square this big, at the right end
COG_R = 5.5 * S       # outer radius of the gear's body
TEETH, TOOTH = 8, (3 * S, 2.5 * S)  # count, (width, how far past COG_R)
FONT, SIZE = "Consolas", 9.5 * S


class StatusBar:
    def __init__(self, batch: pyglet.graphics.Batch, win_w: int) -> None:
        self.batch = batch
        self.left = 0.0
        self.win_w = win_w
        self.hovered = False
        bg, fg = pyglet.graphics.Group(order=7), pyglet.graphics.Group(order=8)
        line = max(1, round(S / 2))
        self.bg = shapes.Rectangle(0, 0, 1, BAR_H, color=T.PICKER_HEADER, batch=batch, group=bg)
        self.border = shapes.Rectangle(0, BAR_H - line, 1, line, color=T.PICKER_BORDER, batch=batch, group=fg)
        self.cog_bg = shapes.Rectangle(0, 0, COG, BAR_H - line, color=T.PICKER_HEADER, batch=batch, group=fg)
        gear = pyglet.graphics.Group(order=9)
        hole = pyglet.graphics.Group(order=10)
        tw, tl = TOOTH
        self.teeth = [shapes.Rectangle(0, 0, tw, 2 * (COG_R + tl), color=T.HELP_TEXT[:3], batch=batch, group=gear)
                      for _ in range(TEETH // 2)]  # each bar is two opposite teeth
        for i, t in enumerate(self.teeth):
            t.anchor_position = (tw / 2, COG_R + tl)
            t.rotation = i * 360 / TEETH
        self.body = shapes.Circle(0, 0, COG_R, segments=32, color=T.HELP_TEXT[:3], batch=batch, group=gear)
        self.hole = shapes.Circle(0, 0, COG_R * 0.45, segments=24, color=T.PICKER_HEADER, batch=batch, group=hole)

        def label(color, anchor_x="left"):
            return pyglet.text.Label("", font_name=FONT, font_size=SIZE, color=color, anchor_x=anchor_x,
                                     anchor_y="center", batch=batch, group=fg)

        self.doc = label(T.PART_TEXT)
        self.stats = label(T.HELP_TEXT, anchor_x="right")
        self._layout()

    # ---- layout --------------------------------------------------------------

    def place(self, left: float, win_w: int) -> None:
        """Span from `left` (the picker's edge) to the window's right edge."""
        if (left, win_w) != (self.left, self.win_w):
            self.left, self.win_w = left, win_w
            self._layout()

    def _layout(self) -> None:
        w = max(1.0, self.win_w - self.left)
        cy = BAR_H / 2
        self.bg.x, self.bg.width = self.left, w
        self.border.x, self.border.width = self.left, w
        cx = self.win_w - COG / 2
        self.cog_bg.x = self.win_w - COG
        for t in self.teeth:
            t.position = (cx, cy)
        self.body.position = self.hole.position = (cx, cy)
        self.doc.position = (self.left + PAD, cy, 0)
        self.stats.position = (self.win_w - COG - PAD, cy, 0)
        # the numbers give way to the name when there isn't room for both
        self.stats.visible = self.stats.x - self.stats.content_width > self.doc.x + self.doc.content_width + 2 * PAD

    # ---- content ---------------------------------------------------------------

    def set_doc(self, text: str) -> None:
        if text != self.doc.text:
            self.doc.text = text
            self._layout()

    def set_stats(self, text: str) -> None:
        if text != self.stats.text:
            self.stats.text = text
            self._layout()

    # ---- hit testing -------------------------------------------------------------

    def contains(self, sx: float, sy: float) -> bool:
        return self.left <= sx <= self.win_w and 0 <= sy < BAR_H

    def cog_hit(self, sx: float, sy: float) -> bool:
        return self.win_w - COG <= sx <= self.win_w and 0 <= sy < BAR_H

    @property
    def cog_anchor(self) -> tuple[float, float]:
        """Where the cog's menu opens from: its top-right corner (the menu flips up and left)."""
        return self.win_w, BAR_H

    def set_hover(self, on: bool) -> None:
        if on != self.hovered:
            self.hovered = on
            color = T.PICKER_HOVER if on else T.PICKER_HEADER
            self.cog_bg.color = self.hole.color = color
            gear = T.PART_TEXT[:3] if on else T.HELP_TEXT[:3]
            self.body.color = gear
            for t in self.teeth:
                t.color = gear
