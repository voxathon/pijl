"""A small box for editing a Number setting: a title, a slider, a typed field and a
hint line. It opens where the menu row was clicked (screen space, HUD batch) and
flips / clamps to stay inside the window.

This module only lays out, draws and hit-tests; the editor runs the edit (see
Editor._open_popover): live while the slider drags, one undo step per release or
Enter. `slider=False` settings get the field alone.
"""

from __future__ import annotations

import pyglet
from pyglet import shapes

from ..parts import Number
from . import theme as T
from .line_edit import LineEdit
from .views import Box

S = T.UI_SCALE
W, PAD = 260 * S, 10 * S
TITLE_H, SLIDER_H, FIELD_H, HINT_H = 24 * S, 26 * S, 28 * S, 22 * S
TRACK_H, KNOB_R = 4 * S, 7 * S
FONT, SIZE, SMALL = "Consolas", 11 * S, 10 * S


class NumberPopover:
    def __init__(self, batch: pyglet.graphics.Batch, win_w: int, win_h: int, anchor: tuple[float, float],
                 title: str, setting: Number, value, hint: str) -> None:
        """`value`: the current value, or None when the edited parts disagree (mixed):
        the field starts empty and there's no knob until the slider is used."""
        self.setting = setting
        self.anchor = anchor
        self.edit = LineEdit("", 40)
        self.caret_on, self._blink_t = True, 0.0
        bg, fg, top = (pyglet.graphics.Group(order=o) for o in (21, 22, 23))
        self.has_slider = setting.slider
        self.h = PAD + TITLE_H + (SLIDER_H if self.has_slider else 0) + FIELD_H + HINT_H + PAD / 2
        self.panel = Box(W, self.h, max(1, round(S)), *T.MENU_PANEL, batch, bg)

        def label(text="", color=T.PART_TEXT, size=SIZE, anchor_x="left"):
            return pyglet.text.Label(text, font_name=FONT, font_size=size, color=color, anchor_x=anchor_x,
                                     anchor_y="center", batch=batch, group=top)

        self.title = label(title)
        self.track = shapes.Rectangle(0, 0, W - 2 * PAD - 2 * KNOB_R, TRACK_H, color=T.PICKER_BORDER,
                                      batch=batch, group=fg)
        self.filled = shapes.Rectangle(0, 0, 0, TRACK_H, color=T.SELECT, batch=batch, group=fg)
        self.knob = shapes.Circle(0, 0, KNOB_R, color=T.CARET, batch=batch, group=top)
        for s in (self.track, self.filled, self.knob):
            s.visible = self.has_slider
        self.field = Box(W - 2 * PAD, FIELD_H, max(1, round(S)), T.PICKER_BG, T.SELECT, batch, fg)
        self.field_text = label()
        self.unit = label(setting.unit, color=T.PICKER_DIM_TEXT, anchor_x="right")
        self.caret = shapes.Rectangle(0, 0, 1.5 * S, 16 * S, color=T.CARET, batch=batch, group=top)
        self.hint = label(hint, color=T.HELP_TEXT, size=SMALL)
        self.value = value
        self.layout(win_w, win_h)
        self.set_value(value)

    # ---- layout ----------------------------------------------------------------

    def layout(self, win_w: int, win_h: int) -> None:
        ax, ay = self.anchor
        left = ax if ax + W <= win_w else ax - W  # down-right of the click, like the menu
        top = ay if ay - self.h >= 0 else ay + self.h
        left = max(0.0, min(left, win_w - W))
        top = max(self.h, min(top, win_h))
        self.panel.position = (left, top - self.h)
        y = top - PAD
        self.title.position = (left + PAD, y - TITLE_H / 2, 0)
        y -= TITLE_H
        if self.has_slider:
            self.track.position = (left + PAD + KNOB_R, y - SLIDER_H / 2 - TRACK_H / 2)
            self.filled.position = self.track.position
            y -= SLIDER_H
        self.field.position = (left + PAD, y - FIELD_H)
        self.field_text.position = (left + PAD + 8 * S, y - FIELD_H / 2, 0)
        self.unit.position = (left + W - PAD - 8 * S, y - FIELD_H / 2, 0)
        y -= FIELD_H
        self.hint.position = (left + PAD, y - HINT_H / 2, 0)
        self._show_knob()
        self._show_field()

    def _show_knob(self) -> None:
        if not self.has_slider:
            return
        known = self.value is not None
        self.knob.visible = known
        f = self.setting.fraction(self.value) if known else 0.0
        self.filled.width = f * self.track.width
        self.knob.position = (self.track.x + f * self.track.width, self.track.y + TRACK_H / 2)

    def _show_field(self) -> None:
        self.field_text.text = self.edit.text
        before = pyglet.text.Label(self.edit.text[:self.edit.caret], font_name=FONT, font_size=SIZE)
        self.caret.position = (self.field_text.x + before.content_width, self.field_text.y - self.caret.height / 2)
        self.caret.visible = self.caret_on

    # ---- state -----------------------------------------------------------------

    def set_value(self, value) -> None:
        """Show `value` (None: mixed) on the slider and in the field."""
        self.value = value
        self.edit = LineEdit(self.shown_text, 40)
        self._show_knob()
        self._show_field()

    @property
    def shown_text(self) -> str:
        """What the field says for the current value (the unit is a label beside it)."""
        if self.value is None:
            return ""
        text = self.setting.show(self.value)
        return text.removesuffix(f" {self.setting.unit}") if self.setting.unit else text

    @property
    def text(self) -> str:
        return self.edit.text

    def type_text(self, text: str) -> None:
        self.edit.insert(text)
        self._typed()

    def motion(self, motion: int) -> None:
        self.edit.motion(motion)
        self._typed()

    def _typed(self) -> None:
        self.caret_on, self._blink_t = True, 0.0
        self._show_field()

    def set_hint(self, text: str, danger: bool = False) -> None:
        self.hint.text = text
        self.hint.color = T.MENU_DANGER if danger else T.HELP_TEXT

    # ---- hit tests ---------------------------------------------------------------

    def contains(self, sx: float, sy: float) -> bool:
        return self.panel.contains(sx, sy)

    def on_slider(self, sx: float, sy: float) -> bool:
        """Is the point on the slider's row (anywhere along the track, knob included)?"""
        if not self.has_slider:
            return False
        cy = self.track.y + TRACK_H / 2
        return (abs(sy - cy) <= SLIDER_H / 2
                and self.track.x - KNOB_R <= sx <= self.track.x + self.track.width + KNOB_R)

    def value_at(self, sx: float):
        """The value under screen x on the slider (clamped to its ends)."""
        return self.setting.at((sx - self.track.x) / self.track.width)

    def tick(self, dt: float) -> None:
        """Caret blink."""
        self._blink_t += dt
        if self._blink_t >= 0.5:
            self._blink_t = 0.0
            self.caret_on = not self.caret_on
            self.caret.visible = self.caret_on

    def delete(self) -> None:
        self.panel.delete()
        for s in (self.title, self.track, self.filled, self.knob, self.field_text, self.unit, self.caret, self.hint):
            s.delete()
        self.field.delete()
