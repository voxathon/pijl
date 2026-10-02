"""What every in-place text field shares on top of LineEdit: drawing the caret and
the selection beside a pyglet Label (FieldCursor), and the selecting that's the
same everywhere: Ctrl+A / C / X / V (shortcut) and the mouse (TextMouse) through a
TextTarget, which tells them where a field is and what's typed in it.

Used by the picker (search, collection names), the prompt, the number popover,
part labels (world space: they draw their own caret, the rest applies) and the
launcher.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import pyglet
from pyglet import shapes
from pyglet.window import key

from . import theme as T
from .line_edit import LineEdit

DOUBLE_CLICK = 0.4  # s: a second press in a field within this selects everything
SELECTION_ALPHA = 0.4  # the selection highlight's opacity


@dataclass
class TextTarget:
    """A field the mouse and the shortcuts can work on."""

    edit: Callable[
        [], LineEdit
    ]  # what's typed in it (asked each time: fields swap theirs)
    contains: Callable[[float, float], bool]  # screen (or world) point -> on the field?
    index_at: Callable[[float, float], int]  # point -> nearest caret position
    changed: Callable[[], None]  # show the text / caret / selection again (and react)


def shortcut(
    target: TextTarget, symbol: int, modifiers: int, window: pyglet.window.Window
) -> bool:
    """Ctrl+A select all, Ctrl+C copy, Ctrl+X cut, Ctrl+V paste. Whether the key was
    one of them (then it's dealt with)."""
    if not modifiers & key.MOD_CTRL or symbol not in (key.A, key.C, key.X, key.V):
        return False
    edit = target.edit()
    if symbol == key.A:
        edit.select_all()
    elif symbol in (key.C, key.X) and edit.selected_text:
        window.set_clipboard_text(edit.selected_text)
        if symbol == key.X:
            edit.insert("")
    elif symbol == key.V:
        edit.insert(window.get_clipboard_text() or "")
    target.changed()
    return True


class TextMouse:
    """Click: caret there. Shift+click: select to there. Drag: select. A quick
    second click: select everything."""

    def __init__(self) -> None:
        self.dragging = False
        self._last = 0.0  # when the last press was (for double-clicks)

    def press(self, target: TextTarget, x: float, y: float, extend: bool) -> None:
        edit, now = target.edit(), time.monotonic()
        if not extend and now - self._last < DOUBLE_CLICK:
            edit.select_all()
            self._last = 0.0  # (a third click starts over)
        else:
            self._last = now
            edit.set_caret(target.index_at(x, y), extend)
            self.dragging = True
        target.changed()

    def drag(self, target: TextTarget, x: float, y: float) -> None:
        if self.dragging:
            target.edit().set_caret(target.index_at(x, y), extend=True)
            target.changed()

    def release(self) -> None:
        self.dragging = False

    def forget(self) -> None:
        """A new field: the next press is a first click."""
        self.dragging, self._last = False, 0.0


class FieldCursor:
    """The caret and the selection highlight for a LineEdit drawn as a pyglet Label.
    `sel_group` should draw over the field's fill and under the text."""

    def __init__(
        self,
        batch: pyglet.graphics.Batch,
        caret_group: pyglet.graphics.Group,
        sel_group: pyglet.graphics.Group,
        font_name: str,
        font_size: float,
        height: float,
        caret_w: float = 1.5 * T.UI_SCALE,
    ) -> None:
        self.measure = pyglet.text.Label("", font_name=font_name, font_size=font_size)
        self.caret = shapes.Rectangle(
            0, 0, caret_w, height, color=T.CARET, batch=batch, group=caret_group
        )
        self.sel = shapes.Rectangle(
            0, 0, 0, height, color=T.SELECT, batch=batch, group=sel_group
        )
        self.sel.opacity = round(255 * SELECTION_ALPHA)
        self.sel.visible = False
        self.start = (
            0  # first character shown (a long text scrolls to keep the caret in)
        )
        self.x0 = 0.0  # where the shown text starts on screen

    def width(self, text: str) -> float:
        """How wide `text` is drawn. (An emptied pyglet label still reports its old
        text's width, so "" is answered without asking it.)"""
        if not text:
            return 0.0
        self.measure.text = text
        return self.measure.content_width

    def place(
        self,
        edit: LineEdit,
        x: float,
        cy: float,
        caret_on: bool = True,
        max_w: float | None = None,
        selection: bool = True,
    ) -> str:
        """Put the caret and the highlight for text starting at screen x `x`, centered
        on `cy`. With `max_w`, text too long for it scrolls so the caret stays in.
        Returns the text to show (from `start` on)."""
        text, caret = edit.text, edit.caret
        start = min(self.start, caret)
        if max_w is not None:
            while start > 0 and self.width(text[start - 1 :]) <= max_w:
                start -= 1  # room again (text deleted): show more of the start
            while start < caret and self.width(text[start:caret]) > max_w:
                start += 1
        else:
            start = 0
        self.start, self.x0 = start, x
        h = self.caret.height
        self.caret.position = (x + self.width(text[start:caret]), cy - h / 2)
        self.caret.visible = caret_on
        sel = edit.selection if selection else None
        self.sel.visible = sel is not None
        if sel is not None:
            x0 = x + self.width(text[start : max(sel[0], start)])
            x1 = x + self.width(text[start : sel[1]])
            if max_w is not None:
                x1 = min(x1, x + max_w)
            self.sel.position = (x0, cy - h / 2)
            self.sel.width = max(0.0, x1 - x0)
        return text[start:]

    def index_at(self, edit: LineEdit, sx: float) -> int:
        """The caret position nearest screen x `sx` (as last placed)."""
        text, start = edit.text, self.start
        best, best_d = start, abs(sx - self.x0)
        for i in range(start + 1, len(text) + 1):
            d = abs(sx - self.x0 - self.width(text[start:i]))
            if d < best_d:
                best, best_d = i, d
        return best

    def set_opacity(self, a: int) -> None:
        self.caret.opacity = a
        self.sel.opacity = round(a * SELECTION_ALPHA)

    def set_groups(
        self, caret_group: pyglet.graphics.Group, sel_group: pyglet.graphics.Group
    ) -> None:
        self.caret.group, self.sel.group = caret_group, sel_group

    def delete(self) -> None:
        self.caret.delete()
        self.sel.delete()
