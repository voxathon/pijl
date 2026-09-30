"""A one-line text buffer with a caret: the typing logic shared by every in-place text field.

Feed it pyglet's on_text / on_text_motion events; drawing is up to the caller.
"""

from __future__ import annotations

from pyglet.window import key


class LineEdit:
    def __init__(self, text: str, max_len: int) -> None:
        self.text = text
        self.caret = len(text)  # insertion index into text
        self.max_len = max_len

    def insert(self, text: str) -> None:
        text = "".join(
            c for c in text if c.isprintable()
        )  # drops Enter's carriage return
        text = text[: max(self.max_len - len(self.text), 0)]
        t, i = self.text, self.caret
        self.text, self.caret = t[:i] + text + t[i:], i + len(text)

    def motion(self, motion: int) -> None:
        t, i = self.text, self.caret
        if motion == key.MOTION_BACKSPACE and i > 0:
            self.text, self.caret = t[: i - 1] + t[i:], i - 1
        elif motion == key.MOTION_DELETE:
            self.text = t[:i] + t[i + 1 :]
        elif motion == key.MOTION_LEFT:
            self.caret = max(0, i - 1)
        elif motion == key.MOTION_RIGHT:
            self.caret = min(len(t), i + 1)
        elif motion in (key.MOTION_BEGINNING_OF_LINE, key.MOTION_BEGINNING_OF_FILE):
            self.caret = 0
        elif motion in (key.MOTION_END_OF_LINE, key.MOTION_END_OF_FILE):
            self.caret = len(t)
