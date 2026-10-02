"""A one-line text buffer with a caret: the typing logic shared by every in-place text field.

Feed it pyglet's on_text / on_text_motion events; drawing is up to the caller.
Optionally a selection too (between `anchor` and `caret`): feed on_text_motion_select
to motion(..., select=True). A field that doesn't draw a selection just never makes
one, and then it behaves exactly as before.
"""

from __future__ import annotations

from pyglet.window import key


class LineEdit:
    def __init__(self, text: str, max_len: int) -> None:
        self.text = text
        self.caret = len(text)  # insertion index into text
        self.max_len = max_len
        self.anchor: int | None = None  # the selection's other end (None: no selection)

    # ---- selection ----------------------------------------------------------------

    @property
    def selection(self) -> tuple[int, int] | None:
        """(start, end) of the selected text, or None if nothing is selected."""
        if self.anchor is None or self.anchor == self.caret:
            return None
        return min(self.anchor, self.caret), max(self.anchor, self.caret)

    @property
    def selected_text(self) -> str:
        sel = self.selection
        return self.text[sel[0] : sel[1]] if sel else ""

    def select_all(self) -> None:
        self.anchor, self.caret = 0, len(self.text)

    def set_caret(self, index: int, extend: bool = False) -> None:
        """Put the caret at `index` (a click); `extend`: select from where it was (or
        from the selection's anchor) to there (Shift+click, a drag)."""
        index = max(0, min(index, len(self.text)))
        if extend:
            if self.anchor is None:
                self.anchor = self.caret
        else:
            self.anchor = None
        self.caret = index

    def _delete_selection(self) -> bool:
        sel = self.selection
        self.anchor = None
        if sel is None:
            return False
        self.text = self.text[: sel[0]] + self.text[sel[1] :]
        self.caret = sel[0]
        return True

    # ---- typing -------------------------------------------------------------------

    def insert(self, text: str) -> None:
        """Type `text` at the caret, replacing the selection if there is one."""
        text = "".join(
            c for c in text if c.isprintable()
        )  # drops Enter's carriage return
        self._delete_selection()
        text = text[: max(self.max_len - len(self.text), 0)]
        t, i = self.text, self.caret
        self.text, self.caret = t[:i] + text + t[i:], i + len(text)

    def motion(self, motion: int, select: bool = False) -> None:
        """A pyglet text motion. `select` (Shift held): moving the caret extends the
        selection instead of dropping it."""
        t, i = self.text, self.caret
        if motion in (key.MOTION_BACKSPACE, key.MOTION_DELETE):
            if self._delete_selection():
                return
            if motion == key.MOTION_BACKSPACE and i > 0:
                self.text, self.caret = t[: i - 1] + t[i:], i - 1
            elif motion == key.MOTION_DELETE:
                self.text = t[:i] + t[i + 1 :]
            return
        sel = self.selection
        if select:
            if self.anchor is None:
                self.anchor = i
        else:
            self.anchor = None
            if sel and motion in (key.MOTION_LEFT, key.MOTION_RIGHT):
                # with a selection, an arrow just goes to that end of it
                self.caret = sel[0] if motion == key.MOTION_LEFT else sel[1]
                return
        if motion == key.MOTION_LEFT:
            self.caret = max(0, i - 1)
        elif motion == key.MOTION_RIGHT:
            self.caret = min(len(t), i + 1)
        elif motion == key.MOTION_PREVIOUS_WORD:
            self.caret = _word_start(t, i)
        elif motion == key.MOTION_NEXT_WORD:
            self.caret = _word_end(t, i)
        elif motion in (key.MOTION_BEGINNING_OF_LINE, key.MOTION_BEGINNING_OF_FILE):
            self.caret = 0
        elif motion in (key.MOTION_END_OF_LINE, key.MOTION_END_OF_FILE):
            self.caret = len(t)


def _word_start(t: str, i: int) -> int:
    """Where Ctrl+Left goes: back over spaces, then over the word before them."""
    while i > 0 and t[i - 1].isspace():
        i -= 1
    while i > 0 and not t[i - 1].isspace():
        i -= 1
    return i


def _word_end(t: str, i: int) -> int:
    """Where Ctrl+Right goes: over the word, then over the spaces after it."""
    while i < len(t) and not t[i].isspace():
        i += 1
    while i < len(t) and t[i].isspace():
        i += 1
    return i
