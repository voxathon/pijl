"""A small modal box near the top of the window: a title, an optional text field,
an optional list that the field filters, and a hint (wrapped to the box's width).

Used for naming a macro, the Ctrl+O quick-open, "unsaved changes?" and
"really delete?". Screen
space (HUD batch), drawn over everything with the board dimmed behind it. This
module only lays out, draws and hit-tests; what Enter and other keys *mean* is
up to the editor (see Editor._open_prompt).
"""

from __future__ import annotations

import textwrap

import pyglet
from pyglet import shapes

from . import theme as T
from .line_edit import LineEdit
from .text_field import FieldCursor, TextTarget
from .views import Box

S = T.UI_SCALE
W, PAD = 440 * S, 12 * S
TITLE_H, FIELD_H, ROW_H, HINT_H = 26 * S, 30 * S, 24 * S, 24 * S
MESSAGE_H = 18 * S  # per line
MAX_ROWS = 10
TOP_MARGIN = 80 * S  # from the top of the window to the box
FONT, SIZE, SMALL = "Consolas", 11 * S, 10 * S


class Prompt:
    def __init__(
        self,
        batch: pyglet.graphics.Batch,
        win_w: int,
        win_h: int,
        title: str,
        *,
        text: str | None = None,
        max_len: int = 40,
        items: list[str] | None = None,
        hint: str = "",
        empty: str = "nothing here yet",
        message: str = "",
        danger: bool = False,
    ) -> None:
        """`message`: a few lines under the title (wrapped); `danger` draws it red.
        A prefilled `text` starts selected: typing replaces it, an arrow keeps it."""
        self.batch = batch
        self.edit = LineEdit(text, max_len) if text is not None else None
        if self.edit is not None:
            self.edit.select_all()
        self.items = items
        self.shown: list[str] = []  # items matching the field, best first
        self.selected = 0  # index into shown
        self.offset = 0  # first shown item on screen (the list scrolls)
        self.empty = empty  # what the list says when nothing matches
        self.caret_on, self._blink_t = True, 0.0
        self._filtered: str | None = None  # the text the list was last filtered by

        shade_g, bg, fg, text_g = (
            pyglet.graphics.Group(order=o) for o in (20, 21, 22, 23)
        )
        n_rows = MAX_ROWS if items is not None else 0
        # Wrapped here, one label per line: pyglet's multiline layout drops spaces.
        # (The font is monospaced, so a line's width is its length.)
        char_w = (
            pyglet.text.Label("M" * 10, font_name=FONT, font_size=SMALL).content_width
            / 10
        )
        self.message = [
            pyglet.text.Label(
                line,
                font_name=FONT,
                font_size=SMALL,
                color=T.MENU_DANGER if danger else T.PART_TEXT,
                anchor_y="center",
                batch=batch,
                group=text_g,
            )
            for line in textwrap.wrap(message, int((W - 2 * PAD) / char_w))
        ]
        self.message_h = len(self.message) * MESSAGE_H + PAD / 2 if self.message else 0
        self._wrap = int((W - 2 * PAD) / char_w)  # (hint lines, in characters)
        n_hint = max(1, len(textwrap.wrap(hint, self._wrap)))
        self.h = (
            PAD
            + TITLE_H
            + self.message_h
            + (FIELD_H + PAD / 2 if self.edit else 0)
            + n_rows * ROW_H
            + HINT_H
            + (n_hint - 1) * MESSAGE_H
            + PAD / 2
        )
        self.shade = shapes.Rectangle(
            0, 0, win_w, win_h, color=(0, 0, 0, 110), batch=batch, group=shade_g
        )
        self.panel = Box(W, self.h, max(1, round(S)), *T.MENU_PANEL, batch, bg)

        def label(text="", color=T.PART_TEXT, size=SIZE):
            return pyglet.text.Label(
                text,
                font_name=FONT,
                font_size=size,
                color=color,
                anchor_y="center",
                batch=batch,
                group=text_g,
            )

        self.title = label(title)
        self.field = self.field_text = self.cursor = None
        if self.edit is not None:
            self.field = Box(
                W - 2 * PAD, FIELD_H, max(1, round(S)), T.PICKER_BG, T.SELECT, batch, fg
            )
            self.field_text = label()
            sel_g = pyglet.graphics.Group(order=22.5)  # over the field, under the text
            self.cursor = FieldCursor(batch, text_g, sel_g, FONT, SIZE, 16 * S)
        self.rows = [
            (
                shapes.Rectangle(
                    0,
                    0,
                    W - 2 * PAD,
                    ROW_H,
                    color=T.MENU_PANEL[0],
                    batch=batch,
                    group=fg,
                ),
                label(),
            )
            for _ in range(n_rows)
        ]
        self.hints = [label(color=T.HELP_TEXT, size=SMALL) for _ in range(n_hint)]
        self.set_hint(hint)
        self.layout(win_w, win_h)
        self._refilter()

    # ---- layout ----------------------------------------------------------------

    def layout(self, win_w: int, win_h: int) -> None:
        self.shade.width, self.shade.height = win_w, win_h
        left = max(0.0, (win_w - W) / 2)
        top = win_h - TOP_MARGIN
        self.panel.position = (left, top - self.h)
        y = top - PAD
        self.title.position = (left + PAD, y - TITLE_H / 2, 0)
        y -= TITLE_H
        for line in self.message:
            line.position = (left + PAD, y - MESSAGE_H / 2, 0)
            y -= MESSAGE_H
        if self.message:
            y -= PAD / 2
        if self.field is not None:
            self.field.position = (left + PAD, y - FIELD_H)
            self.field_text.position = (left + PAD + 8 * S, y - FIELD_H / 2, 0)
            y -= FIELD_H + PAD / 2
        for row, text in self.rows:
            y -= ROW_H
            row.position = (left + PAD, y)
            text.position = (left + PAD + 8 * S, y + ROW_H / 2, 0)
        for i, line in enumerate(self.hints):
            line.position = (left + PAD, y - HINT_H / 2 - i * MESSAGE_H, 0)
        self._show()

    def _show(self) -> None:
        """Bring the field, caret and list rows up to date."""
        if self.edit is not None:
            self.field_text.text = self.cursor.place(
                self.edit, self.field_text.x, self.field_text.y, self.caret_on
            )
        visible = self.shown[self.offset : self.offset + len(self.rows)]
        for i, (row, text) in enumerate(self.rows):
            if i < len(visible):
                text.text, text.color = visible[i], T.PART_TEXT
                row.color = (
                    T.MENU_HOVER
                    if self.offset + i == self.selected
                    else T.MENU_PANEL[0]
                )
            else:
                text.text = self.empty if i == 0 and not self.shown else ""
                text.color = T.PICKER_DIM_TEXT
                row.color = T.MENU_PANEL[0]

    # ---- input -----------------------------------------------------------------

    @property
    def text(self) -> str:
        return self.edit.text if self.edit is not None else ""

    @property
    def choice(self) -> str | None:
        """The highlighted list item, or None (no list, or nothing matches)."""
        return self.shown[self.selected] if self.shown else None

    def type_text(self, text: str) -> None:
        if self.edit is not None:
            self.edit.insert(text)
            self._typed()

    def motion(self, motion: int, select: bool = False) -> None:
        if self.edit is not None:
            self.edit.motion(motion, select)
            self._typed()

    def target(self) -> TextTarget | None:
        """The field, for the mouse and Ctrl+A / C / X / V (see text_field.py)."""
        if self.edit is None:
            return None
        return TextTarget(
            lambda: self.edit,
            self.field.contains,
            lambda x, y: self.cursor.index_at(self.edit, x),
            self._typed,
        )

    def _typed(self) -> None:
        self.caret_on, self._blink_t = True, 0.0
        if self.items is None or self.text == self._filtered:
            self._show()  # (just the caret: the list keeps its highlight)
        else:
            self._refilter()

    def _refilter(self) -> None:
        if self.items is None:
            self._show()
            return
        self._filtered = self.text
        q = self.text.strip().casefold()
        hits = [i for i in self.items if q in i.casefold()]
        self.shown = sorted(
            hits, key=lambda i: not i.casefold().startswith(q)
        )  # prefix matches first
        self.selected, self.offset = 0, 0
        self._show()

    def move(self, delta: int) -> None:
        """Arrow keys / scroll: move the highlight, scrolling the list to keep it visible."""
        if not self.shown:
            return
        self.selected = max(0, min(len(self.shown) - 1, self.selected + delta))
        if self.selected < self.offset:
            self.offset = self.selected
        elif self.selected >= self.offset + len(self.rows):
            self.offset = self.selected - len(self.rows) + 1
        self._show()

    def item_at(self, sx: float, sy: float) -> int | None:
        """Index into `shown` of the list row under the cursor."""
        for i, (row, _) in enumerate(self.rows):
            if (
                self.offset + i < len(self.shown)
                and row.x <= sx <= row.x + row.width
                and row.y <= sy <= row.y + row.height
            ):
                return self.offset + i
        return None

    def hover(self, sx: float, sy: float) -> None:
        i = self.item_at(sx, sy)
        if i is not None and i != self.selected:
            self.selected = i
            self._show()

    def contains(self, sx: float, sy: float) -> bool:
        return self.panel.contains(sx, sy)

    def set_hint(self, text: str, danger: bool = False) -> None:
        """The hint, wrapped over the lines the box has for it (made for the first one:
        a longer one later gets its overflow on the last line)."""
        lines = textwrap.wrap(text, self._wrap) or [""]
        n = len(self.hints)
        lines = lines[: n - 1] + [" ".join(lines[n - 1 :])] + [""] * (n - len(lines))
        for label, line in zip(self.hints, lines):
            label.text = line
            label.color = T.MENU_DANGER if danger else T.HELP_TEXT

    def tick(self, dt: float) -> None:
        """Caret blink."""
        if self.edit is None:
            return
        self._blink_t += dt
        if self._blink_t >= 0.5:
            self._blink_t = 0.0
            self.caret_on = not self.caret_on
            self.cursor.caret.visible = self.caret_on

    def delete(self) -> None:
        self.shade.delete()
        self.panel.delete()
        self.title.delete()
        for line in self.hints:
            line.delete()
        for line in self.message:
            line.delete()
        if self.field is not None:
            self.field.delete()
            self.field_text.delete()
            self.cursor.delete()
        for row, text in self.rows:
            row.delete()
            text.delete()
