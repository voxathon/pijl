"""The console: a strip along the bottom of the window to type commands into (` opens
and closes it).

A line's first word names a command (COMMANDS, see `command`). A line that doesn't
start with one goes to the fallbacks (see `fallback`), newest first, until one takes
it: that's how a mod makes bare lines mean something (the netlist mod: wiring).

While it's open, typing goes to it and the mouse still works the board. Up / Down
walk back through what was typed before (kept between runs, in the data root);
Page Up / Page Down scroll the log; Esc or ` closes it. Ctrl+Z / Ctrl+Y still undo
and redo the board.

Commands see a Context: the editor, the box the line is for (the one box in the
selection, else the box under the cursor, else none: the whole board), and say()
to print to the log.

This module only lays out, draws and keeps the log and history; the editor routes
keys and text to it and runs the lines (Editor._console_*).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyglet
from pyglet import shapes

from . import theme as T
from .line_edit import LineEdit
from .text_field import FieldCursor, TextTarget

log = logging.getLogger("pijl.ui")

S = T.UI_SCALE
LINES = 8  # log lines shown
LINE_H = round(17 * S)
PAD = 6 * S
FONT, SIZE = "Consolas", 10 * S
MAX_LEN = 2000  # characters in a line
KEEP = 500  # lines of history kept
LOG_KEEP = 1000  # lines of log kept

# what say() prints as: text color
KINDS = {
    "out": T.PART_TEXT,
    "echo": T.HELP_TEXT,
    "error": T.MENU_DANGER,
}


# ---- commands -----------------------------------------------------------------------


@dataclass
class Context:
    """What a command gets. `box`: a BoxView or None (the whole board)."""

    editor: Any
    box: Any
    say: Callable[..., None]  # say(text, kind="out"): kind "out", "echo" or "error"

    def error(self, text: str) -> None:
        self.say(text, "error")


@dataclass
class Command:
    name: str
    run: Callable[[Context, str], None]  # (context, the rest of the line)
    help: str = ""


COMMANDS: dict[str, Command] = {}
# fallback(context, line) -> True if it took the line (newest registered asked first)
FALLBACKS: list[Callable[[Context, str], bool]] = []


def command(name: str, help: str = ""):
    """Register a console command: @command("goto", "goto LABEL: center on a part")
    on a function (context, rest of the line). A later one of the same name wins."""

    def register(fn):
        COMMANDS[name.casefold()] = Command(name, fn, help)
        return fn

    return register


def fallback(fn: Callable[[Context, str], bool]):
    """Register a handler for lines that aren't commands. Returns True if it took the
    line; the next (older) one is asked otherwise."""
    FALLBACKS.append(fn)
    return fn


def run_line(ctx: Context, line: str) -> None:
    """Run one typed line: a command, or the fallbacks. Errors are said, not raised."""
    line = line.strip()
    if not line:
        return
    word, _, rest = line.partition(" ")
    cmd = COMMANDS.get(word.casefold())
    try:
        if cmd is not None:
            cmd.run(ctx, rest.strip())
            return
        for fb in reversed(FALLBACKS):
            if fb(ctx, line):
                return
    except Exception as e:  # (a command's bug shouldn't take the editor down)
        log.exception("console: %s", line)
        ctx.error(f"{word}: {type(e).__name__}: {e}")
        return
    ctx.error(f"no command {word!r} (help lists them)")


@command("help", "help: list the commands")
def _help(ctx: Context, rest: str) -> None:
    for name in sorted(COMMANDS):
        c = COMMANDS[name]
        ctx.say(c.help or c.name)
    if FALLBACKS:
        ctx.say("anything else: handed to the mods that take bare lines", "echo")


@command("clear", "clear: empty the log")
def _clear(ctx: Context, rest: str) -> None:
    ctx.editor.console.clear()


# ---- the strip ----------------------------------------------------------------------


class Console:
    def __init__(self, batch: pyglet.graphics.Batch, history_file: Path | None) -> None:
        self.batch = batch
        self.open = False
        self.edit = LineEdit("", MAX_LEN)
        self.history_file = history_file
        self.history = _read_history(history_file)
        self.back = len(self.history)  # where Up / Down are in it (len: the new line)
        self.draft = ""  # the line being typed, while walking the history
        self.log: list[tuple[str, str]] = []  # (text, kind)
        self.scroll = 0  # log lines scrolled back
        self.caret_on, self._blink_t = True, 0.0
        self.scope = "board"
        self.left = self.right = self.bottom = 0.0

        bg, fg, sel_g, text_g = (pyglet.graphics.Group(order=o) for o in (11, 12, 12.5, 13))
        line = max(1, round(S))
        self.h = PAD * 2 + LINE_H * (LINES + 1)
        self.panel = shapes.Rectangle(0, 0, 1, self.h, color=(*T.PICKER_BG, 235), batch=batch, group=bg)
        self.border = shapes.Rectangle(0, 0, 1, line, color=T.PICKER_BORDER, batch=batch, group=fg)
        self.field = shapes.Rectangle(0, 0, 1, LINE_H, color=T.PICKER_HEADER, batch=batch, group=fg)

        def label(color=T.PART_TEXT):
            return pyglet.text.Label(
                "", font_name=FONT, font_size=SIZE, color=color, anchor_y="center",
                batch=batch, group=text_g,
            )

        self.lines = [label() for _ in range(LINES)]
        self.prompt = label(T.HELP_TEXT)
        self.input = label()
        self.cursor = FieldCursor(batch, text_g, sel_g, FONT, SIZE, LINE_H * 0.8)
        self._set_visible(False)

    # ---- placing -------------------------------------------------------------------

    def layout(self, left: float, right: float, bottom: float) -> None:
        """Across from `left` to `right` (screen px), its bottom edge at `bottom`."""
        self.left, self.right, self.bottom = left, right, bottom
        w = max(1.0, right - left)
        self.panel.position, self.panel.width = (left, bottom), w
        self.border.position, self.border.width = (left, bottom + self.h - self.border.height), w
        self.field.position, self.field.width = (left, bottom + PAD), w
        y = bottom + PAD + LINE_H / 2
        self.prompt.position = (left + PAD, y, 0)
        for i, lbl in enumerate(reversed(self.lines)):
            lbl.position = (left + PAD, y + LINE_H * (i + 1), 0)
        self._show()

    def contains(self, x: float, y: float) -> bool:
        return self.open and self.left <= x <= self.right and self.bottom <= y <= self.bottom + self.h

    def _field_contains(self, x: float, y: float) -> bool:
        f = self.field
        return self.open and f.x <= x <= f.x + f.width and f.y <= y <= f.y + f.height

    # ---- showing -------------------------------------------------------------------

    def toggle(self) -> None:
        self.set_open(not self.open)

    def set_open(self, on: bool) -> None:
        self.open = on
        self._set_visible(on)
        if on:
            self.caret_on, self._blink_t = True, 0.0
            self._show()

    def _set_visible(self, on: bool) -> None:
        for s in (self.panel, self.border, self.field, self.prompt, self.input, *self.lines):
            s.visible = on
        self.cursor.caret.visible = on and self.caret_on
        if not on:
            self.cursor.sel.visible = False

    def set_scope(self, name: str) -> None:
        if name != self.scope:
            self.scope = name
            self._show()

    def _show(self) -> None:
        if not self.open:
            return
        end = len(self.log) - self.scroll
        shown = self.log[max(0, end - LINES) : end]
        shown = [("", "out")] * (LINES - len(shown)) + shown
        for lbl, (text, kind) in zip(self.lines, shown):
            lbl.text, lbl.color = text, KINDS.get(kind, T.PART_TEXT)
        self.prompt.text = f"[{self.scope}] >"
        x = self.prompt.x + self.cursor.width(self.prompt.text) + PAD
        y = self.prompt.y
        self.input.position = (x, y, 0)
        self.input.text = self.cursor.place(
            self.edit, x, y, self.caret_on, max_w=max(10.0, self.right - x - PAD)
        )
        self.cursor.caret.visible = self.caret_on

    def tick(self, dt: float) -> None:
        if not self.open:
            return
        self._blink_t += dt
        if self._blink_t >= 0.5:
            self._blink_t = 0.0
            self.caret_on = not self.caret_on
            self.cursor.caret.visible = self.caret_on

    # ---- the log -------------------------------------------------------------------

    def say(self, text: str, kind: str = "out") -> None:
        for line in str(text).splitlines() or [""]:
            self.log.append((line, kind))
        del self.log[:-LOG_KEEP]
        self.scroll = 0
        self._show()

    def clear(self) -> None:
        self.log.clear()
        self.scroll = 0
        self._show()

    def page(self, delta: int) -> None:
        """Scroll the log a page back (+1) or forward (-1)."""
        most = max(0, len(self.log) - LINES)
        self.scroll = max(0, min(most, self.scroll + delta * (LINES - 1)))
        self._show()

    # ---- typing --------------------------------------------------------------------

    def type_text(self, text: str) -> None:
        self.edit.insert(text)
        self._typed()

    def motion(self, motion: int, select: bool = False) -> None:
        self.edit.motion(motion, select)
        self._typed()

    def _typed(self) -> None:
        self.caret_on, self._blink_t = True, 0.0
        self._show()

    def target(self) -> TextTarget:
        return TextTarget(
            lambda: self.edit,
            self._field_contains,
            lambda x, y: self.cursor.index_at(self.edit, x),
            self._typed,
        )

    def walk(self, delta: int) -> None:
        """Up (-1) / Down (+1) through the history."""
        if self.back == len(self.history):
            self.draft = self.edit.text
        self.back = max(0, min(len(self.history), self.back + delta))
        text = self.history[self.back] if self.back < len(self.history) else self.draft
        self.edit = LineEdit(text, MAX_LEN)
        self._typed()

    def take(self) -> str:
        """The typed line (Enter): into the history and the log, the field emptied."""
        line = self.edit.text
        self.edit = LineEdit("", MAX_LEN)
        if line.strip():
            if not self.history or self.history[-1] != line:
                self.history.append(line)
                del self.history[:-KEEP]
                _write_history(self.history_file, self.history)
            self.say(f"> {line}", "echo")
        self.back, self.draft = len(self.history), ""
        self._typed()
        return line

    def delete(self) -> None:
        for s in (self.panel, self.border, self.field, self.prompt, self.input, *self.lines):
            s.delete()
        self.cursor.delete()


def _read_history(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()][-KEEP:]
    except OSError:
        return []


def _write_history(path: Path | None, lines: list[str]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("couldn't save the console history: %s", e)
