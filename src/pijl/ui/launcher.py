"""The launcher's window: the start screen `pijl` opens with (see pijl/launcher.py).

    pijl 0.2.3
    PROJECT
    default
    [            START             ]
    [ LOAD PROJECT ] [ NEW PROJECT ]
    [HEADLESS] [  MODS  ] [SETTINGS]
                              [QUIT]

LOAD PROJECT lists the projects (click one to pick it, "rename" to rename it),
NEW PROJECT makes one, SETTINGS edits the preferences (click a value to change
it; numbers are typed, or nudged with the wheel), and HEADLESS runs one of the
project's macros without the editor: in this terminal if pijl has one, else in a
console window of its own. MODS turns mods on and off, orders them, and has the
safe start switch (start without mods; shown on the main page when it's on).

This module imports only theme (colors) and line_edit from pijl.ui: the editor's
modules read theme.UI_SCALE when they're imported, which happens after this
window closes. Its own size follows the UI scale preference, live.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import pyglet
from pyglet import shapes
from pyglet.window import key, mouse

from .. import prefs
from ..launcher import Launcher, ModRow, has_console, open_console, version
from ..parts.settings import Choice, Number, Toggle
from . import theme as T
from .line_edit import LineEdit
from .text_field import FieldCursor, TextMouse, TextTarget, shortcut

W, H = 520, 440  # the window, at UI scale 1
PAD, GAP = 24, 10
BTN_H, BIG_H, SMALL_BTN_H, ROW_H = 40, 52, 32, 30
STATUS_LINES, STATUS_LINE_H = 3, 15  # the status line wraps onto this many, at most
FONT = "Consolas"

BUTTON = (T.PICKER_HEADER, T.MENU_HOVER)  # (fill, hovered fill)
PRIMARY = ((48, 104, 186), (66, 128, 215))
ROW = (T.PICKER_BG, T.PICKER_HOVER)
ACCENT = (*T.SELECT, 255)


class NoWindow(Exception):
    """pyglet couldn't open a window (no display, no OpenGL)."""


@dataclass
class _Hit:
    x: float
    y: float
    w: float
    h: float
    click: Callable[[], None] | None
    right: Callable[[], None] | None = None
    wheel: Callable[[int], None] | None = None
    hint: str = ""
    rect: shapes.ShapeBase | None = None
    colors: tuple = ()

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h


@dataclass
class _Edit:
    line: LineEdit
    label: pyglet.text.Label
    cursor: FieldCursor
    enter: Callable[[str], None]
    box: tuple[float, float, float, float]  # x, y, w, h

    def contains(self, x: float, y: float) -> bool:
        bx, by, bw, bh = self.box
        return bx <= x < bx + bw and by <= y < by + bh


def run_window(launcher: Launcher) -> list[str] | None:
    """Show the launcher until something's started (its `pijl` arguments) or it's
    closed (None). NoWindow if there's no window to be had."""
    try:
        window = LauncherWindow(launcher)
    except Exception as e:  # (pyglet raises all sorts when there's no display)
        raise NoWindow(str(e) or type(e).__name__) from e
    pyglet.app.run()
    # The event loop stops windows from queueing events until it runs (and doesn't
    # undo that): put it back, so the editor's window starts as it would first thing.
    pyglet.window.Window._enable_event_queue = True
    return window.result


class LauncherWindow(pyglet.window.Window):
    def __init__(self, launcher: Launcher) -> None:
        self.launcher = launcher
        self.s = launcher.prefs["ui.scale"]
        self._ready = False  # (pyglet may send on_resize before __init__ is done)
        super().__init__(round(W * self.s), round(H * self.s), caption="pijl")
        self.result: list[str] | None = None
        self.page = "main"
        self.batch = pyglet.graphics.Batch()
        self.hits: list[_Hit] = []
        self.hovered: _Hit | None = None
        self.edit: _Edit | None = None
        self.text_mouse = TextMouse()  # clicking / dragging in the field
        self.editing: str | None = (
            None  # which row's being edited (a project or a pref)
        )
        self.scroll = 0
        self.message, self.danger = "", False
        self.status: list[pyglet.text.Label] = []  # its lines, top to bottom
        self._mouse = (-1.0, -1.0)
        self._ready = True
        if launcher.problems:
            self.say("; ".join(launcher.problems), danger=True)
        self.show("main")

    # ---- drawing helpers ---------------------------------------------------------

    def _label(
        self,
        text,
        x,
        y,
        size=11,
        color=T.PART_TEXT,
        anchor_x="left",
        anchor_y="baseline",
        bold=False,
    ):
        label = pyglet.text.Label(
            text,
            font_name=FONT,
            font_size=size * self.s,
            weight="bold" if bold else "normal",
            color=color,
            x=x,
            y=y,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            batch=self.batch,
            group=self._text,
        )
        self.drawn.append(label)
        return label

    def _box(
        self,
        x,
        y,
        w,
        h,
        colors,
        click=None,
        right=None,
        wheel=None,
        hint="",
        border=T.PICKER_BORDER,
    ):
        rect = shapes.BorderedRectangle(
            x,
            y,
            w,
            h,
            max(1, round(self.s)),
            color=colors[0],
            border_color=border,
            batch=self.batch,
            group=self._back,
        )
        self.hits.append(_Hit(x, y, w, h, click, right, wheel, hint, rect, colors))
        return rect

    def _button(
        self,
        text,
        x,
        y,
        w,
        h,
        click,
        colors=BUTTON,
        hint="",
        size=11,
        color=T.PART_TEXT,
    ):
        self._box(x, y, w, h, colors, click, hint=hint)
        self._label(
            text,
            x + w / 2,
            y + h / 2,
            size,
            color,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

    def _field(self, x, y, w, h, text: str, max_len: int, enter: Callable[[str], None]):
        field = shapes.BorderedRectangle(
            x,
            y,
            w,
            h,
            max(1, round(self.s)),
            color=T.PICKER_BG,
            border_color=T.SELECT,
            batch=self.batch,
            group=self._back,
        )
        self.drawn.append(field)
        label = self._label("", x + 8 * self.s, y + h / 2, anchor_y="center")
        cursor = FieldCursor(
            self.batch,
            self._text,
            self._sel,
            FONT,
            11 * self.s,
            h * 0.55,
            caret_w=1.5 * self.s,
        )
        line = LineEdit(text, max_len)
        line.select_all()  # typing replaces what's there
        self.edit = _Edit(line, label, cursor, enter, (x, y, w, h))
        self.text_mouse.forget()
        self._show_edit()

    def _show_edit(self) -> None:
        e = self.edit
        e.label.text = e.cursor.place(e.line, e.label.x, e.label.y)

    def _edit_target(self) -> TextTarget:
        """The field, for the mouse and Ctrl+A / C / X / V (see text_field.py)."""
        e = self.edit
        return TextTarget(
            lambda: e.line,
            e.contains,
            lambda x, y: e.cursor.index_at(e.line, x),
            self._show_edit,
        )

    def say(self, text: str, danger: bool = False) -> None:
        """A message on the status line (problems in red), until the next one."""
        self.message, self.danger = text, danger
        self._show_status()

    def _show_status(self) -> None:
        if not self.status:
            return
        if self.message:
            text, color = self.message, T.MENU_DANGER if self.danger else T.HELP_TEXT
        else:
            text, color = (self.hovered.hint if self.hovered else ""), T.PICKER_DIM_TEXT
        # Wrapped by hand, one label per line: pyglet's multiline layout drops spaces.
        # (The font is monospaced, so a line's width is its length.) Lines sit at the
        # bottom, just above the buttons.
        lines = textwrap.wrap(
            text, max(1, int((self.width - 2 * PAD * self.s) / self._char_w))
        )
        if len(lines) > STATUS_LINES:
            lines = lines[: STATUS_LINES - 1] + [lines[STATUS_LINES - 1][:-1] + "…"]
        lines = [""] * (STATUS_LINES - len(lines)) + lines
        for label, line in zip(self.status, lines):
            label.text, label.color = line, color

    # ---- pages -------------------------------------------------------------------

    def show(self, page: str | None = None) -> None:
        """(Re)build a page from scratch: they're small."""
        if page is not None and page != self.page:
            self.page, self.scroll, self.editing, self.message = page, 0, None, ""
        self.batch = pyglet.graphics.Batch()
        self._back = pyglet.graphics.Group(0)
        self._sel = pyglet.graphics.Group(0.5)  # a field's selected text: under the text
        self._text = pyglet.graphics.Group(1)
        self.hits, self.hovered, self.edit = [], None, None
        # what's drawn on the page (pyglet takes a label or shape out of the batch
        # when it's garbage collected)
        self.drawn: list = []
        s = self.s
        self.top = self.height - PAD * s
        self._char_w = (
            pyglet.text.Label("M" * 10, font_name=FONT, font_size=9.5 * s).content_width
            / 10
        )
        self.status = [
            self._label(
                "", PAD * s, (PAD + SMALL_BTN_H + 12 + i * STATUS_LINE_H) * s, size=9.5
            )
            for i in reversed(range(STATUS_LINES))
        ]
        getattr(self, "_page_" + self.page)()
        if self.editing is None or self.edit is None:
            self.editing = None
        self._show_status()
        self._hover(*self._mouse)

    def _title(self, text: str) -> None:
        s = self.s
        self._label(text, PAD * s, self.top, 16, anchor_y="top", bold=True)
        self.top -= 44 * s

    def _bottom(self, right: tuple[str, Callable[[], None]] | None = None) -> None:
        s = self.s
        self._button(
            "BACK",
            PAD * s,
            PAD * s,
            120 * s,
            SMALL_BTN_H * s,
            lambda: self.show("main"),
            hint="Esc",
        )
        if right:
            w = 200 * s
            self._button(
                right[0],
                self.width - PAD * s - w,
                PAD * s,
                w,
                SMALL_BTN_H * s,
                right[1],
            )

    def _list(
        self, rows: list, draw: Callable[[object, float, float, float], None]
    ) -> None:
        """Rows from self.top down to the status line, scrolled by the wheel."""
        s = self.s
        bottom = (PAD + SMALL_BTN_H + 18 + STATUS_LINES * STATUS_LINE_H) * s
        fits = max(1, int((self.top - bottom) // (ROW_H * s)))
        self.scroll = max(0, min(self.scroll, len(rows) - fits))
        for i, row in enumerate(rows[self.scroll : self.scroll + fits]):
            y = self.top - (i + 1) * ROW_H * s
            draw(row, PAD * s, y, self.width - 2 * PAD * s)
        self._fits, self._rows = fits, len(rows)
        if len(rows) > fits:
            more = f"{self.scroll + 1}-{self.scroll + fits} of {len(rows)} (scroll)"
            self._label(
                more,
                self.width - PAD * s,
                self.top + 18 * s,
                9.5,
                T.PICKER_DIM_TEXT,
                anchor_x="right",
            )

    def _page_main(self) -> None:
        s, la = self.s, self.launcher
        x, w = PAD * s, self.width - 2 * PAD * s
        half = (w - GAP * s) / 2
        self._label("pijl", x, self.top, 26, anchor_y="top", bold=True)
        self._label(
            version(),
            self.width - x,
            self.top - 8 * s,
            9.5,
            T.PICKER_DIM_TEXT,
            anchor_x="right",
            anchor_y="top",
        )
        y = self.top - 70 * s
        self._label("PROJECT", x, y, 9.5, T.PICKER_DIM_TEXT)
        if la.safe:
            self._label(
                "SAFE START: NO MODS",
                self.width - x,
                y,
                9.5,
                T.MENU_DANGER,
                anchor_x="right",
                bold=True,
            )
        y -= 30 * s
        self._label(la.project, x, y, 18, ACCENT, bold=True)
        y -= (24 + BIG_H) * s
        self._button(
            "START",
            x,
            y,
            w,
            BIG_H * s,
            self._start,
            PRIMARY,
            "Enter: open the editor",
            size=14,
        )
        y -= (GAP * 1.5 + BTN_H) * s
        self._button(
            "LOAD PROJECT",
            x,
            y,
            half,
            BTN_H * s,
            lambda: self.show("load"),
            hint="pick or rename a project",
        )
        self._button(
            "NEW PROJECT",
            x + half + GAP * s,
            y,
            half,
            BTN_H * s,
            lambda: self.show("new"),
        )
        y -= (GAP + BTN_H) * s
        third = (w - 2 * GAP * s) / 3
        self._button(
            "HEADLESS",
            x,
            y,
            third,
            BTN_H * s,
            lambda: self.show("headless"),
            hint="run a macro without the editor",
        )
        self._button(
            "MODS",
            x + third + GAP * s,
            y,
            third,
            BTN_H * s,
            lambda: self.show("mods"),
            hint="turn mods on and off, order them; safe start",
        )
        self._button(
            "SETTINGS",
            x + 2 * (third + GAP * s),
            y,
            third,
            BTN_H * s,
            lambda: self.show("settings"),
        )
        self._button(
            "QUIT",
            self.width - x - 120 * s,
            PAD * s,
            120 * s,
            SMALL_BTN_H * s,
            self._quit,
            hint="Esc",
        )

    def _page_load(self) -> None:
        s, la = self.s, self.launcher
        self._title("LOAD PROJECT")

        def draw(name: str, x, y, w) -> None:
            h = ROW_H * s - 2 * s
            if self.editing == name:
                self._field(x, y, w, h, name, 40, lambda text: self._rename(name, text))
                return
            self._box(x, y, w, h, ROW, lambda: self._pick(name), hint="click: pick it")
            mark = "  (picked)" if name == la.project else ""
            self._label(
                name + mark,
                x + 8 * s,
                y + h / 2,
                color=ACCENT if mark else T.PART_TEXT,
                anchor_y="center",
            )
            bw = 80 * s
            self._button(
                "rename",
                x + w - bw - 3 * s,
                y + 3 * s,
                bw,
                h - 6 * s,
                lambda: self._start_edit(name),
                ROW,
                size=9.5,
                color=T.HELP_TEXT,
            )

        self._list(la.projects(), draw)
        self._bottom()

    def _page_new(self) -> None:
        s = self.s
        self._title("NEW PROJECT")
        self._label("Its name:", PAD * s, self.top, 10, T.HELP_TEXT, anchor_y="top")
        y = self.top - (24 + 34) * s
        self._field(
            PAD * s, y, self.width - 2 * PAD * s, 34 * s, "", 40, self._make_project
        )
        self._label(
            "Enter: create it   Esc: back", PAD * s, y - 22 * s, 9.5, T.PICKER_DIM_TEXT
        )
        self._bottom(("CREATE", lambda: self._make_project(self.edit.line.text)))

    def _page_settings(self) -> None:
        s, la = self.s, self.launcher
        self._title("SETTINGS")
        rows = []
        for heading, group in prefs.SECTIONS:
            rows.append(heading)
            rows += list(group)

        def draw(row: str, x, y, w) -> None:
            h = ROW_H * s - 2 * s
            setting = prefs.PREFS.get(row)
            if setting is None:  # a heading
                self._label(
                    row.upper(), x, y + 8 * s, 9.5, T.PICKER_DIM_TEXT, bold=True
                )
                return
            value = la.prefs[row]
            changed = value != setting.initial
            name = setting.title(row) + (" *" if changed else "")
            hint = self._pref_hint(row)
            if self.editing == row:
                self._label(name, x + 8 * s, y + h / 2, anchor_y="center")
                fw = 160 * s
                text = str(value) if type(value) is int else f"{value:g}"
                self._field(
                    x + w - fw, y, fw, h, text, 16, lambda t: self._set_typed(row, t)
                )
                return
            self._box(
                x,
                y,
                w,
                h,
                ROW,
                lambda: self._pref_click(row),
                lambda: self._pref_click(row, -1),
                (lambda d: self._pref_wheel(row, d))
                if isinstance(setting, Number)
                else None,
                hint,
            )
            self._label(name, x + 8 * s, y + h / 2, anchor_y="center")
            self._label(
                setting.show(value),
                x + w - 8 * s,
                y + h / 2,
                color=ACCENT,
                anchor_x="right",
                anchor_y="center",
            )

        self._list(rows, draw)
        self._bottom(("RESET TO DEFAULTS", self._reset))

    def _page_headless(self) -> None:
        s, la = self.s, self.launcher
        self._title(f"HEADLESS: {la.project}")
        try:
            macros = list(la.macros().items())
        except OSError as e:
            macros = []
            self.say(f"can't list the macros: {e}", danger=True)
        if not macros:
            self._label(
                "No macros in this project yet.",
                PAD * s,
                self.top,
                10,
                T.HELP_TEXT,
                anchor_y="top",
            )
        where = "in this terminal" if has_console() else "in a console window"

        def draw(item, x, y, w) -> None:
            id, title = item
            h = ROW_H * s - 2 * s
            self._box(
                x,
                y,
                w,
                h,
                ROW,
                lambda: self._headless(id, title, False),
                hint=f"click: type its inputs {where}",
            )
            self._label(title, x + 8 * s, y + h / 2, anchor_y="center")
            bw = 110 * s
            self._button(
                "truth table",
                x + w - bw - 3 * s,
                y + 3 * s,
                bw,
                h - 6 * s,
                lambda: self._headless(id, title, True),
                ROW,
                size=9.5,
                color=T.HELP_TEXT,
            )

        self._list(macros, draw)
        self._bottom()

    def _page_mods(self) -> None:
        s, la = self.s, self.launcher
        self._title("MODS")
        try:
            on, off = la.mods()
        except OSError as e:
            on, off = [], []
            self.say(f"can't read the mods folder: {e}", danger=True)
        if not on and not off:
            self._label(
                "No mods. Put a mod's .py file or folder in",
                PAD * s,
                self.top,
                10,
                T.HELP_TEXT,
                anchor_y="top",
            )
            self._label(
                la.mods_folder(),
                PAD * s,
                self.top - 20 * s,
                9.5,
                T.PICKER_DIM_TEXT,
                anchor_y="top",
            )
        rows: list = (["LOAD ORDER", *on] if on else []) + (["DISABLED", *off] if off else [])

        def draw(row, x, y, w) -> None:
            h = ROW_H * s - 2 * s
            if not isinstance(row, ModRow):  # a heading
                self._label(row, x, y + 8 * s, 9.5, T.PICKER_DIM_TEXT, bold=True)
                return
            hint = "; ".join(filter(None, [row.description, *row.notes]))
            self._box(x, y, w, h, ROW, lambda: None, hint=hint or row.name)
            color = T.MENU_DANGER if row.notes else ACCENT if row.on else T.PART_TEXT
            self._label(
                f"{row.title} {row.tags}".rstrip(),
                x + 8 * s,
                y + h / 2,
                color=color,
                anchor_y="center",
            )
            bw = 54 * s
            buttons = (
                [("up", lambda: self._mod(la.mod_move, row.name, -1)),
                 ("down", lambda: self._mod(la.mod_move, row.name, 1)),
                 ("off", lambda: self._mod(la.mod_off, row.name))]
                if row.on
                else [("on", lambda: self._mod(la.mod_on, row.name))]
            )
            bx = x + w - 3 * s
            for text, click in reversed(buttons):
                bx -= bw
                self._button(
                    text, bx, y + 3 * s, bw, h - 6 * s, click, ROW, size=9.5, color=T.HELP_TEXT
                )
                bx -= 3 * s

        self._list(rows, draw)
        self._bottom(
            (
                f"SAFE START: {'ON' if la.safe else 'OFF'}",
                self._toggle_safe,
            )
        )

    # ---- what the buttons do -----------------------------------------------------

    def _mod(self, action: Callable, *args) -> None:
        if self._try(lambda: action(*args)):
            self.show()

    def _toggle_safe(self) -> None:
        self.launcher.safe = not self.launcher.safe
        self.show()
        self.say("starts without mods" if self.launcher.safe else "starts with the mods in the load order")

    def _finish(self, result: list[str] | None) -> None:
        self.result = result
        self.close()

    def _start(self) -> None:
        self._finish(self.launcher.editor())

    def _quit(self) -> None:
        self._finish(None)

    def _try(self, action: Callable[[], object]) -> bool:
        try:
            action()
        except (OSError, ValueError) as e:
            self.say(f"can't: {e}", danger=True)
            return False
        return True

    def _pick(self, name: str) -> None:
        if self._try(lambda: self.launcher.pick(name)):
            self.show("main")

    def _start_edit(self, row: str) -> None:
        self.editing, self.message = row, ""
        self.show()

    def _rename(self, old: str, text: str) -> None:
        if self._try(lambda: self.launcher.rename(old, text)):
            self.editing = None
            self.show()

    def _make_project(self, text: str) -> None:
        if self._try(lambda: self.launcher.new_project(text)):
            self.show("main")
            self.say(f"made {self.launcher.project}")

    def _pref_hint(self, key: str) -> str:
        setting = prefs.PREFS[key]
        how = {
            Choice: "click: next, right-click: back",
            Toggle: "click: flip it",
            Number: f"click: type it ({setting.show(getattr(setting, 'min', 0))} to {setting.show(getattr(setting, 'max', 0))}), wheel: nudge it",
        }.get(type(setting), "")
        return f"{setting.hint}. {how}" if setting.hint else how

    def _set_pref(self, key: str, value) -> None:
        if self._try(lambda: self.launcher.set_pref(key, value)):
            self.editing = None
            if key == "ui.scale":
                self._rescale()
            self.show()

    def _pref_click(self, key: str, direction: int = 1) -> None:
        setting, value = prefs.PREFS[key], self.launcher.prefs[key]
        if isinstance(setting, Choice):
            i = next(
                i
                for i, v in enumerate(setting.values)
                if v == value and type(v) is type(value)
            )
            self._set_pref(key, setting.values[(i + direction) % len(setting.values)])
        elif isinstance(setting, Toggle):
            self._set_pref(key, not value)
        elif isinstance(setting, Number):
            self._start_edit(key)

    def _pref_wheel(self, key: str, notches: int) -> None:
        setting: Number = prefs.PREFS[key]
        step = (
            setting.step
            if setting.step is not None
            else (1 if setting.whole else (setting.max - setting.min) / 100)
        )
        if self.keys_shift:
            step *= 10
        self._set_pref(key, setting.parse(self.launcher.prefs[key] + notches * step))

    def _set_typed(self, key: str, text: str) -> None:
        try:
            value = prefs.parse_text(key, text)
        except ValueError as e:
            self.say(str(e), danger=True)
            return
        self._set_pref(key, value)

    def _reset(self) -> None:
        if self._try(self.launcher.reset_prefs):
            self._rescale()
            self.show()
            self.say("everything's back to its default")

    def _rescale(self) -> None:
        """The window follows the UI scale preference."""
        s = self.launcher.prefs["ui.scale"]
        if s != self.s:
            self.s = s
            self.set_size(round(W * s), round(H * s))

    def _headless(self, id: str, title: str, table: bool) -> None:
        args = self.launcher.headless(id, table)
        if has_console():
            self._finish(args)  # cli runs it here, once the window's gone
            return
        problem = open_console(args)
        if problem:
            self.say(problem, danger=True)
        else:
            self.say(f"started {title} in a console window")

    # ---- events ------------------------------------------------------------------

    keys_shift = False

    def on_draw(self) -> None:
        r, g, b, _ = T.BACKGROUND
        pyglet.gl.glClearColor(r / 255, g / 255, b / 255, 1)
        self.clear()
        self.batch.draw()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        if self._ready:
            self.show()

    def _hit(self, x: float, y: float) -> _Hit | None:
        for h in reversed(self.hits):  # (buttons inside rows come later: they win)
            if h.contains(x, y):
                return h
        return None

    def _hover(self, x: float, y: float) -> None:
        self._mouse = (x, y)
        hit = self._hit(x, y)
        if hit is self.hovered:
            return
        if self.hovered is not None and self.hovered.rect is not None:
            self.hovered.rect.color = self.hovered.colors[0]
        self.hovered = hit
        if hit is not None and hit.rect is not None and (hit.click or hit.right):
            hit.rect.color = hit.colors[1]
        self._show_status()

    def on_mouse_motion(self, x, y, dx, dy) -> None:
        self._hover(x, y)

    def on_mouse_leave(self, x, y) -> None:
        self._hover(-1, -1)

    def on_mouse_press(self, x, y, button, modifiers) -> None:
        self.keys_shift = bool(modifiers & key.MOD_SHIFT)
        if self.edit is not None and self.edit.contains(x, y):
            if button == mouse.LEFT:
                self.text_mouse.press(self._edit_target(), x, y, self.keys_shift)
            return
        hit = self._hit(x, y)
        if self.editing is not None:  # a click outside the field drops the edit
            self.editing = None
            if hit is None or not (hit.click or hit.right):
                self.show()
                return
        if hit is None:
            return
        if self.message and self.danger:
            self.message = ""
        if button == mouse.LEFT and hit.click:
            hit.click()
        elif button == mouse.RIGHT and hit.right:
            hit.right()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y) -> None:
        notches = int(scroll_y) or (1 if scroll_y > 0 else -1 if scroll_y < 0 else 0)
        if not notches:
            return
        hit = self._hit(x, y)
        if hit is not None and hit.wheel is not None and self.editing is None:
            hit.wheel(notches)
        elif (
            getattr(self, "_rows", 0) > getattr(self, "_fits", 0)
            and self.editing is None
        ):
            self.scroll = max(0, min(self.scroll - notches, self._rows - self._fits))
            self.show()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        if self.text_mouse.dragging and self.edit is not None:
            self.text_mouse.drag(self._edit_target(), x, y)

    def on_mouse_release(self, x, y, button, modifiers) -> None:
        if button == mouse.LEFT:
            self.text_mouse.release()

    def on_key_press(self, symbol, modifiers) -> None:
        self.keys_shift = bool(modifiers & key.MOD_SHIFT)
        if self.edit is not None and shortcut(
            self._edit_target(), symbol, modifiers, self
        ):
            return  # Ctrl+A / C / X / V in the field
        if self.edit is not None and symbol in (key.ENTER, key.NUM_ENTER):
            self.edit.enter(self.edit.line.text)
        elif symbol == key.ESCAPE:
            if self.editing is not None:
                self.editing = None
                self.show()
            elif self.page != "main":
                self.show("main")
            else:
                self._quit()
        elif self.page == "main" and symbol in (key.ENTER, key.NUM_ENTER):
            self._start()

    def on_key_release(self, symbol, modifiers) -> None:
        self.keys_shift = bool(modifiers & key.MOD_SHIFT)

    def on_text(self, text: str) -> None:
        if self.edit is not None:
            self.edit.line.insert(text)
            self._show_edit()

    def on_text_motion(self, motion: int, select: bool = False) -> None:
        if self.edit is not None:
            self.edit.line.motion(motion, select)
            self._show_edit()

    def on_text_motion_select(self, motion: int) -> None:  # (Shift held)
        self.on_text_motion(motion, select=True)

    def on_close(self) -> None:
        self.result = None
        super().on_close()
