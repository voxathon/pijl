"""The Controls sheet: a scrollable list of every mouse and keyboard control, opened
from the status bar's cogwheel menu.

It's the short version of the controls in editor.py's docstring; keep the two in
step. Shown the same way as a Prompt (it has the same methods), so the editor's
PROMPT mode handles it: Esc, Enter or a click outside closes it, the wheel and
the arrow keys scroll it.
"""

from __future__ import annotations

import textwrap

import pyglet
from pyglet import shapes

from . import theme as T
from .picker import ClipGroup
from .views import Box

S = T.UI_SCALE
MAX_W, MARGIN, PAD = 760 * S, 40 * S, 12 * S
TITLE_H, HINT_H, HEAD_H, GAP = 30 * S, 26 * S, 28 * S, 3 * S
KEYS_W = 190 * S  # the key column; descriptions wrap in what's left
LINE_STEP = 30 * S  # one arrow key / wheel notch
FONT, SIZE, SMALL = "Consolas", 10.5 * S, 10 * S

CONTROLS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Parts", [
        ("click a part (picker)", "pick it up; it follows the cursor"),
        ("click", "place it (Shift+click: place it and keep another)"),
        ("right-click / Esc", "put it back"),
        ("click a part", "select it; a plain click on an IN switch toggles it"),
        ("drag a part", "move it (a selected part moves the whole selection)"),
        ("right-click a part", "label, recolor, delete"),
    ]),
    ("Wires", [
        ("click a pin", "start a wire; it follows the cursor"),
        ("  click empty space", "add a bend point"),
        ("  click a pin / wire", "connect (ending on a wire makes a junction)"),
        ("  right-click / Bksp", "remove the last bend point, or cancel"),
        ("press+drag on a wire", "start a branch from that spot (also Alt+click)"),
        ("right-click a wire", "edit bends, branch, recolor, delete from there on"),
    ]),
    ("Selection", [
        ("Shift+click", "add to / remove from the selection"),
        ("drag empty space", "box-select (Shift: add to the selection)"),
        ("Ctrl+A", "select everything"),
        ("Del / Backspace", "delete the selection"),
        ("Esc / click empty space", "clear the selection"),
    ]),
    ("Editing", [
        ("Ctrl+C / Ctrl+X", "copy / cut"),
        ("Ctrl+V", "paste: click to place it (Shift+click: and keep another copy)"),
        ("Ctrl+D", "duplicate into a block: each press doubles it, right then down"),
        ("  Ctrl+scroll", "space the block out (Ctrl+Shift+scroll: the other way)"),
        ("Ctrl+Z / Ctrl+Y", "undo / redo (also Ctrl+Shift+Z); mid-action, Ctrl+Z cancels it"),
    ]),
    ("Files", [
        ("Ctrl+S", "save the board as a macro (the first save asks for a name)"),
        ("Ctrl+Shift+S", "save under another name"),
        ("Ctrl+O", "open a macro (or right-click it in the picker -> Open)"),
        ("Ctrl+N", "new, empty board"),
        ("cogwheel -> Projects", "switch to another project, or make a new one"),
    ]),
    ("View", [
        ("scroll", "zoom"),
        ("right-drag / middle-drag", "pan"),
        ("Home", "reset the camera"),
        ("hold Ctrl", "snap to the grid (Ctrl+Shift: the finer subgrid)"),
        ("Tab", "pin names on parts: hidden -> on hover -> always"),
        ("«  (picker header)", "tuck the part picker away"),
    ]),
]


class ControlsSheet:
    def __init__(self, batch: pyglet.graphics.Batch, win_w: int, win_h: int, pixel_ratio: float) -> None:
        self.batch = batch
        self.pixel_ratio = pixel_ratio
        self.scroll = 0.0
        shade_g, bg = pyglet.graphics.Group(order=20), pyglet.graphics.Group(order=21)
        text_g = pyglet.graphics.Group(order=23)
        self.clip = ClipGroup(order=22)
        body_g = pyglet.graphics.Group(order=0, parent=self.clip)
        self.shade = shapes.Rectangle(0, 0, win_w, win_h, color=(0, 0, 0, 110), batch=batch, group=shade_g)
        self.panel: Box | None = None
        self.bg_group = bg
        self.title = pyglet.text.Label("Controls", font_name=FONT, font_size=11 * S, color=T.PART_TEXT,
                                       anchor_y="center", batch=batch, group=text_g)
        self.hint = pyglet.text.Label("scroll / arrows: more   Esc: close", font_name=FONT, font_size=SMALL,
                                      color=T.HELP_TEXT, anchor_y="center", batch=batch, group=text_g)
        self.body_g = body_g
        self.char_w = pyglet.text.Label("M" * 10, font_name=FONT, font_size=SIZE).content_width / 10
        # (kind, what): "head" a section title label, "keys" the key column's label, "what" the
        # description: its text, plus one single-line label per wrapped line (made by layout).
        # Wrapped here, not by pyglet: its multiline labels lose some of the spaces between words.
        self.labels: list[tuple[str, pyglet.text.Label | list]] = []
        for head, rows in CONTROLS:
            self.labels.append(("head", pyglet.text.Label(head.upper(), font_name=FONT, font_size=SMALL,
                                                          color=T.PICKER_DIM_TEXT, anchor_y="top",
                                                          batch=batch, group=body_g)))
            for keys, what in rows:
                self.labels.append(("keys", pyglet.text.Label(keys, font_name=FONT, font_size=SIZE,
                                                              color=T.PART_TEXT, anchor_y="top",
                                                              batch=batch, group=body_g)))
                self.labels.append(("what", [what]))
        self.layout(win_w, win_h)

    # ---- layout ----------------------------------------------------------------

    def layout(self, win_w: int, win_h: int) -> None:
        self.win = (win_w, win_h)
        self.shade.width, self.shade.height = win_w, win_h
        w = max(200 * S, min(MAX_W, win_w - 2 * MARGIN))
        cols = max(10, int((w - 2 * PAD - KEYS_W) / self.char_w))
        for kind, what in self.labels:
            if kind == "what":
                for line in what[1:]:
                    line.delete()
                what[1:] = [pyglet.text.Label(text, font_name=FONT, font_size=SIZE, color=T.HELP_TEXT,
                                              anchor_y="top", batch=self.batch, group=self.body_g)
                            for text in textwrap.wrap(what[0], cols)]
        # the body's height, top to bottom (the rows' own offsets are worked out in _place)
        self.body_h = self._place(0, 0, dry=True)
        h = min(win_h - 2 * MARGIN, TITLE_H + self.body_h + HINT_H + PAD)
        h = max(h, TITLE_H + HINT_H + 2 * PAD)
        if self.panel is not None:
            self.panel.delete()
        self.panel = Box(w, h, max(1, round(S)), *T.MENU_PANEL, self.batch, self.bg_group)
        left, top = (win_w - w) / 2, (win_h + h) / 2
        self.panel.position = (left, top - h)
        self.box = (left, top - h, w, h)
        self.title.position = (left + PAD, top - TITLE_H / 2, 0)
        self.hint.position = (left + PAD, top - h + HINT_H / 2, 0)
        self.view_top, self.view_h = top - TITLE_H, h - TITLE_H - HINT_H
        self.scroll = max(0.0, min(self.scroll, self.body_h - self.view_h))
        r = self.pixel_ratio
        self.clip.rect = (int(left * r), int((self.view_top - self.view_h) * r), int(w * r), int(self.view_h * r))
        self._place(left + PAD, self.view_top + self.scroll)

    def _place(self, x: float, top: float, dry: bool = False) -> float:
        """Lay the rows out from `top` down; returns the height used."""
        y = top
        keys = None
        for kind, label in self.labels:
            if kind == "head":
                y -= GAP * 2 if y != top else 0
                if not dry:
                    label.position = (x, y - HEAD_H / 2 + label.content_height / 2, 0)
                y -= HEAD_H
            elif kind == "keys":
                keys = label
                if not dry:
                    label.position = (x, y, 0)
            else:
                line_h = keys.content_height
                if not dry:
                    for i, line in enumerate(label[1:]):
                        line.position = (x + KEYS_W, y - i * line_h, 0)
                y -= line_h * max(1, len(label) - 1) + GAP
        return top - y + PAD

    # ---- the Prompt interface (see the module docstring) ----------------------------

    text = ""
    choice = None

    def move(self, delta: int) -> None:
        self.scroll = max(0.0, min(self.scroll + delta * LINE_STEP, self.body_h - self.view_h))
        self._place(self.box[0] + PAD, self.view_top + self.scroll)

    def contains(self, sx: float, sy: float) -> bool:
        x, y, w, h = self.box
        return x <= sx <= x + w and y <= sy <= y + h

    def item_at(self, sx: float, sy: float) -> None:
        return None

    def hover(self, sx: float, sy: float) -> None:
        pass

    def type_text(self, text: str) -> None:
        pass

    def motion(self, motion: int) -> None:
        pass

    def tick(self, dt: float) -> None:
        pass

    def delete(self) -> None:
        for s in (self.shade, self.panel, self.title, self.hint):
            s.delete()
        for kind, what in self.labels:
            for label in (what[1:] if kind == "what" else [what]):
                label.delete()
