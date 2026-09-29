"""Collapsible part picker along the left edge of the window.

Screen space (HUD batch), so it ignores the camera. Shows the Library:
each collection as an expandable section listing its parts, then the loose
parts (in no collection) below a divider. The editor decides what clicks mean
(see the controls in editor.py); this module lays out, draws, hit-tests, and
carries out drags and renames.

Everything is rebuilt by refresh() whenever the content changes. That's a few
dozen shapes at most, and it keeps the drawing code a straight top-to-bottom read.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyglet
from pyglet import shapes
from pyglet.gl import GL_SCISSOR_TEST, glDisable, glEnable, glScissor

from . import theme as T
from .library import Collection, Library
from .line_edit import LineEdit
from .views import Box

PANEL_W, COLLAPSED_W = 190, 24
HEADER_H, SECTION_H, ROW_H = 36, 26, 24
BUTTON = 24          # header buttons are squares this big
PAD, INDENT = 8, 14  # left padding; extra indent for parts inside a collection
SWATCH = 10          # little color chip in front of each part name
LOOSE_GAP = 12       # space between the last collection and the loose parts (holds the divider)
TAIL = 2 * ROW_H     # empty space under the list, so there's always somewhere to drop "loose"
SCROLL_STEP = 30
NAME_MAX = 18
FONT, FONT_SIZE = "Consolas", 11

@dataclass(eq=False)
class Row:
    what: str                     # "section", "part", or "empty" (placeholder in an empty open collection)
    collection: Collection | None  # the section's collection, or the part's (None: loose)
    part: str | None = None
    index: int = 0                # position of the part in its list
    top: float = 0.0              # distance from the top of the list, px, scroll not applied
    h: float = ROW_H


class ClipGroup(pyglet.graphics.Group):
    """Clips its children to a rectangle (the scrollable list, so it slides under the header)."""

    def __init__(self, order: int) -> None:
        super().__init__(order)
        self.rect = (0, 0, 0, 0)  # framebuffer pixels

    def set_state(self) -> None:
        glEnable(GL_SCISSOR_TEST)
        glScissor(*self.rect)

    def unset_state(self) -> None:
        glDisable(GL_SCISSOR_TEST)


def swatch_colors(part: str) -> tuple:
    """(fill, border) of the chip in front of a part's name: the part's own body colors."""
    return {"IN": T.SWITCH_ON, "OUT": T.LED_ON}.get(part, T.PART_BODY)


class PartPicker:
    def __init__(self, library: Library, batch: pyglet.graphics.Batch, win_h: int,
                 pixel_ratio: float = 1.0) -> None:
        self.lib = library
        self.batch = batch
        self.open = True
        self.win_h = win_h
        self.pixel_ratio = pixel_ratio  # framebuffer pixels per window pixel (HiDPI), for the clip
        self.scroll = 0.0               # how far the list is scrolled up, px
        self.rows: list[Row] = []
        self.loose_top = 0.0
        self.content_h = 0.0
        self.hovered: Row | str | None = None
        self.dragging: Row | None = None
        self.drop: tuple | None = None  # where the drag would land: see _drop_at
        self.renaming: Collection | None = None
        self.edit: LineEdit | None = None

        self.panel_group = pyglet.graphics.Group(order=2)
        self.clip = ClipGroup(order=3)
        self.row_bg = pyglet.graphics.Group(order=0, parent=self.clip)
        self.row_fg = pyglet.graphics.Group(order=1, parent=self.clip)
        self.row_text = pyglet.graphics.Group(order=2, parent=self.clip)
        self.marker_group = pyglet.graphics.Group(order=3, parent=self.clip)
        self.header_bg = pyglet.graphics.Group(order=5)
        self.header_fg = pyglet.graphics.Group(order=6)
        self.ghost_group = pyglet.graphics.Group(order=7)

        self._shapes: list = []       # everything refresh() made
        self._bgs: dict = {}          # hover target (Row / button name) -> its background rectangle
        self._buttons: dict[str, tuple[float, float, float, float]] = {}
        self._drag_shapes: list = []  # marker + ghost, rebuilt on every drag move
        self._caret: shapes.Rectangle | None = None
        self._measure = pyglet.text.Label("", font_name=FONT, font_size=FONT_SIZE)
        self.refresh()

    # ---- geometry ------------------------------------------------------------

    @property
    def width(self) -> int:
        return PANEL_W if self.open else COLLAPSED_W

    @property
    def list_top(self) -> float:
        """Screen y of the top of the list area (the bottom of the header)."""
        return self.win_h - HEADER_H

    def contains(self, sx: float, sy: float) -> bool:
        return 0 <= sx < self.width and 0 <= sy <= self.win_h

    def screen_y(self, top: float) -> float:
        """List position (px from the list top) -> screen y."""
        return self.list_top - top + self.scroll

    def row_y(self, row: Row) -> float:
        """Screen y of a row's bottom edge."""
        return self.screen_y(row.top + row.h)

    def row_at(self, sy: float) -> Row | None:
        if sy >= self.list_top:
            return None
        return next((r for r in self.rows if self.row_y(r) <= sy < self.row_y(r) + r.h), None)

    def hit(self, sx: float, sy: float) -> Row | str | None:
        """What's under a screen point: a Row, "toggle" / "new" (header buttons),
        "header", "blank" (list space without a row), or None (not over the picker)."""
        if not self.contains(sx, sy):
            return None
        if not self.open:
            return "toggle"  # the whole collapsed strip opens it
        if sy >= self.list_top:
            for name, (x, y, w, h) in self._buttons.items():
                if x <= sx <= x + w and y <= sy <= y + h:
                    return name
            return "header"
        return self.row_at(sy) or "blank"

    # ---- state changes ---------------------------------------------------------

    def toggle(self) -> None:
        self.open = not self.open
        self.refresh()

    def resize(self, win_h: int, pixel_ratio: float) -> None:
        self.win_h, self.pixel_ratio = win_h, pixel_ratio
        self.refresh()

    def scroll_by(self, dy: float) -> None:
        self.scroll -= dy * SCROLL_STEP  # wheel up (positive) moves toward the top
        self.refresh()

    def set_hover(self, target: Row | str | None) -> None:
        if target is self.hovered or target == self.hovered:
            return
        for old in (self.hovered, target):
            if old in self._bgs:
                self._bgs[old].color = T.PICKER_HOVER if old is target else self._bg_color(old)
        self.hovered = target

    # ---- renaming a collection ----------------------------------------------------

    def start_rename(self, c: Collection, fresh: bool = False) -> None:
        """Type a new name for `c`. `fresh`: it was just made, so start empty (its
        default name shows as a placeholder, and stays if nothing is typed)."""
        self.renaming, self.edit = c, LineEdit("" if fresh else c.name, NAME_MAX)
        self.open = True
        self._scroll_into_view(next(r for r in self._layout() if r.collection is c))
        pyglet.clock.schedule_interval(self._blink, 0.5)
        self.refresh()

    def rename_text(self, text: str) -> None:
        self.edit.insert(text.upper())  # collection names are all caps, like the built-in ones
        self.refresh()

    def rename_motion(self, motion: int) -> None:
        self.edit.motion(motion)
        self.refresh()

    def finish_rename(self, commit: bool) -> None:
        if commit:
            self.lib.rename(self.renaming, self.edit.text)
        self.renaming, self.edit = None, None
        pyglet.clock.unschedule(self._blink)
        self.refresh()

    def _blink(self, dt: float) -> None:
        if self._caret is not None:
            self._caret.visible = not self._caret.visible

    # ---- drag and drop -------------------------------------------------------------

    def begin_drag(self, row: Row) -> None:
        self.dragging = row
        self.refresh()  # dims the row being carried

    def drag_to(self, sx: float, sy: float) -> None:
        self.drop = self._drop_at(sx, sy)
        self._draw_drag(sx, sy)

    def end_drag(self) -> None:
        """Drop the dragged row where the marker shows (nowhere if there's no marker)."""
        row, drop = self.dragging, self.drop
        if drop is not None and row.what == "part":
            self.lib.move_part(row.part, drop[0], drop[1])
        elif drop is not None and row.what == "section":
            self.lib.move_collection(row.collection, drop[0])
        self.cancel_drag()

    def cancel_drag(self) -> None:
        for s in self._drag_shapes:
            s.delete()
        self.dragging, self.drop, self._drag_shapes = None, None, []
        self.refresh()

    def _drop_at(self, sx: float, sy: float) -> tuple | None:
        """Where the dragged row would land if dropped here, as
        parts:        (collection or None for loose, index or None for "at the end", marker)
        collections:  (index, marker)
        marker is ("line", y, indent) or ("box", row). None: not a drop spot."""
        if not self.contains(sx, sy) or sy >= self.list_top:
            return None
        if self.dragging.what == "part":
            row = self.row_at(sy)
            if row is None:  # blank space: loose, at the end
                return None, None, ("line", self.screen_y(self.loose_top + len(self.lib.loose) * ROW_H), 0)
            if row.what == "empty":
                row = next(r for r in self.rows if r.what == "section" and r.collection is row.collection)
            if row.what == "section":  # onto the header: into that collection, at the end
                return row.collection, None, ("box", row)
            after = sy < self.row_y(row) + row.h / 2
            y = self.row_y(row) if after else self.row_y(row) + row.h
            return row.collection, row.index + after, ("line", y, INDENT if row.collection else 0)
        # a collection: insert before the first block (section + its rows) whose middle is below the cursor
        blocks = self._blocks()
        cy = self.list_top - sy + self.scroll  # cursor, as px from the list top
        index = next((i for i, (t, b) in enumerate(blocks) if cy < (t + b) / 2), len(blocks))
        edge = blocks[index][0] if index < len(blocks) else (blocks[-1][1] if blocks else 0)
        return index, ("line", self.screen_y(edge), 0)

    def _blocks(self) -> list[tuple[float, float]]:
        """(top, bottom) of each collection's section plus its rows, in list px."""
        blocks = []
        for r in self.rows:
            if r.what == "section":
                blocks.append([r.top, r.top + r.h])
            elif r.collection is not None:
                blocks[-1][1] = r.top + r.h
        return [tuple(b) for b in blocks]

    def _draw_drag(self, sx: float, sy: float) -> None:
        for s in self._drag_shapes:
            s.delete()
        self._drag_shapes = []
        add = self._drag_shapes.append
        if self.drop is not None:
            marker = self.drop[-1]
            if marker[0] == "line":
                _, y, indent = marker
                add(shapes.Rectangle(PAD + indent, y - 1, PANEL_W - 2 * PAD - indent, 2, color=T.SELECT,
                                     batch=self.batch, group=self.marker_group))
            else:
                box = shapes.Box(2, self.row_y(marker[1]) + 1, PANEL_W - 4, marker[1].h - 2, thickness=2,
                                 color=T.SELECT, batch=self.batch, group=self.marker_group)
                add(box)
        # the ghost: the row's name in a little box following the cursor
        row = self.dragging
        text = row.part if row.what == "part" else row.collection.name
        self._measure.text = text
        w, h = self._measure.content_width + 2 * PAD, ROW_H
        ghost = Box(w, h, 1, *T.MENU_PANEL, self.batch, self.ghost_group)
        ghost.position = (sx + 10, sy - h - 6)
        add(ghost)
        add(pyglet.text.Label(text, font_name=FONT, font_size=FONT_SIZE, color=T.PART_TEXT,
                              x=sx + 10 + PAD, y=sy - h / 2 - 6, anchor_y="center",
                              batch=self.batch, group=pyglet.graphics.Group(order=1, parent=self.ghost_group)))

    # ---- layout + drawing -----------------------------------------------------------

    def _layout(self) -> list[Row]:
        rows, top = [], 4.0
        for c in self.lib.collections:
            rows.append(Row("section", c, top=top, h=SECTION_H))
            top += SECTION_H
            if c.open:
                if not c.parts:
                    rows.append(Row("empty", c, top=top))
                    top += ROW_H
                for i, p in enumerate(c.parts):
                    rows.append(Row("part", c, p, i, top))
                    top += ROW_H
        top += LOOSE_GAP
        self.loose_top = top
        for i, p in enumerate(self.lib.loose):
            rows.append(Row("part", None, p, i, top))
            top += ROW_H
        self.content_h = top + TAIL
        return rows

    def _scroll_into_view(self, row: Row) -> None:
        view_h = self.list_top
        if row.top < self.scroll:
            self.scroll = row.top
        elif row.top + row.h > self.scroll + view_h:
            self.scroll = row.top + row.h - view_h

    def _bg_color(self, target) -> tuple:
        if isinstance(target, Row) and target.what == "section":
            return T.PICKER_SECTION
        return T.PICKER_BG if isinstance(target, Row) else T.PICKER_HEADER

    def _fit(self, text: str, max_w: float) -> str:
        """Shorten text with an ellipsis until it fits in max_w pixels."""
        self._measure.text = text
        while text and self._measure.content_width > max_w:
            text = text[:-1]
            self._measure.text = text + "…"
        return self._measure.text

    def refresh(self) -> None:
        for s in self._shapes:
            s.delete()
        self._shapes, self._bgs, self._buttons, self._caret = [], {}, {}, None
        add = self._shapes.append
        b, h = self.batch, self.win_h
        r = self.pixel_ratio
        self.clip.rect = (0, 0, int(PANEL_W * r), int(max(self.list_top, 0) * r))

        def label(text, x, y, group, color=T.PART_TEXT, size=FONT_SIZE, anchor_x="left"):
            add(pyglet.text.Label(text, font_name=FONT, font_size=size, color=color, x=x, y=y,
                                  anchor_x=anchor_x, anchor_y="center", batch=b, group=group))

        def rect(x, y, w, hh, color, group):
            s = shapes.Rectangle(x, y, w, hh, color=color, batch=b, group=group)
            add(s)
            return s

        if not self.open:
            rect(0, 0, COLLAPSED_W, h, T.PICKER_BG, self.panel_group)
            rect(COLLAPSED_W - 1, 0, 1, h, T.PICKER_BORDER, self.header_fg)
            self._bgs["toggle"] = rect(0, h - HEADER_H, COLLAPSED_W - 1, HEADER_H, T.PICKER_HEADER, self.header_bg)
            label("»", COLLAPSED_W / 2, h - HEADER_H / 2, self.header_fg, anchor_x="center")
            if self.hovered in self._bgs:
                self._bgs[self.hovered].color = T.PICKER_HOVER
            return

        self.rows = self._layout()
        self.scroll = max(0.0, min(self.scroll, self.content_h - self.list_top))

        # panel + header
        rect(0, 0, PANEL_W, h, T.PICKER_BG, self.panel_group)
        rect(0, h - HEADER_H, PANEL_W, HEADER_H, T.PICKER_HEADER, self.header_bg)
        rect(0, h - HEADER_H, PANEL_W, 1, T.PICKER_BORDER, self.header_fg)
        rect(PANEL_W - 1, 0, 1, h, T.PICKER_BORDER, self.header_fg)
        label("PARTS", PAD + 2, h - HEADER_H / 2, self.header_fg, color=T.HELP_TEXT)
        by = h - HEADER_H + (HEADER_H - BUTTON) / 2
        for i, (name, glyph) in enumerate((("toggle", "«"), ("new", "+"))):
            bx = PANEL_W - 6 - (i + 1) * BUTTON - i * 2
            self._buttons[name] = (bx, by, BUTTON, BUTTON)
            self._bgs[name] = rect(bx, by, BUTTON, BUTTON, T.PICKER_HEADER, self.header_bg)
            label(glyph, bx + BUTTON / 2, by + BUTTON / 2, self.header_fg, anchor_x="center")

        # rows
        for row in self.rows:
            y = self.row_y(row)
            if y > self.list_top or y + row.h < 0:
                continue  # scrolled out of sight (the clip would hide it anyway)
            self._bgs[row] = rect(0, y, PANEL_W - 1, row.h, self._bg_color(row), self.row_bg)
            cy = y + row.h / 2
            carried = self.dragging is not None and (
                row is self.dragging or (row.what == self.dragging.what and row.part == self.dragging.part
                                         and row.collection is self.dragging.collection))
            color = T.PICKER_DIM_TEXT if carried else T.PART_TEXT
            if row.what == "section":
                self._draw_section(row, y, cy, color, label, rect, add)
            elif row.what == "empty":
                label("drop parts here", PAD + INDENT, cy, self.row_text, color=T.PICKER_DIM_TEXT, size=10)
            else:
                x = PAD + (INDENT if row.collection else 0)
                chip = Box(SWATCH, SWATCH, 1, *swatch_colors(row.part), b, self.row_fg)
                chip.position = (x, cy - SWATCH / 2)
                add(chip)
                label(self._fit(row.part, PANEL_W - x - SWATCH - 8 - PAD), x + SWATCH + 8, cy, self.row_text,
                      color=color)
        if self.lib.loose and self.lib.collections:  # divider between the collections and the loose parts
            rect(PAD, self.screen_y(self.loose_top - LOOSE_GAP / 2), PANEL_W - 2 * PAD, 1, T.PICKER_BORDER,
                 self.row_fg)

        if self.hovered in self._bgs:
            self._bgs[self.hovered].color = T.PICKER_HOVER
        else:
            self.hovered = None  # it's gone (scrolled away, deleted, ...)

    def _draw_section(self, row: Row, y, cy, color, label, rect, add) -> None:
        c = row.collection
        x = PAD + 2
        # disclosure triangle: pointing down when open, right when closed
        tri = ((x, cy + 4, x + 8, cy + 4, x + 4, cy - 3) if c.open else (x + 1, cy + 5, x + 1, cy - 5, x + 8, cy))
        add(shapes.Triangle(*tri, color=T.HELP_TEXT[:3], batch=self.batch, group=self.row_fg))
        name_x = x + 14
        count_w = 24
        if c is self.renaming:
            field = Box(PANEL_W - name_x - PAD, row.h - 6, 1, T.PICKER_BG, T.SELECT, self.batch, self.row_fg)
            field.position = (name_x - 4, y + 3)
            add(field)
            if self.edit.text:
                label(self.edit.text, name_x, cy, self.row_text)
            else:
                label(c.name, name_x, cy, self.row_text, color=T.PICKER_DIM_TEXT)
            self._measure.text = self.edit.text[:self.edit.caret]
            self._caret = rect(name_x + self._measure.content_width, cy - 8, 1.5, 16, T.CARET, self.row_text)
            return
        label(self._fit(c.name, PANEL_W - name_x - count_w - PAD), name_x, cy, self.row_text, color=color)
        label(str(len(c.parts)), PANEL_W - PAD - 2, cy, self.row_text, color=T.PICKER_DIM_TEXT, size=10,
              anchor_x="right")
