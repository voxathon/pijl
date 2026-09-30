"""Collapsible part picker along the left edge of the window.

Screen space (HUD batch), so it ignores the camera. Shows the Library: each
collection as an expandable section listing its parts, then the loose parts
(in no collection) below a divider. The editor decides what clicks mean (see
the controls in editor.py); this module lays out, draws, hit-tests, and
carries out drags and renames.

Retained + tweened: every row on screen is a Widget that lives as long as its
row does. refresh() only computes where each row *should* be (the layout);
update(dt), called every frame, eases each widget toward that. So scrolling,
expanding, reordering and the panel sliding in/out all animate for free, and a
wheel tick costs a few position updates instead of rebuilding every label.

Dragging: the dragged row(s) are lifted out of the list and pinned to the
cursor, and the layout is computed as if the drop had already happened, so
the other rows slide apart to open a gap where it would land.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pyglet
from pyglet import shapes
from pyglet.gl import GL_SCISSOR_TEST, glDisable, glEnable, glScissor

from . import theme as T
from .library import Collection, Library
from .line_edit import LineEdit
from .views import Box

S = T.UI_SCALE
HEADER_H, SECTION_H, ROW_H = 36 * S, 26 * S, 24 * S
BUTTON = 24 * S  # header buttons are squares this big
BUTTON_GAP = 4 * S  # around the toggle button, which sits at the panel's right edge
PANEL_W = round(190 * S)
COLLAPSED_W = round(
    BUTTON + 2 * BUTTON_GAP
)  # what's left on screen when tucked away: the toggle's column
PAD, INDENT = 8 * S, 14 * S  # left padding; extra indent for parts inside a collection
SWATCH = 10 * S  # little color chip in front of each part name
LOOSE_GAP = (
    12 * S
)  # space between the last collection and the loose parts (holds the divider)
TAIL = (
    2 * ROW_H
)  # empty space under the list, so there's always somewhere to drop "loose"
SCROLL_STEP = 40 * S
EDGE_SCROLL = 30 * S  # dragging this close to the list's top/bottom edge scrolls it
EDGE_SPEED = 600 * S  # px per second
NAME_MAX = 18
FONT, FONT_SIZE, SMALL_SIZE = "Consolas", 11 * S, 10 * S
SPEED = 16.0  # tween rate: each frame closes 1 - e^(-SPEED*dt) of the distance (~95% in 0.19s)


def approach(value: float, target: float, k: float, eps: float) -> float:
    """One easing step from value toward target; snaps once within eps."""
    if abs(target - value) <= eps:
        return target
    return value + (target - value) * k


@dataclass(eq=False)
class Row:
    """One line of the layout: where a row should be, not what's drawn (that's Widget)."""

    what: str  # "section", "part", or "empty" (placeholder in an empty open collection)
    collection: (
        Collection | None
    )  # the section's collection, or the part's (None: loose)
    part: str | None = None
    top: float = 0.0  # distance from the top of the list, px, scroll not applied
    h: float = ROW_H

    @property
    def key(self) -> tuple:
        """Identity across layouts: a part keeps its widget when it moves between collections."""
        return (self.what, self.part if self.what == "part" else self.collection)

    @property
    def indent(self) -> float:
        return INDENT if self.what != "section" and self.collection is not None else 0


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


class Layer:
    """(background, foreground, text) groups: one set per kind of row, one for dragged rows."""

    def __init__(self, order: int, parent: pyglet.graphics.Group | None) -> None:
        self.bg = pyglet.graphics.Group(order=order, parent=parent)
        self.fg = pyglet.graphics.Group(order=order + 1, parent=parent)
        self.text = pyglet.graphics.Group(order=order + 2, parent=parent)


class Widget:
    """What's drawn for one row, plus its animated state."""

    def __init__(self, picker: PartPicker, row: Row, layer: Layer) -> None:
        self.p, self.row, self.what = picker, row, row.what
        self.top = self.target_top = row.top
        self.indent = self.target_indent = row.indent
        self.alpha, self.target_alpha = 0.0, 1.0
        self.dx = 0.0  # horizontal offset, eases back to 0 after a drop
        self.angle = self.target_angle = (
            0.0  # section triangle: 0 points right, 90 down
        )
        self.leaving = False  # fading out; deleted once invisible
        self.hovered = False
        self.pin: tuple | None = None  # while dragged: (x, y) at press, cursor at press
        self.shown_alpha = -1
        b = picker.batch
        base = self._base_color()
        self.color = list(base)
        self.layer = layer
        self.bg = shapes.Rectangle(
            0, 0, PANEL_W - 1, row.h, color=base, batch=b, group=layer.bg
        )
        self.shapes: list = [self.bg]
        self.labels: list[pyglet.text.Label] = []
        self.field: Box | None = None  # rename text field
        self.caret: shapes.Rectangle | None = None

        def label(text, color=T.PART_TEXT, size=FONT_SIZE, anchor_x="left"):
            lb = pyglet.text.Label(
                text,
                font_name=FONT,
                font_size=size,
                color=color,
                anchor_x=anchor_x,
                anchor_y="center",
                batch=b,
                group=layer.text,
            )
            self.labels.append(lb)
            return lb

        if self.what == "part":
            fill, border = picker.swatch(row.part)
            self.chip_border = shapes.Rectangle(
                0, 0, SWATCH, SWATCH, color=border, batch=b, group=layer.fg
            )
            self.chip = shapes.Rectangle(
                0,
                0,
                SWATCH - 2 * S,
                SWATCH - 2 * S,
                color=fill,
                batch=b,
                group=layer.fg,
            )
            self.shapes += [self.chip_border, self.chip]
            self.name = label(picker.name_of(row.part))
        elif self.what == "section":
            self.angle = self.target_angle = 90.0 if row.collection.open else 0.0
            self.tri = shapes.Triangle(
                0, 0, 0, 0, 0, 0, color=T.HELP_TEXT[:3], batch=b, group=layer.fg
            )
            self.shapes.append(self.tri)
            self.name = label("")
            self.count = label(
                "", color=T.PICKER_DIM_TEXT, size=SMALL_SIZE, anchor_x="right"
            )
        else:
            self.name = label(
                "drop parts here", color=T.PICKER_DIM_TEXT, size=SMALL_SIZE
            )
        self.refresh()

    def _base_color(self) -> tuple:
        return T.PICKER_SECTION if self.what == "section" else T.PICKER_BG

    def set_layer(self, layer: Layer) -> None:
        """Move to other draw groups (lifting a dragged row above the rest, and back)."""
        self.layer = layer
        self.bg.group = layer.bg
        for s in self.shapes[1:]:
            s.group = layer.fg
        for lb in self.labels:
            lb.group = layer.text
        if self.field is not None:
            for r in (self.field.fill, *self.field.edges):
                r.group = layer.fg
            self.caret.group = layer.text

    def refresh(self) -> None:
        """Bring text + rename field up to date with the row's data."""
        if (
            self.what == "part"
        ):  # greyed out while it can't be placed (see PartPicker.disabled)
            name = self.p.name_of(self.row.part)  # (a macro can be renamed)
            if self.name.text != name:
                self.name.text = name
            color = T.PICKER_DIM_TEXT if self.p.disabled(self.row.part) else T.PART_TEXT
            if tuple(self.name.color) != tuple(color):
                self.name.color = color
                self.shown_alpha = -1  # the color reset the opacity: re-apply it
        if self.what != "section":
            return
        p, c = self.p, self.row.collection
        self.target_angle = 90.0 if c.open else 0.0
        self.count.text = str(len(c.parts))
        if c is p.renaming:
            text = p.edit.text
            self.name.text = text or c.name  # empty: the default name as a placeholder
            self.name.color = T.PART_TEXT if text else T.PICKER_DIM_TEXT
            if self.field is None:
                self.field = Box(
                    PANEL_W - self._name_x() - PAD + 4 * S,
                    self.row.h - 6 * S,
                    round(S),
                    T.PICKER_BG,
                    T.SELECT,
                    p.batch,
                    self.layer.fg,
                )
                self.caret = shapes.Rectangle(
                    0,
                    0,
                    1.5 * S,
                    16 * S,
                    color=T.CARET,
                    batch=p.batch,
                    group=self.layer.text,
                )
            self.count.visible = False
        else:
            self.name.text = p.fit(c.name, PANEL_W - self._name_x() - 24 * S - PAD)
            self.name.color = T.PART_TEXT
            self.count.visible = True
            if self.field is not None:
                self.field.delete()
                self.caret.delete()
                self.field = self.caret = None
        self.shown_alpha = -1  # label colors were reset: re-apply the opacity

    @staticmethod
    def _name_x() -> float:
        return PAD + 16 * S

    def tick(self, k: float) -> bool:
        """Ease toward the targets. Returns whether anything is still moving."""
        before = (self.top, self.indent, self.alpha, self.dx, self.angle, *self.color)
        self.top = approach(self.top, self.target_top, k, 0.3)
        self.indent = approach(self.indent, self.target_indent, k, 0.3)
        self.alpha = approach(self.alpha, self.target_alpha, k, 0.01)
        self.dx = approach(self.dx, 0.0, k, 0.3)
        self.angle = approach(self.angle, self.target_angle, k, 0.5)
        want = (
            T.PICKER_LIFT
            if self.pin
            else T.PICKER_HOVER
            if self.hovered
            else self._base_color()
        )
        self.color = [approach(c, w, k, 1.0) for c, w in zip(self.color, want)]
        return before != (
            self.top,
            self.indent,
            self.alpha,
            self.dx,
            self.angle,
            *self.color,
        )

    def screen_pos(self, cursor: tuple[float, float]) -> tuple[float, float]:
        """Bottom-left corner on screen."""
        if self.pin is not None:
            x0, y0, px, py = self.pin
            return x0 + cursor[0] - px, y0 + cursor[1] - py
        p = self.p
        return p.x_off + self.dx, p.list_top - self.top - self.row.h + p.scroll

    def place(self, cursor: tuple[float, float]) -> None:
        x, y = self.screen_pos(cursor)
        cy = y + self.row.h / 2
        self.bg.position = (x, y)
        self.bg.color = tuple(round(c) for c in self.color)
        if self.what == "part":
            cx = x + PAD + self.indent
            self.chip_border.position = (cx, cy - SWATCH / 2)
            self.chip.position = (cx + S, cy - SWATCH / 2 + S)
            self.name.position = (cx + SWATCH + 8 * S, cy, 0)
        elif self.what == "section":
            tx, r = x + PAD + 6 * S, 5 * S
            a = -math.radians(self.angle)  # clockwise: right -> down
            (x1, y1), (x2, y2), (x3, y3) = [
                (
                    tx + px * math.cos(a) - py * math.sin(a),
                    cy + px * math.sin(a) + py * math.cos(a),
                )
                for px, py in ((r, 0), (-0.6 * r, r), (-0.6 * r, -r))
            ]
            (
                self.tri.x,
                self.tri.y,
                self.tri.x2,
                self.tri.y2,
                self.tri.x3,
                self.tri.y3,
            ) = x1, y1, x2, y2, x3, y3
            nx = x + self._name_x()
            self.name.position = (nx, cy, 0)
            self.count.position = (x + PANEL_W - PAD - 2 * S, cy, 0)
            if self.field is not None:
                self.field.position = (nx - 4 * S, y + 3 * S)
                self.p.measure.text = self.p.edit.text[: self.p.edit.caret]
                self.caret.position = (nx + self.p.measure.content_width, cy - 8 * S)
                self.caret.visible = self.p.caret_on
        else:
            self.name.position = (x + PAD + self.indent, cy, 0)
        a = round(255 * self.alpha)
        if a != self.shown_alpha:
            self.shown_alpha = a
            for s in self.shapes:
                s.opacity = a
            for lb in self.labels:
                lb.opacity = a

    def delete(self) -> None:
        for s in self.shapes:
            s.delete()
        for lb in self.labels:
            lb.delete()
        if self.field is not None:
            self.field.delete()
            self.caret.delete()


class PartPicker:
    def __init__(
        self,
        library: Library,
        batch: pyglet.graphics.Batch,
        win_h: int,
        pixel_ratio: float = 1.0,
        swatch: Callable[[str], tuple] = lambda part: T.PART_BODY,
        name_of: Callable[[str], str] = lambda part: part,
        disabled: Callable[[str], bool] = lambda part: False,
    ) -> None:
        self.lib = library
        self.disabled = disabled  # part -> shown greyed out (can't be placed right now); refresh() re-asks
        self.swatch = swatch  # part -> (fill, border) of the little color chip in front of its name
        self.name_of = name_of  # part -> the text shown for it
        self.batch = batch
        self.open = True
        self.open_t = 1.0  # animated: 1 = fully out, 0 = tucked away
        self.win_h = win_h
        self.pixel_ratio = (
            pixel_ratio  # framebuffer pixels per window pixel (HiDPI), for the clip
        )
        self.scroll = self.scroll_target = 0.0  # how far the list is scrolled up, px
        self.rows: list[
            Row
        ] = []  # the current layout (during a drag: as if already dropped)
        self.loose_top = 0.0
        self.content_h = 0.0
        self.widgets: dict[tuple, Widget] = {}
        self.cursor = (0.0, 0.0)
        self.hovered: Row | str | None = None
        # dragging
        self.dragging: Row | None = None
        self.drop: tuple | None = (
            None  # parts: (collection or None, index or None = end); collections: (index,)
        )
        self.grab_dy = 0.0  # grabbed row's center minus the cursor, screen px
        self._drop_c: Collection | None = None
        # renaming
        self.renaming: Collection | None = None
        self.edit: LineEdit | None = None
        self.caret_on, self._blink_t = True, 0.0

        self.clip = ClipGroup(order=3)
        self.part_layer = Layer(0, self.clip)
        self.section_layer = Layer(
            3, self.clip
        )  # above parts: collapsing rows slide under their header
        self.marker_group = pyglet.graphics.Group(order=6, parent=self.clip)
        self.panel_group = pyglet.graphics.Group(order=2)
        self.header_bg = pyglet.graphics.Group(order=5)
        self.header_fg = pyglet.graphics.Group(order=6)
        self.drag_layer = Layer(
            0, pyglet.graphics.Group(order=9)
        )  # not clipped: can leave the panel

        self.measure = pyglet.text.Label("", font_name=FONT, font_size=FONT_SIZE)
        self.divider = shapes.Rectangle(
            0,
            0,
            PANEL_W - 2 * PAD,
            max(1, round(S / 2)),
            color=T.PICKER_BORDER,
            batch=batch,
            group=self.part_layer.fg,
        )
        self.divider_top = self.divider_target = 0.0
        self.drop_box = shapes.Box(
            0,
            0,
            PANEL_W - 4 * S,
            SECTION_H - 2 * S,
            thickness=2 * S,
            color=T.SELECT,
            batch=batch,
            group=self.marker_group,
        )
        self.drop_box.visible = False
        self.chrome: list = []  # panel + header: (shape, base x), slid by x_off
        self.chevron: list[
            shapes.Line
        ] = []  # the toggle's «, turning into » as the panel tucks away
        self.buttons: dict[str, tuple[float, float, float, float]] = {}
        self.button_bgs: dict[str, shapes.Rectangle] = {}
        self._build_chrome()
        self.refresh()
        for w in self.widgets.values():  # the first layout doesn't animate in
            w.alpha, w.top = 1.0, w.target_top
        self.divider_top = self.divider_target
        self._place_all()

    # ---- geometry ------------------------------------------------------------

    @property
    def x_off(self) -> float:
        """How far the panel is slid out to the left: 0 = fully open. Tucked away, only
        its right edge is still on screen (COLLAPSED_W, the toggle button's column)."""
        return -(1 - self.open_t) * (PANEL_W - COLLAPSED_W)

    @property
    def width(self) -> float:
        """How much of the window's left edge the picker covers right now."""
        return PANEL_W + self.x_off

    @property
    def list_top(self) -> float:
        """Screen y of the top of the list area (the bottom of the header)."""
        return self.win_h - HEADER_H

    def contains(self, sx: float, sy: float) -> bool:
        return 0 <= sx < self.width and 0 <= sy <= self.win_h

    def to_list(self, sy: float) -> float:
        """Screen y -> distance from the list top."""
        return self.list_top - sy + self.scroll

    def row_at(self, sy: float) -> Row | None:
        if sy >= self.list_top:
            return None
        ly = self.to_list(sy)
        return next((r for r in self.rows if r.top <= ly < r.top + r.h), None)

    def hit(self, sx: float, sy: float) -> Row | str | None:
        """What's under a screen point: a Row, "toggle" / "new" (header buttons),
        "header", "blank" (list space without a row), or None (not over the picker)."""
        if not self.contains(sx, sy):
            return None
        if not self.open:
            return "toggle"  # the whole tucked-away edge opens it
        if sy >= self.list_top:
            for name, (x, y, w, h) in self.buttons.items():
                if x + self.x_off <= sx <= x + self.x_off + w and y <= sy <= y + h:
                    return name
            return "header"
        return self.row_at(sy) or "blank"

    def fit(self, text: str, max_w: float) -> str:
        """Shorten text with an ellipsis until it fits in max_w pixels."""
        self.measure.text = text
        while text and self.measure.content_width > max_w:
            text = text[:-1]
            self.measure.text = text + "…"
        return self.measure.text

    # ---- state changes ---------------------------------------------------------

    def toggle(self) -> None:
        self.open = not self.open
        self.set_hover(None)

    def resize(self, win_h: int, pixel_ratio: float) -> None:
        self.win_h, self.pixel_ratio = win_h, pixel_ratio
        self._build_chrome()
        self.refresh()
        self._place_all()

    def set_library(self, library: Library) -> None:
        """Show another library (another project's); no animation from the old one."""
        for w in self.widgets.values():
            w.delete()
        self.widgets.clear()
        self.lib, self.hovered = library, None
        self.scroll = self.scroll_target = 0.0
        self.refresh()
        for w in self.widgets.values():
            w.alpha, w.top = 1.0, w.target_top
        self.divider_top = self.divider_target
        self._place_all()

    def scroll_by(self, clicks: float) -> None:
        self.scroll_target = self._clamp_scroll(
            self.scroll_target - clicks * SCROLL_STEP
        )

    def _clamp_scroll(self, s: float) -> float:
        return max(0.0, min(s, self.content_h - self.list_top))

    def set_hover(self, target: Row | str | None) -> None:
        key = target.key if isinstance(target, Row) else target
        for k, w in self.widgets.items():
            w.hovered = k == key
        for name, bg in self.button_bgs.items():
            bg.color = T.PICKER_HOVER if name == key else T.PICKER_HEADER
        self.hovered = target

    def set_all_open(self, open_: bool) -> None:
        for c in self.lib.collections:
            c.open = open_
        self.refresh()

    # ---- renaming a collection ----------------------------------------------------

    def start_rename(self, c: Collection, fresh: bool = False) -> None:
        """Type a new name for `c`. `fresh`: it was just made, so start empty (its
        default name shows as a placeholder, and stays if nothing is typed)."""
        self.renaming, self.edit = c, LineEdit("" if fresh else c.name, NAME_MAX)
        self.open = True
        self.caret_on, self._blink_t = True, 0.0
        self.refresh()
        row = next(r for r in self.rows if r.collection is c)
        if row.top < self.scroll_target:
            self.scroll_target = row.top
        elif row.top + row.h > self.scroll_target + self.list_top:
            self.scroll_target = row.top + row.h - self.list_top

    def rename_text(self, text: str) -> None:
        self.edit.insert(
            text.upper()
        )  # collection names are all caps, like the built-in ones
        self._rename_changed()

    def rename_motion(self, motion: int) -> None:
        self.edit.motion(motion)
        self._rename_changed()

    def _rename_changed(self) -> None:
        self.caret_on, self._blink_t = True, 0.0  # keep the caret visible while typing
        self.widgets[("section", self.renaming)].refresh()
        self._place_all()

    def finish_rename(self, commit: bool) -> None:
        if commit:
            self.lib.rename(self.renaming, self.edit.text)
        self.renaming, self.edit = None, None
        self.refresh()
        self._place_all()

    # ---- drag and drop -------------------------------------------------------------

    def _carried(self, row: Row) -> list[Widget]:
        """The widgets that travel with a dragged row: a collection brings its visible rows."""
        if row.what == "part":
            return [self.widgets[row.key]]
        return [
            w
            for w in self.widgets.values()
            if not w.leaving and w.row.collection is row.collection
        ]

    def begin_drag(self, row: Row, press: tuple[float, float]) -> None:
        self.dragging = row
        if row.what == "part":
            where = row.collection.parts if row.collection else self.lib.loose
            self.drop = (row.collection, where.index(row.part))  # right where it was
        else:
            self.drop = (self.lib.collections.index(row.collection),)
        self.grab_dy = (self.list_top - row.top - row.h / 2 + self.scroll) - press[1]
        self.cursor = press
        for w in self._carried(row):
            x, y = w.screen_pos(press)
            w.pin = (x, y, *press)
            w.set_layer(self.drag_layer)
        self.set_hover(None)
        self.refresh()

    def drag_to(self, sx: float, sy: float) -> None:
        self.cursor = (sx, sy)
        drop = self._drop_at(sy + self.grab_dy)
        if drop != self.drop:
            self.drop = drop
            self.refresh()
        self._place_all()

    def end_drag(self) -> None:
        """Drop the dragged row where the gap is."""
        row, drop = self.dragging, self.drop
        if row.what == "part":
            self.lib.move_part(row.part, *drop)
        else:
            self.lib.move_collection(row.collection, drop[0])
        self.cancel_drag()

    def cancel_drag(self) -> None:
        """Let go: the carried rows ease from wherever they are into their (new) places."""
        for w in self._carried(self.dragging):
            x, y = w.screen_pos(self.cursor)
            w.pin = None
            w.top = self.list_top - y - w.row.h + self.scroll
            w.dx = x - self.x_off
            w.set_layer(self.section_layer if w.what == "section" else self.part_layer)
        self.dragging, self.drop = None, None
        self.refresh()

    def _drop_at(self, sy: float) -> tuple:
        """Where the drag lands if the grabbed row's center is at screen y `sy`. Decided
        against the current layout, which already has the gap in it: hovering the gap
        keeps the drop where it is, so the rows don't flicker back and forth."""
        ly = self.to_list(min(sy, self.list_top - 1))
        drag = self.dragging
        if drag.what == "part":
            row = next((r for r in self.rows if r.top <= ly < r.top + r.h), None)
            if row is None:
                if self.rows and ly < self.rows[0].top:
                    return self.drop
                return (
                    (None, 0) if ly < self.loose_top else (None, None)
                )  # just above / below the loose parts
            if row.key == drag.key:
                return self.drop
            if (
                row.what == "section"
            ):  # on a header: first in an open collection, into a closed one
                return (
                    (row.collection, 0)
                    if row.collection.open
                    else (row.collection, None)
                )
            if row.what == "empty":
                return row.collection, 0
            rest = [
                p
                for p in (row.collection.parts if row.collection else self.lib.loose)
                if p != drag.part
            ]
            return row.collection, rest.index(row.part) + (ly > row.top + row.h / 2)
        # a collection: before / after the block (section + its rows) under the grabbed header
        rest = [c for c in self.lib.collections if c is not drag.collection]
        for c, top, bottom in self._blocks():
            if top <= ly < bottom:
                if c is drag.collection:
                    return self.drop
                return (rest.index(c) + (ly > (top + bottom) / 2),)
        return (0,) if self.rows and ly < self.rows[0].top else (len(rest),)

    def _blocks(self) -> list[tuple[Collection, float, float]]:
        """(collection, top, bottom) of each section plus its rows, in list px."""
        blocks = []
        for r in self.rows:
            if r.what == "section":
                blocks.append([r.collection, r.top, r.top + r.h])
            elif r.collection is not None:
                blocks[-1][2] = r.top + r.h
        return [tuple(b) for b in blocks]

    # ---- layout --------------------------------------------------------------------

    def _order(self) -> tuple[list[tuple[Collection, list[str]]], list[str]]:
        """The library's order (collections + their parts, loose parts); during a
        drag, as if it had already been dropped."""
        cols = [(c, list(c.parts)) for c in self.lib.collections]
        loose = list(self.lib.loose)
        d = self.dragging
        if d is not None and d.what == "part":
            dest, index = self.drop
            for _, parts in cols:
                if d.part in parts:
                    parts.remove(d.part)
            if d.part in loose:
                loose.remove(d.part)
            target = (
                loose if dest is None else next(parts for c, parts in cols if c is dest)
            )
            target.insert(len(target) if index is None else index, d.part)
        elif d is not None:
            entry = next(e for e in cols if e[0] is d.collection)
            cols.remove(entry)
            cols.insert(self.drop[0], entry)
        return cols, loose

    def _layout(self) -> list[Row]:
        cols, loose = self._order()
        rows, top = [], 4 * S
        for c, parts in cols:
            rows.append(Row("section", c, top=top, h=SECTION_H))
            top += SECTION_H
            if c.open:
                if not parts:
                    rows.append(Row("empty", c, top=top))
                    top += ROW_H
                for p in parts:
                    rows.append(Row("part", c, p, top))
                    top += ROW_H
        top += LOOSE_GAP
        self.loose_top = top
        for p in loose:
            rows.append(Row("part", None, p, top))
            top += ROW_H
        self.content_h = top + TAIL
        return rows

    def refresh(self) -> None:
        """Recompute the layout after any change; widgets then ease into it."""
        self.rows = self._layout()
        self.scroll_target = self._clamp_scroll(self.scroll_target)
        section_top = {r.collection: r.top for r in self.rows if r.what == "section"}
        carried = (
            {w.row.key for w in self._carried(self.dragging)}
            if self.dragging
            else set()
        )
        seen = set()
        for row in self.rows:
            seen.add(row.key)
            w = self.widgets.get(row.key)
            if w is None:
                w = Widget(
                    self,
                    row,
                    self.section_layer if row.what == "section" else self.part_layer,
                )
                # new rows grow out from under their collection's header (expanding), else fade in place
                w.top = (
                    section_top.get(row.collection, row.top)
                    if row.what != "section"
                    else row.top
                )
                self.widgets[row.key] = w
            w.row, w.leaving = row, False
            w.target_top, w.target_indent, w.target_alpha = row.top, row.indent, 1.0
            w.refresh()
        for key, w in self.widgets.items():
            if key in seen or key in carried:
                continue
            # gone from the layout: slide under the header of the collection it's in now
            # (collapsed, or dropped into a closed one), or just fade where it is
            w.leaving, w.target_alpha = True, 0.0
            owner = self.lib.where(w.row.part) if w.what == "part" else w.row.collection
            if w.what != "section" and owner in section_top:
                w.target_top = section_top[owner]
        self.divider_target = self.loose_top - LOOSE_GAP / 2
        self.divider.visible = bool(self.lib.loose and self.lib.collections)
        drag = self.dragging
        self._drop_c = (
            self.drop[0]
            if drag and drag.what == "part" and self.drop[1] is None
            else None
        )
        self.drop_box.visible = self._drop_c is not None and not self._drop_c.open

    # ---- animation + drawing ----------------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the tweens one frame."""
        k = 1 - math.exp(-SPEED * dt)
        moving = False
        if self.renaming is not None:
            self._blink_t += dt
            if self._blink_t >= 0.5:
                self._blink_t, self.caret_on = 0.0, not self.caret_on
                moving = True
        if self.dragging is not None and self.contains(
            *self.cursor
        ):  # near an edge: scroll along
            sy = self.cursor[1]
            if sy > self.list_top - EDGE_SCROLL:
                self.scroll_target = self._clamp_scroll(
                    self.scroll_target - EDGE_SPEED * dt
                )
            elif sy < EDGE_SCROLL:
                self.scroll_target = self._clamp_scroll(
                    self.scroll_target + EDGE_SPEED * dt
                )
        old = (self.open_t, self.scroll, self.divider_top)
        self.open_t = approach(self.open_t, 1.0 if self.open else 0.0, k, 0.002)
        self.scroll = approach(self.scroll, self.scroll_target, k, 0.3)
        self.divider_top = approach(self.divider_top, self.divider_target, k, 0.3)
        moving |= old != (self.open_t, self.scroll, self.divider_top)
        if old[1] != self.scroll and self.dragging is not None:
            self.drag_to(*self.cursor)  # the list moved under the cursor
        for key, w in list(self.widgets.items()):
            moving |= w.tick(k)
            if w.leaving and w.alpha <= 0.01:
                w.delete()
                del self.widgets[key]
        if moving:
            self._place_all()

    def _place_all(self) -> None:
        r = self.pixel_ratio
        x_off = self.x_off
        # The rows disappear under the edge that stays behind: it's empty once tucked away.
        right = self.width - (1 - self.open_t) * COLLAPSED_W
        self.clip.rect = (0, 0, max(0, int(right * r)), int(max(self.list_top, 0) * r))
        for shape, base_x in self.chrome:
            shape.x = base_x + x_off
        self._place_chevron()
        for w in self.widgets.values():
            w.place(self.cursor)
        self.divider.position = (
            x_off + PAD,
            self.list_top - self.divider_top + self.scroll,
        )
        w = self.widgets.get(("section", self._drop_c))
        if self.drop_box.visible and w is not None:
            x, y = w.screen_pos(self.cursor)
            self.drop_box.position = (x + 2 * S, y + S)

    def _build_chrome(self) -> None:
        """Panel background, and the header with its buttons."""
        for s, _ in self.chrome:
            s.delete()
        self.chrome, self.buttons, self.button_bgs = [], {}, {}
        b, h = self.batch, self.win_h

        def rect(x, y, w, hh, color, group, chrome=True):
            s = shapes.Rectangle(x, y, w, hh, color=color, batch=b, group=group)
            if chrome:
                self.chrome.append((s, x))
            return s

        def label(text, x, y, group, color=T.PART_TEXT, chrome=True, anchor_x="left"):
            lb = pyglet.text.Label(
                text,
                font_name=FONT,
                font_size=FONT_SIZE,
                color=color,
                x=x,
                y=y,
                anchor_x=anchor_x,
                anchor_y="center",
                batch=b,
                group=group,
            )
            if chrome:
                self.chrome.append((lb, x))
            return lb

        line = max(1, round(S / 2))
        rect(0, 0, PANEL_W, h, T.PICKER_BG, self.panel_group)
        rect(0, h - HEADER_H, PANEL_W, HEADER_H, T.PICKER_HEADER, self.header_bg)
        rect(0, h - HEADER_H, PANEL_W, line, T.PICKER_BORDER, self.header_fg)
        rect(PANEL_W - line, 0, line, h, T.PICKER_BORDER, self.header_fg)
        label("PARTS", PAD + 2 * S, h - HEADER_H / 2, self.header_fg, color=T.HELP_TEXT)
        by = h - HEADER_H + (HEADER_H - BUTTON) / 2
        for i, name in enumerate(("toggle", "new")):
            bx = PANEL_W - BUTTON_GAP - (i + 1) * BUTTON - i * 2 * S
            self.buttons[name] = (bx, by, BUTTON, BUTTON)
            self.button_bgs[name] = rect(
                bx, by, BUTTON, BUTTON, T.PICKER_HEADER, self.header_bg
            )
        label("+", *self._button_center("new"), self.header_fg, anchor_x="center")
        # The toggle's « is drawn, not typed, so it can turn around its own center (see _place_all).
        for s in self.chevron:
            s.delete()
        self.chevron = [
            shapes.Line(
                0,
                0,
                0,
                0,
                thickness=1.25 * S,
                color=T.PART_TEXT[:3],
                batch=b,
                group=self.header_fg,
            )
            for _ in range(4)
        ]

    def _button_center(self, name: str) -> tuple[float, float]:
        """A header button's center, with the panel fully open."""
        x, y, w, h = self.buttons[name]
        return x + w / 2, y + h / 2

    def _place_chevron(self) -> None:
        """The toggle's «: pointing left when open, turned half a turn (») once tucked away."""
        cx, cy = self._button_center("toggle")
        cx += self.x_off
        a = math.pi * (1 - self.open_t)
        ca, sa = math.cos(a), math.sin(a)
        arm_x, arm_y, apart = (
            2.5 * S,
            4 * S,
            5 * S,
        )  # each chevron: < with its tip at -arm_x
        lines = iter(self.chevron)
        for dx in (-apart / 2, apart / 2):
            tip = (dx - arm_x, 0.0)
            for end in ((dx + arm_x, arm_y), (dx + arm_x, -arm_y)):
                line = next(lines)
                (line.x, line.y), (line.x2, line.y2) = (
                    (cx + px * ca - py * sa, cy + px * sa + py * ca)
                    for px, py in (tip, end)
                )
