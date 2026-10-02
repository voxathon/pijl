"""The picker's search field (Ctrl+F, type, Enter / Esc, its "x") and its refresh
button. Needs GL (a hidden window, see hidden_editor.py); skipped where there's none."""

import json

import pytest
from hidden_editor import hidden_editor


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    yield editor
    editor.close()


@pytest.fixture
def fresh(ed):
    from pijl.ui.editor import Mode

    ed._cancel()
    if ed.mode is Mode.IDLE:
        ed.picker.stop_search(clear=True)
    if not ed.picker.open:
        ed.picker.toggle()
    ed.text_mouse.forget()  # (a click from the last test would be a double-click)
    return ed


def key_press(ed, symbol, modifiers=0):
    ed.dispatch_event("on_key_press", symbol, modifiers)


def click(ed, x, y):
    from pyglet.window import mouse

    ed.dispatch_event("on_mouse_press", x, y, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_release", x, y, mouse.LEFT, 0)


def point_on(ed, what: str):
    """A screen point where the picker's hit() says `what`."""
    p = ed.picker
    for sy in range(int(p.list_top), ed.height, 2):
        for sx in range(0, int(p.width), 2):
            if p.hit(sx, sy) == what:
                return sx, sy
    raise AssertionError(f"no {what} on screen")


def shown_parts(ed) -> list[str]:
    return [r.part for r in ed.picker.rows if r.what == "part"]


def test_ctrl_f_filters_and_enter_picks_the_top_match(fresh):
    from pyglet.window import key

    from pijl.ui.editor import Mode

    ed = fresh
    everything = shown_parts(ed)
    key_press(ed, key.F, key.MOD_CTRL)
    assert ed.mode is Mode.SEARCHING
    ed.dispatch_event("on_text", "and")
    shown = shown_parts(ed)
    assert shown and len(shown) < len(everything)
    assert all("and" in ed._entry_title(p).casefold() for p in shown)
    key_press(ed, key.ENTER)
    assert ed.mode is Mode.PLACING_PART
    assert ed.placing_kind == shown[0]
    assert ed.picker.query.text == "and"  # the filter stays


def test_esc_clears_the_search(fresh):
    from pyglet.window import key

    from pijl.ui.editor import Mode

    ed = fresh
    everything = shown_parts(ed)
    click(ed, *point_on(ed, "search"))
    assert ed.mode is Mode.SEARCHING
    ed.dispatch_event("on_text", "zzzz")
    assert [r.what for r in ed.picker.rows] == ["none"]
    key_press(ed, key.ESCAPE)
    assert ed.mode is Mode.IDLE
    assert shown_parts(ed) == everything


def test_clicking_away_keeps_the_filter_and_x_clears_it(fresh):
    from pijl.ui.editor import Mode

    ed = fresh
    everything = shown_parts(ed)
    click(ed, *point_on(ed, "search"))
    ed.dispatch_event("on_text", "or")
    click(ed, ed.width * 0.6, ed.height * 0.6)  # the board
    assert ed.mode is Mode.IDLE
    assert ed.picker.filtering
    click(ed, *point_on(ed, "clear"))
    assert not ed.picker.filtering
    assert shown_parts(ed) == everything


def test_caret_starts_at_the_left_over_the_placeholder(fresh):
    ed = fresh
    ed.picker.measure.text = "something long"  # (what an emptied label still measures)
    click(ed, *point_on(ed, "search"))
    p = ed.picker
    assert p.search_text.text == "search"  # the placeholder
    assert p.search_cursor.caret.x == p.search_cursor.x0


def test_selecting_in_the_search_field(fresh):
    from pyglet.window import key

    ed = fresh
    p = ed.picker
    click(ed, *point_on(ed, "search"))
    ed.dispatch_event("on_text", "nand")
    ed.dispatch_event("on_text_motion_select", key.MOTION_LEFT)
    ed.dispatch_event("on_text_motion_select", key.MOTION_LEFT)
    assert p.query.selected_text == "nd"
    sel = p.search_cursor.sel
    assert sel.visible and sel.width > 0
    ed.dispatch_event("on_text", "x")
    assert p.query.text == "nax"
    key_press(ed, key.A, key.MOD_CTRL)
    assert p.query.selected_text == "nax"
    ed.dispatch_event("on_text", "or")
    assert p.query.text == "or"


def test_clicking_and_dragging_in_the_search_field(fresh):
    from pyglet.window import mouse

    ed = fresh
    p = ed.picker
    sx, sy = point_on(ed, "search")
    click(ed, sx, sy)
    ed.dispatch_event("on_text", "abcdef")
    ed.text_mouse.forget()  # (not a double-click)
    c = p.search_cursor
    at = lambda i: c.x0 + c.width("abcdef"[:i])
    ed.dispatch_event("on_mouse_press", at(1), sy, mouse.LEFT, 0)
    assert p.query.caret == 1
    ed.dispatch_event("on_mouse_drag", at(4), sy, 0, 0, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_release", at(4), sy, mouse.LEFT, 0)
    assert p.query.selected_text == "bcd"
    # a second click right after: everything
    ed.dispatch_event("on_mouse_press", at(2), sy, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_release", at(2), sy, mouse.LEFT, 0)
    assert p.query.selected_text == "abcdef"


def test_refresh_picks_up_macro_files_changed_on_disk(fresh):
    from pijl.snapshot import MACRO

    ed = fresh
    path = ed.store.folder / "outside.json"
    ed.store.folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"title": "Outside"}), encoding="utf-8")
    assert MACRO + "outside" not in shown_parts(ed)
    click(ed, *point_on(ed, "refresh"))
    assert MACRO + "outside" in shown_parts(ed)
    path.unlink()
    click(ed, *point_on(ed, "refresh"))
    assert MACRO + "outside" not in shown_parts(ed)
