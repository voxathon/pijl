"""Clicking the carried part in the picker again stacks another below it: a column,
spaced like Ctrl+D's, that Ctrl+scroll spaces out and that's a Ctrl+D block once
placed. Needs a GL window; skipped where one can't be made."""

import os

import pytest
import itertools


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    os.environ["PIJL_DATA"] = str(tmp_path_factory.mktemp("pijl-data"))
    try:
        from pijl.ui.editor import Editor

        editor = Editor()
    except Exception as e:  # (no display / GL)
        pytest.skip(f"no editor window here: {e}")
    editor._enable_event_queue = False  # dispatch synthetic events right away
    yield editor
    editor.close()


@pytest.fixture
def fresh(ed):
    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    ed.tiling = None
    ed.last_picker_click = (None, 0.0)  # (a click from the last test would be a bounce)
    if not ed.picker.open:
        ed.picker.toggle()
    return ed


def row_point(ed, kind: str):
    """A screen point on the picker row for `kind`."""
    p = ed.picker
    row = next(r for r in p.rows if r.what == "part" and r.part == kind)
    sy = p.list_top - (row.top + row.h / 2) + p.scroll
    sx = next(x for x in range(0, ed.width, 2) if p.hit(x, sy) is row)
    return sx, sy


def click(ed, x, y, modifiers=0):
    from pyglet.window import mouse

    ed.dispatch_event("on_mouse_press", x, y, mouse.LEFT, modifiers)
    ed.dispatch_event("on_mouse_release", x, y, mouse.LEFT, modifiers)


def ctrl_scroll(ed, notches: int):
    from pyglet.window import key

    ed.keys.data[key.LCTRL] = True
    try:
        for _ in range(abs(notches)):
            x, y = ed.mouse
            ed.dispatch_event("on_mouse_scroll", x, y, 0, 1 if notches > 0 else -1)
    finally:
        ed.keys.data[key.LCTRL] = False


def board_point(ed):
    return ed.width * 0.6, ed.height * 0.6


def column_gaps(views):
    ys = sorted((v.y for v in views), reverse=True)
    return {round(a - b, 6) for a, b in itertools.pairwise(ys)}


def carry_column(ed, kind: str, count: int):
    from pyglet.window import mouse

    px, py = row_point(ed, kind)
    for _ in range(count):
        click(ed, px, py)
    bx, by = board_point(ed)
    ed.dispatch_event("on_mouse_motion", bx, by, 0, 0)
    return mouse


def test_clicking_the_carried_part_stacks_a_column(fresh):
    from pijl.ui import theme as T
    from pijl.ui.editor import Mode

    ed = fresh
    carry_column(ed, "NOT", 4)
    assert ed.mode is Mode.PLACING_PART
    views = ed.placing_views
    assert len(views) == 4 and all(v.part.kind == "NOT" for v in views)
    assert len({v.x for v in views}) == 1  # a straight line...
    assert all(b.y < a.y for a, b in itertools.pairwise(views))  # ...going down
    t = ed.column
    step = (t.size[1] + t.gap[1]) * T.GRID
    assert column_gaps(views) == {round(step, 6)}  # Ctrl+D's spacing

    # the top part stays where the cursor holds it as the column grows
    top = views[0]
    held = (top.x + ed.drag_delta[0], top.y + ed.drag_delta[1])
    click(ed, *row_point(ed, "NOT"))
    bx, by = board_point(ed)
    ed.dispatch_event("on_mouse_motion", bx, by, 0, 0)
    assert len(ed.placing_views) == 5 and ed.placing_views[0] is top
    assert (top.x + ed.drag_delta[0], top.y + ed.drag_delta[1]) == held

    click(ed, bx, by)
    assert ed.mode is Mode.IDLE
    placed = list(ed.part_views.values())
    assert len(placed) == 5 and all(v.table.opacity[v.row] == 255 for v in placed)
    assert (top.x, top.y) == held
    assert len({v.x for v in placed}) == 1
    assert column_gaps(placed) == {round(step, 6)}


def test_ctrl_scroll_spaces_the_carried_column(fresh):
    from pijl.ui import theme as T

    ed = fresh
    carry_column(ed, "NOT", 3)
    t = ed.column
    gap = t.gap[1]
    ctrl_scroll(ed, 2)
    assert t.gap[1] == gap + 2
    assert column_gaps(ed.placing_views) == {round((t.size[1] + gap + 2) * T.GRID, 6)}
    ctrl_scroll(ed, -50)
    assert t.gap[1] == 1  # MIN_GAP
    assert column_gaps(ed.placing_views) == {round((t.size[1] + 1) * T.GRID, 6)}


def test_a_single_part_ctrl_scroll_still_zooms(fresh):
    ed = fresh
    carry_column(ed, "NOT", 1)
    zoom = ed.camera.zoom
    ctrl_scroll(ed, 1)
    assert ed.camera.zoom != zoom


def test_placed_column_is_one_undo_step_and_a_ctrl_d_block(fresh):
    from pyglet.window import key

    from pijl.ui import theme as T

    ed = fresh
    carry_column(ed, "NOT", 3)
    steps = len(ed.history.undo_stack)
    click(ed, *board_point(ed))
    assert len(ed.part_views) == 3
    assert len(ed.history.undo_stack) == steps + 1
    assert len(ed.selection.parts) == 3 and ed._tiling_active()

    # Ctrl+scroll after placing: still spaces it, as one more undo step
    t = ed.tiling
    gap = t.gap[1]
    ctrl_scroll(ed, 3)
    assert t.gap[1] == gap + 3
    views = list(ed.part_views.values())
    assert column_gaps(views) == {round((t.size[1] + gap + 3) * T.GRID, 6)}
    assert len(ed.history.undo_stack) == steps + 2

    # Ctrl+D doubles it to the right: 2 columns of 3
    ed.dispatch_event("on_key_press", key.D, key.MOD_CTRL)
    views = list(ed.part_views.values())
    assert len(views) == 6
    assert len({v.x for v in views}) == 2 and len({v.y for v in views}) == 3

    ed.dispatch_event("on_key_press", key.Z, key.MOD_CTRL)
    ed.dispatch_event("on_key_press", key.Z, key.MOD_CTRL)
    ed.dispatch_event("on_key_press", key.Z, key.MOD_CTRL)
    assert not ed.part_views


def test_shift_click_keeps_another_column_like_it(fresh):
    from pyglet.window import key

    from pijl.ui.editor import Mode

    ed = fresh
    carry_column(ed, "NOT", 3)
    ctrl_scroll(ed, 1)
    gap = ed.column.gap[1]
    click(ed, *board_point(ed), modifiers=key.MOD_SHIFT)
    assert ed.mode is Mode.PLACING_PART
    assert len(ed.placing_views) == 3 and ed.column.gap[1] == gap
    assert len(ed.part_views) == 6  # 3 placed + 3 ghosts


def test_esc_drops_the_whole_column_and_another_part_swaps(fresh):
    from pyglet.window import key

    from pijl.ui.editor import Mode

    ed = fresh
    carry_column(ed, "NOT", 4)
    ed.dispatch_event("on_key_press", key.ESCAPE, 0)
    assert ed.mode is Mode.IDLE and not ed.part_views and ed.column is None

    carry_column(ed, "NOT", 3)
    other = next(r.part for r in ed.picker.rows if r.what == "part" and r.part != "NOT")
    click(ed, *row_point(ed, other))
    assert ed.mode is Mode.PLACING_PART
    assert len(ed.part_views) == 1 and ed.placing_views[0].part.kind == other
