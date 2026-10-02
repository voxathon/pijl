"""Dragging through a real editor: the drop is recorded as a move (a delta in the undo
history, not a copy of every part), and undo / redo put everything back exactly.
Needs GL (a hidden window, see hidden_editor.py); skipped where there's none."""

import numpy as np

import pytest
from hidden_editor import hidden_editor


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    editor._clear_board()
    editor._reset_history(None)
    yield editor
    editor.close()


def drag_all(ed, dx_px: float):
    from pyglet.window import key, mouse

    ed.dispatch_event("on_key_press", key.A, key.MOD_CTRL)
    ed.dispatch_event("on_key_release", key.A, key.MOD_CTRL)
    v = min(ed.selection.parts, key=lambda v: v.part.uid)
    sx, sy = ed.camera.world_to_screen(v.x + v.w / 2, v.y + v.h / 2)
    ed.dispatch_event("on_mouse_press", sx, sy, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_drag", sx + dx_px, sy, dx_px, 0, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_release", sx + dx_px, sy, mouse.LEFT, 0)


def board(ed):
    from pijl.ui.document import capture

    snap = capture(ed)
    return repr(sorted(snap.parts.items())), repr(sorted(snap.wires.items()))


def in_sync(ed):
    h = ed.history.current
    return board(ed) == (repr(sorted(h.parts.items())), repr(sorted(h.wires.items())))


def test_a_dragged_board_is_one_move_and_undoes_exactly(ed):
    from pyglet.window import key

    # off-grid floats, a bend and a junction: things a move has to carry exactly
    views = [ed.add_part("NOT", 100.5 + 120 * i, 33.25 * i) for i in range(12)]
    for a, b in zip(views, views[1:]):
        ed.connect(a.part.outputs[0], b.part.inputs[0], bends=[(a.x + 90.5, a.y + 7.0)])
    ed._record()
    drag_all(ed, 0)  # (picks them up and puts them back: make every x a float first)
    start = board(ed)
    steps = len(ed.history.undo_stack)

    drag_all(ed, 37)
    moved = board(ed)
    assert moved != start and in_sync(ed)
    assert len(ed.history.undo_stack) == steps + 1
    parts_section = ed.history.undo_stack[-1][0][0]
    assert parts_section[2], "the drop should be stored as a move"
    assert sum(len(u) for u, _, _ in parts_section[2]) + len(parts_section[1]) == 12

    ed.dispatch_event("on_key_press", key.Z, key.MOD_CTRL)
    assert board(ed) == start and in_sync(ed)
    ed.dispatch_event("on_key_press", key.Y, key.MOD_CTRL)
    assert board(ed) == moved and in_sync(ed)


def test_pin_tags_follow_tab_hover_and_drags(ed):
    from pyglet.window import key

    from pijl.ui.document import capture

    ed._clear_board()
    ed._reset_history(None)
    views = [ed.add_part("AND", 200.0 + 160 * i, 100.0) for i in range(4)]
    glyphs = ed.text.buf
    base = int(glyphs.used.sum())

    def tab_to(mode):
        while ed.pin_label_mode != mode:
            ed.dispatch_event("on_key_press", key.TAB, 0)

    tab_to(2)  # always
    # (made while tags are shown: they're placed as the part's are made)
    views.append(ed.add_part("OR", 200.0, 300.0))
    ed._record()
    base += len("OR")  # (its title)
    assert all(v.pin_labels_shown for v in views)
    with_tags = int(glyphs.used.sum())
    assert with_tags > base
    drag_all(ed, 23)  # tags move with their parts: same as made where they land
    moved = {
        s: glyphs.f["rect"][s].tolist() for s in np.flatnonzero(glyphs.used[: glyphs.top])
    }
    tab_to(0)  # hidden: every tag gone
    assert not any(v.pin_labels_shown for v in views) and int(glyphs.used.sum()) == base
    tab_to(2)
    assert int(glyphs.used.sum()) == with_tags
    again = {
        s: glyphs.f["rect"][s].tolist() for s in np.flatnonzero(glyphs.used[: glyphs.top])
    }
    assert sorted(again.values()) == sorted(moved.values())
    tab_to(1)  # hover: only the part under the cursor
    v = views[2]
    sx, sy = ed.camera.world_to_screen(v.x + v.w / 2, v.y + v.h / 2)
    ed.dispatch_event("on_mouse_motion", sx, sy, 0, 0)
    assert [w.pin_labels_shown for w in views] == [False, False, True, False, False]
    assert capture(ed).parts  # (still a sound board)
