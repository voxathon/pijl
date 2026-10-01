"""Dragging through a real editor: the drop is recorded as a move (a delta in the undo
history, not a copy of every part), and undo / redo put everything back exactly.
Needs a GL window; skipped where one can't be made."""

import os

import pytest


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    os.environ["PIJL_DATA"] = str(tmp_path_factory.mktemp("pijl-data"))
    try:
        from pijl.ui.editor import Editor

        editor = Editor()
    except Exception as e:  # (no display / GL)
        pytest.skip(f"no editor window here: {e}")
    editor._enable_event_queue = False  # dispatch synthetic events right away
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
