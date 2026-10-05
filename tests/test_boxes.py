"""Boxes (ui/boxes.py): the save format and history without a window, then the real
editor: wrapping the selection, clicking inside, carrying by the header, resizing,
drawing one, undo, copy / paste. Editor tests need GL (see hidden_editor.py)."""

import json

import pytest
from hidden_editor import hidden_editor

from pijl.snapshot import Snapshot
from pijl.storage import decode, dumps, encode
from pijl.ui.document import History

BOX = ("adder", 100.0, 200.0, 300.0, 150.0, "green", {})


def test_boxes_save_and_load():
    snap = Snapshot({}, {}, boxes={3: BOX, 5: ("", 0.5, 0.0, 40.0, 40.0, None, {"m": {"k": [1]}})})
    text = dumps(encode(snap))
    data = json.loads(text)
    assert data["boxes"] == [
        {"uid": 3, "label": "adder", "rect": [100, 200, 300, 150], "color": "green"},
        {"uid": 5, "rect": [0.5, 0, 40, 40], "data": {"m": {"k": [1]}}},
    ]
    loaded = decode(data, None)
    assert loaded.warnings == [] and loaded.snapshot.boxes == snap.boxes


def test_no_boxes_no_key():
    assert "boxes" not in dumps(encode(Snapshot({}, {})))


def test_bad_boxes_are_dropped():
    data = {
        "pijl": 1,
        "parts": [],
        "wires": [],
        "boxes": [
            {"uid": 1, "rect": [0, 0, 10, 10]},
            {"uid": 1, "rect": [0, 0, 10, 10]},  # duplicate
            {"uid": 2, "rect": [0, 0, -5, 10]},  # empty
            {"uid": 3, "rect": [0, 0]},  # not a rectangle
            {"uid": 4, "rect": [0, 0, 1, 1], "color": 7},
        ],
    }
    loaded = decode(data, None)
    assert list(loaded.snapshot.boxes) == [1] and len(loaded.warnings) == 4


def test_history_has_a_box_section():
    h = History(Snapshot({}, {}))
    assert h.record({}, {}, {}, {}, {7: BOX})
    moved = (*BOX[:1], 0.0, 0.0, *BOX[3:])
    assert h.record({}, {}, {}, {}, {7: moved})
    h.undo()
    assert h.current.boxes == {7: BOX}
    h.undo()
    assert h.current.boxes == {}
    h.redo()
    h.redo()
    assert h.current.boxes == {7: moved}


# ---- the editor -------------------------------------------------------------------


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    yield editor
    editor.close()


@pytest.fixture
def board(ed):
    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    ed.empty_click = ed.header_click = (0.0, (0.0, 0.0))  # (no double-clicks across tests)
    return ed


def screen(ed, p):
    return ed.camera.world_to_screen(*p)


def press(ed, p, button=None, modifiers=0):
    from pyglet.window import mouse

    x, y = screen(ed, p)
    ed.dispatch_event("on_mouse_motion", x, y, 0, 0)
    ed.dispatch_event("on_mouse_press", x, y, button or mouse.LEFT, modifiers)


def release(ed, p, button=None, modifiers=0):
    from pyglet.window import mouse

    ed.dispatch_event("on_mouse_release", *screen(ed, p), button or mouse.LEFT, modifiers)


def click(ed, p, button=None, modifiers=0):
    press(ed, p, button, modifiers)
    release(ed, p, button, modifiers)


def drag(ed, a, b, modifiers=0):
    from pyglet.window import mouse

    (ax, ay), (bx, by) = screen(ed, a), screen(ed, b)
    press(ed, a, modifiers=modifiers)
    ed.dispatch_event("on_mouse_drag", (ax + bx) / 2, (ay + by) / 2, 0, 0, mouse.LEFT, modifiers)
    ed.dispatch_event("on_mouse_drag", bx, by, 0, 0, mouse.LEFT, modifiers)
    ed.dispatch_event("on_mouse_release", bx, by, mouse.LEFT, modifiers)


def key_press(ed, symbol, modifiers=0):
    ed.dispatch_event("on_key_press", symbol, modifiers)


def type_label(ed, text):
    from pyglet.window import key

    for ch in text:
        ed.dispatch_event("on_text", ch)
    key_press(ed, key.ENTER)


def undo(ed, redo=False):
    from pyglet.window import key

    key_press(ed, key.Y if redo else key.Z, key.MOD_CTRL)


def two_gates(ed):
    a, b = ed.add_part("NOT", 200, 300), ed.add_part("NOT", 300, 300)
    w = ed.connect(a.part.outputs[0], b.part.inputs[0])
    ed._record()
    return a, b, ed.wire_views[w]


def wrapped(ed):
    """Two wired gates with a box around them, labeled "pair"."""
    from pyglet.window import key

    a, b, w = two_gates(ed)
    ed.selection.set(parts=[a, b])
    key_press(ed, key.B)
    assert ed.mode.name == "EDITING_LABEL"
    type_label(ed, "pair")
    (box,) = ed.box_views.values()
    return a, b, w, box


def test_b_boxes_the_selection(board):
    ed = board
    a, b, _, box = wrapped(ed)
    assert box.label == "pair" and ed.mode.name == "IDLE"
    for v in (a, b):  # around both, on the grid
        assert box.x < v.x and v.x + v.w < box.x + box.w
        assert box.y < v.y and v.y + v.h < box.y + box.h
    assert box.x % 10 == 0 and box.y % 10 == 0
    undo(ed)
    assert ed.box_views == {}
    undo(ed, redo=True)
    (box,) = ed.box_views.values()
    assert box.label == "pair"


def test_a_click_inside_selects_whats_in_it(board):
    ed = board
    a, b, w, box = wrapped(ed)
    outside = ed.add_part("NOT", box.x + box.w + 100, 300)
    ed.selection.clear()
    click(ed, (box.x + 4, box.y + 4))  # empty, inside
    assert ed.selection.parts == {a, b} and ed.selection.wires == {w}
    assert ed.selection.boxes == {box} and outside not in ed.selection
    click(ed, (box.x + box.w + 300, box.y))  # empty, outside: cleared
    assert not ed.selection


def test_dragging_inside_is_still_a_box_select(board):
    ed = board
    a, _, _, _box = wrapped(ed)
    ed.selection.clear()
    drag(ed, (a.x - 5, a.y - 5), (a.x + a.w + 5, a.y + a.h + 5))
    assert ed.selection.parts == {a} and not ed.selection.boxes


def test_dragging_the_header_carries_what_is_in_it(board):
    ed = board
    a, b, w, box = wrapped(ed)
    ed.selection.clear()
    was = (box.x, box.y), (a.x, a.y), (b.x, b.y), w.points
    top = (box.x + box.w / 2, box.y + box.h - 4)
    drag(ed, top, (top[0] + 100, top[1] + 50))
    assert ed.mode.name == "IDLE"
    assert (box.x, box.y) == (was[0][0] + 100, was[0][1] + 50)
    assert (a.x, a.y) == (was[1][0] + 100, was[1][1] + 50)
    assert (b.x, b.y) == (was[2][0] + 100, was[2][1] + 50)
    assert w.points[0] == (was[3][0][0] + 100, was[3][0][1] + 50)
    undo(ed)  # one step for all of it
    box = next(iter(ed.box_views.values()))
    a = ed.part_views[ed.circuit.part_by_uid[a.part.uid]]
    assert (box.x, box.y) == was[0] and (a.x, a.y) == was[1]


def test_a_click_on_the_header_selects_and_a_double_click_labels(board):
    ed = board
    a, b, _w, box = wrapped(ed)
    ed.selection.clear()
    top = (box.x + box.w / 2, box.y + box.h - 4)
    click(ed, top)
    assert ed.selection.boxes == {box} and ed.selection.parts == {a, b}
    click(ed, top)
    assert ed.mode.name == "EDITING_LABEL" and ed.edit_view is box
    ed._cancel()
    assert box.label == "pair"


def test_resizing_by_an_edge(board):
    ed = board
    _, _, _, box = wrapped(ed)
    x0, y0, w0, h0 = box.x, box.y, box.w, box.h
    drag(ed, (x0 + w0, y0 + h0 / 2), (x0 + w0 + 200, y0 + h0 / 2 + 70))  # right edge
    assert (box.x, box.y, box.w, box.h) == (x0, y0, w0 + 200, h0)
    drag(ed, (x0, y0), (x0 - 30, y0 - 40))  # bottom left corner
    assert (box.x, box.y, box.w, box.h) == (x0 - 30, y0 - 40, w0 + 230, h0 + 40)
    undo(ed)
    undo(ed)
    assert (box.x, box.y, box.w, box.h) == (x0, y0, w0, h0)


def test_drawing_a_new_box(board):
    ed = board
    ed.selection.clear()
    click(ed, (500, 500))  # (an empty click first: the menu's anchor)
    from pijl.ui.editor import Mode

    ed.mode = Mode.DRAWING_BOX
    drag(ed, (100, 100), (260, 30))
    assert ed.mode.name == "EDITING_LABEL"
    type_label(ed, "new")
    (box,) = ed.box_views.values()
    assert (box.label, box.x, box.y, box.w, box.h) == ("new", 100, 30, 160, 70)
    undo(ed)
    assert ed.box_views == {}


def test_copy_paste_takes_the_box_along(board):
    from pyglet.window import key

    ed = board
    _a, _b, _w, box = wrapped(ed)
    click(ed, (box.x + 4, box.y + 4))
    key_press(ed, key.C, key.MOD_CTRL)
    key_press(ed, key.V, key.MOD_CTRL)
    assert ed.mode.name == "PLACING_PART"
    click(ed, (1000, 1000))
    assert len(ed.box_views) == 2 and len(ed.part_views) == 4
    copy = next(v for v in ed.box_views.values() if v is not box)
    assert copy.label == "pair" and copy.uid != box.uid and not copy.ghost
    undo(ed)
    assert list(ed.box_views.values()) == [box]


def test_delete_and_remove(board):
    ed = board
    _a, _b, _w, box = wrapped(ed)
    ed.remove_boxes([box])
    ed._record()
    assert ed.box_views == {} and len(ed.part_views) == 2
    undo(ed)
    (box,) = ed.box_views.values()
    click(ed, (box.x + 4, box.y + 4))
    ed.delete_selection()
    ed._record()
    assert ed.box_views == {} and ed.part_views == {}


def test_boxes_survive_save_and_load(board):
    from pijl.ui.document import capture

    ed = board
    _, _, _, box = wrapped(ed)
    box.set_color("blue")
    snap = capture(ed)
    loaded = decode(json.loads(dumps(encode(snap))), ed.circuit.registry)
    assert loaded.snapshot.boxes == {box.uid: ("pair", box.x, box.y, box.w, box.h, "blue", {})}


def test_box_data_is_saved_undone_and_copied(board):
    from pijl.ui.document import capture

    ed = board
    _, _, _, box = wrapped(ed)
    ed._record()
    ed.set_box_data(box, "mymod", {"script": "A.q -> B.a"})
    ed._record()
    assert box.mod_data == {"mymod": {"script": "A.q -> B.a"}}
    snap = capture(ed)
    loaded = decode(json.loads(dumps(encode(snap))), ed.circuit.registry)
    assert loaded.snapshot.boxes[box.uid][6] == box.mod_data
    with pytest.raises(TypeError):
        ed.set_box_data(box, "mymod", object())  # (not JSON)
    ed.set_box_data(box, "mymod", None)
    ed._record()
    assert box.mod_data == {}
    undo(ed)
    assert box.mod_data == {"mymod": {"script": "A.q -> B.a"}}
    undo(ed)
    assert box.mod_data == {}
