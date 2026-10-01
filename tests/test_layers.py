"""Wire layers (2.5D) and vias, stacked like a chip's metal over its silicon: the data
(saving, undo), and through a real editor drawing across layers (a via at each
change), pins only on layer 0, placing and deleting vias, picking only what reaches
the active layer, and moving wires between layers. The editor tests need a GL
window; they're skipped where one can't be made."""

import json
import os

import numpy as np
import pytest

from pijl.parts import builtin_registry
from pijl.snapshot import LAYER_COUNT, Snapshot, is_via
from pijl.storage import decode, dumps, encode
from pijl.ui import theme as T
from pijl.ui.document import History

# ---- data ---------------------------------------------------------------------------

PARTS = {1: ("IN", "", 200.0, 300.0, {}), 2: ("NOT", "", 400.0, 300.0, {})}


def layered() -> Snapshot:
    return Snapshot(
        PARTS,
        {
            1: (("w", 1), ("w", 1), (), (300.0, 300.0), (300.0, 300.0)),  # a via
            2: (("p", 1, False, 0), ("w", 1), (), None, (300.0, 300.0)),
            3: (("w", 1), ("w", 3), ((350.0, 320.0),), (300.0, 300.0), (400.0, 320.0)),
        },
        {},
        {3: 4},  # (the via and wire 2 are on layer 0: not in it)
    )


def test_a_via_is_a_wire_of_no_length_on_nothing():
    snap = layered()
    assert [uid for uid, d in snap.wires.items() if is_via(d, uid)] == [1]


def test_layers_are_saved_and_read_back():
    data = encode(layered())
    assert [w.get("layer") for w in data["wires"]] == [None, None, 4]
    loaded = decode(json.loads(dumps(data)), builtin_registry())
    assert loaded.warnings == []
    assert loaded.snapshot == layered()


def test_a_bad_layer_drops_the_wire_with_a_warning():
    data = encode(layered())
    data["wires"][2]["layer"] = 16  # (there are 16: 0 to 15)
    loaded = decode(data, builtin_registry())
    assert 3 not in loaded.snapshot.wires and len(loaded.warnings) == 1


def test_undo_brings_a_layer_back():
    h = History(layered())
    assert h.record({}, {}, {}, {3: 7, 1: 2})
    assert h.current.wire_layers == {1: 2, 3: 7}
    h.undo()
    assert h.current == layered()
    h.record({}, {}, {}, {3: None})  # (back on layer 0)
    assert h.current.wire_layers == {}


# ---- editor -------------------------------------------------------------------------


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
def board(ed):
    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    ed.set_layer(0)
    return ed


def click(ed, p, modifiers=0):
    from pyglet.window import mouse

    x, y = ed.camera.world_to_screen(*p)
    ed.dispatch_event("on_mouse_motion", x, y, 0, 0)
    ed.dispatch_event("on_mouse_press", x, y, mouse.LEFT, modifiers)
    ed.dispatch_event("on_mouse_release", x, y, mouse.LEFT, modifiers)


def press(ed, symbol, modifiers=0):
    ed.dispatch_event("on_key_press", symbol, modifiers)


def go_to(ed, layer):
    """PgUp / PgDn to that layer."""
    from pyglet.window import key

    while ed.layer < layer:
        press(ed, key.PAGEUP)
    while ed.layer > layer:
        press(ed, key.PAGEDOWN)


def undo(ed, redo=False):
    from pyglet.window import key

    press(ed, key.Y if redo else key.Z, key.MOD_CTRL)


def two_parts(ed):
    a = ed.add_part("IN", 200, 300)
    b = ed.add_part("NOT", 500, 400)
    ed._record()
    return a, b


def vias(ed):
    return sorted((v for v in ed.wire_views.values() if v.is_via), key=lambda v: v.wire.uid)


def lines(ed):
    """The wires that aren't vias, by uid."""
    return sorted(
        (v for v in ed.wire_views.values() if not v.is_via), key=lambda v: v.wire.uid
    )


def one_net(ed) -> bool:
    nets = ed.circuit.wire_nets(np.array([w.slot for w in ed.circuit.wires], np.intp))
    return len(set(nets.tolist())) == 1


def shapes(view):
    """The layer bytes (flags.x, sel.y) of a view's shapes: segments, then dots."""
    t = view.table
    dots = t.dot[view.row]
    slots = [*t.slots(view.row).tolist(), *dots[dots >= 0].tolist()]
    f = t.buf.f
    return [(int(f["flags"][s, 0]), int(f["sel"][s, 1])) for s in slots]


def up_and_over(ed, a, b, layer=1):
    """IN -> NOT: from the pin on layer 0, up to `layer` at a bend (350, y), across to
    (450, y + 50), and back down to layer 0 there for the NOT's input."""
    start = ed.pin_pos(a.part.outputs[0])
    click(ed, start)
    click(ed, (350, start[1]))
    go_to(ed, layer)
    click(ed, (450, start[1] + 50))
    go_to(ed, 0)
    click(ed, ed.pin_pos(b.part.inputs[0]))
    assert ed.mode.name == "IDLE"
    return start


def test_going_up_a_layer_and_back_leaves_a_via_at_each_change(board):
    ed = board
    a, b = two_parts(ed)
    start = up_and_over(ed, a, b)
    v1, v2 = vias(ed)
    assert (v1.src, v2.src) == ((350, start[1]), (450, start[1] + 50))
    assert v1.layer == v2.layer == 0  # (a via's floor: the lower of its two layers)
    w1, w2, w3 = lines(ed)
    assert (w1.wire.src, w1.wire.dst) == (a.part.outputs[0], v1.wire)
    assert (w2.wire.src, w2.wire.dst) == (v1.wire, v2.wire)
    assert (w3.wire.src, w3.wire.dst) == (v2.wire, b.part.inputs[0])
    assert [w.layer for w in (w1, w2, w3)] == [0, 1, 0]
    assert set(shapes(w2)) == {(1, 0)}
    assert one_net(ed)  # (vias are junctions)
    # a via's look: a ring in its floor's color, reaching from its floor to the top
    t = v1.table
    ring = int(t.dot[v1.row, 0])
    assert tuple(t.buf.f["ca"][ring][:3].tolist()) == T.LAYER_COLORS[0]
    assert set(shapes(v1)) == {(0, LAYER_COUNT - 1)}
    # and it all came in one step
    undo(ed)
    assert not ed.wire_views
    undo(ed, redo=True)
    assert len(vias(ed)) == 2 and [w.layer for w in lines(ed)] == [0, 1, 0]


def test_only_what_reaches_the_active_layer_is_clickable(board):
    from pijl.sim import FREE

    ed = board
    a, b = two_parts(ed)
    start = up_and_over(ed, a, b, layer=2)
    w1, w2, _ = lines(ed)
    on_0 = (275, start[1])
    on_2 = (400, start[1] + 25)  # (halfway along the diagonal)
    high = ed.wire_views[ed.connect(FREE, FREE, [], (600.0, 600.0), (600.0, 600.0), layer=3)]
    assert ed.layer == 0
    assert ed.wire_at(*on_0) is w1 and ed.wire_at(*on_2) is None
    assert ed.wire_at(600, 600) is None  # (a via doesn't reach below its floor)
    go_to(ed, 2)
    assert ed.wire_at(*on_2) is w2 and ed.wire_at(*on_0) is None
    assert ed.wire_at(600, 600) is None
    go_to(ed, 5)
    assert ed.wire_at(600, 600) is high  # (... and does reach every layer above it)


def test_pins_only_connect_to_layer_0(board):
    from pyglet.window import key

    ed = board
    a, b = two_parts(ed)
    pin = ed.pin_pos(b.part.inputs[0])
    go_to(ed, 1)
    click(ed, ed.pin_pos(a.part.outputs[0]))
    assert ed.mode.name != "WIRING"  # (no wire from a pin off layer 0)
    ed._cancel()
    go_to(ed, 0)
    click(ed, ed.pin_pos(a.part.outputs[0]))
    assert ed.mode.name == "WIRING"
    press(ed, key.PAGEUP)
    assert ed.layer == 0  # refused: no bend yet, and the pin only reaches layer 0
    click(ed, (350, 300))
    press(ed, key.PAGEUP)
    assert ed.layer == 1  # past a bend: a via goes there
    assert ed.wire_target(*pin) is None  # (not a target from layer 1)
    click(ed, pin)
    assert ed.mode.name == "WIRING" and not ed.wire_views  # (a bend, not the end)
    ed._cancel()


def test_drawn_from_the_input_each_stretch_keeps_its_layer(board):
    ed = board
    a, b = two_parts(ed)
    click(ed, ed.pin_pos(b.part.inputs[0]))  # from the NOT's input ...
    click(ed, (450, 400))
    go_to(ed, 2)
    click(ed, (350, 300))
    go_to(ed, 0)
    click(ed, ed.pin_pos(a.part.outputs[0]))  # ... to the IN's output
    assert sorted(w.layer for w in lines(ed)) == [0, 0, 2]
    assert [v.layer for v in vias(ed)] == [0, 0] and one_net(ed)


def test_backspace_takes_the_layer_back_with_the_bend(board):
    from pyglet.window import key

    ed = board
    a, _ = two_parts(ed)
    start = ed.pin_pos(a.part.outputs[0])
    click(ed, start)
    click(ed, (350, start[1]))
    press(ed, key.PAGEUP)
    click(ed, (350, 450))
    press(ed, key.PAGEUP)
    assert ed.wire_layers == [0, 1] and ed.layer == 2
    press(ed, key.BACKSPACE)
    assert ed.wire_bends == [(350, start[1])] and ed.layer == 1
    ed._cancel()


def test_a_branch_needs_a_bend_before_it_changes_layer(board):
    from pyglet.window import key

    ed = board
    a, b = two_parts(ed)
    w = ed.connect(a.part.outputs[0], b.part.inputs[0])
    ed._record()
    p = ed.pin_pos(a.part.outputs[0])
    ed._start_wiring(w, (p[0] + 50, p[1]))
    press(ed, key.PAGEUP)
    assert ed.layer == 0  # refused: it would join its wire on another layer
    click(ed, (300, 500))
    press(ed, key.PAGEUP)
    assert ed.layer == 1
    ed._cancel()


def test_placing_a_via_and_wiring_from_it(board):
    from pyglet.window import mouse

    ed = board
    _, b = two_parts(ed)
    go_to(ed, 2)
    opened = []
    real = ed._open_menu
    ed._open_menu = lambda x, y, items: (opened.append(items), real(x, y, items))
    x, y = ed.camera.world_to_screen(350, 350)
    ed.dispatch_event("on_mouse_press", x, y, mouse.RIGHT, 0)
    ed.dispatch_event("on_mouse_release", x, y, mouse.RIGHT, 0)
    del ed._open_menu
    place = next(i for i in opened[0] if i.text.startswith("Place via"))
    ed._close_menu()
    place.action()
    ed._record()
    (via,) = vias(ed)
    assert via.src == (350, 350) and via.layer == 2
    go_to(ed, 3)
    click(ed, (350, 350))  # a click on a via starts a wire from it, like a pin
    assert ed.mode.name == "WIRING" and ed.wire_start is via.wire
    click(ed, (450, 350))
    go_to(ed, 0)  # (a bend, then down to the pins: another via there)
    click(ed, ed.pin_pos(b.part.inputs[0]))
    assert sorted(w.layer for w in lines(ed)) == [0, 3]
    assert sorted(v.layer for v in vias(ed)) == [0, 2]
    # an end on a via draws no junction dot over it
    (from_via,) = [w for w in lines(ed) if via.wire in w.wire.ends]
    k = 0 if from_via.wire.src is via.wire else 1
    assert from_via.table.buf.f["radius"][from_via.table.dot[from_via.row, k]] == 0


def test_deleting_a_via_unplugs_what_was_on_it(board):
    ed = board
    a, b = two_parts(ed)
    up_and_over(ed, a, b)
    v1, _ = vias(ed)
    ed.selection.set(wires=[v1])
    ed.delete_selection()
    ed._record()
    assert len(vias(ed)) == 1 and len(lines(ed)) == 3  # the wires stay, their ends free
    assert sum(any(e is w.wire for e in w.wire.ends) for w in lines(ed)) == 2
    undo(ed)
    assert len(vias(ed)) == 2 and one_net(ed)


def test_shift_page_up_moves_the_selected_wires_a_layer_up(board):
    from pyglet.window import key

    from pijl.sim import FREE

    ed = board
    a, b = two_parts(ed)
    on_pin = ed.wire_views[ed.connect(a.part.outputs[0], b.part.inputs[0])]
    free = ed.wire_views[ed.connect(FREE, FREE, [], (300.0, 500.0), (400.0, 500.0), layer=3)]
    ed._record()
    ed.selection.set(wires=[free, on_pin])
    press(ed, key.PAGEUP, key.MOD_SHIFT)
    assert free.layer == 4 and on_pin.layer == 0  # (wires on pins stay on layer 0)
    assert ed.layer == 1  # (the view follows: from 0, one up)
    assert set(shapes(free)) == {(4, 0)}
    undo(ed)
    assert free.layer == 3


def test_saving_keeps_the_layers(board):
    ed = board
    a, b = two_parts(ed)
    up_and_over(ed, a, b, layer=5)
    data = encode(ed.history.current)
    assert sorted(w.get("layer", 0) for w in data["wires"]) == [0, 0, 0, 0, 5]


def test_the_preview_shows_layers_and_vias(board):
    from pyglet.window import key

    ed = board
    a, _ = two_parts(ed)
    start = ed.pin_pos(a.part.outputs[0])
    click(ed, start)
    click(ed, (350, start[1]))
    press(ed, key.PAGEUP)
    press(ed, key.PAGEUP)
    line = ed.preview
    assert line.buf.layered
    f = line.buf.f
    assert f["flags"][line._slots, 0].tolist() == [0, 2]
    ring = int(line._vias[0])
    assert (int(f["flags"][ring, 0]), int(f["sel"][ring, 1])) == (0, LAYER_COUNT - 1)
    assert tuple(f["a"][ring].tolist()) == (350.0, start[1])
    press(ed, key.BACKSPACE)  # the bend goes, and with it the via
    assert line._vias is None and f["flags"][line._slots, 0].tolist() == [0]
    ed._cancel()


def test_a_layered_board_draws(board):
    ed = board
    a, b = two_parts(ed)
    up_and_over(ed, a, b, layer=3)
    ed.selection.set(wires=list(ed.wire_views.values()))  # (the halo is layered too)
    assert ed.wire_table.buf.layer_mask & 0b1001 == 0b1001
    for layer in (0, 3, 15):
        ed.set_layer(layer)
        ed.switch_to()
        ed.on_draw()
