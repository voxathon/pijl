"""Buses in the real editor: a settings edit that changes a part's pins rebuilds it
(dropping wires that don't fit), undo puts the old pins and wires back, and wire
widths ride along through undo, copies and the free-wire Width prompt. Needs GL (a
hidden window, see hidden_editor.py); skipped where there's none."""

import pytest
from hidden_editor import hidden_editor

from pijl.logic import X, ints
from pijl.sim import FREE


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    yield editor
    editor.close()


@pytest.fixture(autouse=True)
def fresh(ed):
    ed._clear_board()
    ed._reset_history(None)


def in_sync(ed) -> bool:
    """Is the board what the undo history thinks it is (widths included)?"""
    from pijl.ui.document import capture

    snap, h = capture(ed), ed.history.current
    return (snap.parts, snap.wires, snap.wire_widths) == (h.parts, h.wires, h.wire_widths)


def undo(ed):
    from pyglet.window import key

    ed.dispatch_event("on_key_press", key.Z, key.MOD_CTRL)


def redo(ed):
    from pyglet.window import key

    ed.dispatch_event("on_key_press", key.Y, key.MOD_CTRL)


def part(ed, uid):
    return ed.circuit.part_by_uid[uid]


def test_a_width_edit_rebuilds_the_part_and_drops_wires_that_dont_fit(ed):
    src, n, led = ed.add_part("IN", 0, 0), ed.add_part("NOT", 200, 0), ed.add_part("OUT", 400, 0)
    ed.connect(src.part.outputs[0], n.part.inputs[0])
    ed.connect(n.part.outputs[0], led.part.inputs[0])
    ed._record()
    steps = len(ed.history.undo_stack)

    assert ed._set_setting([src], "width", 4)
    ed._record()
    s = part(ed, src.part.uid)
    assert s.outputs[0].width == 4 and s.props["width"] == 4
    assert len(ed.circuit.wires) == 1 and in_sync(ed)  # (IN -> NOT went: 4 lanes on 1)
    assert len(ed.history.undo_stack) == steps + 1  # one step for all of it

    undo(ed)
    s = part(ed, src.part.uid)
    assert s.outputs[0].width == 1 and len(ed.circuit.wires) == 2 and in_sync(ed)
    s.outputs[0].state = 1
    ed.circuit.run_until_stable(10)
    assert str(part(ed, led.part.uid).inputs[0].state) == "0"
    redo(ed)
    assert part(ed, src.part.uid).outputs[0].width == 4 and len(ed.circuit.wires) == 1
    assert in_sync(ed)


def test_a_new_split_pattern_keeps_the_wires_that_still_fit(ed):
    src = ed.add_part("IN", 0, 0)
    assert ed._set_setting([src], "width", 8)
    src = ed.part_views[part(ed, src.part.uid)]
    sp = ed.add_part("SPLIT", 200, 0)
    assert ed._set_setting([sp], "pattern", "4,4")
    sp = ed.part_views[part(ed, sp.part.uid)]
    lo, hi = ed.add_part("OUT", 400, 0), ed.add_part("OUT", 400, -100)
    for v in (lo, hi):
        assert ed._set_setting([v], "width", 4)
    lo, hi = (ed.part_views[part(ed, v.part.uid)] for v in (lo, hi))
    bus = ed.connect(src.part.outputs[0], sp.part.inputs[0])
    a = ed.connect(sp.part.outputs[0], lo.part.inputs[0])
    b = ed.connect(sp.part.outputs[1], hi.part.inputs[0])
    assert bus.width == 8 and a.width == b.width == 4
    ed._record()
    src.part.outputs[0].state = 0x5A
    ed.circuit.step()
    assert int(ints(lo.part.inputs[0].state)[0]) == 0xA

    assert ed._set_setting([sp], "pattern", "4,2,2")  # pin 1 is lanes 4-5 now: 2 wide
    ed._record()
    sp2 = part(ed, sp.part.uid)
    assert sp2.layout.outs == ("0-3", "4-5", "6-7")
    assert sorted(w.uid for w in ed.circuit.wires) == [bus.uid, a.uid] and in_sync(ed)
    part(ed, src.part.uid).outputs[0].state = 0x5A
    ed.circuit.step()
    assert int(ints(part(ed, lo.part.uid).inputs[0].state)[0]) == 0xA  # (still wired)

    undo(ed)
    assert part(ed, sp.part.uid).layout.outs == ("0-3", "4-7")
    assert sorted(w.uid for w in ed.circuit.wires) == sorted([bus.uid, a.uid, b.uid])
    assert in_sync(ed)


def test_free_wires_get_their_width_from_the_prompt_and_undo_takes_it_back(ed):
    from pijl.ui.document import free_tree, set_wire_width

    w = ed.connect(FREE, FREE, [], (0, 0), (100, 0))
    branch = ed.connect(w, FREE, [], (50, 0), (50, 80))
    ed._record()
    tree = free_tree(ed.circuit, w)
    assert tree == [w, branch]
    set_wire_width(ed, tree, 8)
    ed._record()
    assert [x.width for x in ed.circuit.wires] == [8, 8] and in_sync(ed)
    assert ed.history.current.wire_widths == {w.uid: 8, branch.uid: 8}
    # an 8-lane wire takes an 8-lane pin, and then its tree isn't free any more
    led = ed.add_part("OUT", 300, 0)
    assert ed._set_setting([led], "width", 8)
    led = part(ed, led.part.uid)
    trunk = ed.circuit.wire_by_uid[w.uid]
    assert ed.connect(trunk, led.inputs[0], [], (100, 0)) is not None
    assert free_tree(ed.circuit, trunk) is None
    ed._record()
    undo(ed)  # (the OUT and its wire)
    assert [x.width for x in ed.circuit.wires] == [8, 8] and in_sync(ed)
    undo(ed)  # (the width)
    assert [x.width for x in ed.circuit.wires] == [1, 1] and in_sync(ed)


def test_copies_keep_their_widths(ed):
    from pijl.ui.document import capture, instantiate

    src, led = ed.add_part("IN", 0, 0), ed.add_part("OUT", 300, 0)
    for v in (src, led):
        assert ed._set_setting([v], "width", 16)
    src, led = (ed.part_views[part(ed, v.part.uid)] for v in (src, led))
    ed.connect(src.part.outputs[0], led.part.inputs[0])
    ed._record()
    clip = capture(ed, [src, led])
    assert list(clip.wire_widths.values()) == [16]
    parts, wires = instantiate(ed, clip)
    assert [p.part.pins[0].width for p in parts] == [16, 16]
    assert [v.wire.width for v in wires] == [16]
    ed._record()
    assert in_sync(ed)


def test_clicking_a_bit_cell_flips_that_lane(ed):
    from pyglet.window import mouse

    from pijl.ui.views import cell_at

    src = ed.add_part("IN", 0, 0)
    assert ed._set_setting([src], "width", 4)
    src = ed.part_views[part(ed, src.part.uid)]
    assert src.h == 5 * 20  # (four cells, a pin's spacing each, plus one)
    assert len(ed.part_table.face_slots(src.row)) == 4
    ed._record()
    for lane in (2, 0, 2):
        wx = src.x + src.w / 2
        wy = src.y + (4 - lane) * 20  # (lane 0 on top)
        assert cell_at(src, wx, wy) == lane
        sx, sy = ed.camera.world_to_screen(wx, wy)
        ed.dispatch_event("on_mouse_press", sx, sy, mouse.LEFT, 0)
        ed.dispatch_event("on_mouse_release", sx, sy, mouse.LEFT, 0)
    assert int(ints(src.part.outputs[0].state)[0]) == 0b0001
    assert not ed.selection.parts  # (a cell click isn't a select)


def test_buses_show_how_many_lanes_are_lit_and_read_out_under_the_cursor(ed):
    from pijl.ui import theme as T
    from pijl.ui.editor import _bus_text
    from pijl.ui.sdf_shapes import SHOW_OFF, SHOW_ON, SHOW_X

    src, led = ed.add_part("IN", 0, 0), ed.add_part("OUT", 300, 0)
    for v in (src, led):
        assert ed._set_setting([v], "width", 8)
    src, led = (ed.part_views[part(ed, v.part.uid)] for v in (src, led))
    w = ed.connect(src.part.outputs[0], led.part.inputs[0])
    ed._record()
    view = ed.wire_views[w]
    buf = ed.wire_table.buf
    slots = ed.wire_table.slots(view.row)
    assert (buf.f["radius"][slots] == T.BUS_THICKNESS / 2).all()

    def shown(value):
        src.part.outputs[0].state = value
        ed.update(1 / 60)
        return int(buf.state[slots[0]])

    assert shown(0) == SHOW_OFF and shown(255) == SHOW_ON and shown(X) == SHOW_X
    half, one = shown(0x0F), shown(0x01)
    assert SHOW_OFF < one < half < SHOW_ON
    assert _bus_text(src.part.outputs[0].state) == "8 lanes  0x01  1  00000001"
    assert _bus_text(ed.circuit.wire_lanes(w)) == "8 lanes  0x01  1  00000001"

    # the readout, with the cursor over the wire
    sx, sy = ed.camera.world_to_screen(150, view.src[1])
    ed.dispatch_event("on_mouse_motion", sx, sy, 0, 0)
    ed.update(1 / 60)
    assert ed.bus_readout.text.startswith("8 lanes  0x01")
    ed.dispatch_event("on_mouse_motion", sx, sy + 300, 0, 0)
    ed.update(1 / 60)
    assert ed.bus_readout.text == ""
