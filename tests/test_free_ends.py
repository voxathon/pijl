"""Free wire ends through a real editor: ending a wire on nothing, unplugging by
deleting a part, carrying a free end onto a pin, and undo / save of all that.
Needs GL (a hidden window, see hidden_editor.py); skipped where there's none."""

import json

import pytest
from hidden_editor import hidden_editor

from pijl.sim import FREE


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
    return ed


def screen(ed, p):
    return ed.camera.world_to_screen(*p)


def click(ed, p, button=None, modifiers=0):
    from pyglet.window import mouse

    x, y = screen(ed, p)
    ed.dispatch_event("on_mouse_motion", x, y, 0, 0)
    ed.dispatch_event("on_mouse_press", x, y, button or mouse.LEFT, modifiers)
    ed.dispatch_event("on_mouse_release", x, y, button or mouse.LEFT, modifiers)


def drag(ed, a, b):
    from pyglet.window import mouse

    (ax, ay), (bx, by) = screen(ed, a), screen(ed, b)
    ed.dispatch_event("on_mouse_motion", ax, ay, 0, 0)
    ed.dispatch_event("on_mouse_press", ax, ay, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_drag", (ax + bx) / 2, (ay + by) / 2, 0, 0, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_drag", bx, by, 0, 0, mouse.LEFT, 0)
    ed.dispatch_event("on_mouse_release", bx, by, mouse.LEFT, 0)


def undo(ed, redo=False):
    from pyglet.window import key

    ed.dispatch_event("on_key_press", key.Y if redo else key.Z, key.MOD_CTRL)


def free_ends(ed):
    return sorted(
        (w.uid, side)
        for w in ed.circuit.wires
        for side in ("src", "dst")
        if getattr(w, side) is w
    )


def test_clicking_the_last_bend_again_ends_the_wire_free(board):
    ed = board
    a = ed.add_part("IN", 200, 300)
    ed._record()
    click(ed, ed.pin_pos(a.part.outputs[0]))
    assert ed.mode.name == "WIRING"
    click(ed, (400, 320))
    click(ed, (400, 320))  # the same spot again
    assert ed.mode.name == "IDLE"
    (w,) = ed.circuit.wires
    assert w.src is a.part.outputs[0] and w.dst is w
    view = ed.wire_views[w]
    assert view.dst == (400, 320) and view.bends == ()
    assert ed.free_end_at(400, 320) == (view, "dst")


def test_deleting_a_part_leaves_its_outside_wires_with_free_ends(board):
    ed = board
    a, g, h = ed.add_part("IN", 200, 300), ed.add_part("NOT", 400, 300), ed.add_part("NOT", 400, 400)
    w1 = ed.connect(a.part.outputs[0], g.part.inputs[0])
    w2 = ed.connect(g.part.outputs[0], h.part.inputs[0], [(520, 320), (520, 420)])
    ed._record()
    spot = ed.wire_views[w2].src
    ed.selection.set(parts=[g])
    ed.delete_selection()
    ed._record()
    assert set(ed.circuit.wires) == {w1, w2}  # both lead somewhere else
    assert w1.dst is w1 and w2.src is w2
    assert ed.wire_views[w2].src == spot and ed.wire_views[w2].bends == ((520, 320), (520, 420))
    undo(ed)
    g2 = next(p for p in ed.circuit.parts if p.kind == "NOT" and p.uid == g.part.uid)
    assert free_ends(ed) == []
    assert {(w.src, w.dst) for w in ed.circuit.wires} == {
        (a.part.outputs[0], g2.inputs[0]),
        (g2.outputs[0], h.part.inputs[0]),
    }
    undo(ed, redo=True)
    assert len(free_ends(ed)) == 2


def test_deleting_both_ends_still_removes_the_wire(board):
    ed = board
    a, g = ed.add_part("IN", 200, 300), ed.add_part("NOT", 400, 300)
    ed.connect(a.part.outputs[0], g.part.inputs[0])
    ed.selection.set(parts=[a, g])
    ed.delete_selection()
    assert ed.circuit.wires == []


def test_dragging_a_free_end_onto_a_pin_plugs_it_in(board):
    from pijl.logic import ONE, Level

    ed = board
    a, g = ed.add_part("IN", 200, 300), ed.add_part("NOT", 500, 300)
    stub = ed.connect(a.part.outputs[0], FREE, [], None, (350, 320))
    ed._record()
    drag(ed, (350, 320), ed.pin_pos(g.part.inputs[0]))
    assert ed.mode.name == "IDLE"
    (w,) = ed.circuit.wires
    assert w.src is a.part.outputs[0] and w.dst is g.part.inputs[0]
    assert w.uid > stub.uid  # (made anew: see document.rewire)
    a.part.outputs[0].state = True
    for _ in range(5):
        ed.circuit.step()
    assert g.part.inputs[0].state is Level(ONE)
    undo(ed)
    (w,) = ed.circuit.wires
    assert w.dst is w and ed.wire_views[w].dst == (350, 320)


def test_a_free_end_dropped_on_nothing_just_moves(board):
    ed = board
    a = ed.add_part("IN", 200, 300)
    stub = ed.connect(a.part.outputs[0], FREE, [], None, (350, 320))
    ed._record()
    drag(ed, (350, 320), (360, 400))
    assert ed.circuit.wires == [stub] and stub.dst is stub
    assert ed.wire_views[stub].dst == ed.snapped(360, 400)
    undo(ed)
    assert ed.wire_views[stub].dst == (350, 320)


def test_a_free_end_wont_plug_into_its_own_wire(board):
    ed = board
    a = ed.add_part("IN", 200, 300)
    stub = ed.connect(a.part.outputs[0], FREE, [(300, 400)], None, (400, 400))
    assert not ed.can_rewire(stub, "dst", stub)
    branch = ed.connect(stub, FREE, [], (300, 350), (250, 350))
    assert not ed.can_rewire(stub, "dst", branch)
    g = ed.add_part("NOT", 500, 300)
    assert ed.can_rewire(branch, "dst", g.part.inputs[0])
    assert not ed.can_rewire(stub, "dst", g.part.outputs[0])  # output to output


def test_unplug_end_from_the_menu_and_put_it_back(board):
    ed = board
    a, g = ed.add_part("IN", 200, 300), ed.add_part("NOT", 500, 300)
    w = ed.connect(a.part.outputs[0], g.part.inputs[0])
    ed._record()
    ed._unplug(ed.wire_views[w], "dst")
    assert ed.mode.name == "DRAGGING_END" and w.dst is w
    ed._cancel()  # Esc: back where it was, as it was
    assert w.dst is g.part.inputs[0] and ed.mode.name == "IDLE"
    assert not ed._record()  # nothing changed


def test_free_ends_survive_save_and_load(board):
    from pijl.storage import decode, dumps, encode
    from pijl.ui.document import capture, instantiate

    ed = board
    a = ed.add_part("IN", 200, 300)
    stub = ed.connect(a.part.outputs[0], FREE, [(300, 300)], None, (300, 400))
    ed.connect(stub, FREE, [], (300, 350), (250, 350))  # a free branch
    ed.connect(FREE, FREE, [], (100, 100), (150, 100))  # attached to nothing at all
    snap = capture(ed)
    data = json.loads(dumps(encode(snap)))
    loaded = decode(data, ed.circuit.registry)
    assert loaded.warnings == []
    assert loaded.snapshot.wires == snap.wires
    ed._clear_board()
    instantiate(ed, loaded.snapshot)
    assert len(free_ends(ed)) == 4 and len(ed.circuit.wires) == 3


def test_cut_delete_splices_onto_a_free_end(board):
    ed = board
    a, g = ed.add_part("IN", 200, 300), ed.add_part("NOT", 600, 300)
    trunk = ed.connect(a.part.outputs[0], g.part.inputs[0], [(400, 320)])
    view = ed.wire_views[trunk]
    stub = ed.connect(trunk, FREE, [], (400, 320), (400, 450))
    ed.cut_wire(view, (500, 320))  # past the junction: the trunk ends up down the stub
    assert ed.circuit.wires == [trunk] and trunk.dst is trunk
    assert ed.wire_views[trunk].dst == (400, 450)
    assert stub not in ed.wire_views


def test_a_copied_part_takes_its_free_stubs_along(board):
    from pijl.ui.document import capture, instantiate

    ed = board
    a = ed.add_part("IN", 200, 300)
    ed.connect(a.part.outputs[0], FREE, [], None, (350, 320))
    ed.connect(FREE, FREE, [], (0.0, 0.0), (10.0, 0.0))  # (on its own: not a's)
    clip = capture(ed, [a])
    assert len(clip.wires) == 1
    parts, wires = instantiate(ed, clip)
    (copy,) = wires
    assert copy.wire.dst is copy.wire and copy.wire.src is parts[0].part.outputs[0]


def test_double_click_on_empty_board_starts_a_wire_from_nothing(board):
    ed = board
    g = ed.add_part("NOT", 600, 300)
    click(ed, (400, 400))
    click(ed, (400, 400))  # double-click: a wire from nothing
    assert ed.mode.name == "WIRING" and ed.wire_start is FREE
    click(ed, (400, 400))  # (a third click right there: nothing)
    assert ed.mode.name == "WIRING" and ed.wire_bends == []
    click(ed, (500, 400))
    click(ed, ed.pin_pos(g.part.inputs[0]))
    (w,) = ed.circuit.wires
    assert w.src is w and w.dst is g.part.inputs[0]
    assert ed.wire_views[w].src == (400, 400) and ed.wire_views[w].bends == ((500, 400),)


def test_a_wire_from_nothing_to_nothing(board):
    ed = board
    click(ed, (400, 400))
    click(ed, (400, 400))
    click(ed, (480, 400))
    click(ed, (480, 400))  # end it there too
    (w,) = ed.circuit.wires
    assert w.src is w.dst is w and ed.wire_views[w].points == [(400, 400), (480, 400)]
    undo(ed)
    assert ed.circuit.wires == []


def test_right_click_on_empty_board_offers_a_new_wire(board):
    from pyglet.window import mouse

    ed = board
    opened = []
    real = ed._open_menu
    ed._open_menu = lambda x, y, items: (opened.append(items), real(x, y, items))
    x, y = screen(ed, (400, 400))
    ed.dispatch_event("on_mouse_press", x, y, mouse.RIGHT, 0)
    ed.dispatch_event("on_mouse_release", x, y, mouse.RIGHT, 0)
    del ed._open_menu
    assert ed.mode.name == "MENU"
    ((item, *rest),) = opened
    assert item.text == "New wire" and [i.text for i in rest] == ["New box"]
    ed._close_menu()
    item.action()
    assert ed.mode.name == "WIRING" and ed.wire_start is FREE and ed.wire_start_pos == (400, 400)
    ed._cancel()
    # a right-drag still just pans
    ed.dispatch_event("on_mouse_press", x, y, mouse.RIGHT, 0)
    ed.dispatch_event("on_mouse_drag", x + 40, y, 40, 0, mouse.RIGHT, 0)
    ed.dispatch_event("on_mouse_release", x + 40, y, mouse.RIGHT, 0)
    assert ed.mode.name == "IDLE"


def key_press(ed, symbol, modifiers=0):
    ed.dispatch_event("on_key_press", symbol, modifiers)
    ed.dispatch_event("on_key_release", symbol, modifiers)


def test_a_wire_on_its_own_copies_and_pastes(board):
    from pyglet.window import key

    ed = board
    a, g = ed.add_part("IN", 200, 300), ed.add_part("NOT", 500, 300)
    w = ed.connect(a.part.outputs[0], g.part.inputs[0], [(300, 320), (300, 400), (480, 400)])
    floating = ed.connect(FREE, FREE, [], (100, 500), (200, 500))
    ed._record()
    ed.selection.set(wires=[ed.wire_views[w], ed.wire_views[floating]])
    key_press(ed, key.C, key.MOD_CTRL)
    clip = ed.clipboard
    assert set(clip.wires) == {w.uid, floating.uid} and not clip.parts
    src, dst, *_ = clip.wires[w.uid]
    assert src == dst == ("w", w.uid)  # its ends' parts weren't copied: free in the copy
    key_press(ed, key.V, key.MOD_CTRL)
    assert ed.mode.name == "PLACING_PART" and len(ed.placing_wires) == 2
    click(ed, (700, 600))
    assert ed.mode.name == "IDLE" and len(ed.circuit.wires) == 4
    new = [x for x in ed.circuit.wires if x not in (w, floating)]
    assert all(x.src is x.dst is x for x in new)
    assert {v.wire for v in ed.selection.wires} == set(new)
    # pasting and then changing your mind leaves nothing behind
    key_press(ed, key.V, key.MOD_CTRL)
    ed._cancel()
    assert len(ed.circuit.wires) == 4 and len(ed.wire_views) == 4


def test_ctrl_d_lays_wires_side_by_side_across_their_run(board):
    from pyglet.window import key

    ed = board
    flat = ed.connect(FREE, FREE, [(300, 300)], (200, 300), (400, 320))  # runs left-right
    ed.selection.set(wires=[ed.wire_views[flat]])
    key_press(ed, key.D, key.MOD_CTRL)
    key_press(ed, key.D, key.MOD_CTRL)
    starts = sorted(ed.wire_views[x].src for x in ed.circuit.wires)
    xs = {x for x, _ in starts}
    assert len(ed.circuit.wires) == 4 and xs == {200}  # all stacked down: no column
    ys = sorted(y for _, y in starts)
    assert ys[1] - ys[0] == ys[2] - ys[1] == ys[3] - ys[2] > 0

    ed._clear_board()
    ed._reset_history(None)
    tall = ed.connect(FREE, FREE, [], (200, 200), (220, 400))  # runs up-down
    ed.selection.set(wires=[ed.wire_views[tall]])
    key_press(ed, key.D, key.MOD_CTRL)
    key_press(ed, key.D, key.MOD_CTRL)
    starts = sorted(ed.wire_views[x].src for x in ed.circuit.wires)
    assert {y for _, y in starts} == {200} and len(starts) == 4  # side by side


def test_ctrl_double_click_starts_on_the_grid_ctrl_shift_on_the_subgrid(board):
    from pyglet.window import key

    from pijl.ui import theme as T

    ed = board
    off = (403.0, 397.0)
    for mods, step in ((key.MOD_CTRL, T.GRID), (key.MOD_CTRL | key.MOD_SHIFT, T.GRID / T.SUBGRID_DIVISIONS)):
        held = [key.LCTRL] + ([key.LSHIFT] if mods & key.MOD_SHIFT else [])
        for k in held:
            ed.dispatch_event("on_key_press", k, mods)
        click(ed, off, modifiers=mods | key.MOD_NUMLOCK)
        click(ed, off, modifiers=mods | key.MOD_NUMLOCK)
        assert ed.mode.name == "WIRING" and ed.wire_start is FREE
        x, y = ed.wire_start_pos
        assert x % step == 0 and y % step == 0 and abs(x - off[0]) <= step / 2
        for k in held:
            ed.dispatch_event("on_key_release", k, mods)
        ed._cancel()
        ed.empty_click = (0.0, (0.0, 0.0))


def test_snapping_follows_the_grid_lines_on_screen():
    from pijl.ui import theme as T
    from pijl.ui.grid import snap_step

    big = T.GRID * T.GRID_MAJOR_EVERY
    assert snap_step(1.0) == T.GRID and snap_step(1.0, T.SUBGRID_DIVISIONS) == T.GRID / 2
    zoom = 0.5  # GRID is 5 px apart on screen: too fine, the next level is used
    assert T.GRID * zoom < T.GRID_SNAP_MIN_PX <= big * zoom
    assert snap_step(zoom) == big and snap_step(zoom, T.SUBGRID_DIVISIONS) == T.GRID
    assert snap_step(4.0) == T.GRID  # never finer than the pin grid without Shift


def test_ctrl_double_click_zoomed_out_starts_on_the_coarser_grid(board):
    from pyglet.window import key

    from pijl.ui import theme as T

    ed = board
    level = ed.camera.level
    ed.camera.level = -8  # zoom 0.5
    try:
        big = T.GRID * T.GRID_MAJOR_EVERY
        off = (403.0, 397.0)
        ed.dispatch_event("on_key_press", key.LCTRL, key.MOD_CTRL)
        click(ed, off, modifiers=key.MOD_CTRL)
        click(ed, off, modifiers=key.MOD_CTRL)
        assert ed.mode.name == "WIRING" and ed.wire_start_pos == (400, 400)
        assert all(c % big == 0 for c in ed.wire_start_pos)
        ed.dispatch_event("on_key_release", key.LCTRL, 0)
        ed._cancel()
        ed.empty_click = (0.0, (0.0, 0.0))
    finally:
        ed.camera.level = level
