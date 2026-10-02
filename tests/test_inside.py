"""Looking inside placed macros (right-click -> View, see ui/inside.py): the views show
the instance's own hidden parts, live; nothing inside can be edited; backing out
puts the board back exactly. Needs GL (a hidden window, see hidden_editor.py); skipped
where there's none."""

import pytest
from hidden_editor import hidden_editor

from pijl.logic import ONE, ZERO
from pijl.snapshot import Snapshot

from test_macros import half_adder, w


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    editor = hidden_editor(tmp_path_factory)
    editor.store.save("ha", half_adder(), editor.catalog, "ha")
    wrap = Snapshot(
        {
            1: ("IN", "a", 0.0, 100.0, {}),
            2: ("IN", "b", 0.0, 0.0, {}),
            3: ("macro:ha", "inner", 100.0, 40.0, {}),
            10: ("OUT", "sum", 300.0, 100.0, {}),
            11: ("OUT", "carry", 300.0, 0.0, {}),
        },
        {1: w(1, 0, 3, 0), 2: w(2, 0, 3, 1), 3: w(3, 0, 10, 0), 4: w(3, 1, 11, 0)},
    )
    editor.store.save("wrap", wrap, editor.catalog, "wrap")
    editor.circuit.settle_ticks = 0
    yield editor
    editor.close()


@pytest.fixture
def board(ed):
    """Two switches into a placed `wrap`, its outputs into two LEDs."""
    ed._clear_board()
    ed.circuit.settle_ticks = 0
    ed._reset_history(None)
    a, b = ed.add_part("IN", 0, 100), ed.add_part("IN", 0, 0)
    m = ed.add_part("macro:wrap", 200, 40)
    s, c = ed.add_part("OUT", 500, 100), ed.add_part("OUT", 500, 0)
    ed.connect(a.part.outputs[0], m.part.inputs[0])
    ed.connect(b.part.outputs[0], m.part.inputs[1])
    ed.connect(m.part.outputs[0], s.part.inputs[0])
    ed.connect(m.part.outputs[1], c.part.inputs[0])
    ed._record()
    yield a, b, m
    ed._leave_inside(everything=True)


def settle(ed, n=30):
    for _ in range(n):
        ed.circuit.step()
    ed.view_sync(ed.circuit, ed.world)


def shown_pins(ed) -> set[int]:
    """The circuit pin slots the current scene's shapes show."""
    out = set()
    for buf in ed.world.buffers():
        if buf.pin_src is not None:
            used = buf.used[: buf.end]
            out |= {int(p) for p in buf.pin_src[: buf.end][used] if p >= 0}
    return out


def by_label(ed, label):
    return next(v for v in ed.part_views.values() if v.part.label == label)


def test_view_shows_the_instances_own_hidden_parts(ed, board):
    _, _, m = board
    outer_views = dict(ed.part_views)
    ed._view_inside(m)
    inner = m.part.inner
    assert set(ed.part_views) == set(inner.values())
    assert len(ed.wire_views) == len(m.part.type.body.wires)
    mine = {p.slot for part in inner.values() for p in part.pins}
    assert shown_pins(ed) == mine
    # (positions from the body)
    for uid, (_k, _l, x, y, _p) in m.part.type.body.parts.items():
        v = ed.part_views[inner[uid]]
        assert (v.x, v.y) == (x, y)
    ed._leave_inside()
    assert ed.part_views == outer_views and not ed.inside


def test_inside_follows_the_board_live(ed, board):
    a, b, m = board
    ed._view_inside(m)
    ed._view_inside(by_label(ed, "inner"))  # the half adder inside wrap
    assert len(ed.inside) == 2
    sum_port = by_label(ed, "sum")
    carry_port = by_label(ed, "carry")
    ed.circuit.click(a.part)
    ed.circuit.click(b.part)
    settle(ed)
    assert sum_port.part.inputs[0].state == ZERO
    assert carry_port.part.inputs[0].state == ONE
    ed.circuit.click(b.part)
    settle(ed)
    assert sum_port.part.inputs[0].state == ONE
    assert carry_port.part.inputs[0].state == ZERO
    # what's drawn is those pins
    from pijl.ui.sdf_shapes import SHOW_BY_CODE

    for buf in ed.world.buffers():
        if buf.pin_src is None:
            continue
        used = buf.used[: buf.end].nonzero()[0]
        src = buf.pin_src[used]
        keep = src >= 0
        codes = ed.circuit.pin_codes(src[keep].astype("intp"))
        assert (buf.state[used[keep]] == SHOW_BY_CODE[codes]).all()


def test_nothing_inside_can_be_edited(ed, board):
    from pyglet.window import key

    _, _, m = board
    state = ed.history.state
    ed._view_inside(m)
    n = len(ed.part_views)
    ed.dispatch_event("on_key_press", key.A, key.MOD_CTRL)
    ed.dispatch_event("on_key_press", key.DELETE, 0)
    ed.dispatch_event("on_key_press", key.Z, key.MOD_CTRL)
    ed.dispatch_event("on_key_press", key.D, key.MOD_CTRL)
    assert len(ed.part_views) == n and not ed.selection
    assert ed.history.state == state and ed.inside
    ed.dispatch_event("on_key_press", key.ESCAPE, 0)
    assert not ed.inside and m in ed.part_views.values()
    assert ed.history.state == state


def test_backing_out_puts_the_board_and_camera_back(ed, board):
    from pyglet.window import key

    _, _, m = board
    cam = (ed.camera.x, ed.camera.y, ed.camera.level)
    views = dict(ed.part_views)
    ed._view_inside(m)
    ed._view_inside(by_label(ed, "inner"))
    ed.dispatch_event("on_key_press", key.BACKSPACE, 0)
    assert len(ed.inside) == 1
    ed.dispatch_event("on_key_press", key.ESCAPE, 0)
    assert not ed.inside
    assert ed.part_views == views
    assert (ed.camera.x, ed.camera.y, ed.camera.level) == cam


def test_the_instance_going_away_backs_out(ed, board):
    _, _, m = board
    ed._view_inside(m)
    ed.circuit.close_part(m.part)  # (as if it were gone, behind the views' back)
    ed.update(1 / 60)
    assert not ed.inside


def test_loading_another_board_backs_out_first(ed, board):
    _, _, m = board
    ed._view_inside(m)
    ed._clear_board()
    assert not ed.inside and not ed.part_views


def test_probe_shows_the_hovered_parts_pin_levels(ed, board):
    a, _, m = board
    ed.circuit.click(a.part)
    settle(ed)
    ed.probe.update(m)
    texts = [label.text for label in ed.probe.labels]
    assert texts == ["Z01X"[c] for c in ed.circuit.pin_codes(ed.probe.slots).tolist()]
    assert texts[0] == "1"
    ed.circuit.click(a.part)
    settle(ed)
    ed.probe.update(m)
    assert ed.probe.labels[0].text == "0"
    ed.probe.update(None)
    assert not ed.probe.labels


def test_right_click_menus(ed, board, monkeypatch):
    a, _, m = board
    opened = []
    monkeypatch.setattr(ed, "_open_menu", lambda x, y, items: opened.append(items))

    def menu_on(view):
        opened.clear()
        sx, sy = ed.camera.world_to_screen(view.x + view.w / 2, view.y + view.h / 2)
        wx, wy = ed.camera.screen_to_world(sx, sy)
        ed._item_menu(sx, sy, wx, wy, 0)
        return {i.text: i for i in opened[0]} if opened else {}

    on_board = menu_on(m)
    assert {"View", "Open definition", "Label...", "Delete"} <= set(on_board)
    assert "View" not in menu_on(a)  # (not a macro)
    on_board["View"].action()
    assert len(ed.inside) == 1
    inner = by_label(ed, "inner")
    assert set(menu_on(inner)) == {"View", "Open definition"}  # read-only in here
    assert menu_on(by_label(ed, "a")) == {}  # nothing to do with a plain part
