"""The netlist bundled mod: the language against a plain circuit, then the editor's box
menu (needs GL: see hidden_editor.py; skipped where there's none)."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest
from hidden_editor import hidden_editor

from pijl import mods
from pijl.parts import TEMPLATES, registry
from pijl.sim import Circuit

SHIPPED = Path(mods.__file__).parent / "bundled"


def _lang(name="lang"):
    """A module of the mod on its own (importing the package would run its patches)."""
    spec = importlib.util.spec_from_file_location(f"netlist_{name}", SHIPPED / "netlist" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m  # (dataclasses look their module up)
    spec.loader.exec_module(m)
    return m


lang = _lang()
rt = _lang("router")
NetlistError, Scope, parse, resolve = lang.NetlistError, lang.Scope, lang.parse, lang.resolve


def board(*specs, boxes=None):
    """Parts from (kind, label, x, y); a Scope over all of them."""
    c = Circuit(registry.load(TEMPLATES))
    pos, parts = {}, []
    for kind, label, x, y in specs:
        p = c.add_part(kind)
        p.label = label
        pos[p] = (x, y)
        parts.append(p)
    named = boxes or {}

    def other(label):
        if label not in named:
            raise KeyError(f"no box labeled {label!r}")
        return [parts[i] for i in named[label]]

    scope = Scope(parts, other, pos.__getitem__, lambda pin: bool(c.ends_on(pin)))
    return c, parts, scope


def plan(script, scope):
    return resolve(script, scope).pairs


def names(pairs):
    def n(pin):
        lay = pin.part.layout
        return f"{pin.part.label or pin.part.kind}.{(lay.ins if pin.is_input else lay.outs)[pin.index]}"

    return [(n(a), n(b)) for a, b in pairs]


def test_parse():
    sts = parse('A.out -> B.a; # c\n\n"my reg"*.q, regs/<NAND>X?.b->C.a # trailing\nchain ADD*: cout -> cin')
    assert [s.line for s in sts] == [1, 3, 4]
    assert str(sts[1].left[0]) == "my reg*.q" and str(sts[1].left[1]) == "regs/<NAND>X?.b"
    assert sts[2].parts.label == "ADD*" and (sts[2].out, sts[2].into) == ("cout", "cin")


@pytest.mark.parametrize("bad", ["A.out", "A -> B.a", "A.out -> B.a junk", "A.out => B.a", "chain X: a b"])
def test_parse_errors(bad):
    with pytest.raises(NetlistError):
        parse(bad)


def test_fan_out_and_pairwise():
    _, _, scope = board(
        ("NOT", "S", 0, 0), ("NOT", "G2", 0, 0), ("NOT", "G10", 0, 0), ("NOT", "G1", 0, 0),
        ("NOT", "H1", 0, 0), ("NOT", "H2", 0, 0), ("NOT", "H10", 0, 0),
    )
    assert names(plan("S.out -> G*.a", scope)) == [
        ("S.out", "G1.a"), ("S.out", "G2.a"), ("S.out", "G10.a")
    ]
    _, _, scope = board(
        ("NOT", "G2", 0, 0), ("NOT", "G10", 0, 0), ("NOT", "G1", 0, 0),
        ("NOT", "H1", 0, 0), ("NOT", "H2", 0, 0), ("NOT", "H10", 0, 0),
    )
    # (written input first: the pairs come out output first anyway)
    assert names(plan("H*.a -> G*.out", scope)) == [
        ("G1.out", "H1.a"), ("G2.out", "H2.a"), ("G10.out", "H10.a")
    ]


def test_unlabeled_parts_go_by_kind_and_position():
    _, _, scope = board(
        ("NOT", "", 0, 0), ("NOT", "", 50, 100), ("NOT", "", 0, 100), ("NAND", "", 0, 0)
    )
    pairs = resolve("chain NOT: out -> a", scope).pairs
    xy = [(scope.pos(a.part), scope.pos(b.part)) for a, b in pairs]
    assert xy == [((0, 100), (50, 100)), ((50, 100), (0, 0))]  # top to bottom, left to right
    with pytest.raises(NetlistError, match="3 pins on the left, 2 on the right"):
        resolve("<NOT>.out -> NAND.?", _fresh(scope))


def test_other_boxes():
    _, _, scope = board(("NOT", "A", 0, 0), ("NOT", "B", 0, 0), boxes={"far": [1]})
    assert names(plan("A.out -> far/B.a", scope)) == [("A.out", "B.a")]
    with pytest.raises(NetlistError, match="no box labeled 'near'"):
        resolve("A.out -> near/B.a", _fresh(scope))


@pytest.mark.parametrize(
    "script, problem",
    [
        ("A.out -> B.out", "both are outputs"),
        ("A.a -> B.a", "both are inputs"),
        ("A.out -> A.a", "itself"),
        ("A.out -> B.a; C.out -> B.a", "fed twice"),
        ("A.out -> Q.a", "no part matches 'Q'"),
        ("A.out -> B.zz", "no pin of B matches 'zz'"),
        ("chain A: out -> a", "at least two"),
    ],
)
def test_problems(script, problem):
    _, _, scope = board(("NOT", "A", 0, 0), ("NOT", "B", 0, 0), ("NOT", "C", 0, 0))
    with pytest.raises(NetlistError, match=problem):
        resolve(script, scope)


def test_wired_inputs_and_repeats():
    c, (a, b), scope = board(("NOT", "A", 0, 0), ("NOT", "B", 0, 0))
    assert len(resolve("A.out -> B.a; B.a -> A.out", scope)) == 1  # (twice: once)
    c.connect(a.outputs[0], b.inputs[0])
    with pytest.raises(NetlistError, match="wired already"):
        resolve("A.out -> B.a", _fresh(scope))


# ---- the editor -------------------------------------------------------------------


@pytest.fixture(scope="module")
def ed(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mods, "BUNDLED", SHIPPED)
        mp.setenv("PIJL_DATA", str(tmp_path_factory.mktemp("data")))
        mp.setenv("PIJL_MODS", "")
        mp.setenv("PIJL_SAFE", "")
        from pijl.ui import editor as editor_module

        from pijl.ui import console, menus

        # (the mod's patches are taken back afterwards)
        mp.setattr(editor_module.Editor, "_box_items", editor_module.Editor._box_items)
        mp.setattr(console, "COMMANDS", dict(console.COMMANDS))
        mp.setattr(console, "FALLBACKS", [])
        mp.setattr(menus, "PROVIDERS", {k: [] for k in menus.KINDS})
        mods.plan()
        mods.enable("netlist")
        rep = mods.load()
        assert [m.name for m in rep.loaded] == ["netlist"] and not rep.problems, rep.problems
        editor = hidden_editor(tmp_path_factory)
        yield editor
        editor.close()
        mods._report = None
        mods._hooks.clear()
        mods._tracebacks.clear()
        for name in [n for n in sys.modules if n == mods.PACKAGE or n.startswith(mods.PACKAGE + ".")]:
            del sys.modules[name]


def test_box_menu_wires_and_undoes(ed):
    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    src = ed.add_part("NOT", 100, 300)
    src.part.label = "S"
    sinks = [ed.add_part("NOT", 300, 400 - 60 * i) for i in range(3)]
    for i, v in enumerate(sinks):
        v.part.label = f"G{i}"
    box = ed.add_box(("logic", 0, 0, 600, 600, None))
    ed._record()
    items = {i.text: i for i in ed._box_items(box)}
    sub = {i.text: i for i in items["Netlist"].submenu}
    assert set(sub) == {"Type lines (console)", "Run script", "Edit script"}

    netlist = sys.modules["pijl_mods.netlist"]
    assert netlist.run(ed, box, "S.out -> G*.a")
    ed._record()
    assert len(ed.circuit.wires) == 3 and len(ed.selection.wires) == 3
    for v in sinks:
        (w,) = ed.circuit.ends_on(v.part.inputs[0])
        assert ed.wire_views[w].color != netlist.UNROUTED
        pts = ed.wire_views[w].points
        assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(pts, pts[1:]))  # (on the grid's lines)
    assert "not routed" not in ed.status.text

    assert not netlist.run(ed, box, "S.out -> G0.a")  # (wired now)
    assert "wired already" in ed.status.text
    ed._undo()
    assert len(ed.circuit.wires) == 0  # one step

    # an old netlists/<label>.txt: taken in once, then the script lives on the box
    path = netlist.script_file(ed.project.path, "logic")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# from the file\nchain G*: out -> a\n", encoding="utf-8")
    sub["Run script"].action()
    assert len(ed.circuit.wires) == 2
    assert netlist.script_of(box) == "# from the file\nchain G*: out -> a\n"
    path.unlink()
    ed._record()
    ed._undo()  # (the wires and taking the script in: one step)
    assert len(ed.circuit.wires) == 0 and netlist.script_of(box) is None


def test_edit_goes_through_a_scratch_file(ed, monkeypatch):
    import os
    import time

    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    for label, x, y in (("A", 0, 400), ("B", 300, 500)):
        ed.add_part("NOT", x, y).part.label = label
    box = ed.add_box(("logic", -100, 0, 600, 700, None, {}))
    ed._record()
    netlist = sys.modules["pijl_mods.netlist"]
    opened = []
    monkeypatch.setattr(netlist, "_open_in_editor", opened.append)
    said = []
    say = lambda text, kind="out": said.append((text, kind))
    netlist.edit_box(ed, box, say)
    (path,) = opened
    assert "Netlist for the box" in path.read_text(encoding="utf-8")
    path.write_text("A.out -> B.a\n", encoding="utf-8")
    t = time.time() + 5
    os.utime(path, (t, t))  # (a save, surely after the write)
    netlist.run_box(ed, box, say)
    assert netlist.script_of(box) == "A.out -> B.a\n" and len(ed.circuit.wires) == 1
    assert said[-1] == ("1 connection", "out")


def test_board_fallback_default_pin_names_and_case():
    _, parts, scope = board(("NOT", "A", 0, 0), ("NOT", "B", 0, 0), ("NOT", "C", 0, 0))
    scope = Scope(parts[:1], scope.boxes, scope.pos, scope.wired, board=lambda: parts)
    # B and C aren't in the box: found on the board; out1 / in1 by position; any case
    assert names(plan("a.OUT1 -> b.IN1, C.A", scope)) == [("A.out", "B.a"), ("A.out", "C.a")]
    with pytest.raises(NetlistError, match=re.escape("it has a (in1), out (out1)")):
        resolve("A.q -> B.a", _fresh(scope))


def _fresh(scope):
    return Scope(scope.parts, scope.boxes, scope.pos, scope.wired, scope.board)


def test_wildcards_fall_through_a_box_without_the_pin():
    _, parts, scope = board(("CLK", "", 0, 0), ("NOT", "A", 0, 0), ("NOT", "B", 0, 0))
    scope = Scope(parts[:1], scope.boxes, scope.pos, scope.wired, board=lambda: parts)
    assert names(plan("CLK.out1 -> *.A", scope)) == [("CLK.clk", "A.a"), ("CLK.clk", "B.a")]



# ---- the router -------------------------------------------------------------------


def axis_runs(path):
    return [(a, b) for a, b in zip(path, path[1:])]


def routes_of(board, nets):
    return [res.routes for res in rt.route(board, nets)]


def test_route_straight_and_around():
    (r,), = routes_of(rt.Board(), [rt.Net([rt.Pin((10, 0), -1)], root=rt.Pin((0, 0), 1))])
    assert r.path[0] == (0, 0) and r.path[-1] == (10, 0) and rt.corners(r.path) == []
    board = rt.Board()
    board.add_rect(4, -2, 6, 2)
    (r,), = routes_of(board, [rt.Net([rt.Pin((10, 0), -1)], root=rt.Pin((0, 0), 1))])
    assert not set(r.path) & board.blocked and len(rt.corners(r.path)) == 4
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in axis_runs(r.path))


def test_fan_out_is_a_tree():
    sinks = [rt.Pin((10, y), -1) for y in (6, 0, -6)]
    (routes,) = routes_of(rt.Board(), [rt.Net(sinks, root=rt.Pin((0, 0), 1))])
    assert [r.sink for r in routes] == [1, 0, 2]  # (nearest first)
    assert routes[0].joins is None and all(r.joins == 0 for r in routes[1:])
    assert all(r.path[0] in routes[0].path for r in routes[1:])  # (they start on the trunk)


def test_nets_cross_but_never_share_a_run():
    nets = [
        rt.Net([rt.Pin((20, 0), -1)], root=rt.Pin((0, 0), 1)),
        rt.Net([rt.Pin((12, 10), -1)], root=rt.Pin((8, -10), 1)),
    ]
    a, b = (r[0].path for r in routes_of(rt.Board(), nets))
    assert a and b
    shared = set(a) & set(b)
    assert len(shared) == 1  # (one crossing)
    (c,) = shared
    i, j = a.index(c), b.index(c)
    assert rt._axis(a[i - 1], a[i + 1]) != rt._axis(b[j - 1], b[j + 1])


def test_unroutable_comes_back_empty():
    board = rt.Board()
    for x0, y0, x1, y1 in ((5, -5, 15, -5), (5, 5, 15, 5), (5, -5, 5, 5), (15, -5, 15, 5)):
        board.add_rect(x0, y0, x1, y1)  # (a closed ring around the sink)
    (r,), = routes_of(board, [rt.Net([rt.Pin((10, 0), -1)], root=rt.Pin((0, 0), 1))])
    assert r.path is None


def test_parse_nets():
    a, b, c = parse("net DATA: R*.q, ALU.a; bus: X.out; net.q -> B.a")
    assert (a.kind, a.name, [str(e) for e in a.ends]) == ("net", "DATA", ["R*.q", "ALU.a"])
    assert (b.kind, b.name) == ("bus", None)
    assert str(c.left[0]) == "net.q"  # (a part called net)


def test_net_problems():
    _, _, scope = board(("NOT", "A", 0, 0), ("NOT", "B", 0, 0), ("SPLIT", "W", 0, 0))
    with pytest.raises(NetlistError, match="needs two pins"):
        resolve("net: A.out", scope)
    with pytest.raises(NetlistError, match="fed twice"):
        resolve("A.out -> B.a; net: B.a, A.out", _fresh(scope))
    p = resolve("net N: A.out, B.a, A.a", _fresh(scope))
    assert p.pairs == [] and [n.name for n in p.nets] == ["N"] and len(p.nets[0].pins) == 3


def test_bus_spine_and_taps():
    pins = [rt.Pin((10, y), -1) for y in (0, 4, 8, 12)]
    (res,) = rt.route(rt.Board(), [rt.Net(pins, spine=True)])
    assert res.spine is not None and res.rooted is None
    (x0, y0), (x1, y1) = res.spine
    assert x0 == x1 == 9 and (y0, y1) == (0, 12)  # (down the pins' front)
    assert all(r.seed == rt.SPINE and r.path for r in res.routes)


def test_net_without_a_root_starts_at_an_output():
    pins = [rt.Pin((10, 0), -1), rt.Pin((0, 5), 1), rt.Pin((10, 10), -1)]
    (res,) = rt.route(rt.Board(), [rt.Net(pins)])
    assert res.rooted == 1 and sorted(r.sink for r in res.routes) == [0, 2]
    assert all(r.path for r in res.routes)


def test_sinks_join_the_wires_a_net_has():
    seed = [(0, 0), (20, 0)]
    (res,) = rt.route(rt.Board(), [rt.Net([rt.Pin((10, 6), -1)], seeds=[("w", seed)])])
    (r,) = res.routes
    assert r.seed == "w" and r.path[0][1] == 0


def test_bus_and_net_in_the_editor(ed):
    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    names = {}
    for label, x, y in (("A", 0, 400), ("B", 300, 500), ("C", 300, 300), ("D", 300, 100)):
        names[label] = ed.add_part("NOT", x, y)
        names[label].part.label = label
    box = ed.add_box(("all", -100, 0, 600, 700, None))
    ed._record()
    netlist = sys.modules["pijl_mods.netlist"]
    assert netlist.run(ed, box, "bus D: A.out, B.a, C.a")
    ed._record()
    c = ed.circuit
    free = [w for w in c.wires if w.src is w and w.dst is w]
    assert len(free) == 1 and len(c.wires) == 4  # (the spine, and a tap per pin)
    assert "not routed" not in ed.status.text
    # a pin that's wired already brings its wires: D joins those
    assert netlist.run(ed, box, "net: B.a, D.a")
    ed._record()
    assert c.ends_on(names["D"].part.inputs[0]) and len(c.wires) == 5
    assert "not routed" not in ed.status.text
    ed._undo()
    ed._undo()
    assert len(c.wires) == 0


def test_outputs_on_a_net_join_wires_not_the_root():
    pins = [rt.Pin((0, 0), 1), rt.Pin((0, 10), 1), rt.Pin((20, 5), -1)]
    (res,) = rt.route(rt.Board(), [rt.Net(pins)])
    assert res.rooted == 0
    first, second = res.routes
    assert first.sink == 2 and first.joins is None and first.seed is None  # (input: to the root)
    assert second.sink == 1 and second.joins == 0  # (output: onto that wire)


def test_whole_buses(ed):
    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    a, b, c = ed.add_parts(
        [("IN", 0, 300, None, "A", {"width": 8}), ("OUT", 300, 400, None, "B", {"width": 8}),
         ("OUT", 300, 200, None, "C", None)]
    )
    box = ed.add_box(("all", -100, 0, 600, 600, None))
    ed._record()
    netlist = sys.modules["pijl_mods.netlist"]
    assert netlist.run(ed, box, "A.* -> B.*")
    ed._record()
    (w,) = ed.circuit.wires
    assert w.width == 8 and ed.status.text == "netlist: 1 connection"
    assert not netlist.run(ed, box, "A.* -> C.*")
    assert "8 lanes into 1" in ed.status.text


def test_the_console(ed):
    from pyglet.window import key

    ed._cancel()
    ed._clear_board()
    ed._reset_history(None)
    for label, x, y in (("A", 0, 400), ("B", 300, 500), ("C", 300, 300)):
        ed.add_part("NOT", x, y).part.label = label
    box = ed.add_box(("logic", -100, 0, 600, 700, None))
    ed._record()
    ed.selection.set(boxes=[box])
    press = lambda sym, mods=0: ed.dispatch_event("on_key_press", sym, mods)

    def type_line(text):
        for ch in text:
            ed.dispatch_event("on_text", ch)
        press(key.ENTER)

    press(key.GRAVE)
    ed.dispatch_event("on_text", "`")  # (the key's own text: not typed)
    assert ed.console.open and ed.console.edit.text == ""
    ed.update(0.0)
    assert ed.console.scope == "logic"
    type_line("A.out -> B.a, C.a")  # (bare: the netlist)
    assert len(ed.circuit.wires) == 2
    assert ed.console.log[-1] == ("2 connections", "out")
    type_line("nope")
    assert ed.console.log[-1][1] == "error"
    press(key.Z, key.MOD_CTRL)  # (undo still reaches the board)
    assert len(ed.circuit.wires) == 0
    press(key.UP)
    press(key.UP)
    assert ed.console.edit.text == "A.out -> B.a, C.a"
    press(key.ESCAPE)
    assert not ed.console.open
