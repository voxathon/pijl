"""Buses: wide pins (PartType.widths), wires of several lanes, and everything that
has to keep working when a pin is more than one net."""

import itertools
import json
import random

import numpy as np
import pytest

from pijl.logic import ONE, X, Z, ZERO, Logic, ints, stack
from pijl.macros import Catalog
from pijl.parts import TEMPLATES, PartType, Registry, layout_of, load
from pijl.sim import Circuit
from pijl.sim.circuit import FREE
from pijl.sim.config import OPTIONS, EngineConfig
from pijl.snapshot import MACRO, Snapshot
from pijl.storage import decode, dumps, encode


class Add8(PartType):
    kind, ins, outs = "ADD8", ("a", "b"), ("s", "c")
    widths = {"a": 8, "b": 8, "s": 8}
    pure = True

    def eval(self, ctx, a, b):
        (va, ka), (vb, kb) = ints(a), ints(b)
        total = va + vb
        s = Logic.of_ints(total, 8)
        known = ka & kb
        s = Logic.of_codes(np.where(known[:, None], s.codes, np.uint8(X)))
        c = Logic.of_codes(np.where(known, ((total >> np.uint64(8)) & np.uint64(1)) + 1, X).astype(np.uint8))
        return s, c


class Inv(PartType):
    """As wide as its prop says."""

    kind, ins, outs = "INV", ("d",), ("q",)
    props = {"width": 4}
    widths = {"d": "width", "q": "width"}
    pure = True

    def eval(self, ctx, d):
        return ~d


class Lanes(PartType):
    """Splits a 4-lane bus into its lanes, the long way round (not pure)."""

    kind, ins, outs = "LANES", ("d",), ("q0", "q1", "q2", "q3")
    widths = {"d": 4}

    def eval(self, ctx, d):
        return tuple(d[:, i] for i in range(4))


class Pack(PartType):
    kind, ins, outs = "PACK", ("a", "b", "c", "d"), ("q",)
    widths = {"q": 4}
    pure = True

    def eval(self, ctx, a, b, c, d):
        return stack([a, b, c, d])


class Pull8(PartType):
    kind, ins, outs = "PULL8", ("in",), ("out",)
    widths = {"in": 8, "out": 8}
    joins = (("in", "out"),)
    weak = ("out",)

    def eval(self, ctx, net):
        return ONE


class Bad(PartType):
    """Returns a number for a wide output."""

    kind, ins, outs = "BAD", ("d",), ("q",)
    widths = {"d": 4, "q": 4}

    def eval(self, ctx, d):
        return 5


def registry() -> Registry:
    reg = load(TEMPLATES)
    for t in (Add8, Inv, Lanes, Pack, Pull8, Bad):
        reg.add(t)
    return reg


def bus_in(c, width, uid=None):
    return c.add_parts([c.registry.get("IN")], [uid], props=[{"width": width}])[0]


def bus_out(c, width, uid=None):
    return c.add_parts([c.registry.get("OUT")], [uid], props=[{"width": width}])[0]


def value(pin) -> int:
    v, known = ints(pin.state)
    assert known, pin.state
    return int(v)


# ---- logic -------------------------------------------------------------------------


def test_ints_and_back():
    v = Logic.of_ints([0, 5, 255, -1], 8)
    assert v.shape == (4, 8)
    assert repr(v[1]) == "Logic(10100000)"  # (a 1-d row prints lane 0 first)
    assert repr(Logic.of_ints([5], 4)) == "Logic[0101]"  # rows: most significant first
    values, known = ints(v)
    assert values.tolist() == [0, 5, 255, 255] and known.all()
    blurred = Logic.of_codes(v.codes.copy())
    blurred.codes[1, 3] = X
    values, known = ints(blurred)
    assert known.tolist() == [True, False, True, True] and values[1] == 5
    assert ints(Logic.of_ints(2**64 - 1, 64))[0] == 2**64 - 1


def test_stack_and_lane_ops():
    a = Logic([True, False])
    b = Logic([ONE, X])
    s = stack([a, b, ZERO])
    assert s.shape == (2, 3)
    assert list(s[0]) == [ONE, ONE, ZERO] and list(s[1]) == [ZERO, X, ZERO]
    assert list((~s)[1]) == [ONE, X, ONE]


# ---- the contract ------------------------------------------------------------------


@pytest.mark.parametrize(
    "widths, why",
    [
        ({"nope": 2}, "no pin called"),
        ({"d": 0}, "1 to 64"),
        ({"d": 65}, "1 to 64"),
        ({"d": True}, "1 to 64"),
        ({"d": "missing"}, "no prop or setting"),
    ],
)
def test_bad_widths_dont_register(widths, why):
    class T(PartType):
        kind, ins, outs = "T", ("d",), ("q",)

    T.widths = widths
    with pytest.raises(ValueError, match=why):
        Registry().add(T)


def test_joined_pins_must_be_equally_wide():
    class J(PartType):
        kind, ins, outs = "J", ("a",), ("b",)
        joins = (("a", "b"),)
        widths = {"a": 2}

    with pytest.raises(ValueError, match="equally wide"):
        Registry().add(J)


def test_each_type_gets_its_own_widths(tmp_path):
    """A mod editing a type's widths in place must not edit every type's."""
    reg = registry()
    reg.get("NAND").widths["out"] = 4
    assert PartType.widths == {} and reg.get("NOT").widths == {}


def test_api_1_scripts_cant_have_wide_pins(tmp_path):
    (tmp_path / "old.py").write_text(
        "from pijl.parts import PartType\nAPI = 1\n"
        "class W(PartType):\n    kind, ins, outs = 'W', ('d',), ('q',)\n    widths = {'d': 2}\n"
        "def register(reg):\n    reg.add(W)\n"
    )
    reg = load(tmp_path)
    assert "W" not in reg and "API = 2" in reg.errors[0]


# ---- wiring ------------------------------------------------------------------------


def test_a_bus_carries_numbers_through_a_wide_part():
    c = Circuit(registry())
    a, b, add = bus_in(c, 8), bus_in(c, 8), c.add_part("ADD8")
    s, carry = bus_out(c, 8), c.add_part("OUT")
    assert [p.width for p in add.pins] == [8, 8, 8, 1]
    for src, dst in ((a.outputs[0], add.inputs[0]), (b.outputs[0], add.inputs[1])):
        wire, _ = c.connect(src, dst)
        assert wire.width == 8
    c.connect(add.outputs[0], s.inputs[0])
    c.connect(add.outputs[1], carry.inputs[0])
    for x, y in ((200, 100), (3, 4), (255, 1)):
        a.outputs[0].state, b.outputs[0].state = x, y
        c.run_until_stable(10)
        assert value(s.inputs[0]) == (x + y) % 256
        assert carry.inputs[0].state is (ONE if x + y > 255 else ZERO)
    a.outputs[0].state = X  # every lane
    c.run_until_stable(10)
    assert set(s.inputs[0].state.codes.tolist()) == {int(X)} and carry.inputs[0].state is X


def test_widths_must_match():
    c = Circuit(registry())
    a, n, wide = bus_in(c, 8), c.add_part("NOT"), bus_out(c, 4)
    assert not c.can_connect(a.outputs[0], n.inputs[0])
    assert c.connect(a.outputs[0], n.inputs[0]) == (None, [])
    assert c.connect(a.outputs[0], wide.inputs[0], check=False) == (None, [])  # never
    good, _ = c.connect(a.outputs[0], bus_out(c, 8).inputs[0])
    assert not c.can_connect(good, n.inputs[0])  # a narrow pin on a bus
    assert c.connect(n.outputs[0], FREE, width=2) == (None, [])
    assert c.wires == [good]


def test_branches_and_free_wires_have_lanes_too():
    c = Circuit(registry())
    a, o1, o2 = bus_in(c, 8), bus_out(c, 8), bus_out(c, 8)
    trunk, _ = c.connect(a.outputs[0], o1.inputs[0])
    branch, _ = c.connect(trunk, o2.inputs[0])
    stub, _ = c.connect(branch, FREE)
    loose, _ = c.connect(FREE, FREE, width=3)
    assert (branch.width, stub.width, loose.width) == (8, 8, 3)
    a.outputs[0].state = 0b1011_0110
    c.step()
    assert value(o1.inputs[0]) == value(o2.inputs[0]) == 0b1011_0110
    # lane i is its own net all the way through
    lanes = c.wire_nets(np.arange(branch.slot, branch.slot + 8))
    assert len(set(lanes.tolist())) == 8
    assert c.with_descendants(np.array([trunk.slot])).tolist() == sorted(
        [trunk.slot, branch.slot, stub.slot]
    )
    assert c.remove_wire(trunk) == [trunk, branch, stub]
    c.step()
    assert set(o2.inputs[0].state.codes.tolist()) == {int(Z)}
    assert c.hidden_wires == [] and c.wires == [loose]


def test_merge_and_detach_keep_lanes_lined_up():
    c = Circuit(registry())
    a, o1, o2 = bus_in(c, 4), bus_out(c, 4), bus_out(c, 4)
    trunk, _ = c.connect(a.outputs[0], o1.inputs[0])
    branch, _ = c.connect(trunk, o2.inputs[0])
    c.detach(trunk, "dst")
    c.merge(trunk, branch)
    a.outputs[0].state = 0b1001
    c.step()
    assert value(o2.inputs[0]) == 0b1001
    assert c.wires == [trunk] and trunk.dst is o2.inputs[0]


def test_widths_from_props_and_one_batch_per_shape():
    c = Circuit(registry())
    inv = c.add_parts([c.registry.get("INV")] * 3, [None] * 3, props=[{"width": 2}, None, {"width": 8}])
    assert [p.inputs[0].width for p in inv] == [2, 4, 8]
    batches = [b for b in c._eval_batches() if b.type.kind == "INV"]
    assert sorted(len(b.parts) for b in batches) == [1, 1, 1]
    for p, w in zip(inv, (2, 4, 8)):
        src = bus_in(c, w)
        c.connect(src.outputs[0], p.inputs[0])
        src.outputs[0].state = 1
    c.run_until_stable(10)
    assert [value(p.outputs[0]) for p in inv] == [0b10, 0b1110, 0b1111_1110]


def test_lanes_in_and_out_of_scripts():
    c = Circuit(registry())
    src, split, pack, dst = bus_in(c, 4), c.add_part("LANES"), c.add_part("PACK"), bus_out(c, 4)
    c.connect(src.outputs[0], split.inputs[0])
    for i in range(4):
        c.connect(split.outputs[i], pack.inputs[3 - i])  # reversed
    c.connect(pack.outputs[0], dst.inputs[0])
    src.outputs[0].state = 0b0011
    c.run_until_stable(10)
    assert value(dst.inputs[0]) == 0b1100
    assert [q.state for q in split.outputs] == [ONE, ONE, ZERO, ZERO]


def test_a_wide_pull():
    c = Circuit(registry())
    pull, o = c.add_part("PULL8"), bus_out(c, 8)
    c.connect(pull.outputs[0], o.inputs[0])
    c.run_until_stable(10)
    assert value(o.inputs[0]) == 255
    src = bus_in(c, 8)
    c.connect(src.outputs[0], pull.inputs[0])
    src.outputs[0].state = 0x42
    c.run_until_stable(10)
    assert value(o.inputs[0]) == 0x42  # the strong driver wins


def test_a_number_for_a_wide_output_faults_the_kind():
    c = Circuit(registry())
    b = c.add_part("BAD")
    c.step()
    assert "Logic.of_ints" in c.faults["BAD"]
    assert set(b.outputs[0].state.codes.tolist()) == {int(X)}


def test_a_bad_layout_doesnt_register_or_faults_the_kind():
    class Odd(PartType):
        kind, ins, outs = "ODD", ("d",), ("q",)

        def layout(self, props):
            return {"widths": (3,)}  # one short

    with pytest.raises(ValueError, match="1 widths for 2 pins"):
        registry().add(Odd)
    # a mod patching it in later: the kind is faulted, its pins one lane each
    Odd.layout = PartType.layout
    reg = registry()
    reg.add(Odd)
    Odd.layout = lambda self, props: {"widths": (3,)}
    try:
        c = Circuit(reg)
        p = c.add_part("ODD")
        assert "ODD.layout" in c.faults["ODD"] and p.inputs[0].width == 1
    finally:
        Odd.layout = PartType.layout


def test_removing_wide_parts_frees_every_lane():
    c = Circuit(registry())
    a, o = bus_in(c, 8), bus_out(c, 8)
    c.connect(a.outputs[0], o.inputs[0])
    c.remove_parts([a])
    assert not c._pins.alive[a.outputs[0].slot : a.outputs[0].slot + 8].any()
    c.step()
    assert set(o.inputs[0].state.codes.tolist()) == {int(Z)}
    heads, counts = c.pin_slots_of([o])
    assert heads.tolist() == [o.inputs[0].slot] and counts.tolist() == [1]


# ---- macros ------------------------------------------------------------------------


def _inverter_macro() -> Snapshot:
    """IN (8) -> INV (8) -> OUT (8), plus a narrow pin for good measure."""
    pin = lambda uid, i, inp: ("p", uid, inp, i)  # noqa: E731
    return Snapshot(
        {
            1: ("IN", "d", 0, 0, {"width": 8}),
            2: ("INV", "", 100, 0, {"width": 8}),
            3: ("OUT", "q", 200, 0, {"width": 8}),
            4: ("IN", "e", 0, -100, {"width": 1}),
            5: ("OUT", "f", 200, -100, {"width": 1}),
        },
        {
            1: (pin(1, 0, False), pin(2, 0, True), (), None, None),
            2: (pin(2, 0, False), pin(3, 0, True), (), None, None),
            3: (pin(4, 0, False), pin(5, 0, True), (), None, None),
        },
        wire_widths={1: 8, 2: 8},
    )


def test_macro_pins_are_as_wide_as_their_ports():
    cat = Catalog(registry(), {"inv8": _inverter_macro()}.__getitem__)
    t = cat.get(MACRO + "inv8")
    assert layout_of(t, {}).widths == (8, 1, 8, 1)
    c = Circuit(cat)
    for _ in range(2):  # (the second one is stamped from the first's blueprint)
        m = c.add_part(MACRO + "inv8")
        src, dst = bus_in(c, 8), bus_out(c, 8)
        c.connect(src.outputs[0], m.inputs[0])
        c.connect(m.outputs[0], dst.inputs[0])
        src.outputs[0].state = 0x0F
        c.run_until_stable(10)
        assert value(dst.inputs[0]) == 0xF0
        assert not c.can_connect(src.outputs[0], m.inputs[1])


def test_a_stale_macro_body_leaves_mismatched_wires_out():
    body = _inverter_macro()
    body.wire_widths[1] = 4  # (a file edited by hand)
    c = Circuit(Catalog(registry(), {"m": body}.__getitem__))
    m = c.add_part(MACRO + "m")
    assert len(m.inner_wires) == 2


# ---- files -------------------------------------------------------------------------


def test_widths_round_trip_through_files():
    reg = registry()
    snap = _inverter_macro()
    data = encode(snap, reg)
    assert [d.get("width") for d in data["wires"]] == [8, 8, None]
    assert data["parts"][3] == {"uid": 4, "kind": "IN", "label": "e", "pos": [0, -100]}
    back = decode(json.loads(dumps(data)), reg)
    assert back.snapshot == snap and not back.warnings


def test_a_wire_whose_ends_arent_its_width_is_dropped():
    reg = registry()
    data = encode(_inverter_macro(), reg)
    data["wires"][0]["width"] = 4
    data["wires"][2]["width"] = 2
    back = decode(json.loads(dumps(data)), reg)
    assert sorted(back.snapshot.wires) == [2]
    assert len(back.warnings) == 2 and "lanes wide" in back.warnings[0]


# ---- engines -----------------------------------------------------------------------


def test_every_engine_agrees_on_buses():
    """Random boards of wide parts, tick for tick, like test_circuit's big one."""
    reg = registry()
    configs = [EngineConfig(**dict(zip(OPTIONS, combo))) for combo in itertools.product(*OPTIONS.values())]
    reference = EngineConfig(dirty="off", eval="batches")
    configs.remove(reference)

    def build(seed, config):
        rng = random.Random(seed)
        c = Circuit(reg, settle_ticks=rng.choice([0, 6]), seed=seed, config=config)
        parts = []
        for _ in range(rng.randint(4, 24)):
            kind = rng.choice(["ADD8", "INV", "INV", "LANES", "PACK", "PULL8", "IN", "NAND"])
            props = {"width": rng.choice([1, 4, 8])} if kind in ("IN", "INV") else None
            parts += c.add_parts([reg.get(kind)], [None], props=[props])
        outs = [q for p in parts for q in p.outputs]
        for pin in [q for p in parts for q in p.inputs]:
            fits = [o for o in outs if o.width == pin.width and o.part is not pin.part]
            if fits and rng.random() < 0.9:
                c.connect(rng.choice(fits), pin, check=False)
        return rng, c, [p for p in parts if p.kind == "IN"]

    def tick(rng, c, switches):
        if switches and rng.random() < 0.3:
            pin = rng.choice(switches).outputs[0]
            pin.state = rng.choice([ZERO, ONE, X, rng.randrange(256)])
        c.step()

    for seed in range(30):
        ref = build(seed, reference)
        others = [build(seed, cfg) for cfg in configs]
        for t in range(40):
            tick(*ref)
            for cfg, board in zip(configs, others):
                tick(*board)
                a, b = board[1], ref[1]
                where = (str(cfg), seed, t)
                assert np.array_equal(a._pins.states[: b._pins.n], b._pins.states[: b._pins.n]), where
                assert np.array_equal(a.net_value, b.net_value), where


# ---- headless ----------------------------------------------------------------------


def test_the_harness_drives_and_reads_buses_by_name():
    from pijl.engine import Harness

    cat = Catalog(registry(), {"inv8": _inverter_macro()}.__getitem__)
    h = Harness(cat.get(MACRO + "inv8"), cat, settle_ticks=0)
    assert h.wide and [p.width for p in h.part.pins] == [8, 1, 8, 1]
    out = h.apply({"d": 0x0F, "e": 1})
    assert int(ints(out["q"])[0]) == 0xF0 and out["f"] is ONE
    for value in (X, "x", "0xFF", 255):
        h.set(d=value)
        h.settle()
        assert h.get("q").codes.tolist() == ([int(X)] * 8 if value in (X, "x") else [int(ZERO)] * 8)
    with pytest.raises(ValueError, match="number"):
        h.set(d="ten")


def test_bit_strings_are_a_bus_lane_by_lane():
    from pijl.engine import Harness

    cat = Catalog(registry(), {"inv8": _inverter_macro()}.__getitem__)
    h = Harness(cat.get(MACRO + "inv8"), cat, settle_ticks=0)
    assert h.in_lanes == (*(f"d[{i}]" for i in range(8)), "e")
    assert h.out_lanes == (*(f"q[{i}]" for i in range(8)), "f")
    h.set_bits("110000001")  # d = 0b11 (lane 0 first), e = 1
    h.settle()
    assert int(ints(h.get("q"))[0]) == 0b1111_1100
    assert h.bits() == "001111111"
    h.set_bits("X", start=8)
    h.settle()
    assert h.bits()[-1] == "X"


# ---- the command line and the pipe ------------------------------------------------------


@pytest.fixture
def bus_project(tmp_path, monkeypatch):
    """A project with "pass": a 4-lane bus straight through, and a NOT beside it."""
    from pijl.project import Project
    from pijl.storage import MacroStore

    monkeypatch.setenv("PIJL_DATA", str(tmp_path / "data"))
    p = Project.open("bench")
    pin = lambda uid, i, inp: ("p", uid, inp, i)  # noqa: E731
    snap = Snapshot(
        {
            1: ("IN", "d", 0, 100, {"width": 4}),
            2: ("OUT", "q", 200, 100, {"width": 4}),
            3: ("IN", "e", 0, 0, {}),
            4: ("NOT", "", 100, 0, {}),
            5: ("OUT", "f", 200, 0, {}),
        },
        {
            1: (pin(1, 0, False), pin(2, 0, True), (), None, None),
            2: (pin(3, 0, False), pin(4, 0, True), (), None, None),
            3: (pin(4, 0, False), pin(5, 0, True), (), None, None),
        },
        wire_widths={1: 4},
    )
    MacroStore(p.macros_dir).save("pass", snap)
    return p


def test_the_command_line_takes_and_shows_buses(bus_project, capsys, monkeypatch):
    from test_engine import cli

    from pijl import mods

    # (main() loads mods: leave no loaded-mods state behind for later test files)
    monkeypatch.setattr(mods, "_report", mods._report)
    monkeypatch.delenv("PIJL_MODS", raising=False)
    assert "pass  (d:4, e) -> (q:4, f)" in cli(capsys, monkeypatch, "list")[1]  # (a bus: name:lanes)
    assert cli(capsys, monkeypatch, "run", "pass", "d=0xA", "e=1") ==(0, "q=1010 f=0\n", "")
    assert cli(capsys, monkeypatch, "run", "pass", "d=x", "e=0", "--json")[1] == '{"q": "XXXX", "f": 1}\n'
    assert cli(capsys, monkeypatch, "run", "pass", "01001")[1] == "q=0010 f=0\n"  # lanes, lane 0 first
    status, out, _ = cli(capsys, monkeypatch, "run", "pass", "--table")
    rows = out.splitlines()
    assert status == 0 and len(rows) == 33
    assert rows[0] == "d[0] d[1] d[2] d[3] e | q[0] q[1] q[2] q[3] f"
    assert rows[-1].split() == ["1"] * 5 + ["|"] + ["1"] * 4 + ["0"]


def test_the_pipe_names_lanes(bus_project):
    import io

    from pijl import pipe as P
    from pijl.engine import Engine

    eng = Engine("bench")
    out = io.BytesIO()
    assert P.serve(lambda n, s: eng.harness("pass"), 0, 0, io.BytesIO(b""), out) == 0
    body = out.getvalue()[P.HEADER.size :]
    assert P.HELLO_HEAD.unpack_from(body)[3:] == (5, 5)
    names = body[P.HELLO_HEAD.size :].split(b"\0")[2:-1]
    assert names == [b"d[0]", b"d[1]", b"d[2]", b"d[3]", b"e", b"q[0]", b"q[1]", b"q[2]", b"q[3]", b"f"]


# ---- SPLIT ---------------------------------------------------------------------------


def split(c, **props):
    return c.add_parts([c.registry.get("SPLIT")], [None], props=[{**c.registry.get("SPLIT").props, "width": 8, "pattern": "", "flip": False, **props}])[0]


@pytest.mark.parametrize(
    "pattern, width, want",
    [("", 4, [1, 1, 1, 1]), ("4,4", 8, [4, 4]), ("3,5", 8, [3, 5]), ("1x8", 8, [1] * 8),
     ("2x2", 8, [2, 2, 4]), ("4,4,4", 8, [4, 4]), ("6", 4, [4]), (" 2 , 1x2 ", 5, [2, 1, 1, 1])],
)
def test_split_patterns(pattern, width, want):
    from pijl.parts.registry import load as load_parts

    reg = load_parts(TEMPLATES)
    mod = type(reg.get("SPLIT")).__module__
    import sys

    assert sys.modules[mod].sizes(pattern, width) == want


def test_a_bad_pattern_doesnt_parse():
    s = registry().get("SPLIT").settings["pattern"]
    for bad in ("4;4", "x", "0", "2x0", "four"):
        with pytest.raises(ValueError):
            s.parse(bad)
    assert s.parse(" 4,4 ") == "4,4"


def test_split_and_merge_are_lanes_straight_through():
    c = Circuit(registry())
    src, sp = bus_in(c, 8), split(c, pattern="4,4")
    assert sp.layout.ins == ("bus",) and sp.layout.outs == ("0-3", "4-7")
    assert [p.width for p in sp.pins] == [8, 4, 4]
    lo, hi = bus_out(c, 4), bus_out(c, 4)
    c.connect(src.outputs[0], sp.inputs[0])
    c.connect(sp.outputs[0], lo.inputs[0])
    c.connect(sp.outputs[1], hi.inputs[0])
    src.outputs[0].state = 0xA5
    c.step()  # (no delay: the same step that carries the switch carries the halves)
    assert (value(lo.inputs[0]), value(hi.inputs[0])) == (0x5, 0xA)
    # and back together, flipped, the halves swapped
    mg, out = split(c, pattern="4,4", flip=True), bus_out(c, 8)
    assert mg.layout.ins == ("0-3", "4-7") and mg.layout.outs == ("bus",)
    c.connect(sp.outputs[0], mg.inputs[1])
    c.connect(sp.outputs[1], mg.inputs[0])
    c.connect(mg.outputs[0], out.inputs[0])
    c.step()
    assert value(out.inputs[0]) == 0x5A
    assert not c._eval_batches() or all(b.type.kind != "SPLIT" for b in c._eval_batches())


def test_a_split_into_single_lanes_feeds_gates():
    c = Circuit(registry())
    src, sp = bus_in(c, 4), split(c, width=4)
    nots = [c.add_part("NOT") for _ in range(4)]
    c.connect(src.outputs[0], sp.inputs[0])
    for pin, n in zip(sp.outputs, nots):
        c.connect(pin, n.inputs[0])
    src.outputs[0].state = 0b0110
    c.run_until_stable(10)
    assert [n.outputs[0].state for n in nots] == [ONE, ZERO, ZERO, ONE]


def test_reshaped_says_when_a_settings_edit_changes_pins():
    c = Circuit(registry())
    sp, i = split(c, pattern="4,4"), bus_in(c, 8)
    assert not c.reshaped(sp, {**sp.props})
    assert c.reshaped(sp, {**sp.props, "pattern": "1x8"})
    assert c.reshaped(sp, {**sp.props, "flip": True})
    assert c.reshaped(i, {"width": 4}) and not c.reshaped(i, {"width": 8})


def test_splits_round_trip_through_files():
    reg = registry()
    pin = lambda uid, i, inp: ("p", uid, inp, i)  # noqa: E731
    snap = Snapshot(
        {
            1: ("IN", "", 0, 0, {"width": 16}),
            2: ("SPLIT", "", 100, 0, {"width": 16, "pattern": "4x4", "flip": False}),
            3: ("OUT", "", 200, 0, {"width": 4}),
        },
        {
            1: (pin(1, 0, False), pin(2, 0, True), (), None, None),
            2: (pin(2, 3, False), pin(3, 0, True), (), None, None),
        },
        wire_widths={1: 16, 2: 4},
    )
    data = encode(snap, reg)
    assert data["parts"][1]["props"] == {"width": 16, "pattern": "4x4"}  # (flip: default)
    back = decode(json.loads(dumps(data)), reg)
    assert back.snapshot == snap and not back.warnings
    data["parts"][1]["props"]["pattern"] = "8,8"  # (its pin 3 is gone now)
    back = decode(json.loads(dumps(data)), reg)
    assert sorted(back.snapshot.wires) == [1]
