import random

import pytest

from pijl.logic import ONE, X, ZERO, Level
from pijl.macros import Catalog
from pijl.parts import builtin_registry, load, TEMPLATES
from pijl.sim import Circuit
from pijl.snapshot import MACRO, Snapshot
from pijl.storage import decode, dumps, encode

import json
from pathlib import Path

GATES = ["NAND", "AND", "OR", "NOT"]
ARITY = {"NAND": 2, "AND": 2, "OR": 2, "NOT": 1}


def level(b: bool) -> Level:
    return ONE if b else ZERO


def catalog(defs: dict[str, Snapshot], registry=None) -> Catalog:
    return Catalog(registry or builtin_registry(), lambda name: defs[name])


def build(c: Circuit, snap: Snapshot) -> dict[int, object]:
    """Put a snapshot of pin-to-pin wires on a bare circuit (what the editor's restore does)."""
    parts = {uid: c.add_part(kind, uid) for uid, (kind, *_rest) in snap.parts.items()}
    for uid in sorted(snap.wires):
        (_, su, _, si), (_, du, _, di) = snap.wires[uid][:2]
        c.connect(parts[su].outputs[si], parts[du].inputs[di], uid)
    return parts


def pin(src_uid, index, is_input):
    return ("p", src_uid, is_input, index)


def w(src, si, dst, di):
    return (pin(src, si, False), pin(dst, di, True), (), None, None)


# ---- settling ------------------------------------------------------------------------


def latch(c: Circuit):
    """Cross-coupled NANDs with both set/reset inactive (high): should hold a value."""
    one = c.add_part("IN")
    one.outputs[0].state = True
    a, b = c.add_part("NAND"), c.add_part("NAND")
    c.connect(one.outputs[0], a.inputs[0])
    c.connect(one.outputs[0], b.inputs[1])
    c.connect(a.outputs[0], b.inputs[0])
    c.connect(b.outputs[0], a.inputs[1])
    return a.outputs[0], b.outputs[0]


def test_without_settling_a_fresh_latch_stays_unknown():
    c = Circuit()
    q, qb = latch(c)
    for _ in range(200):
        c.step()
    assert q.state is qb.state is X  # nothing ever says which way it falls


def test_settling_breaks_the_tie_and_then_holds():
    outcomes = set()
    for seed in range(20):
        c = Circuit(settle_ticks=64, seed=seed)
        q, qb = latch(c)
        for _ in range(64):
            c.step()
        held = []
        for _ in range(50):
            c.step()
            held.append((q.state, qb.state))
        assert len(set(held)) == 1 and {*held[0]} == {
            ZERO,
            ONE,
        }  # settled into a real state
        outcomes.add(held[0])
    assert outcomes == {(ONE, ZERO), (ZERO, ONE)}  # either one, depending on the noise


def test_settling_is_reproducible():
    def run(seed):
        c = Circuit(settle_ticks=64, seed=seed)
        q, qb = latch(c)
        out = []
        for _ in range(80):
            c.step()
            out.append((q.state, qb.state))
        return out

    assert run(7) == run(7)


# ---- macro types -----------------------------------------------------------------------


def half_adder() -> Snapshot:
    """Inputs a (top), b; outputs sum (XOR from NANDs), carry. Port uids 1, 2 / 10, 11."""
    parts = {
        1: ("IN", "a", 0.0, 100.0, {}),
        2: ("IN", "", 0.0, 0.0, {}),
        3: ("NAND", "", 100.0, 50.0, {}),
        4: ("NAND", "", 200.0, 100.0, {}),
        5: ("NAND", "", 200.0, 0.0, {}),
        6: ("NAND", "", 300.0, 50.0, {}),
        7: ("NOT", "", 300.0, -50.0, {}),
        10: ("OUT", "sum", 400.0, 50.0, {}),
        11: ("OUT", "carry", 400.0, -50.0, {}),
    }
    wires = {
        1: w(1, 0, 3, 0),
        2: w(2, 0, 3, 1),
        3: w(1, 0, 4, 0),
        4: w(3, 0, 4, 1),
        5: w(3, 0, 5, 0),
        6: w(2, 0, 5, 1),
        7: w(4, 0, 6, 0),
        8: w(5, 0, 6, 1),
        9: w(6, 0, 10, 0),
        10: w(3, 0, 7, 0),
        11: w(7, 0, 11, 0),
    }
    return Snapshot(parts, wires)


def test_pins_come_from_the_ports_top_to_bottom_and_unlabeled_ones_are_numbered():
    t = catalog({"ha": half_adder()}).get(MACRO + "ha")
    assert t.kind == "macro:ha" and t.title == "ha"
    assert t.in_ids == (1, 2) and t.ins == ("a", "2")
    assert t.out_ids == (10, 11) and t.outs == ("sum", "carry")


def test_a_macro_computes_like_its_body():
    cat = catalog({"ha": half_adder()})
    for a in (False, True):
        for b in (False, True):
            c = Circuit(cat)
            ia, ib, m = c.add_part("IN"), c.add_part("IN"), c.add_part("macro:ha")
            s_led, c_led = c.add_part("OUT"), c.add_part("OUT")
            ia.outputs[0].state, ib.outputs[0].state = a, b
            for src, dst in (
                (ia.outputs[0], m.inputs[0]),
                (ib.outputs[0], m.inputs[1]),
                (m.outputs[0], s_led.inputs[0]),
                (m.outputs[1], c_led.inputs[0]),
            ):
                c.connect(src, dst)
            for _ in range(
                10
            ):  # (exact tick-for-tick timing: see the wrapping test below)
                c.step()
            assert (s_led.inputs[0].state, c_led.inputs[0].state) == (
                level(a != b),
                level(a and b),
            )
            assert m.outputs[0].state is level(
                a != b
            )  # the instance's pins show the value too


def test_nested_macros_and_removal():
    two = Snapshot(
        {
            1: ("IN", "x", 0.0, 0.0, {}),
            2: ("macro:ha", "", 50.0, 0.0, {}),
            3: ("OUT", "s", 100.0, 0.0, {}),
        },
        {1: w(1, 0, 2, 0), 2: w(1, 0, 2, 1), 3: w(2, 0, 3, 0)},
    )  # x XOR x = 0 ... then NOT it
    cat = catalog({"ha": half_adder(), "two": two})
    c = Circuit(cat)
    m = c.add_part("macro:two")
    assert len(c.hidden_parts) == 3 + len(half_adder().parts)
    led = c.add_part("OUT")
    c.connect(m.outputs[0], led.inputs[0])
    for _ in range(5):
        c.step()
    assert led.inputs[0].state is X  # x floats: X XOR X could be anything
    x = c.add_part("IN")
    c.connect(x.outputs[0], m.inputs[0])
    for _ in range(5):
        c.step()
    assert led.inputs[0].state is ZERO
    c.remove_part(m)
    assert c.hidden_parts == [] and c.hidden_wires == []


def test_a_macro_cannot_contain_itself():
    loop = Snapshot({1: ("macro:loop", "", 0.0, 0.0, {})}, {})
    cat = catalog(
        {"loop": loop, "outer": Snapshot({1: ("macro:loop", "", 0.0, 0.0, {})}, {})}
    )
    assert "macro:loop" not in cat
    with pytest.raises(KeyError, match="contains itself"):
        cat.get("macro:loop")
    assert (
        cat.book.contains("outer", "loop") is False
    )  # unloadable: treated as not containing


def test_contains_looks_all_the_way_down():
    mid = Snapshot({1: ("macro:ha", "", 0.0, 0.0, {})}, {})
    top = Snapshot({1: ("macro:mid", "", 0.0, 0.0, {})}, {})
    cat = catalog({"ha": half_adder(), "mid": mid, "top": top})
    assert cat.book.contains("top", "ha") and cat.book.contains("mid", "ha")
    assert not cat.book.contains("ha", "top")


def test_hooks_reach_parts_inside_macros():
    reg = load(TEMPLATES, Path(__file__).parent / "part_scripts" / "good")
    cat = catalog({"box": Snapshot({1: ("COUNTER", "", 0.0, 0.0, {})}, {})}, reg)
    c = Circuit(cat)
    ghost = c.add_part("macro:box", live=False)
    counter = reg.get("COUNTER")
    assert counter.log == []
    c.open_part(ghost)
    inner = ghost.inner[1]
    assert counter.log == [("open", inner.uid)]
    c.close_all()
    assert counter.log[-1] == ("close", inner.uid)


# ---- save files -------------------------------------------------------------------------


def test_macro_instances_in_files_and_pins_by_port_uid():
    defs = {"ha": half_adder()}
    board = Snapshot(
        {
            1: ("IN", "", 0.0, 0.0, {}),
            2: ("macro:ha", "", 100.0, 0.0, {}),
            3: ("OUT", "", 200.0, 0.0, {}),
        },
        {1: w(1, 0, 2, 1), 2: w(2, 1, 3, 0)},
    )  # into b, carry out
    data = encode(board, catalog(defs))
    assert data["parts"][1] == {"uid": 2, "macro": "ha", "pos": [100, 0]}
    assert data["wires"][0]["to"] == {"part": 2, "pin": 2} and data["wires"][1][
        "from"
    ] == {"part": 2, "pin": 11}
    assert decode(json.loads(dumps(data)), catalog(defs)).snapshot == board

    # Move b above a inside the macro: pin order flips, but the wires follow their pins.
    moved = half_adder()
    moved.parts[2] = ("IN", "", 0.0, 500.0, {})
    loaded = decode(data, catalog({"ha": moved}))
    assert loaded.warnings == []
    assert loaded.snapshot.wires[1][1] == ("p", 2, True, 0)  # b is now pin 0

    # Delete the carry output inside the macro: only the wire to it is dropped.
    gone = half_adder()
    del gone.parts[11]
    del gone.wires[11]
    loaded = decode(data, catalog({"ha": gone}))
    assert set(loaded.snapshot.wires) == {1} and loaded.warnings


def test_missing_macro_is_dropped_with_a_warning():
    data = {
        "pijl": 1,
        "parts": [{"uid": 1, "macro": "nope", "pos": [0, 0]}],
        "wires": [],
    }
    loaded = decode(data, catalog({}))
    assert loaded.snapshot.parts == {} and "nope" in loaded.warnings[0]


# ---- the big one: wrapping never changes behavior ------------------------------------------


def random_board(rng: random.Random, n_in: int, n_gates: int) -> Snapshot:
    parts = {i + 1: ("IN", "", 0.0, float(i * 40), {}) for i in range(n_in)}
    for u in range(n_in + 1, n_in + n_gates + 1):
        parts[u] = (
            rng.choice(GATES),
            "",
            float(rng.randrange(100, 900)),
            float(rng.randrange(0, 900)),
            {},
        )
    wires, wu = {}, 1
    for u, (kind, *_rest) in parts.items():
        for i in range(ARITY.get(kind, 0)):
            src = rng.choice(
                [s for s in parts if s != u and parts[s][0] != "OUT"]
            )  # feedback allowed
            wires[wu] = w(src, 0, u, i)
            wu += 1
    return Snapshot(parts, wires)


def wrap(snap: Snapshot, inside: set[int], name: str) -> tuple[Snapshot, Snapshot, int]:
    """Cut `inside` out of `snap` into a macro `name`. Returns (outer board, macro body, instance uid)."""
    body_parts = {u: snap.parts[u] for u in inside}
    body_wires, outer_wires = {}, {}
    uid = max(snap.parts) + 1
    wid = max(snap.wires, default=0) + 1
    ins: dict[tuple, int] = {}  # outside source pin -> IN port uid
    outs: dict[tuple, int] = {}  # inside source pin -> OUT port uid
    inst = uid + 1000
    for wuid, data in snap.wires.items():
        src, dst = data[0], data[1]
        s_in, d_in = src[1] in inside, dst[1] in inside
        if s_in and d_in:
            body_wires[wuid] = data
        elif not s_in and not d_in:
            outer_wires[wuid] = data
        elif d_in:  # coming in
            if src not in ins:
                ins[src] = port = uid
                uid += 1
                body_parts[port] = (
                    "IN",
                    "",
                    -100.0,
                    10000.0 - 10 * len(ins),
                    {},
                )  # creation order = top to bottom
                outer_wires[wid] = (
                    src,
                    ("p", inst, True, len(ins) - 1),
                    (),
                    None,
                    None,
                )
                wid += 1
            body_wires[wid] = (("p", ins[src], False, 0), dst, (), None, None)
            wid += 1
        else:  # going out
            if src not in outs:
                outs[src] = port = uid
                uid += 1
                body_parts[port] = ("OUT", "", 2000.0, 10000.0 - 10 * len(outs), {})
                body_wires[wid] = (src, ("p", port, True, 0), (), None, None)
                wid += 1
            k = list(outs).index(src)
            outer_wires[wid] = (("p", inst, False, k), dst, (), None, None)
            wid += 1
    outer_parts = {u: d for u, d in snap.parts.items() if u not in inside}
    outer_parts[inst] = (MACRO + name, "", 500.0, 500.0, {})
    return Snapshot(outer_parts, outer_wires), Snapshot(body_parts, body_wires), inst


def build_any(c: Circuit, snap: Snapshot):
    parts = {uid: c.add_part(d[0], uid) for uid, d in snap.parts.items()}
    for uid in sorted(snap.wires):
        (_, su, _, si), (_, du, _, di) = snap.wires[uid][:2]
        c.connect(parts[su].outputs[si], parts[du].inputs[di], uid)
    return parts


@pytest.mark.parametrize("seed", range(25))
def test_wrapping_parts_in_macros_never_changes_behavior(seed):
    rng = random.Random(seed)
    flat = random_board(rng, n_in=3, n_gates=14)
    gates = [u for u, d in flat.parts.items() if d[0] != "IN"]
    outer_set = set(rng.sample(gates, 8))
    inner_set = set(rng.sample(sorted(outer_set), 3))  # a macro inside the macro

    outer, body, inst = wrap(flat, outer_set, "outer")
    body2, body_inner, inst2 = wrap(body, inner_set, "inner")
    cat = catalog({"outer": body2, "inner": body_inner})

    from pijl.sim.config import EngineConfig, default

    flat_config = EngineConfig.parse("compile=off", default())  # (compiled: timing changes)
    c_flat, c_wrap = Circuit(config=flat_config), Circuit(cat, config=flat_config)
    p_flat = build_any(c_flat, flat)
    p_wrap = build_any(c_wrap, outer)

    def where(u):
        if u in inner_set:
            return p_wrap[inst].inner[inst2].inner[u]
        if u in outer_set:
            return p_wrap[inst].inner[u]
        return p_wrap[u]

    sources = [u for u, d in flat.parts.items() if d[0] == "IN"]
    for tick in range(60):
        if rng.random() < 0.3:
            s = rng.choice(sources)
            v = not p_flat[s].outputs[0].state
            p_flat[s].outputs[0].state = where(s).outputs[0].state = v
        c_flat.step()
        c_wrap.step()
        for u in flat.parts:
            assert [p.state for p in p_flat[u].pins] == [
                p.state for p in where(u).pins
            ], (seed, tick, u)


def _tree(p):
    yield p
    for q in p.inner.values():
        yield from _tree(q)


@pytest.mark.parametrize("seed", range(10))
def test_later_instances_of_a_macro_are_built_like_the_first(seed):
    # The first instance of a macro type is built part by part; later ones are stamped
    # from it (Circuit._stamp). They must be the same parts, wiring and behavior.
    rng = random.Random(seed)
    flat = random_board(rng, n_in=3, n_gates=14)
    gates = [u for u, d in flat.parts.items() if d[0] != "IN"]
    outer_set = set(rng.sample(gates, 8))
    inner_set = set(rng.sample(sorted(outer_set), 3))
    _outer, body, _inst = wrap(flat, outer_set, "outer")
    body2, body_inner, _inst2 = wrap(body, inner_set, "inner")
    c = Circuit(catalog({"outer": body2, "inner": body_inner}))
    a, b = c.add_part("macro:outer"), c.add_part("macro:outer")

    def shape(p):
        return [
            (q.kind, q.uid, q.label, q.props, [x.passive for x in q.pins],
             [(w.uid, [type(e).__name__ for e in w.ends]) for w in q.inner_wires],
             [(x.index, x.is_input, y.index, y.is_input) for x, y in q.links])
            for q in _tree(p)
        ]  # fmt: skip

    assert shape(a)[1:] == shape(b)[1:] and len(list(_tree(a))) > 10
    feeds = [[c.add_part("IN") for _ in m.inputs] for m in (a, b)]
    for m, ins in zip((a, b), feeds):
        for i, src in enumerate(ins):
            c.connect(src.outputs[0], m.inputs[i])
    for tick in range(60):
        if rng.random() < 0.3:
            i = rng.randrange(len(a.inputs))
            v = not feeds[0][i].outputs[0].state
            feeds[0][i].outputs[0].state = feeds[1][i].outputs[0].state = v
        c.step()
        for pa, pb in zip(_tree(a), _tree(b)):
            assert [x.state for x in pa.pins] == [x.state for x in pb.pins], (seed, tick)


def test_a_macro_inside_that_changed_is_built_anew():
    from pijl.macros import MacroType

    outer = Snapshot(
        {1: ("IN", "", 0.0, 0.0, {}), 2: ("macro:inner", "", 100.0, 0.0, {}), 3: ("OUT", "", 200.0, 0.0, {})},
        {1: w(1, 0, 2, 0), 2: w(2, 0, 3, 0)},
    )
    inv = Snapshot(
        {1: ("IN", "", 0.0, 0.0, {}), 2: ("NOT", "", 100.0, 0.0, {}), 3: ("OUT", "", 200.0, 0.0, {})},
        {1: w(1, 0, 2, 0), 2: w(2, 0, 3, 0)},
    )
    buf = Snapshot(
        {1: ("IN", "", 0.0, 0.0, {}), 3: ("OUT", "", 200.0, 0.0, {})}, {1: w(1, 0, 3, 0)}
    )
    cat = catalog({"outer": outer, "inner": inv})

    class Swapped:  # the same outer type, but "inner" is now something else
        swap = {}

        def get(self, kind):
            return self.swap.get(kind) or cat.get(kind)

        def __contains__(self, kind):
            return kind in cat

    reg = Swapped()
    c = Circuit(reg)
    first = c.add_part("macro:outer")
    Swapped.swap["macro:inner"] = MacroType("inner", buf, cat)
    second = c.add_part("macro:outer")
    assert first.inner[2].type is cat.get("macro:inner")
    assert second.inner[2].type is Swapped.swap["macro:inner"]
    assert [q.kind for q in _tree(second)] == ["macro:outer", "IN", "macro:inner", "IN", "OUT", "OUT"]


def test_free_ends_inside_a_macro_join_nothing():
    """A body wire with a free end (it names its own uid), and one free at both ends:
    the macro works as if they weren't there, and their handles come out free."""
    body = Snapshot(
        {1: ("IN", "a", 0, 0, {}), 2: ("NOT", "", 0, 0, {}), 3: ("OUT", "y", 0, 0, {})},
        {
            1: w(1, 0, 2, 0),
            2: w(2, 0, 3, 0),
            3: (("w", 2), ("w", 3), (), (5.0, 5.0), (9.0, 9.0)),  # a stub off wire 2
            4: (("w", 4), ("w", 4), (), (0.0, 0.0), (1.0, 0.0)),  # attached to nothing
        },
    )
    c = Circuit(catalog({"inv": body}))
    a, m, out = c.add_part("IN"), c.add_part(MACRO + "inv"), c.add_part("OUT")
    c.connect(a.outputs[0], m.inputs[0])
    c.connect(m.outputs[0], out.inputs[0])
    for v in (False, True):
        a.outputs[0].state = v
        for _ in range(10):
            c.step()
        assert out.inputs[0].state is level(not v)
    stub, floating = sorted((x for x in m.inner_wires if x.uid in (3, 4)), key=lambda x: x.uid)
    assert stub.dst is stub and stub.src.uid == 2
    assert floating.src is floating.dst is floating
