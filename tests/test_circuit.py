import numpy as np
import pytest

from pijl.logic import ONE, X, Z, ZERO, Level
from pijl.sim import Circuit
from pijl.sim.config import EngineConfig


def settle(c: Circuit, steps: int = 10) -> None:
    for _ in range(steps):
        c.step()


def test_nand_truth_table():
    c = Circuit()
    a, b, g, out = (
        c.add_part("IN"),
        c.add_part("IN"),
        c.add_part("NAND"),
        c.add_part("OUT"),
    )
    c.connect(a.outputs[0], g.inputs[0])
    c.connect(g.inputs[1], b.outputs[0])  # reversed order must work too
    c.connect(g.outputs[0], out.inputs[0])
    for va in (False, True):
        for vb in (False, True):
            a.outputs[0].state, b.outputs[0].state = va, vb
            settle(c)
            assert out.inputs[0].state is Level(ZERO + (not (va and vb)))


def test_invalid_connections_rejected():
    c = Circuit()
    a, b = c.add_part("IN"), c.add_part("IN")
    g = c.add_part("NOT")
    assert c.connect(a.outputs[0], b.outputs[0]) == (None, [])  # out -> out
    assert c.connect(g.inputs[0], g.outputs[0]) == (None, [])  # same part
    assert c.wires == []


def test_pin_to_pin_wires_are_normalized():
    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    w, _ = c.connect(g.inputs[0], a.outputs[0])
    assert w.src is a.outputs[0] and w.dst is g.inputs[0]


def test_rewiring_input_replaces_old_wire():
    c = Circuit()
    a, b, g = c.add_part("IN"), c.add_part("IN"), c.add_part("NOT")
    first, _ = c.connect(a.outputs[0], g.inputs[0])
    second, replaced = c.connect(b.outputs[0], g.inputs[0])
    assert replaced == [first] and c.wires == [second]


def test_remove_part_removes_its_wires():
    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    c.connect(a.outputs[0], g.inputs[0])
    c.remove_part(g)
    assert c.wires == [] and c.parts == [a]


def test_sr_latch_from_nands_holds_state():
    # Active-low SR latch: feedback loops must settle without blowing up.
    c = Circuit()
    s, r = c.add_part("IN"), c.add_part("IN")
    n1, n2 = c.add_part("NAND"), c.add_part("NAND")
    c.connect(s.outputs[0], n1.inputs[0])
    c.connect(r.outputs[0], n2.inputs[0])
    c.connect(n1.outputs[0], n2.inputs[1])
    c.connect(n2.outputs[0], n1.inputs[1])
    q = n1.outputs[0]

    s.outputs[0].state, r.outputs[0].state = (
        True,
        True,
    )  # hold, fresh: nothing says which way
    settle(c)
    assert q.state is X
    s.outputs[0].state, r.outputs[0].state = False, True  # set
    settle(c)
    assert q.state is ONE
    s.outputs[0].state = True  # release: hold
    settle(c)
    assert q.state is ONE
    r.outputs[0].state = False  # reset
    settle(c)
    assert q.state is ZERO
    r.outputs[0].state = True  # release: hold
    settle(c)
    assert q.state is ZERO


# ---- junctions / nets ------------------------------------------------------


def test_branch_fans_out_one_driver():
    c = Circuit()
    a, n1, n2 = c.add_part("IN"), c.add_part("NOT"), c.add_part("NOT")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    branch, _ = c.connect(trunk, n2.inputs[0])  # wire -> input pin
    assert branch.src is trunk and branch.dst is n2.inputs[0]
    a.outputs[0].state = True
    settle(c)
    assert n1.inputs[0].state is n2.inputs[0].state is ONE
    assert c.wire_state(branch) == (ONE, False)


def test_wire_ending_on_wire_joins_nets():
    c = Circuit()
    a, n1, n2 = c.add_part("IN"), c.add_part("NOT"), c.add_part("NOT")
    w1, _ = c.connect(a.outputs[0], n1.inputs[0])
    stub, _ = c.connect(n2.inputs[0], w1)  # input pin -> wire, either order
    assert stub.src is w1 and stub.dst is n2.inputs[0]
    a.outputs[0].state = True
    settle(c)
    assert n2.inputs[0].state


def test_two_drivers_agreeing_is_fine_disagreeing_is_a_conflict():
    c = Circuit()
    a, b, led = c.add_part("IN"), c.add_part("IN"), c.add_part("OUT")
    w, _ = c.connect(a.outputs[0], led.inputs[0])
    w2, _ = c.connect(b.outputs[0], w)  # second driver onto the same net
    for va, vb, expect_value, expect_conflict in [
        (False, False, ZERO, False),
        (True, True, ONE, False),
        (True, False, X, True),
        (False, True, X, True),
    ]:
        a.outputs[0].state, b.outputs[0].state = va, vb
        settle(c)
        assert led.inputs[0].state is expect_value
        assert c.wire_state(w) == c.wire_state(w2) == (expect_value, expect_conflict)


def test_undriven_net_floats():
    c = Circuit()
    n1, n2 = c.add_part("NOT"), c.add_part("NOT")
    assert not c.can_connect(
        n1.inputs[0], n2.inputs[0]
    )  # in -> in pin-to-pin is still rejected...
    a = c.add_part("IN")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    c.connect(trunk, n2.inputs[0])
    c.remove_part(a)  # ...but a net can lose its driver
    settle(c)
    assert n1.inputs[0].state is n2.inputs[0].state is Z
    assert n1.outputs[0].state is X  # a gate reads a floating input as X
    lone = c.add_part("NOT")  # an input with no wire at all floats too
    settle(c)
    assert lone.inputs[0].state is Z and lone.outputs[0].state is X


def test_removing_a_wire_removes_its_branches():
    c = Circuit()
    a, n1, n2, n3 = (
        c.add_part("IN"),
        c.add_part("NOT"),
        c.add_part("NOT"),
        c.add_part("NOT"),
    )
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    b1, _ = c.connect(trunk, n2.inputs[0])
    b2, _ = c.connect(b1, n3.inputs[0])  # branch of a branch
    assert c.remove_wire(trunk) == [trunk, b1, b2]
    assert c.wires == []


def test_with_descendants_matches_descendants():
    c = Circuit()
    a = c.add_part("IN")
    nots = [c.add_part("NOT") for _ in range(6)]
    trunk, _ = c.connect(a.outputs[0], nots[0].inputs[0])
    b1, _ = c.connect(trunk, nots[1].inputs[0])
    b2, _ = c.connect(b1, nots[2].inputs[0])
    b3, _ = c.connect(b2, nots[3].inputs[0])  # four deep
    other, _ = c.connect(nots[4].outputs[0], nots[5].inputs[0])
    c.remove_wire(b3)  # (a dead wire's slot is never found again)
    for start in ([trunk], [b1], [b2], [other], [b1, other]):
        want = sorted(w.slot for w in [*start, *c.descendants(*start)])
        got = c.with_descendants(np.array([w.slot for w in start])).tolist()
        assert got == want


def test_replacing_input_wire_cannot_saw_off_own_branch():
    c = Circuit()
    a, n1 = c.add_part("IN"), c.add_part("NOT")
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    b = c.add_part("NOT")
    branch, _ = c.connect(trunk, b.inputs[0])
    # wiring n1's input from `branch` would first remove trunk (and thus branch)
    assert not c.can_connect(n1.inputs[0], branch)
    assert not c.can_connect(n1.inputs[0], trunk)


def test_merge_splices_branch_onto_trunk():
    c = Circuit()
    a, n1, n2, n3 = (
        c.add_part("IN"),
        c.add_part("NOT"),
        c.add_part("NOT"),
        c.add_part("NOT"),
    )
    trunk, _ = c.connect(a.outputs[0], n1.inputs[0])
    branch, _ = c.connect(trunk, n2.inputs[0])
    twig, _ = c.connect(branch, n3.inputs[0])  # hangs off the branch
    c.merge(trunk, branch)
    assert trunk.src is a.outputs[0] and trunk.dst is n2.inputs[0]
    assert twig.src is trunk and branch not in c.wires
    assert c.wires == [trunk, twig]  # parents still before children
    # the lookups follow the splice
    assert c.attachments(trunk) == [twig] and c.attachments(branch) == []
    assert c.wires_at(n2.inputs[0]) == [trunk] and c.wires_at(n1.inputs[0]) == []
    assert c.descendants(trunk) == [twig]
    assert c.with_descendants(np.array([trunk.slot])).tolist() == [trunk.slot, twig.slot]
    a.outputs[0].state = True
    settle(c)
    assert (
        n2.inputs[0].state and n3.inputs[0].state and not n1.inputs[0].state
    )  # n1 was cut off


def test_wire_uids_are_stable():
    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    w, _ = c.connect(a.outputs[0], g.inputs[0])
    c.remove_wire(w)
    again, _ = c.connect(a.outputs[0], g.inputs[0], uid=w.uid)
    assert again.uid == w.uid
    b = c.add_part("NOT")
    assert c.connect(a.outputs[0], b.inputs[0])[0].uid == w.uid + 1


def test_take_changes_reports_only_what_changed():
    c = Circuit()
    a, g, out, idle = (
        c.add_part("IN"),
        c.add_part("NOT"),
        c.add_part("OUT"),
        c.add_part("NOT"),
    )
    w1, _ = c.connect(a.outputs[0], g.inputs[0])
    w2, _ = c.connect(g.outputs[0], out.inputs[0])
    assert c.take_changes() == (True, set(), [])  # new wiring: redraw everything
    settle(c)
    c.take_changes()
    settle(c)
    assert c.take_changes() == (False, set(), [])  # settled: nothing to redraw
    c.click(a)
    settle(c)
    everything, parts, wires = c.take_changes()
    assert not everything
    assert parts == {a, g, out}  # not `idle`
    assert set(wires) == {w1, w2}


def test_revision_counts_edits_not_clicks():
    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    w, _ = c.connect(a.outputs[0], g.inputs[0])
    r = c.revision
    c.click(a)
    settle(c)
    assert c.revision == r  # using the circuit isn't editing it
    c.remove_wire(w)
    assert c.revision > r


def test_rebuilding_wiring_skips_the_drawing_rules():
    c = Circuit()
    g = c.add_part("NAND")
    assert c.connect(g.outputs[0], g.inputs[0]) == (
        None,
        [],
    )  # can't be drawn by hand ...
    w, _ = c.connect(
        g.outputs[0], g.inputs[0], check=False
    )  # ... but undo must bring it back
    assert w is not None and c.wires_at(g.inputs[0]) == [w]


# ---- four-state vs. two-state ------------------------------------------------------


def test_known_values_match_a_plain_bool_simulation():
    """With every input wired and no X anywhere, four-state logic must do exactly what
    the old two-state engine did, tick for tick: checked against a tiny bool reference
    of the same timing (outputs from current inputs, then nets carry them). Random
    boards, feedback loops included."""
    import random

    fns = {
        "NAND": lambda a, b: not (a and b),
        "AND": lambda a, b: a and b,
        "OR": lambda a, b: a or b,
        "NOT": lambda a: not a,
    }
    arity = {"NAND": 2, "AND": 2, "OR": 2, "NOT": 1}
    for seed in range(30):
        rng = random.Random(seed)
        c = Circuit()
        ins = [c.add_part("IN") for _ in range(3)]
        gates = [c.add_part(rng.choice(list(fns))) for _ in range(rng.randint(3, 20))]
        sources = [p.outputs[0] for p in ins + gates]
        feed = {}  # input pin -> the output pin that drives it
        for g in gates:
            for pin in g.inputs:
                feed[pin] = src = rng.choice(
                    sources
                )  # (can be the gate's own output: a loop)
                c.connect(src, pin, check=False)
        for g in gates:
            for pin in g.pins:
                pin.state = ZERO  # the old engine powered up at 0, inputs included
        c.take_changes()  # (builds the nets now: that carries values at once, outside the timing)
        ref = {pin: False for p in ins + gates for pin in p.pins}
        for tick in range(60):
            if rng.random() < 0.3:
                s = rng.choice(ins).outputs[0]
                ref[s] = not ref[s]
                s.state = ref[s]
            c.step()
            new = {
                g.outputs[0]: fns[g.kind](*(ref[p] for p in g.inputs)) for g in gates
            }
            ref.update(new)
            ref.update({pin: ref[src] for pin, src in feed.items()})
            assert all(pin.state is (ONE if v else ZERO) for pin, v in ref.items()), (
                seed,
                tick,
            )


def _pin_rows(c: Circuit) -> list[tuple]:
    """Every part's pins as plain data: what the arrays hold for them."""
    s = c._pins
    return [
        (
            p.kind,
            p.uid,
            p.slot,
            [
                (
                    q.slot,
                    q.is_input,
                    q.index,
                    int(s.states[q.slot]),
                    bool(s.reader[q.slot]),
                    bool(s.weak[q.slot]),
                )
                for q in p.inputs + p.outputs + p.drives
            ],
        )
        for p in c.parts
    ]


def test_add_parts_matches_adding_one_by_one():
    kinds = ["IN", "NAND", "NOT", "PULLUP", "OUT", "NAND", "PULLDOWN", "IN", "XOR"]
    one = Circuit()
    for k in kinds:
        one.add_part(k)
    many = Circuit()
    parts = many.add_parts([many.registry.get(k) for k in kinds], [None] * len(kinds))
    assert [p.kind for p in parts] == kinds  # in order, whatever the grouping by type
    assert _pin_rows(many) == _pin_rows(one)
    assert all(p.live for p in parts)


def test_add_parts_gives_every_part_its_own_props():
    c = Circuit()
    t = c.registry.get("PULLUP")
    a, b = c.add_parts([t, t], [None, None])
    a.props["priority"] = 5
    assert (
        b.props["priority"] == 0 and a.props is not b.props and a.state is not b.state
    )


def test_wire_states_of_slots_it_never_had():
    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    w, _ = c.connect(a.outputs[0], g.inputs[0])
    value, conflict, has = c.wire_states(np.array([w.slot, 99], np.intp))
    assert has.tolist() == [True, False]
    assert value[0] == ZERO and not conflict[0]


def test_removed_parts_and_wires_are_let_go():
    c = Circuit()
    a, b = c.add_part("IN"), c.add_part("NOT")
    w, _ = c.connect(a.outputs[0], b.inputs[0])
    pin_slots = [p.slot for p in b.pins]
    c.remove_parts([b])  # (takes its wires along)
    c.step()
    # nothing in the circuit holds them any more (slots stay taken: never reused)
    assert all(c._pins.pins[s] is None for s in pin_slots)
    assert c._wire_slots.wires[w.slot] is None
    assert not c._at and c.pin_count == 3


# ---- free ends: a wire end attached to nothing is the wire itself -----------------


def test_free_end_is_the_wire_itself_and_joins_nothing():
    from pijl.sim import FREE

    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    stub, _ = c.connect(FREE, a.outputs[0])
    assert stub.src is a.outputs[0] and stub.dst is stub and stub.is_free(stub.dst)
    assert c.ends_on(stub) == ()  # (not its own branch)
    c.connect(stub, g.inputs[0])  # a branch off the stub carries a's value
    a.outputs[0].state = True
    settle(c)
    assert g.inputs[0].state is Level(ONE)
    floating, _ = c.connect(FREE, FREE)
    assert floating.src is floating.dst is floating
    settle(c)  # (nets with no pins at all)


def test_detach_keeps_branches_and_frees_the_end():
    c = Circuit()
    a, g, h = c.add_part("IN"), c.add_part("NOT"), c.add_part("NOT")
    w, _ = c.connect(a.outputs[0], g.inputs[0])
    branch, _ = c.connect(w, h.inputs[0])
    c.detach(w, "src")
    assert w.src is w and c.wires_at(a.outputs[0]) == [] and c.attachments(w) == [branch]
    settle(c)
    assert g.inputs[0].state is h.inputs[0].state is Level(Z)  # no driver left
    c.remove_part(a)  # nothing of it was attached any more
    assert c.wires == [w, branch]


def test_merge_onto_a_free_end_leaves_a_free_end():
    from pijl.sim import FREE

    c = Circuit()
    a, g = c.add_part("IN"), c.add_part("NOT")
    trunk, _ = c.connect(a.outputs[0], g.inputs[0])
    stub, _ = c.connect(trunk, FREE)
    c.merge(trunk, stub)
    assert trunk.dst is trunk and c.wires == [trunk]


# ---- dirty-set evaluation --------------------------------------------------------------


def _test_registry():
    """The shipped parts plus some odd ones: a part that isn't pure (it blinks), a
    pure one with 3 inputs, one with 5 (too many for a table), and one whose eval
    fails on Z (no table; it faults when a Z reaches it)."""
    from pijl.logic import Logic
    from pijl.parts import TEMPLATES, PartType, part
    from pijl.parts.registry import load

    class Blink(PartType):
        kind, outs = "BLINK", ("out",)

        def eval(self, ctx):
            return bool((ctx.tick // 3) % 2)

    def picky(a, b):
        if (a.codes == Z).any() or (b.codes == Z).any():
            raise ValueError("no Z, please")
        return a ^ b

    reg = load(TEMPLATES)
    reg.add(Blink())
    reg.add(part("MAJ", ins=("a", "b", "c"), outs=("out", "nout"), eval=lambda a, b, c: ((a & b) | (b & c) | (a & c), ~((a & b) | (b & c) | (a & c)))))
    reg.add(part("AND5", ins=tuple("abcde"), outs=("out",), eval=lambda a, b, c, d, e: a & b & c & d & e))
    reg.add(part("PICKY", ins=("a", "b"), outs=("out",), eval=picky))
    return reg


def test_every_engine_matches_the_plain_one():
    """Every engine config must do exactly what the plain one (dirty=off,
    eval=batches: everything runs, a kind at a time) does, tick for tick: random
    boards with loops, fights, tri-states, pulls, parts that aren't pure, kinds too
    big for a table, a kind that faults, pins written from outside (inputs and outputs
    alike, X and Z included), settling, and wiring edits while it runs."""
    import itertools
    import random

    from pijl.sim.config import OPTIONS

    reg = _test_registry()
    kinds = ["IN", "NAND", "AND", "OR", "NOT", "XOR", "BUF", "TRI", "PULLUP", "PULLDOWN", "OUT"]
    kinds += ["BLINK", "MAJ", "AND5", "PICKY"]
    levels = [ZERO, ONE, X, Z]
    configs = [EngineConfig(**dict(zip(OPTIONS, combo))) for combo in itertools.product(*OPTIONS.values())]
    reference = EngineConfig(dirty="off", eval="batches")
    configs.remove(reference)
    # (boards this small mostly run everything: also try each dirty-set config with a
    # cost model that always picks the dirty parts out)
    configs += [(cfg, "always pick") for cfg in configs if cfg.dirty == "adaptive"]

    class AlwaysPick:
        FULL_FIXED, FULL_PER_PART = 1e12, 0.0
        PICK_FIXED, PICK_PER_KIND, PICK_PER_PART = 0.0, 0.0, 1.0

    def build(seed: int, config):
        rng = random.Random(seed)
        config, costs = config if isinstance(config, tuple) else (config, None)
        c = Circuit(reg, settle_ticks=rng.choice([0, 0, 6]), seed=seed, config=config)
        if costs:
            c._costs = AlwaysPick
        parts = [c.add_part(rng.choice(kinds if seed % 3 else kinds[:-4])) for _ in range(rng.randint(4, 40))]
        outs = [q for p in parts for q in p.outputs]
        ins = [q for p in parts for q in p.inputs]
        for pin in ins:
            if outs and rng.random() < 0.9:
                c.connect(rng.choice(outs), pin, check=False)
        for _ in range(rng.randint(0, 4)):  # outputs wired together: fights
            if len(outs) > 1:
                c.connect(rng.choice(outs), rng.choice(outs), check=False)
        return rng, c, parts, outs, ins

    def tick(rng, c, parts, outs, ins):
        r = rng.random()
        if r < 0.25:  # drive a switch, or poke any pin at all
            pins = [p.outputs[0] for p in parts if p.kind == "IN"] or outs
            if rng.random() < 0.3:
                pins = outs + ins
            if pins:
                rng.choice(pins).state = rng.choice(levels)
        elif r < 0.27 and outs and ins:  # an edit while it runs
            c.connect(rng.choice(outs), rng.choice(ins), check=False)
        c.step()

    for seed in range(60):
        ref = build(seed, reference)
        others = [build(seed, cfg) for cfg in configs]
        b = ref[1]
        for t in range(80):
            tick(*ref)
            n = b._pins.n
            for cfg, board in zip(configs, others):
                tick(*board)
                a = board[1]
                where = (str(cfg), seed, t)
                assert np.array_equal(a._pins.states[:n], b._pins.states[:n]), where
                assert np.array_equal(a.net_value, b.net_value), where
                assert np.array_equal(a.net_conflict, b.net_conflict), where
                assert a.faults == b.faults, where
        want = b.run_until_stable(200)
        for cfg, board in zip(configs, others):
            assert board[1].run_until_stable(200) == want, (str(cfg), seed)


def test_lut_tables():
    from pijl.sim import lut

    reg = _test_registry()
    nand = lut.tabulate(reg.get("NAND"))
    assert nand.shape == (1, 16)
    row = lambda a, b: int(a) + 4 * int(b)  # noqa: E731
    assert nand[0, row(ONE, ONE)] == ZERO and nand[0, row(ZERO, X)] == ONE
    assert nand[0, row(ONE, X)] == X and nand[0, row(ONE, Z)] == X  # (Z reads as X)
    assert lut.tabulate(reg.get("MAJ")).shape == (2, 64)
    for kind in ("BLINK", "AND5", "PICKY", "IN", "OUT"):
        assert lut.tabulate(reg.get(kind)) is None, kind


def test_engine_config_text_and_binding():
    cfg = EngineConfig.parse("dirty=off")
    assert cfg == EngineConfig(dirty="off") and EngineConfig.parse(str(cfg)) == cfg
    assert EngineConfig.parse("") == EngineConfig()  # (defaults: the fastest proven)
    for bad in ("dirty=maybe", "turbo=on", "dirty"):
        with pytest.raises(ValueError):
            EngineConfig.parse(bad)
    from pijl.sim import dirty, plain

    for name, module in (("off", plain), ("adaptive", dirty)):
        c = Circuit(config=EngineConfig(dirty=name))
        assert c.step.__func__ is module.step and c.run_until_stable.__func__ is module.run_until_stable
