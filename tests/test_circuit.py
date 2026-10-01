import numpy as np

from pijl.logic import ONE, X, Z, ZERO, Level
from pijl.sim import Circuit


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
