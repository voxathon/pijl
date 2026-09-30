import time
from pathlib import Path

import pytest

from pijl.logic import ONE, X, Z, ZERO
from pijl.parts import TEMPLATES, builtin_registry, load
from pijl.sim import Circuit

SCRIPTS = Path(__file__).parent / "part_scripts"


@pytest.fixture
def circuit():
    return Circuit(load(TEMPLATES, SCRIPTS / "good"))


def settle(c: Circuit, steps: int = 10) -> None:
    for _ in range(steps):
        c.step()


def level(b: bool):
    return ONE if b else ZERO


# ---- loading ---------------------------------------------------------------------


def test_builtins_are_the_ports_plus_the_template_scripts():
    reg = builtin_registry()
    assert reg.errors == []
    assert [t.kind for t in reg] == ["IN", "OUT", "AND", "BUF", "NAND", "NOT", "OR", "TRI", "XOR",
                                     "PULLUP", "PULLDOWN"]  # ports, then by path
    assert {t.category for t in reg} == {"I/O", "GATES", "WIRING"}
    assert {t.api for t in reg} == {2}


def test_scripts_load_recursively_and_helpers_are_skipped():
    reg = load(SCRIPTS / "good")
    assert reg.errors == []
    assert {"COUNTER", "BRIDGE", "INV", "HIGH", "AND16", "SPLIT", "BOOM"} <= set(reg.types)


def test_broken_scripts_are_skipped_whole_and_reported():
    reg = load(SCRIPTS / "bad")
    assert set(reg.types) == {"IN", "OUT", "FINE"}  # nothing from the bad ones, not even half of partial.py
    errors = "\n".join(reg.errors)
    for name in ("broken.py", "old_api.py", "partial.py", "port.py", "sink.py"):
        assert name in errors
    assert len(reg.errors) == 5


def test_two_folders_cannot_define_the_same_part():
    reg = load(TEMPLATES, TEMPLATES)
    assert [t.kind for t in reg].count("NAND") == 1
    assert len(reg.errors) == len(list(TEMPLATES.rglob("*.py")))  # every template script, the second time


def test_missing_folder_is_just_empty(tmp_path):
    assert set(load(tmp_path / "nope").types) == {"IN", "OUT"}


# ---- evaluation ------------------------------------------------------------------


def test_every_gate_on_arrays():
    c = Circuit()
    cases = {"AND": lambda a, b: a and b, "NAND": lambda a, b: not (a and b),
             "OR": lambda a, b: a or b}
    rows = [(a, b) for a in (False, True) for b in (False, True)]
    for kind, expect in cases.items():
        gates = []
        for a, b in rows:
            ia, ib, g = c.add_part("IN"), c.add_part("IN"), c.add_part(kind)
            ia.outputs[0].state, ib.outputs[0].state = a, b
            c.connect(ia.outputs[0], g.inputs[0])
            c.connect(ib.outputs[0], g.inputs[1])
            gates.append(g)
        settle(c, 3)
        assert [g.outputs[0].state for g in gates] == [level(expect(a, b)) for a, b in rows], kind
    n = c.add_part("NOT")
    settle(c, 1)
    assert n.outputs[0].state is X  # its input floats


def test_scalars_many_inputs_and_many_outputs(circuit):
    high = [circuit.add_part("HIGH") for _ in range(3)]
    wide = circuit.add_part("AND16")
    split = circuit.add_part("SPLIT")
    circuit.connect(high[0].outputs[0], split.inputs[0])
    settle(circuit, 3)
    assert all(h.outputs[0].state is ONE for h in high)
    assert wide.outputs[0].state is X  # nothing wired: an API 1 part can't know what X does
    assert [p.state for p in split.outputs] == [ONE, ZERO]


def test_relative_imports_inside_a_script_folder(circuit):
    inv, high = circuit.add_part("INV"), circuit.add_part("HIGH")
    circuit.connect(high.outputs[0], inv.inputs[0])
    settle(circuit, 2)
    assert inv.outputs[0].state is ZERO


def test_a_raising_eval_disables_only_its_own_kind(circuit):
    boom, n, high = circuit.add_part("BOOM"), circuit.add_part("NOT"), circuit.add_part("HIGH")
    circuit.connect(high.outputs[0], n.inputs[0])
    settle(circuit, 3)
    assert "BOOM" in circuit.faults and "ZeroDivisionError" in circuit.faults["BOOM"]
    assert circuit.errors == [circuit.faults["BOOM"]]  # reported once
    assert boom.outputs[0].state is X  # broken: nobody knows what it would say
    assert n.outputs[0].state is ZERO  # everything else keeps running


# ---- lifecycle -------------------------------------------------------------------


def test_ghosts_are_never_opened_and_opened_parts_always_closed(circuit):
    counter = circuit.registry.get("COUNTER")
    ghost = circuit.add_part("COUNTER", live=False)
    circuit.frame()
    settle(circuit)
    assert counter.log == [] and ghost.state == {}
    circuit.remove_part(ghost)  # a cancelled placement
    assert counter.log == []

    a = circuit.add_part("COUNTER", live=False)
    circuit.open_part(a)  # placed
    b = circuit.add_part("COUNTER")
    assert counter.log == [("open", a.uid), ("open", b.uid)]
    circuit.remove_part(a)
    circuit.close_all()
    assert counter.log[2:] == [("close", a.uid), ("close", b.uid)]


def test_frame_state_and_props(circuit):
    a, b = circuit.add_part("COUNTER"), circuit.add_part("COUNTER")
    b.props["step"] = 2
    circuit.frame()
    circuit.step()
    assert (a.state["count"], b.state["count"]) == (1, 2)
    assert (a.outputs[0].state, b.outputs[0].state) == (ONE, ZERO)
    assert circuit.registry.get("COUNTER").props == {"step": 1}  # instances got copies


def test_impure_ghosts_are_not_evaluated(circuit):
    ghost = circuit.add_part("COUNTER", live=False)
    settle(circuit)  # its eval would need state["count"], which only open() sets
    assert circuit.faults == {}
    assert ghost.outputs[0].state is X  # as it powered up


def test_a_bridge_thread_feeds_eval(circuit):
    bridge = circuit.add_part("BRIDGE")
    deadline = time.monotonic() + 2
    while not bridge.state["value"] and time.monotonic() < deadline:
        time.sleep(0.01)
    circuit.step()
    assert bridge.outputs[0].state is ONE
    circuit.remove_part(bridge)
    assert not bridge.state["thread"].is_alive()


def test_clicks_go_to_clickable_placed_parts():
    c = Circuit()
    switch, gate, ghost = c.add_part("IN"), c.add_part("NAND"), c.add_part("IN", live=False)
    assert c.click(switch) and switch.outputs[0].state is ONE
    assert not c.click(gate)
    assert not c.click(ghost) and ghost.outputs[0].state is ZERO  # switches start off


# ---- four-state ------------------------------------------------------------------


def test_api_1_and_2_scripts_see_x_differently(circuit):
    """AND(0, X) is 0 -- if the script can see the X. API 1 gets bools, so it says X."""
    zero, old, new = circuit.add_part("IN"), circuit.add_part("AND16"), circuit.add_part("AND4S")
    for g in (old, new):
        circuit.connect(zero.outputs[0], g.inputs[0])  # (the other inputs float: X)
    settle(circuit, 3)
    assert new.outputs[0].state is ZERO
    assert old.outputs[0].state is X
    assert circuit.registry.get("AND4S").api == 2 and circuit.registry.get("AND16").api == 1


def bus(circuit, *kinds):
    """Parts whose outputs all drive one net, read by an OUT."""
    led = circuit.add_part("OUT")
    parts = [circuit.add_part(k) for k in kinds]
    trunk, _ = circuit.connect(parts[0].outputs[0], led.inputs[0])
    for p in parts[1:]:
        circuit.connect(p.outputs[0], trunk)
    return led.inputs[0], trunk, parts


def test_tri_state_bus(circuit):
    seen, trunk, (t1, t2) = bus(circuit, "TRI", "TRI")
    ins = {}
    for t in (t1, t2):
        a, en = circuit.add_part("IN"), circuit.add_part("IN")
        circuit.connect(a.outputs[0], t.inputs[0])
        circuit.connect(en.outputs[0], t.inputs[1])
        ins[t] = en.outputs[0], a.outputs[0]

    def drive(t, en, a):
        ins[t][0].state, ins[t][1].state = en, a

    drive(t1, False, True), drive(t2, False, False)
    settle(circuit, 3)
    assert seen.state is Z and circuit.wire_state(trunk) == (Z, False)  # nobody drives
    drive(t1, True, True)
    settle(circuit, 3)
    assert seen.state is ONE  # t2's Z doesn't count
    drive(t2, True, False)
    settle(circuit, 3)
    assert seen.state is X and circuit.wire_state(trunk) == (X, True)  # a fight


def test_pulls_only_count_when_nobody_drives(circuit):
    seen, trunk, (tri, pull) = bus(circuit, "TRI", "PULLUP")
    x, en = circuit.add_part("NOT"), circuit.add_part("IN")  # (x's input floats: it says X)
    circuit.connect(x.outputs[0], tri.inputs[0])
    circuit.connect(en.outputs[0], tri.inputs[1])
    settle(circuit, 3)
    assert seen.state is ONE  # tri is off: the pull-up wins
    en.outputs[0].state = True
    settle(circuit, 3)
    assert seen.state is X and circuit.wire_state(trunk) == (X, False)  # strong X beats a pull, no fight


def test_pull_priority(circuit):
    seen, trunk, (up, down) = bus(circuit, "PULLUP", "PULLDOWN")
    assert up.props == down.props == {"priority": 0}
    settle(circuit, 3)
    assert seen.state is X and circuit.wire_state(trunk) == (X, True)  # equal priority: a conflict
    down.props["priority"] = 1
    circuit.props_changed(down)
    settle(circuit, 3)
    assert seen.state is ZERO and circuit.wire_state(trunk) == (ZERO, False)
    up.props["priority"] = 2
    circuit.props_changed(up)
    settle(circuit, 3)
    assert seen.state is ONE


def test_weak_pins_must_be_outputs():
    from pijl.parts import PartType, Registry

    class Bad(PartType):
        kind, ins, weak = "BAD", ("a",), ("a",)
    with pytest.raises(ValueError, match="weak"):
        Registry().add(Bad)



def test_shipped_gates_in_four_states():
    """Every input combination, 0 / 1 / X / Z, through the shipped templates."""
    from itertools import product

    from pijl.logic import Level

    levels = (ZERO, ONE, X, Z)
    c = Circuit()
    xs = c.add_part("NOT")  # input floats: says X
    source = {ZERO: None, ONE: None, X: xs.outputs[0], Z: None}  # Z: leave the input unwired

    def wire(pin, v):
        if v in (ZERO, ONE):
            s = c.add_part("IN")
            s.outputs[0].state = v
            c.connect(s.outputs[0], pin)
        elif v is X:
            c.connect(source[X], pin)

    cases = []
    for kind, n in (("BUF", 1), ("XOR", 2), ("TRI", 2)):
        for vals in product(levels, repeat=n):
            g = c.add_part(kind)
            for pin, v in zip(g.inputs, vals):
                wire(pin, v)
            cases.append((kind, vals, g))
    settle(c, 3)

    def known(v):
        return v in (ZERO, ONE)

    for kind, vals, g in cases:
        got = g.outputs[0].state
        if kind == "BUF":
            expect = vals[0] if known(vals[0]) else X
        elif kind == "XOR":
            a, b = vals
            expect = Level(ZERO + (a != b)) if known(a) and known(b) else X
        else:  # TRI: (a, en). On, it passes a as it is (a floating a stays Z)
            a, en = vals
            expect = {ONE: a, ZERO: Z}.get(en, X)
        assert got is expect, (kind, vals, got)


def test_inline_pull_is_one_net_both_ways_with_no_delay():
    c = Circuit()
    # left: a TRI, then a pull inline, then an LED. right: the same, but a plain wire.
    rows = []
    for inline in (True, False):
        a, en, tri, led = c.add_part("IN"), c.add_part("IN"), c.add_part("TRI"), c.add_part("OUT")
        c.connect(a.outputs[0], tri.inputs[0])
        c.connect(en.outputs[0], tri.inputs[1])
        if inline:
            pull = c.add_part("PULLUP")
            c.connect(tri.outputs[0], pull.inputs[0])
            c.connect(pull.outputs[0], led.inputs[0])
        else:
            c.connect(tri.outputs[0], led.inputs[0])
        rows.append((a.outputs[0], en.outputs[0], led.inputs[0]))
    (a1, en1, led1), (a2, en2, led2) = rows
    settle(c, 3)
    assert led1.state is ONE and led2.state is Z  # off: the pull decides
    assert [p.state for p in pull.pins] == [ONE, ONE]  # both of its pins show the net
    for a, en in ((a1, en1), (a2, en2)):
        a.state, en.state = False, True
    for _ in range(4):  # the 0 arrives on the same tick as through a plain wire: no delay
        c.step()
        assert (led1.state is ZERO) == (led2.state is ZERO)
    assert led1.state is ZERO and [p.state for p in pull.pins] == [ZERO, ZERO]

    # the other way round: driven from the `out` side, read on the `in` side
    d = Circuit()
    src, pull, led = d.add_part("IN"), d.add_part("PULLDOWN"), d.add_part("OUT")
    d.connect(src.outputs[0], led.inputs[0])
    trunk = d.wires[0]
    d.connect(pull.outputs[0], trunk)  # (out side on the driven wire)
    probe = d.add_part("OUT")
    d.connect(pull.inputs[0], probe.inputs[0], check=False)  # (in side on to another reader)
    src.outputs[0].state = True
    settle(d, 2)
    assert probe.inputs[0].state is ONE  # the strong 1 reaches through the pull


def test_joins_must_name_pins_once():
    from pijl.parts import PartType, Registry

    class Bad(PartType):
        kind, ins, outs, joins = "BAD", ("a",), ("b",), (("a", "c"),)

    class Twice(PartType):
        kind, ins, outs, joins = "TWICE", ("a",), ("b",), (("a", "b"), ("b",))
    for t in (Bad, Twice):
        with pytest.raises(ValueError, match="joins"):
            Registry().add(t)


def test_choices_were_replaced_by_settings():
    from pijl.parts import PartType, Registry

    class Old(PartType):
        kind, props, choices = "OLD", {"speed": 1}, {"speed": (1, 2)}
    with pytest.raises(ValueError, match="replaced by `settings`"):
        Registry().add(Old)
    priority = builtin_registry().get("PULLUP").settings["priority"]
    assert priority.values == tuple(range(10)) and priority.default == 0
    assert (priority.show(0), priority.show(5), priority.show(9)) == ("0 (weakest)", "5", "9 (strongest)")
