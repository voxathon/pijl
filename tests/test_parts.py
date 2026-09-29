import time
from pathlib import Path

import pytest

from pijl.parts import TEMPLATES, builtin_registry, load
from pijl.sim import Circuit

SCRIPTS = Path(__file__).parent / "part_scripts"


@pytest.fixture
def circuit():
    return Circuit(load(TEMPLATES, SCRIPTS / "good"))


def settle(c: Circuit, steps: int = 10) -> None:
    for _ in range(steps):
        c.step()


# ---- loading ---------------------------------------------------------------------


def test_builtins_are_the_ports_plus_the_template_scripts():
    reg = builtin_registry()
    assert reg.errors == []
    assert [t.kind for t in reg] == ["IN", "OUT", "AND", "NAND", "NOT", "OR"]  # ports, then by path
    assert {t.category for t in reg} == {"I/O", "GATES"}


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
    assert len(reg.errors) == 4  # every template script, the second time


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
        assert [g.outputs[0].state for g in gates] == [expect(a, b) for a, b in rows], kind
    n = c.add_part("NOT")
    settle(c, 1)
    assert n.outputs[0].state is True


def test_scalars_many_inputs_and_many_outputs(circuit):
    high = [circuit.add_part("HIGH") for _ in range(3)]
    wide = circuit.add_part("AND16")
    split = circuit.add_part("SPLIT")
    circuit.connect(high[0].outputs[0], split.inputs[0])
    settle(circuit, 3)
    assert all(h.outputs[0].state is True for h in high)
    assert wide.outputs[0].state is False  # nothing wired: all inputs 0
    assert [p.state for p in split.outputs] == [True, False]


def test_relative_imports_inside_a_script_folder(circuit):
    inv = circuit.add_part("INV")
    settle(circuit, 1)
    assert inv.outputs[0].state is True


def test_a_raising_eval_disables_only_its_own_kind(circuit):
    boom, n = circuit.add_part("BOOM"), circuit.add_part("NOT")
    settle(circuit, 3)
    assert "BOOM" in circuit.faults and "ZeroDivisionError" in circuit.faults["BOOM"]
    assert circuit.errors == [circuit.faults["BOOM"]]  # reported once
    assert boom.outputs[0].state is False
    assert n.outputs[0].state is True  # everything else keeps running


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
    assert (a.outputs[0].state, b.outputs[0].state) == (True, False)
    assert circuit.registry.get("COUNTER").props == {"step": 1}  # instances got copies


def test_impure_ghosts_are_not_evaluated(circuit):
    ghost = circuit.add_part("COUNTER", live=False)
    settle(circuit)  # its eval would need state["count"], which only open() sets
    assert circuit.faults == {}
    assert ghost.outputs[0].state is False


def test_a_bridge_thread_feeds_eval(circuit):
    bridge = circuit.add_part("BRIDGE")
    deadline = time.monotonic() + 2
    while not bridge.state["value"] and time.monotonic() < deadline:
        time.sleep(0.01)
    circuit.step()
    assert bridge.outputs[0].state is True
    circuit.remove_part(bridge)
    assert not bridge.state["thread"].is_alive()


def test_clicks_go_to_clickable_placed_parts():
    c = Circuit()
    switch, gate, ghost = c.add_part("IN"), c.add_part("NAND"), c.add_part("IN", live=False)
    assert c.click(switch) and switch.outputs[0].state is True
    assert not c.click(gate)
    assert not c.click(ghost) and ghost.outputs[0].state is False
