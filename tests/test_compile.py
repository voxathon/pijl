"""Compiled macros (sim/compile.py): engine option compile=mixed / zero.

Compiling changes timing (that's the point), so these don't compare tick by tick
with flattened macros. What must hold: once settled, a macro without loops gives
exactly what its flattened gates give; macros with loops still do their job (the
bogobips circuits check themselves); what can't be compiled is flattened instead;
and a compiled instance keeps its state through edits elsewhere."""

import random

import numpy as np
import pytest

from pijl.engine import Harness
from pijl.parts import TEMPLATES, PartType
from pijl.parts.registry import load
from pijl.sim import Circuit
from pijl.sim.config import EngineConfig, default
from pijl.snapshot import MACRO, Snapshot

from test_macros import catalog, w

MODES = ["mixed", "zero"]
FLAT = EngineConfig.parse("compile=off", default())


def config(mode: str, dirty: str = "adaptive") -> EngineConfig:
    return EngineConfig.parse(f"compile={mode},dirty={dirty}", default())


def harness(defs: dict[str, Snapshot], name: str, cfg: EngineConfig, registry=None) -> Harness:
    cat = catalog(defs, registry)
    return Harness(cat.get(MACRO + name), cat, settle_ticks=0, config=cfg)


def compiled(h: Harness) -> bool:
    h.circuit.step()  # (programs are chosen when the nets are built)
    return bool(h.circuit._groups)


# ---- bodies without loops: settled, compiled == flattened -----------------------------

KINDS = ["NAND", "AND", "OR", "NOT", "XOR", "BUF", "TRI", "PULLUP", "PULLDOWN"]
ARITY = {"NOT": 1, "BUF": 1, "PULLUP": 1, "PULLDOWN": 1}


def random_body(rng: random.Random, n_in: int, n_gates: int, n_out: int) -> Snapshot:
    """IN ports, gates that only read what came before them (no loops), OUT ports.
    Some inputs are buses: two fresh tri-state buffers, read by nothing else, both
    wired to it (two drivers on one net: Z, fights). Pulls sit inline on gates' nets
    (on an input line they'd keep the macro flat: see
    test_something_inside_driving_an_input_line_keeps_its_macro_flat)."""
    parts = {i + 1: ("IN", f"i{i}", 0.0, float(1000 - 10 * i), {}) for i in range(n_in)}
    wires: dict = {}
    sources = list(parts)
    nxt = n_in + 1

    def add(kind: str) -> int:
        nonlocal nxt
        parts[nxt] = (kind, "", 100.0, 0.0, {})
        nxt += 1
        return nxt - 1

    def wire(src: int, dst: int, i: int) -> None:
        wires[len(wires) + 1] = w(src, 0, dst, i)

    for _ in range(n_gates):
        gates = sources[n_in:]
        kind = rng.choice(KINDS if gates else KINDS[:7])
        u = add(kind)
        for i in range(ARITY.get(kind, 2)):
            if kind.startswith("PULL"):
                wire(rng.choice(gates), u, i)
            elif rng.random() < 0.15:  # a bus
                for _ in range(2):
                    t = add("TRI")
                    wire(rng.choice(sources), t, 0)
                    wire(rng.choice(sources), t, 1)
                    wire(t, u, i)
            else:
                wire(rng.choice(sources), u, i)
        sources.append(u)
    for j in range(n_out):
        port = add("OUT")
        parts[port] = ("OUT", f"o{j}", 900.0, float(1000 - 10 * j), {})
        wire(rng.choice(sources[n_in:] or sources), port, 0)
    return Snapshot(parts, wires)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dirty", ["adaptive", "off"])
def test_settled_compiled_equals_flattened(mode, dirty):
    for seed in range(40):
        rng = random.Random(seed)
        body = random_body(rng, n_in=rng.randint(1, 5), n_gates=rng.randint(1, 25), n_out=rng.randint(1, 4))
        flat = harness({"m": body}, "m", FLAT)
        comp = harness({"m": body}, "m", config(mode, dirty))
        assert compiled(comp), seed
        for vector in range(30):
            bits = "".join(rng.choice("01XZ") for _ in flat.inputs)
            flat.set_bits(bits)
            comp.set_bits(bits)
            assert flat.settle(1000) and comp.settle(1000), (seed, vector)
            assert comp.bits() == flat.bits(), (seed, vector, bits)


# ---- bodies with loops: the bogobips circuits check themselves --------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("nest", [False, True])
def test_bogobips_circuits_still_work_compiled(tmp_path, mode, nest):
    from pijl import bogobips as B

    plan = [("sipo", 5), ("piso", 4), ("counter", 4), ("lfsr", 5), ("tree-xor", 3), ("decoder", 3), ("adder", 5)]
    eng = B.make_project(tmp_path / "bogo", plan, nest)
    for kind, n in plan:
        h = eng.harness(f"{kind} {n}", settle_ticks=0, config=config(mode))
        assert compiled(h), kind
        steps = B.KINDS[kind].script(n, np.random.default_rng(1), 2)
        reads = 0
        while reads < 60:
            sets, read, expect = next(steps)
            for start, bits in sets:
                h.set_bits(bits, start)
            assert h.settle(10_000), kind
            if read:
                assert h.bits() == expect(), (kind, reads)
                reads += 1


def test_a_deep_chain_takes_a_tick_or_two():
    n = 40
    parts = {1: ("IN", "a", 0.0, 0.0, {}), n + 2: ("OUT", "y", 900.0, 0.0, {})}
    wires = {}
    for i in range(n):
        parts[i + 2] = ("NOT", "", 100.0, 0.0, {})
        wires[i + 1] = w(i + 1, 0, i + 2, 0)
    wires[n + 1] = w(n + 1, 0, n + 2, 0)
    body = Snapshot(parts, wires)
    flat = harness({"chain": body}, "chain", FLAT)
    for mode in MODES:
        comp = harness({"chain": body}, "chain", config(mode))
        for h in (flat, comp):
            h.set_bits("1")
            assert h.settle(1000)
        assert comp.bits() == flat.bits() == "1"
        comp.set_bits("0")
        assert comp.settle(1000) and comp.bits() == "0" and comp.last_ticks <= 2, mode
    flat.set_bits("0")
    assert flat.settle(1000) and flat.last_ticks > n


# ---- what isn't compiled ---------------------------------------------------------------


def test_a_part_that_cant_be_tabulated_keeps_its_macro_flat():
    class Blink(PartType):
        kind, ins, outs = "BLINKER", ("a",), ("out",)

        def eval(self, ctx, a):
            return a

    reg = load(TEMPLATES)
    reg.add(Blink())
    body = Snapshot(
        {1: ("IN", "a", 0.0, 0.0, {}), 2: ("BLINKER", "", 100.0, 0.0, {}), 3: ("OUT", "y", 200.0, 0.0, {})},
        {1: w(1, 0, 2, 0), 2: w(2, 0, 3, 0)},
    )
    h = harness({"m": body}, "m", config("mixed"), reg)
    assert not compiled(h)
    h.set_bits("1")
    assert h.settle(100) and h.bits() == "1"


def test_something_inside_driving_an_input_line_keeps_its_macro_flat():
    # a tri-state buffer inside drives the same net the input port does
    body = Snapshot(
        {
            1: ("IN", "a", 0.0, 0.0, {}),
            2: ("IN", "en", 0.0, -40.0, {}),
            3: ("TRI", "", 100.0, 0.0, {}),
            4: ("BUF", "", 200.0, 0.0, {}),
            5: ("OUT", "y", 300.0, 0.0, {}),
        },
        {1: w(1, 0, 4, 0), 2: w(3, 0, 4, 0), 3: w(2, 0, 3, 1), 4: w(4, 0, 5, 0), 5: w(2, 0, 3, 0)},
    )
    flat = harness({"m": body}, "m", FLAT)
    comp = harness({"m": body}, "m", config("mixed"))
    assert not compiled(comp)
    for bits in ("10", "01", "11", "00"):
        for h in (flat, comp):
            h.set_bits(bits)
            assert h.settle(100)
        assert comp.bits() == flat.bits()


# ---- state -----------------------------------------------------------------------------


def sr_latch() -> Snapshot:
    """Set / reset (active low) -> q."""
    return Snapshot(
        {
            1: ("IN", "s", 0.0, 0.0, {}),
            2: ("IN", "r", 0.0, -40.0, {}),
            3: ("NAND", "", 100.0, 0.0, {}),
            4: ("NAND", "", 100.0, -40.0, {}),
            5: ("OUT", "q", 200.0, 0.0, {}),
        },
        {1: w(1, 0, 3, 0), 2: w(4, 0, 3, 1), 3: w(2, 0, 4, 1), 4: w(3, 0, 4, 0), 5: w(3, 0, 5, 0)},
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_compiled_latch_holds_through_edits_elsewhere(mode):
    cat = catalog({"sr": sr_latch()})
    c = Circuit(cat, config=config(mode))
    s, r, m, q = c.add_part("IN"), c.add_part("IN"), c.add_part(MACRO + "sr"), c.add_part("OUT")
    c.connect(s.outputs[0], m.inputs[0])
    c.connect(r.outputs[0], m.inputs[1])
    c.connect(m.outputs[0], q.inputs[0])
    for value, (sv, rv) in ((1, (0, 1)), (0, (1, 0))):
        s.outputs[0].state, r.outputs[0].state = sv, rv
        assert c.run_until_stable(50) is not None
        s.outputs[0].state = r.outputs[0].state = 1  # hold
        assert c.run_until_stable(50) is not None
        assert c._groups and str(q.inputs[0].state)[-1] == str(value)
        for _ in range(3):  # edits elsewhere rebuild the nets (and the programs' groups)
            c.connect(c.add_part("NOT").outputs[0], c.add_part("OUT").inputs[0])
            assert c.run_until_stable(50) is not None
            assert str(q.inputs[0].state)[-1] == str(value), mode


def test_a_compiled_macros_insides_show_its_values():
    cat = catalog({"sr": sr_latch()})
    c = Circuit(cat, config=config("mixed"))
    s, r, m = c.add_part("IN"), c.add_part("IN"), c.add_part(MACRO + "sr")
    c.connect(s.outputs[0], m.inputs[0])
    c.connect(r.outputs[0], m.inputs[1])
    s.outputs[0].state, r.outputs[0].state = 0, 1
    assert c.run_until_stable(50) is not None
    nand = next(p for p in m.inner.values() if p.kind == "NAND" and p.uid == 3)
    codes = c.pin_codes(np.array([q.slot for q in nand.outputs]))  # (pin_codes syncs)
    assert codes.tolist() == [int(m.outputs[0].state)]
