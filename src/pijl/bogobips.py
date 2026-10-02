"""bogobips: BIt-shifts Per Second. A play on bogomips, and about as scientific.

    pijl bench bogobips [--kind sipo,piso,counter,lfsr,tree,decoder,adder] [--depth ...]
                        [--layers engine,settle,pipe] [--seconds 0.5] [--seed 0]
                        [--flips 1] [--nest] [--engines "dirty=off;dirty=adaptive"]

Circuits built from gates are driven with a seeded random stimulus and read back,
and every piece of the circuit that does its job once counts as one unit of work:

    BIPS = clocks (or vectors) per second x work per clock

Only compare BIPS within one kind: a flip-flop stage and a gate aren't the same
amount of work.

Kinds, and what --depth means for each:
  - sipo (serial in, parallel out; depth = stages): one bit in per clock, every
    stage read back every clock. Work: the stages, all of which shift. Stresses
    readout. Flip-flops are NAND master-slave ones.
  - piso (parallel in, serial out; depth = stages): a word loaded at once (a mux in
    front of every stage), then shifted out a bit per clock. Stresses driving.
  - counter (depth = bits): a synchronous binary counter, E random every clock.
    Every bit's next value goes through logic that reads it back (a ripple of
    ANDs), so the whole thing is one loop: the shape of real sequential logic.
    Few bits change per clock, but a carry can run the whole width.
  - lfsr (depth = stages): a shift register feeding the XNOR of its last two
    stages back into the first. One loop through every stage, every stage busy.
  - tree-and, tree-xor (depth = levels: 2^depth inputs, one output): a binary tree
    of gates, everything merging into one. Each vector flips --flips random
    inputs. AND is where changes die out (one 0 settles a gate, so a random tree is
    mostly quiet past its bottom levels); XOR is where every change runs all the way
    to the root. The two ends of how busy a circuit is. "tree" means both.
  - decoder (depth = inputs: 2^depth outputs, one of them 1): built from smaller
    decoders whose outputs are ANDed pairwise, so every line fans out to many
    gates. Each vector flips --flips inputs and reads every output.
  - adder (depth = bits): ripple carry, a + b + carry in, flipping --flips of its
    inputs a vector. Carries split and join again, so with unit delay it glitches,
    and how long it takes depends on the data: a carry may run the whole width.
For the combinational kinds the work per vector is the gate count: each gate
passing its value on once.

--nest builds trees and adders out of macros (a depth-k tree is two depth-(k-1)
trees and a gate; an adder is a row of full-adder macros) instead of flat gates:
the same circuit, so the same answers, but built by stamping nested macros.

Three layers, each a column:
  - engine: driven in-process, a fixed number of ticks per edge or vector: the
    most the settle run needed, or the circuit's longest path if that's more (an
    adder's full-width carry is rare in random data, but it must fit).
  - settle: the same, but run-to-stable after every edge (Harness.settle): what
    not knowing the timing costs -- or saves, where most of a circuit is idle.
  - pipe:   a child `pijl run --raw` process fed over stdin, answers read back
    from stdout. The whole external I/O round trip.

--engines runs every circuit once per engine config (see pijl/sim/config.py), a row
each, so code paths can be compared in one run; the pipe's child gets the same one.

Each circuit is its own oracle (a register gives back its input, delayed; a tree
is an AND or a parity; a decoder one-hot; an adder a sum). Every clock is read,
and every 64th read is checked, so a speed-up that breaks timing shows up as FAIL,
not as a better number. Only every 64th: what a deep circuit should show can be a
string as long as it is wide, and building it would be timed as engine work.

Everything is generated into a throwaway data folder: no project of yours is read
or touched.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import Engine, Harness
from .sim.config import EngineConfig, default
from .snapshot import MACRO, Snapshot

PROJECT = "bogo"
LAYERS = ("engine", "settle", "pipe")
CHECK_EVERY = 64  # reads; the rest are read but not compared

# what to drive (input index, levels from there on), read now?, what must come out (made
# when asked)
Step = tuple[tuple[tuple[int, str], ...], bool, Callable[[], str] | None]


# ---- building blocks -----------------------------------------------------------------


def _w(src: int, dst: int, di: int = 0, si: int = 0):
    return (("p", src, False, si), ("p", dst, True, di), (), None, None)


class _Board:
    """A body being put together: parts and wires, uids handed out in order."""

    def __init__(self) -> None:
        self.parts: dict = {}
        self.wires: list = []

    def add(self, kind: str, label: str = "", y: float = 0.0, x: float = 0.0) -> int:
        """A part; for ports, y sets the pin order (top to bottom: see macros.py)."""
        uid = len(self.parts) + 1
        self.parts[uid] = (kind, label, x, y, {})
        return uid

    def wire(self, src: int, dst: int, di: int = 0, si: int = 0) -> None:
        self.wires.append(_w(src, dst, di, si))

    def gate(self, kind: str, a: int, b: int | None = None, a_out: int = 0, b_out: int = 0) -> int:
        g = self.add(kind)
        self.wire(a, g, 0, a_out)
        if b is not None:
            self.wire(b, g, 1, b_out)
        return g

    def snapshot(self) -> Snapshot:
        return Snapshot(self.parts, {i: w for i, w in enumerate(self.wires, 1)})


def _ins(board: _Board, labels: list[str]) -> list[int]:
    return [board.add("IN", lab, -i) for i, lab in enumerate(labels)]


def _outs(board: _Board, labels: list[str], x: float = 1000.0) -> list[int]:
    return [board.add("OUT", lab, -i, x) for i, lab in enumerate(labels)]


def _random_bits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, n).astype(np.uint8)


def _text(bits: np.ndarray) -> str:
    return (bits + 48).astype(np.uint8).tobytes().decode()


def _flips(rng: np.random.Generator, cur: np.ndarray, k: int) -> list[int]:
    """Flip k random inputs (in place); their indices."""
    at = rng.integers(0, len(cur), k).tolist()
    for i in at:
        cur[i] ^= 1
    return at


def _sets(cur: np.ndarray, at: list[int]) -> tuple[tuple[int, str], ...]:
    return tuple((i, "01"[cur[i]]) for i in at)


# ---- shift registers -----------------------------------------------------------------


def dff() -> Snapshot:
    """Positive edge D flip-flop: two gated D latches (4 NANDs each), the master
    open while C is low, the slave while it's high. Pins: D, C -> Q."""
    b = _Board()
    d, c = _ins(b, ["D", "C"])
    nc = b.gate("NOT", c)  # the master's enable
    s = b.gate("NAND", d, nc)
    r = b.gate("NAND", s, nc)
    mq, mqb = b.add("NAND"), b.add("NAND")
    b.wire(s, mq, 0), b.wire(mqb, mq, 1), b.wire(r, mqb, 0), b.wire(mq, mqb, 1)
    s2 = b.gate("NAND", mq, c)  # the slave, enabled by C
    r2 = b.gate("NAND", s2, c)
    q, qb = b.add("NAND"), b.add("NAND")
    b.wire(s2, q, 0), b.wire(qb, q, 1), b.wire(r2, qb, 0), b.wire(q, qb, 1)
    (out,) = _outs(b, ["Q"])
    b.wire(q, out)
    return b.snapshot()


def stage() -> Snapshot:
    """One PISO stage: a mux (L ? P : S) into a flip-flop. Pins: P, S, L, C -> Q."""
    b = _Board()
    p, s, load, c = _ins(b, ["P", "S", "L", "C"])
    mux = b.gate("NAND", b.gate("NAND", p, load), b.gate("NAND", s, b.gate("NOT", load)))
    ff = b.add(MACRO + "dff")
    b.wire(mux, ff, 0), b.wire(c, ff, 1)
    (out,) = _outs(b, ["Q"])
    b.wire(ff, out)
    return b.snapshot()


def sipo(n: int) -> Snapshot:
    """D, C -> q1 .. qn: n flip-flops in a row, every one read out."""
    b = _Board()
    d, c = _ins(b, ["D", "C"])
    outs = _outs(b, [f"q{j}" for j in range(1, n + 1)])
    prev = d
    for out in outs:
        ff = b.add(MACRO + "dff")
        b.wire(prev, ff, 0), b.wire(c, ff, 1), b.wire(ff, out)
        prev = ff
    return b.snapshot()


def piso(n: int) -> Snapshot:
    """SI, L, C, p1 .. pn -> out: L high loads p1..pn on the clock, low shifts them
    along (SI coming in at the front); out is the last stage."""
    b = _Board()
    si, load, c, *ps = _ins(b, ["SI", "L", "C"] + [f"p{j}" for j in range(1, n + 1)])
    prev = si
    for p in ps:
        st = b.add(MACRO + "pstage")
        b.wire(p, st, 0), b.wire(prev, st, 1), b.wire(load, st, 2), b.wire(c, st, 3)
        prev = st
    (out,) = _outs(b, ["out"])
    b.wire(prev, out)
    return b.snapshot()


def sipo_script(n: int, rng: np.random.Generator, flips: int) -> Iterator[Step]:
    """Inputs D, C; outputs q1..qn. One clock: C low (the masters take D), then high
    (everything moves one along)."""
    fed = bytearray()  # every bit so far

    def window(k: int) -> str:
        """q1..qn after k bits went in: the last n, newest first; X where none got yet."""
        got = fed[max(0, k - n) : k][::-1].decode()
        return got + "X" * (n - len(got))

    while True:
        for bit in rng.integers(0, 2, 4096).tolist():
            d = "01"[bit]
            yield ((0, d + "0"),), False, None
            fed.append(48 + bit)
            k = len(fed)
            yield ((0, d + "1"),), True, lambda k=k: window(k)


def piso_script(n: int, rng: np.random.Generator, flips: int) -> Iterator[Step]:
    """Inputs SI, L, C, p1..pn; output: the last stage. A load clock, then n - 1 shifts."""
    while True:
        word = _text(_random_bits(rng, n))
        yield ((0, "010" + word),), False, None  # L high: the masters take p1..pn
        yield ((0, "011"),), True, lambda w=word: w[-1]  # the load clock: out = pn
        for k in range(1, n):
            yield ((0, "000"),), False, None
            yield ((0, "001"),), True, lambda w=word, k=k: w[n - 1 - k]  # out = p(n-k)


# ---- state fed back through logic ----------------------------------------------------


def counter(n: int) -> Snapshot:
    """R, E, C -> q0 .. q(n-1) (q0 the lowest bit): a synchronous binary counter.
    Bit i toggles when E and every bit below it are 1 (a ripple of ANDs, so a carry
    can run the whole width); R high clears it on the clock. Every bit's next value
    goes through logic that reads it: one loop through the whole thing."""
    b = _Board()
    r, e, c = _ins(b, ["R", "E", "C"])
    outs = _outs(b, [f"q{i}" for i in range(n)])
    keep = b.gate("NOT", r)
    t = e
    for out in outs:
        ff = b.add(MACRO + "dff")
        b.wire(b.gate("AND", b.gate("XOR", ff, t), keep), ff, 0)
        b.wire(c, ff, 1), b.wire(ff, out)
        t = b.gate("AND", t, ff)
    return b.snapshot()


def counter_script(n: int, rng: np.random.Generator, flips: int) -> Iterator[Step]:
    """Inputs R, E, C. A clearing clock, then clocks with E random."""
    yield ((0, "100"),), False, None
    yield ((0, "101"),), True, lambda: "0" * n
    value, mask = 0, (1 << n) - 1
    while True:
        for e in rng.integers(0, 2, 4096).tolist():
            yield ((0, "0" + "01"[e] + "0"),), False, None
            value = (value + e) & mask
            yield ((2, "1"),), True, lambda v=value: format(v, f"0{n}b")[::-1]


def lfsr(n: int) -> Snapshot:
    """R, C -> q1 .. qn: a shift register whose first stage takes the XNOR of the last
    two (Fibonacci style; all zeros is a fine state with XNOR). R high clears it on the
    clock. The feedback closes a loop through every stage."""
    b = _Board()
    r, c = _ins(b, ["R", "C"])
    outs = _outs(b, [f"q{j}" for j in range(1, n + 1)])
    keep = b.gate("NOT", r)
    ffs = [b.add(MACRO + "dff") for _ in outs]
    feedback = b.gate("NOT", b.gate("XOR", ffs[-1], ffs[-2]))
    for ff, prev, out in zip(ffs, [feedback] + ffs[:-1], outs):
        b.wire(b.gate("AND", prev, keep), ff, 0)
        b.wire(c, ff, 1), b.wire(ff, out)
    return b.snapshot()


def lfsr_script(n: int, rng: np.random.Generator, flips: int) -> Iterator[Step]:
    """Inputs R, C. A clearing clock, then free running (no stimulus to draw)."""
    yield ((0, "10"),), False, None
    yield ((0, "11"),), True, lambda: "0" * n
    state, mask = 0, (1 << n) - 1  # bit i = q(i+1)
    while True:
        yield ((0, "00"),), False, None
        state = ((state << 1) | (1 ^ (state >> (n - 1) & 1) ^ (state >> (n - 2) & 1))) & mask
        yield ((1, "1"),), True, lambda s=state: format(s, f"0{n}b")[::-1]


# ---- trees ---------------------------------------------------------------------------


def tree(gate: str, depth: int, nest: bool) -> Snapshot:
    """x1 .. x(2^depth) -> out, through `depth` levels of `gate`. Nested: two trees of
    depth - 1 (macros "tree-<gate> <depth - 1>") and a gate."""
    b = _Board()
    xs = _ins(b, [f"x{i}" for i in range(1, 2**depth + 1)])
    if nest and depth > 1:
        half = len(xs) // 2
        sub = MACRO + f"tree-{gate.lower()} {depth - 1}"
        left, right = b.add(sub), b.add(sub)
        for i, x in enumerate(xs):
            b.wire(x, left if i < half else right, i % half)
        level = [left, right]
    else:
        level = xs
    while len(level) > 1:
        level = [b.gate(gate, level[i], level[i + 1]) for i in range(0, len(level), 2)]
    (out,) = _outs(b, ["out"])
    b.wire(level[0], out)
    return b.snapshot()


def tree_script(gate: str) -> Callable[[int, np.random.Generator, int], Iterator[Step]]:
    def script(depth: int, rng: np.random.Generator, flips: int) -> Iterator[Step]:
        cur = _random_bits(rng, 2**depth)
        zeros, parity = int(np.count_nonzero(cur == 0)), int(cur.sum()) & 1

        def value() -> str:
            return "01"[zeros == 0] if gate == "AND" else "01"[parity]

        v = value()
        yield ((0, _text(cur)),), True, lambda v=v: v
        while True:
            at = _flips(rng, cur, flips)
            for i in at:
                zeros += -1 if cur[i] else 1
                parity ^= 1
            v = value()
            yield _sets(cur, at), True, lambda v=v: v

    return script


# ---- decoder -------------------------------------------------------------------------


def decoder(n: int) -> Snapshot:
    """a1 .. an (a1 the top bit) -> y0 .. y(2^n - 1): y_v is 1 when the inputs spell v.
    Two half-width decoders, every pair of their outputs ANDed."""
    b = _Board()
    ins = _ins(b, [f"a{i}" for i in range(1, n + 1)])

    def dec(inputs: list[int]) -> list[int]:
        if len(inputs) == 1:
            return [b.gate("NOT", inputs[0]), inputs[0]]
        h = len(inputs) // 2
        top, bottom = dec(inputs[:h]), dec(inputs[h:])
        return [b.gate("AND", t, u) for t in top for u in bottom]

    for src, out in zip(dec(ins), _outs(b, [f"y{v}" for v in range(2**n)])):
        b.wire(src, out)
    return b.snapshot()


def _decoder_depth(n: int) -> int:
    return 1 if n == 1 else max(_decoder_depth(n // 2), _decoder_depth(n - n // 2)) + 1


def _decoder_gates(n: int) -> int:
    return 1 if n == 1 else _decoder_gates(n // 2) + _decoder_gates(n - n // 2) + 2**n


def decoder_script(n: int, rng: np.random.Generator, flips: int) -> Iterator[Step]:
    cur = _random_bits(rng, n)
    value = int("".join(map(str, cur.tolist())), 2)
    size = 2**n

    def one_hot(v: int) -> str:
        return "0" * v + "1" + "0" * (size - v - 1)

    yield ((0, _text(cur)),), True, lambda v=value: one_hot(v)
    while True:
        at = _flips(rng, cur, flips)
        for i in at:
            value ^= 1 << (n - 1 - i)
        yield _sets(cur, at), True, lambda v=value: one_hot(v)


# ---- adder ---------------------------------------------------------------------------


def _full_adder(b: _Board, a: int, x: int, c: int, c_out: int = 0) -> tuple[int, int]:
    """Gates for one bit: (sum, carry out). `c_out`: the carry's output pin."""
    half = b.gate("XOR", a, x)
    s = b.gate("XOR", half, c, b_out=c_out)
    carry = b.gate("OR", b.gate("AND", a, x), b.gate("AND", half, c, b_out=c_out))
    return s, carry


def full_adder() -> Snapshot:
    """a, b, c -> s, co."""
    b = _Board()
    a, x, c = _ins(b, ["a", "b", "c"])
    s, co = _full_adder(b, a, x, c)
    out_s, out_co = _outs(b, ["s", "co"])
    b.wire(s, out_s), b.wire(co, out_co)
    return b.snapshot()


def adder(n: int, nest: bool) -> Snapshot:
    """a0 .. a(n-1), b0 .. b(n-1), cin -> s0 .. s(n-1), cout (bit 0 the lowest)."""
    b = _Board()
    ins = _ins(b, [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)] + ["cin"])
    outs = _outs(b, [f"s{i}" for i in range(n)] + ["cout"])
    carry, carry_out = ins[-1], 0
    for i in range(n):
        if nest:
            fa = b.add(MACRO + "fa")
            b.wire(ins[i], fa, 0), b.wire(ins[n + i], fa, 1), b.wire(carry, fa, 2, carry_out)
            b.wire(fa, outs[i], 0, 0)
            carry, carry_out = fa, 1
        else:
            s, carry = _full_adder(b, ins[i], ins[n + i], carry)
            b.wire(s, outs[i])
    b.wire(carry, outs[-1], 0, carry_out)
    return b.snapshot()


def adder_script(n: int, rng: np.random.Generator, flips: int) -> Iterator[Step]:
    cur = _random_bits(rng, 2 * n + 1)
    a = int(_text(cur[:n])[::-1] or "0", 2)
    x = int(_text(cur[n : 2 * n])[::-1] or "0", 2)
    c = int(cur[-1])

    def total(a: int, x: int, c: int) -> str:
        return format(a + x + c, f"0{n + 1}b")[::-1]  # s0 .. s(n-1), cout

    yield ((0, _text(cur)),), True, lambda a=a, x=x, c=c: total(a, x, c)
    while True:
        at = _flips(rng, cur, flips)
        for i in at:
            if i < n:
                a ^= 1 << i
            elif i < 2 * n:
                x ^= 1 << (i - n)
            else:
                c ^= 1
        yield _sets(cur, at), True, lambda a=a, x=x, c=c: total(a, x, c)


# ---- the kinds -----------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    name: str
    build: Callable[[int, bool], Snapshot]  # depth, nest -> the macro
    script: Callable[[int, np.random.Generator, int], Iterator[Step]]  # depth, rng, flips
    work: Callable[[int], int]  # per clock / vector
    path: Callable[[int], int] | None  # ticks the longest path needs (None: measure it)
    depths: tuple[int, ...]  # the default
    most: int  # the deepest it'll build
    edges: int = 1  # edges (runs) per clock


KINDS = {
    k.name: k
    for k in (
        Kind("sipo", lambda n, _: sipo(n), sipo_script, lambda n: n, None, (8, 64, 512, 4096), 1 << 20, 2),
        Kind("piso", lambda n, _: piso(n), piso_script, lambda n: n, None, (8, 64, 512, 4096), 1 << 20, 2),
        Kind("counter", lambda n, _: counter(n), counter_script, lambda n: n, lambda n: n + 6, (8, 64, 512, 4096), 1 << 16, 2),
        Kind("lfsr", lambda n, _: lfsr(n), lfsr_script, lambda n: n, None, (8, 64, 512, 4096), 1 << 20, 2),
        Kind("tree-and", lambda n, nest: tree("AND", n, nest), tree_script("AND"), lambda n: 2**n - 1, lambda n: n + 1, (4, 8, 12, 16), 22),
        Kind("tree-xor", lambda n, nest: tree("XOR", n, nest), tree_script("XOR"), lambda n: 2**n - 1, lambda n: n + 1, (4, 8, 12, 16), 22),
        Kind("decoder", lambda n, _: decoder(n), decoder_script, _decoder_gates, lambda n: _decoder_depth(n) + 1, (4, 8, 12, 16), 18),
        Kind("adder", adder, adder_script, lambda n: 5 * n, lambda n: 2 * n + 3, (8, 64, 512), 1 << 16),
    )
}
ALIASES = {"tree": ("tree-and", "tree-xor"), "shift": ("sipo", "piso"), "loop": ("counter", "lfsr")}


def make_project(root: Path, plan: list[tuple[str, int]], nest: bool = False) -> Engine:
    """A throwaway project under data root `root` with the circuits in `plan` (kind,
    depth) in it, as macros "<kind> <depth>", plus the macros they're made of."""
    path = root / "projects" / PROJECT
    (path / "macros").mkdir(parents=True)
    (path / "project.json").write_text('{"pijl": 1}\n', encoding="utf-8")
    eng = Engine(path)
    save = eng.store.save
    kinds = {k for k, _ in plan}
    if kinds & {"sipo", "piso", "counter", "lfsr"}:
        save("dff", dff())
        save("pstage", stage(), eng.catalog)
    if nest and "adder" in kinds:
        save("fa", full_adder())
    for gate in ("and", "xor"):
        deepest = max((n for k, n in plan if k == f"tree-{gate}"), default=0)
        if nest:  # every level, from the bottom: each is made of the one below
            for n in range(1, deepest):
                save(f"tree-{gate} {n}", tree(gate.upper(), n, True), eng.catalog)
    for kind, n in plan:
        save(f"{kind} {n}", KINDS[kind].build(n, nest), eng.catalog)
    return eng


# ---- the layers ----------------------------------------------------------------------


class Failed(Exception):
    pass


def _drive(h: Harness, steps: Iterator[Step], ticks: int | None, seconds: float, ticks_seen: list[int]) -> float:
    """Reads (clocks, vectors) per second, in-process: fixed `ticks` per run, or
    settle (None)."""
    clocks, t0, warm = 0, 0.0, 8
    for sets, read, expect in steps:
        for start, bits in sets:
            h.set_bits(bits, start)
        if ticks is None:
            if not h.settle(1 << 20):
                raise Failed("never settled")
            ticks_seen.append(h.last_ticks)
        else:
            h.step(ticks)
        if read:
            got = h.bits()
            if clocks % CHECK_EVERY == 0 and got != (want := expect()):
                raise Failed(f"read {clocks}: got {_short(got)}, expected {_short(want)}")
            clocks += 1
            if clocks == warm:
                t0 = time.perf_counter()
            elif clocks > warm and (elapsed := time.perf_counter() - t0) >= seconds:
                return (clocks - warm) / elapsed
    raise AssertionError("scripts don't end")


def _command() -> list[str]:
    """How to start pijl again: this interpreter -m pijl, or a frozen build itself."""
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "pijl"]


def encode(sets: tuple[tuple[int, str], ...], n_in: int) -> bytes:
    """One step as --raw stream bytes: a whole vector as is (it runs by itself);
    anything else as a prefix and/or @n=L addresses, then ";" to run."""
    if len(sets) == 1 and sets[0][0] == 0 and len(sets[0][1]) == n_in:
        return sets[0][1].encode()
    out = bytearray()
    for start, bits in sets:
        if start == 0:
            out += bits.encode()
        else:
            for j, c in enumerate(bits):
                out += b"@%d=%s" % (start + j + 1, c.encode())
    return bytes(out + b";")


def _pipe(
    root: Path, macro: str, n_in: int, n_out: int, steps: Iterator[Step], ticks: int, seconds: float, config: EngineConfig
) -> float:
    """Reads per second through a child `pijl run --raw`: steps down stdin, a "?"
    after each read, the answers read back on a thread and (every 64th) checked at
    the end."""
    cmd = _command() + ["--data", str(root), "-p", PROJECT, "--engine", str(config)]
    cmd += ["run", macro, "--raw", "--ticks", str(ticks), "--noise", "0"]
    with tempfile.TemporaryFile() as errors:
        return _talk(cmd, errors, n_in, n_out, steps, seconds)


def _talk(cmd: list[str], errors, n_in: int, n_out: int, steps: Iterator[Step], seconds: float) -> float:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, bufsize=0)
    got = bytearray()
    done = threading.Event()
    arrived = threading.Condition()
    finished_at = [0.0]

    def read() -> None:
        while chunk := proc.stdout.read(1 << 16):
            with arrived:
                got.extend(chunk)
                arrived.notify_all()
        finished_at[0] = time.perf_counter()
        with arrived:
            done.set()
            arrived.notify_all()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    checks: list[tuple[int, bytes]] = []  # (read number, what it must be), every 64th
    reads = 0

    def clocks_of(k: int) -> bytes:
        """The next k reads' worth of the script, as raw stream bytes."""
        nonlocal reads
        out = bytearray()
        while k:
            sets, is_read, expect = next(steps)
            out += encode(sets, n_in)
            if is_read:
                out += b"?"
                if reads % CHECK_EVERY == 0:
                    checks.append((reads, expect().encode()))
                reads += 1
                k -= 1
        return bytes(out)

    try:
        proc.stdin.write(clocks_of(8))  # warm-up: also waits out the child's start
        with arrived:
            arrived.wait_for(lambda: len(got) >= reads * n_out or done.is_set(), timeout=600)
        if len(got) < reads * n_out:
            raise Failed("the child didn't answer")
        t0, clocks = time.perf_counter(), 0
        while time.perf_counter() - t0 < seconds:
            proc.stdin.write(clocks_of(16))
            clocks += 16
        proc.stdin.close()
        reader.join(600)
        if proc.wait(60):
            errors.seek(0)
            raise Failed(f"the child failed: {errors.read().decode(errors='replace').strip()}")
        if len(got) != reads * n_out:
            raise Failed(f"{len(got)} answer bytes for {reads} reads of {n_out}")
        for i, want in checks:
            if got[i * n_out : (i + 1) * n_out] != want:
                raise Failed(f"answer {i} is wrong")
        return clocks / (finished_at[0] - t0)
    finally:
        if proc.poll() is None:
            proc.kill()


# ---- the run -------------------------------------------------------------------------


def main(args) -> int:
    kinds = _kinds(args.kind)
    layers = _pick(args.layers, LAYERS, "layer")
    if kinds is None or layers is None:
        return 2
    try:
        given = [int(d) for d in args.depth.split(",") if d.strip()] if args.depth else None
        if given is not None and (not given or min(given) < 1):
            raise ValueError
    except ValueError:
        print(f"pijl: --depth: {args.depth!r} isn't a list of depths (8,64,...)", file=sys.stderr)
        return 2
    if args.flips < 1:
        print("pijl: --flips must be at least 1", file=sys.stderr)
        return 2
    try:
        configs = [EngineConfig.parse(t, default()) for t in args.engines.split(";") if t.strip()] or [default()]
    except ValueError as e:
        print(f"pijl: --engines: {e}", file=sys.stderr)
        return 2
    width = max(len(str(c)) for c in configs) if len(configs) > 1 else 0
    plan = [(k, n) for k in kinds for n in (given or KINDS[k].depths)]
    too_deep = [f"{k} {n} (at most {KINDS[k].most})" for k, n in plan if n > KINDS[k].most]
    if too_deep:
        print(f"pijl: too deep: {', '.join(too_deep)}", file=sys.stderr)
        return 2
    print(f"bogobips: BIt-shifts Per Second ({_versions()})", flush=True)
    if args.nest:
        print("trees and adders built from nested macros", flush=True)
    root = Path(tempfile.mkdtemp(prefix="pijl-bogobips-"))
    status, peak = 0, {}
    try:
        t0 = time.perf_counter()
        eng = make_project(root, plan, args.nest)
        print(f"(generated in {time.perf_counter() - t0:.1f}s)", flush=True)
        head = f"{'kind':8} {'depth':>6} " + (f"{'engine':{width}} " if width else "")
        head += f"{'work':>8} {'build':>8} {'ticks/clock':>11}"
        print(head + "".join(f" {lay:>9}" for lay in layers), flush=True)
        for kind, n in plan:
            for config in configs:
                row, rates = _measure(eng, root, KINDS[kind], n, layers, args, config, width)
                print(row, flush=True)
                if rates is None:
                    status = 1
                    continue
                for layer, bips in rates.items():
                    if bips > peak.get(kind, (0.0, ""))[0]:
                        where = f"{n}, {layer}" + (f", {config}" if width else "")
                        peak[kind] = (bips, where)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for kind, (bips, where) in peak.items():
        print(f"peak {kind}: {_si(bips)} BogoBIPS (depth {where})")
    return status


def _measure(
    eng: Engine, root: Path, k: Kind, n: int, layers: list[str], args, config: EngineConfig, width: int
) -> tuple[str, dict[str, float] | None]:
    """One row of the table, and BIPS by layer (None if a cell failed)."""
    macro = f"{k.name} {n}"
    t0 = time.perf_counter()
    h = eng.harness(macro, settle_ticks=0, config=config)
    built = time.perf_counter() - t0
    work = k.work(n)
    rng = np.random.default_rng(args.seed)
    steps = k.script(n, rng, args.flips)
    seen: list[int] = []
    cells: dict[str, str] = {}
    rates: dict[str, float] = {}
    try:  # settle first: it measures the ticks the fixed-tick layers use
        rate = _drive(h, steps, None, args.seconds if "settle" in layers else 0, seen)
        if "settle" in layers:
            cells["settle"] = _si(rate * work)
            rates["settle"] = rate * work
    except Failed as e:
        cells["settle"] = "FAIL"
        print(f"  {macro}, settle: {e}", file=sys.stderr)
    ticks = max(max(seen, default=0), k.path(n) if k.path else 0) or 1
    for layer in ("engine", "pipe"):
        if layer not in layers or cells.get("settle") == "FAIL":
            continue
        try:
            if layer == "engine":
                rate = _drive(h, steps, ticks, args.seconds, [])
            else:
                fresh = k.script(n, np.random.default_rng(args.seed), args.flips)
                rate = _pipe(root, macro, len(h.inputs), len(h.outputs), fresh, ticks, args.seconds, config)
            cells[layer] = _si(rate * work)
            rates[layer] = rate * work
        except Failed as e:
            cells[layer] = "FAIL"
            print(f"  {macro}, {layer}: {e}", file=sys.stderr)
    row = f"{k.name:8} {n:>6} " + (f"{str(config):{width}} " if width else "")
    row += f"{_si(work):>8} {built * 1000:>6.0f}ms {k.edges * ticks:>11}"
    row += "".join(f" {cells.get(lay, '-'):>9}" for lay in layers)
    return row, (None if "FAIL" in cells.values() else rates)


def _kinds(text: str) -> list[str] | None:
    picked: list[str] = []
    for t in (t.strip() for t in text.split(",")):
        if t:
            picked += ALIASES.get(t, (t,))
    bad = [t for t in picked if t not in KINDS]
    if bad or not picked:
        known = ", ".join([*KINDS, *ALIASES])
        print(f"pijl: unknown kind {', '.join(bad) or '(none)'}; pick from {known}", file=sys.stderr)
        return None
    return [k for k in KINDS if k in picked]


def _pick(text: str, known: tuple[str, ...], what: str) -> list[str] | None:
    picked = [t.strip() for t in text.split(",") if t.strip()]
    bad = [t for t in picked if t not in known]
    if bad or not picked:
        print(f"pijl: unknown {what} {', '.join(bad) or '(none)'}; pick from {', '.join(known)}", file=sys.stderr)
        return None
    return [k for k in known if k in picked]


def _si(v: float) -> str:
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= scale:
            return f"{v / scale:.2f}{unit}"
    return f"{v:.0f}"


def _short(bits: str) -> str:
    return bits if len(bits) <= 24 else bits[:24] + f"...({len(bits)})"


def _versions() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        pijl = version("pijl")
    except PackageNotFoundError:
        pijl = "?"
    return f"pijl {pijl}, numpy {np.__version__}, Python {sys.version.split()[0]}"
