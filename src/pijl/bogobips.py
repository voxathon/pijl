"""bogobips: BIt-shifts Per Second. A play on bogomips, and about as scientific.

    pijl bench bogobips [--kind sipo,piso] [--depth 8,64,512,4096]
                        [--layers engine,settle,pipe] [--seconds 0.5] [--seed 0]

A seeded random bit stream is pushed through shift registers built from NANDs and
read back out. Every stage that moves its bit on a clock is one bit-shift, so

    BIPS = clocks per second x depth

With unit delay all stages shift at once: a clock costs the same number of ticks
at any depth, and what grows with depth is what a tick costs (array work over
every gate). So BIPS rises while numpy's batching pays for itself, then flattens.

Two registers, for the two ends of the I/O:
  - sipo (serial in, parallel out): one bit in per clock, all `depth` outputs read
    back every clock. Stresses readout.
  - piso (parallel in, serial out): a whole word loaded at once (a mux in front of
    every stage), then shifted out a bit per clock. Stresses the drive side.

Three layers, each a column:
  - engine: the macro driven in-process, a fixed number of ticks per clock edge
    (measured by the settle run). The engine's own speed.
  - settle: the same, but run-to-stable after every edge (Harness.settle): what
    not knowing the timing costs.
  - pipe:   a child `pijl run --raw` process fed over stdin, answers read back
    from stdout. The whole external I/O round trip.

A shift register is its own oracle: what comes out is what went in, delayed. Every
clock is read, and every 64th read is checked (a register starts out X, and that's
expected too), so a speed-up that breaks timing shows up as FAIL, not as a better
number. Only every 64th: building what a deep SIPO should show is a string as long
as the register, which would otherwise be timed as if it were the engine's work.

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
from pathlib import Path

import numpy as np

from .engine import Engine, Harness
from .snapshot import MACRO, Snapshot

PROJECT = "bogo"
LAYERS = ("engine", "settle", "pipe")
CHECK_EVERY = 64  # reads; the rest are read but not compared
KINDS = ("sipo", "piso")

# input bits (from the first input on), read now?, what must come out (made when asked)
Step = tuple[str, bool, Callable[[], str] | None]


# ---- the macros ----------------------------------------------------------------------


def _w(src: int, dst: int, di: int = 0, si: int = 0):
    return (("p", src, False, si), ("p", dst, True, di), (), None, None)


def _snap(parts: dict, wires: list) -> Snapshot:
    return Snapshot(parts, {i: w for i, w in enumerate(wires, 1)})


def _port(kind: str, label: str, y: float, x: float = 0.0):
    """A port: macro pins go top to bottom by y (see macros.py), so y sets the order."""
    return (kind, label, x, y, {})


def dff() -> Snapshot:
    """Positive edge D flip-flop: two gated D latches (4 NANDs each), the master
    open while C is low, the slave while it's high. Pins: D, C -> Q."""
    nand = ("NAND", "", 0.0, 0.0, {})
    parts = {1: _port("IN", "D", 0), 2: _port("IN", "C", -1), 3: ("NOT", "", 0.0, 0.0, {})}
    parts |= {uid: nand for uid in range(4, 12)}
    parts[12] = _port("OUT", "Q", 0, 100)
    wires = [
        _w(2, 3),  # nc = not C: the master's enable
        _w(1, 4, 0), _w(3, 4, 1),  # master: s = nand(D, nc)
        _w(4, 5, 0), _w(3, 5, 1),  # r = nand(s, nc)
        _w(4, 6, 0), _w(7, 6, 1),  # mq = nand(s, mqb)
        _w(5, 7, 0), _w(6, 7, 1),  # mqb = nand(r, mq)
        _w(6, 8, 0), _w(2, 8, 1),  # slave, enabled by C: s2 = nand(mq, C)
        _w(8, 9, 0), _w(2, 9, 1),  # r2 = nand(s2, C)
        _w(8, 10, 0), _w(11, 10, 1),  # q = nand(s2, qb)
        _w(9, 11, 0), _w(10, 11, 1),  # qb = nand(r2, q)
        _w(10, 12),
    ]
    return _snap(parts, wires)


def stage() -> Snapshot:
    """One PISO stage: a mux (L ? P : S) into a flip-flop. Pins: P, S, L, C -> Q."""
    nand = ("NAND", "", 0.0, 0.0, {})
    parts = {
        1: _port("IN", "P", 0),
        2: _port("IN", "S", -1),
        3: _port("IN", "L", -2),
        4: _port("IN", "C", -3),
        5: ("NOT", "", 0.0, 0.0, {}),
        6: nand,
        7: nand,
        8: nand,
        9: (MACRO + "dff", "", 0.0, 0.0, {}),
        10: _port("OUT", "Q", 0, 100),
    }
    wires = [
        _w(3, 5),  # not L
        _w(1, 6, 0), _w(3, 6, 1),  # nand(P, L)
        _w(2, 7, 0), _w(5, 7, 1),  # nand(S, not L)
        _w(6, 8, 0), _w(7, 8, 1),  # the mux
        _w(8, 9, 0), _w(4, 9, 1),  # into the flip-flop's D, C
        _w(9, 10),
    ]
    return _snap(parts, wires)


def sipo(n: int) -> Snapshot:
    """D, C -> q1 .. qn: n flip-flops in a row, every one read out."""
    parts = {1: _port("IN", "D", 0), 2: _port("IN", "C", -1)}
    wires = []
    for j in range(1, n + 1):
        ff, out = 2 + j, 2 + n + j
        parts[ff] = (MACRO + "dff", "", 100.0 * j, 0.0, {})
        parts[out] = _port("OUT", f"q{j}", -j, 100.0 * (n + 1))
        wires += [_w(1 if j == 1 else ff - 1, ff, 0), _w(2, ff, 1), _w(ff, out)]
    return _snap(parts, wires)


def piso(n: int) -> Snapshot:
    """SI, L, C, p1 .. pn -> out: L high loads p1..pn on the clock, low shifts them
    along (SI coming in at the front); out is the last stage."""
    parts = {1: _port("IN", "SI", 0), 2: _port("IN", "L", -1), 3: _port("IN", "C", -2)}
    wires = []
    for j in range(1, n + 1):
        p, st = 3 + j, 3 + n + j
        parts[p] = _port("IN", f"p{j}", -2 - j)
        parts[st] = (MACRO + "pstage", "", 100.0 * j, 0.0, {})
        wires += [_w(p, st, 0), _w(1 if j == 1 else st - 1, st, 1), _w(2, st, 2), _w(3, st, 3)]
    parts[4 + 2 * n] = _port("OUT", "out", 0, 100.0 * (n + 1))
    wires.append(_w(3 + 2 * n, 4 + 2 * n))
    return _snap(parts, wires)


def make_project(root: Path, kinds, depths) -> Engine:
    """A throwaway project under data root `root` with the registers in it."""
    path = root / "projects" / PROJECT
    (path / "macros").mkdir(parents=True)
    (path / "project.json").write_text('{"pijl": 1}\n', encoding="utf-8")
    eng = Engine(path)
    eng.store.save("dff", dff())
    eng.store.save("pstage", stage(), eng.catalog)
    for kind in kinds:
        for n in depths:
            eng.store.save(f"{kind} {n}", (sipo if kind == "sipo" else piso)(n), eng.catalog)
    return eng


# ---- the stimulus: one script, run by every layer ------------------------------------


def script(kind: str, n: int, seed: int) -> Iterator[Step]:
    """Edges to apply (input bits from the first input on: a short string leaves the
    rest as they are), each with whether to read after it and what must come out."""
    rng = np.random.default_rng(seed)
    if kind == "sipo":  # inputs D, C; outputs q1..qn
        fed = bytearray()  # every bit so far

        def window(k: int) -> str:
            """q1..qn after k bits went in: the last n, newest first; X where none got yet."""
            got = fed[max(0, k - n) : k][::-1].decode()
            return got + "X" * (n - len(got))

        while True:
            for b in rng.integers(0, 2, 4096).tolist():
                d = "01"[b]
                yield d + "0", False, None  # C low: the masters take D
                fed.append(48 + b)
                k = len(fed)
                yield d + "1", True, lambda k=k: window(k)  # C high: all move one along
    else:  # inputs SI, L, C, p1..pn; output: the last stage
        while True:
            word = "".join("01"[b] for b in rng.integers(0, 2, n).tolist())
            yield "010" + word, False, None  # L high: the masters take p1..pn
            yield "011", True, lambda w=word: w[-1]  # the load clock: out = pn
            for k in range(1, n):
                yield "000", False, None
                yield "001", True, lambda w=word, k=k: w[n - 1 - k]  # shifted: out = p(n-k)


# ---- the layers ----------------------------------------------------------------------


class Failed(Exception):
    pass


def _drive(h: Harness, steps: Iterator[Step], ticks: int | None, seconds: float, ticks_seen: list[int]) -> float:
    """Clocks per second, in-process: fixed `ticks` per edge, or settle (None)."""
    clocks, t0, warm = 0, 0.0, 8
    for bits, read, expect in steps:
        h.set_bits(bits)
        if ticks is None:
            if not h.settle(1000):
                raise Failed("never settled")
            ticks_seen.append(h.last_ticks)
        else:
            h.step(ticks)
        if read:
            got = h.bits()
            if clocks % CHECK_EVERY == 0 and got != (want := expect()):
                raise Failed(f"clock {clocks}: read {_short(got)}, expected {_short(want)}")
            clocks += 1
            if clocks == warm:
                t0 = time.perf_counter()
            elif clocks > warm and clocks % 16 == 0:
                elapsed = time.perf_counter() - t0
                if elapsed >= seconds:
                    return (clocks - warm) / elapsed
    raise AssertionError("scripts don't end")


def _command() -> list[str]:
    """How to start pijl again: this interpreter -m pijl, or a frozen build itself."""
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "pijl"]


def _pipe(root: Path, macro: str, n_in: int, n_out: int, steps: Iterator[Step], ticks: int, seconds: float) -> float:
    """Clocks per second through a child `pijl run --raw`: vectors down stdin, a "?"
    after each clock, the answers read back on a thread and (every 64th) checked at
    the end."""
    cmd = _command() + ["--data", str(root), "-p", PROJECT, "run", macro, "--raw", "--ticks", str(ticks), "--noise", "0"]
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
        """The next k clocks of the script, as raw stream bytes."""
        nonlocal reads
        out = bytearray()
        while k:
            bits, is_read, expect = next(steps)
            out += bits.encode() + (b";" if len(bits) < n_in else b"")
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
    kinds = _pick(args.kind, KINDS, "kind")
    layers = _pick(args.layers, LAYERS, "layer")
    try:
        depths = [int(d) for d in args.depth.split(",") if d.strip()]
        if not depths or min(depths) < 1:
            raise ValueError
    except ValueError:
        print(f"pijl: --depth: {args.depth!r} isn't a list of depths (8,64,...)", file=sys.stderr)
        return 2
    if kinds is None or layers is None:
        return 2
    print(f"bogobips: BIt-shifts Per Second ({_versions()})", flush=True)
    root = Path(tempfile.mkdtemp(prefix="pijl-bogobips-"))
    status, peak = 0, (0.0, "")
    try:
        eng = make_project(root, kinds, depths)
        print(f"{'kind':5} {'depth':>6} {'build':>8} {'ticks/clock':>11}" + "".join(f" {lay:>9}" for lay in layers), flush=True)
        for kind in kinds:
            for n in depths:
                macro = f"{kind} {n}"
                t0 = time.perf_counter()
                h = eng.harness(macro, settle_ticks=0)
                built = time.perf_counter() - t0
                steps = script(kind, n, args.seed)
                seen: list[int] = []
                cells: dict[str, str] = {}
                try:  # settle first: it measures the ticks the fixed-tick layers use
                    rate = _drive(h, steps, None, args.seconds if "settle" in layers else 0, seen)
                    if "settle" in layers:
                        cells["settle"] = _si(rate * n)
                except Failed as e:
                    cells["settle"] = "FAIL"
                    print(f"  {macro}, settle: {e}", file=sys.stderr)
                ticks = max(seen, default=0) or 1
                for layer in ("engine", "pipe"):
                    if layer not in layers or cells.get("settle") == "FAIL":
                        continue
                    try:
                        if layer == "engine":
                            rate = _drive(h, steps, ticks, args.seconds, [])
                        else:
                            rate = _pipe(root, macro, len(h.inputs), len(h.outputs), script(kind, n, args.seed), ticks, args.seconds)
                        cells[layer] = _si(rate * n)
                        if rate * n > peak[0]:
                            peak = (rate * n, f"{kind} {n}, {layer}")
                    except Failed as e:
                        cells[layer] = "FAIL"
                        print(f"  {macro}, {layer}: {e}", file=sys.stderr)
                if "FAIL" in cells.values():
                    status = 1
                row = f"{kind:5} {n:>6} {built * 1000:>6.0f}ms {2 * ticks:>11}"
                print(row + "".join(f" {cells.get(lay, '-'):>9}" for lay in layers), flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if peak[0]:
        print(f"peak: {_si(peak[0])} BogoBIPS ({peak[1]})")
    return status


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
