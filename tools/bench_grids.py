"""Two big-board workloads, timed end to end through a real editor window.

    uv run python tools/bench_grids.py rings [loops=4096]
    uv run python tools/bench_grids.py nand [gates=10000]
    uv run python tools/bench_grids.py all          (both, at both sizes)
    ... [--json=out.json]                           (also write the numbers)
    ... [--repeat=3]                                (each line: its best of 3 runs)

rings: independent loops of 3 inverters, made the way a user makes them -- one
    loop, selected, then Ctrl+D until there are `loops` of them. Many tiny nets;
    nearly every pin toggles every tick.
nand: a grid of NAND gates whose inputs come from random gates anywhere on the
    board (every 10th part an IN switch, so there are 0s to start from). Long
    wires, big fan-out, one tangled net graph; loaded from a save file.

They stress different things, so neither stands in for the other. Uses a throwaway
data folder (PIJL_DATA). Times are milliseconds; "xN" lines are medians of N.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time

os.environ["PIJL_DATA"] = tempfile.mkdtemp(prefix="pijl-bench-")

from pyglet import gl  # noqa: E402
from pyglet.window import key  # noqa: E402

from pijl.snapshot import Snapshot  # noqa: E402
from pijl.ui.document import capture, internal_wires  # noqa: E402
from pijl.ui.editor import Editor  # noqa: E402

RESULTS: dict[str, float] = {}
GC = {"ms": 0.0, "t0": 0.0, "last": 0.0}  # time in the garbage collector (see timed)


def _gc_watch(phase: str, info: dict) -> None:
    if phase == "start":
        GC["t0"] = time.perf_counter()
    else:
        GC["ms"] += 1000 * (time.perf_counter() - GC["t0"])


gc.callbacks.append(_gc_watch)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = next(
        (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--json=")), None
    )
    repeat = next(
        (int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--repeat=")), 1
    )
    what = args[0] if args else "all"
    size = int(args[1]) if len(args) > 1 else None
    runs = []
    if what in ("rings", "all"):
        runs += [
            (rings, n)
            for n in (
                [size] if size else ([4096] if what == "rings" else [4096, 16384])
            )
        ]
    if what in ("nand", "all"):
        runs += [
            (nand, n)
            for n in (
                [size] if size else ([10000] if what == "nand" else [10000, 40000])
            )
        ]
    for fn, n in runs:
        for _ in range(repeat):
            fn(n)
    if repeat > 1:
        print(f"\n== best of {repeat}")
        for label, ms in RESULTS.items():
            print(f"  {label:58s} {ms:9.2f} ms")
    if out:
        with open(out, "w") as f:
            json.dump(RESULTS, f, indent=1)


# ---- the workloads ---------------------------------------------------------------


def rings(loops: int) -> None:
    ed = fresh_editor()
    tag = f"rings{loops}"
    print(f"\n== {loops} loops of 3 inverters")
    vs = [ed.add_part("NOT", k * 120, 0) for k in range(3)]
    for a, b in zip(vs, vs[1:] + vs[:1]):
        ed.connect(a.part.outputs[0], b.part.inputs[0])
    ed.update(1 / 60)
    ed._reset_history(None)
    ed.selection.set(vs, internal_wires(ed, vs))
    presses = int(math.log2(loops))
    t_all = time.perf_counter()
    for i in range(presses):
        ms = timed(lambda: press(ed, key.D, key.MOD_CTRL))
        if i >= presses - 3:
            report(tag, f"ctrl+d press {i + 1} (-> {2 ** (i + 1)} loops)", ms)
        frame_ms = timed(lambda: frame(ed))
        if i == presses - 1:
            report(tag, "  the frame after it", frame_ms)
    report(
        tag,
        f"all {presses} presses, with their frames",
        1000 * (time.perf_counter() - t_all),
    )
    print(f"  ({len(ed.circuit.parts)} parts, {len(ed.circuit.wires)} wires)")
    steady(ed, tag)
    report(tag, "undo the last press", timed(lambda: press(ed, key.Z, key.MOD_CTRL)))
    report(tag, "redo it", timed(lambda: press(ed, key.Y, key.MOD_CTRL)))
    common(ed, tag)
    ed.close()


def nand(gates: int) -> None:
    ed = fresh_editor()
    tag = f"nand{gates}"
    print(f"\n== {gates} randomly wired NAND gates")
    cols = int(math.sqrt(gates) * 1.5)
    rng = random.Random(1)
    parts, wires = {}, {}
    outs = []  # uids of parts with an output
    for i in range(gates):
        uid = i + 1
        kind = "IN" if i % 10 == 0 else "NAND"
        parts[uid] = (kind, "", (i % cols) * 120, -(i // cols) * 80, {})
        outs.append(uid)
    w = 1
    for uid, (kind, *_rest) in parts.items():
        if kind != "NAND":
            continue
        for pin in (0, 1):
            src = rng.choice(outs)
            while src == uid:
                src = rng.choice(outs)
            wires[w] = (("p", src, False, 0), ("p", uid, True, pin), (), None, None)
            w += 1
    snap = Snapshot(parts, wires)
    ed.store.save("bench", snap, ed.catalog, "bench")
    report(tag, "load from disk", timed(lambda: ed._load("bench")))
    print(f"  ({len(ed.circuit.parts)} parts, {len(ed.circuit.wires)} wires)")
    for _ in range(30):  # let the switches' 0s spread
        frame(ed)
    steady(ed, tag)
    report(
        tag,
        "save to disk",
        timed(lambda: ed.store.save("bench2", capture(ed), ed.catalog, "b2")),
    )
    press(ed, key.A, key.MOD_CTRL)
    report(tag, "ctrl+d the whole board", timed(lambda: press(ed, key.D, key.MOD_CTRL)))
    report(tag, "  the frame after it", timed(lambda: frame(ed)))
    report(tag, "undo it", timed(lambda: press(ed, key.Z, key.MOD_CTRL)))
    common(ed, tag)
    ed.close()


# ---- shared parts -----------------------------------------------------------------


def steady(ed: Editor, tag: str) -> None:
    """Frames once nothing is being edited."""
    for _ in range(5):
        frame(ed)
    report(tag, "sim step", median(ed.circuit.step, 30), 30)
    report(tag, "update() (sim + sync + ui)", median(lambda: ed.update(1 / 60), 30), 30)
    report(tag, "on_draw()", median(lambda: (ed.switch_to(), ed.on_draw()), 30), 30)
    report(tag, "  upload after a step", statistics.median(upload_only(ed) for _ in range(30)), 30)
    report(tag, "frame: update + draw + glFinish", median(lambda: full_frame(ed), 30), 30)


def upload_only(ed: Editor) -> float:
    """One sim step and sync, then (timed) only the buffers' uploads."""
    ed.update(1 / 60)
    ed.switch_to()
    gl.glFinish()
    t = time.perf_counter()
    for buf in ed.world.buffers():
        buf._upload()
    gl.glFinish()
    return 1000 * (time.perf_counter() - t)


def full_frame(ed: Editor) -> None:
    ed.update(1 / 60)
    ed.switch_to()
    ed.on_draw()
    gl.glFinish()


def common(ed: Editor, tag: str) -> None:
    report(tag, "select all", timed(lambda: press(ed, key.A, key.MOD_CTRL)))
    report(tag, "drag all: pick up + one move", timed(lambda: drag(ed, 0)))
    report(tag, "  ... and let go", timed(lambda: drag(ed, 1)))
    report(tag, "delete all", timed(lambda: press(ed, key.DELETE, 0)))
    report(tag, "  ... and undo that", timed(lambda: press(ed, key.Z, key.MOD_CTRL)))


def fresh_editor() -> Editor:
    ed = Editor()
    ed._enable_event_queue = False  # dispatch synthetic events right away
    ed._clear_board()
    ed._reset_history(None)
    ed.update(1 / 60)
    return ed


def press(ed: Editor, sym: int, mod: int) -> None:
    ed.dispatch_event("on_key_press", sym, mod)
    ed.dispatch_event("on_key_release", sym, mod)


def frame(ed: Editor) -> None:
    ed.update(1 / 60)
    ed.switch_to()
    ed.on_draw()


def drag(ed: Editor, phase: int) -> None:
    """phase 0: press on a selected part and move; 1: let go there."""
    from pyglet.window import mouse

    v = min(ed.selection.parts, key=lambda v: v.part.uid)  # (the same one every run)
    sx, sy = ed.camera.world_to_screen(v.x + v.w / 2, v.y + v.h / 2)
    if phase == 0:
        ed.dispatch_event("on_mouse_press", sx, sy, mouse.LEFT, 0)
        ed.dispatch_event("on_mouse_drag", sx + 40, sy, 40, 0, mouse.LEFT, 0)
    else:
        ed.dispatch_event("on_mouse_release", sx, sy, mouse.LEFT, 0)


def timed(fn) -> float:
    """fn's time; how much of it the garbage collector took is left in GC["last"]."""
    before = GC["ms"]
    t0 = time.perf_counter()
    fn()
    ms = 1000 * (time.perf_counter() - t0)
    GC["last"] = GC["ms"] - before
    return ms


def median(fn, reps: int) -> float:
    return statistics.median(timed(fn) for _ in range(reps))


def report(tag: str, label: str, ms: float, reps: int = 1) -> None:
    name = f"{tag}: {label.strip()}"
    RESULTS[name] = round(min(ms, RESULTS.get(name, ms)), 3)
    gc_ms, GC["last"] = GC["last"], 0.0
    note = f"  (x{reps})" if reps > 1 else ""
    if gc_ms >= 1:
        note += f"  [gc {gc_ms:.0f} ms]"
    print(f"  {label:44s} {ms:9.2f} ms" + note)


if __name__ == "__main__":
    main()
