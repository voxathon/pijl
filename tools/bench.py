"""How fast is pijl with a big board? Drives a real editor window through its own event
handlers and times the things that get slow as boards grow.

    uv run python tools/bench.py [parts=3000] [--only name,name]

Uses a throwaway data folder (PIJL_DATA), so your projects are never touched. The board
is rows of NAND gates chained output -> input, with an IN switch every 10 parts. Times
are milliseconds per action (averaged where it says xN). A window flashes up: pyglet
needs a GL context.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ["PIJL_DATA"] = tempfile.mkdtemp(prefix="pijl-bench-")

from pyglet.window import key, mouse  # noqa: E402

from pijl.ui.editor import Editor  # noqa: E402

COLS = 60  # parts per row


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 3000
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)

    ed = Editor()
    ed._enable_event_queue = False  # dispatch synthetic events right away (the event loop does this)
    t0 = time.perf_counter()
    views = [ed.add_part("NAND" if i % 10 else "IN", (i % COLS) * 120, (i // COLS) * 80) for i in range(n)]
    for a, b in zip(views, views[1:]):
        if b.part.inputs:
            ed.connect(a.part.outputs[0], b.part.inputs[0])
    ed._reset_history(None)
    ed.update(1 / 60)
    print(f"{n} parts, {len(ed.wire_views)} wires")
    report("build the board", 1000 * (time.perf_counter() - t0))

    def scr(wx, wy):
        return ed.camera.world_to_screen(wx, wy)

    def ev(name, *args):
        ed.dispatch_event(name, *args)

    ed.camera.center_on(600, 400, ed.width, ed.height)
    target, switch = views[62], views[60]
    tx, ty = scr(target.x + 40, target.y + 20)
    sx, sy = scr(switch.x + 10, switch.y + 10)

    def drag_one():
        ev("on_mouse_press", tx, ty, mouse.LEFT, 0)
        ev("on_mouse_drag", tx + 30, ty + 20, 30, 20, mouse.LEFT, 0)
        ev("on_mouse_release", tx + 30, ty + 20, mouse.LEFT, 0)

    def undo():
        ev("on_key_press", key.Z, key.MOD_CTRL)
        ev("on_key_release", key.Z, key.MOD_CTRL)

    def rewire():  # wiring changed: the nets get rebuilt on the next step
        ed.connect(views[5].part.outputs[0], views[7].part.inputs[1])
        ed.circuit.step()

    cases = {
        "step": ("sim step", ed.circuit.step, 20),
        "rewire": ("rewire an input + step (net rebuild)", rewire, 5),
        "frame": ("frame update (sim + sync)", lambda: ed.update(1 / 60), 20),
        "draw": ("draw a frame", lambda: (ed.switch_to(), ed.on_draw()), 20),
        "hover": ("mouse move over a part", lambda: ev("on_mouse_motion", tx, ty, 0, 0), 50),
        "click": ("click a switch", lambda: (ev("on_mouse_press", sx, sy, mouse.LEFT, 0),
                                             ev("on_mouse_release", sx, sy, mouse.LEFT, 0)), 20),
        "box": ("box-select query (1000 x 600 world units)",
                lambda: (ed.part_index.query(0, 0, 1000, 600), ed.wire_index.query(0, 0, 1000, 600)), 20),
        "edit": ("drag one part (an edit)", drag_one, 1),
        "undo": ("undo it", undo, 1),
    }
    for name, (label, fn, reps) in cases.items():
        if only is None or name in only:
            report(label, timed(fn, reps), reps)

    if only is None or "drag" in only:
        # everything selected, dragged around: the per-mouse-move cost
        ev("on_key_press", key.A, key.MOD_CTRL)
        report(f"pick up all {n} parts", timed(lambda: (ev("on_mouse_press", tx, ty, mouse.LEFT, 0),
                                                        ev("on_mouse_drag", tx + 10, ty, 10, 0, mouse.LEFT, 0))))
        moves = iter(range(1, 1000))
        report(f"drag all {n} parts: one mouse move",
               timed(lambda: ev("on_mouse_drag", tx + 10 + next(moves), ty, 1, 0, mouse.LEFT, 0), 10), 10)
        report("... and let go (an edit)", timed(lambda: ev("on_mouse_release", tx + 20, ty, mouse.LEFT, 0)))
        # half the board: the wire between the halves stretches
        ed.selection.set(parts=views[: n // 2])
        ev("on_mouse_press", tx, ty, mouse.LEFT, 0)
        ev("on_mouse_drag", tx + 10, ty, 10, 0, mouse.LEFT, 0)
        report(f"drag {n // 2} parts: one mouse move",
               timed(lambda: ev("on_mouse_drag", tx + 10 + next(moves), ty, 1, 0, mouse.LEFT, 0), 10), 10)
        ev("on_mouse_release", tx + 20, ty, mouse.LEFT, 0)
    if only is None or "delete" in only:
        ev("on_key_press", key.A, key.MOD_CTRL)
        report(f"delete all {n} parts", timed(lambda: ev("on_key_press", key.DELETE, 0)))
        report("... and undo that", timed(undo))
    ed.close()


def timed(fn, reps: int = 1) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1000 * (time.perf_counter() - t0) / reps


def report(label: str, ms: float, reps: int = 1) -> None:
    print(f"  {label:42s} {ms:9.2f} ms" + (f"  (x{reps})" if reps > 1 else ""))


if __name__ == "__main__":
    main()
