"""Macro nesting through the editor: a three-level macro, a half adder and plain parts on
a board, wired (bends, a junction off a macro output), through many edits. After each
step, digest what any version of the app has in common: the board snapshot, simulated
pin states, every view's geometry (pins too) and flags, wire points, the history being
in sync with the board, the rendered board (not the HUD: it shows live timings), and
a uid-blind digest. Semantic, so it compares across big refactors (equiv_buffers.py is
the byte-for-byte check, for refactors that shouldn't change a single slot).

    uv run python tools/equiv_macros.py out.json

To compare against another commit, run it from an extracted copy of that commit's src:

    git archive <commit> src | tar -x -C <dir>
    PYTHONPATH=<dir>/src uv run python tools/equiv_macros.py before.json
    uv run python tools/equiv_macros.py after.json
    uv run python tools/equiv_cmp.py before.json after.json

On a version with PartTable it also checks internal consistency: pins held per row =
the part's pins, dots and pin-end wires at pin_pos (printed as "table problems").
MEQ_DUMP="step,step" MEQ_OUT=prefix saves those steps' frames as .npy. Settling (the
sim's power-on noise) is off unless MEQ_SETTLE=1 (deterministic either way: seeded)."""
import hashlib
import json
import os
import sys

sys.path.insert(0, "tools")
import bench_grids as B  # noqa: E402  (sets its own PIJL_DATA)
import numpy as np  # noqa: E402
import pyglet  # noqa: E402
from pyglet.window import key, mouse  # noqa: E402

import pijl.ui.canvas as _C  # noqa: E402


class _T:  # (frozen clock: patterns that animate draw the same every run)
    monotonic = staticmethod(lambda: 1234.5)
    perf_counter = staticmethod(lambda: 1234.5)


_C.time = _T

from pijl.sim.circuit import Pin  # noqa: E402
import pijl.ui.editor as _E  # noqa: E402

if not os.environ.get("MEQ_SETTLE"):  # (no settling noise: every run the same)
    _E.SETTLE_TICKS = 0
from pijl.snapshot import Snapshot  # noqa: E402
from pijl.ui.document import capture  # noqa: E402

out = []
problems = []


def pinref(uid, index, is_input):
    return ("p", uid, is_input, index)


def w(src, si, dst, di, bends=()):
    return (pinref(src, si, False), pinref(dst, di, True), tuple(bends), None, None)


HALF_ADDER = Snapshot(
    {
        1: ("IN", "a", 0.0, 100.0, {}),
        2: ("IN", "b", 0.0, 0.0, {}),
        3: ("NAND", "", 100.0, 50.0, {}),
        4: ("NAND", "", 200.0, 100.0, {}),
        5: ("NAND", "", 200.0, 0.0, {}),
        6: ("NAND", "", 300.0, 50.0, {}),
        7: ("NOT", "", 300.0, -50.0, {}),
        10: ("OUT", "sum", 400.0, 50.0, {}),
        11: ("OUT", "carry", 400.0, -50.0, {}),
    },
    {
        1: w(1, 0, 3, 0), 2: w(2, 0, 3, 1), 3: w(1, 0, 4, 0), 4: w(3, 0, 4, 1),
        5: w(3, 0, 5, 0), 6: w(2, 0, 5, 1), 7: w(4, 0, 6, 0), 8: w(5, 0, 6, 1),
        9: w(6, 0, 10, 0), 12: w(3, 0, 7, 0), 13: w(7, 0, 11, 0),
    },
)  # fmt: skip
TWO = Snapshot(  # two half adders chained: a + b, then sum + b
    {
        1: ("IN", "a", 0.0, 0.0, {}),
        2: ("IN", "b", 0.0, 100.0, {}),
        3: ("macro:ha", "", 100.0, 0.0, {}),
        4: ("macro:ha", "", 300.0, 0.0, {}),
        5: ("OUT", "s", 500.0, 0.0, {}),
        6: ("OUT", "c1", 500.0, 100.0, {}),
        7: ("OUT", "c2", 500.0, 200.0, {}),
    },
    {
        1: w(1, 0, 3, 0), 2: w(2, 0, 3, 1), 3: w(3, 0, 4, 0), 4: w(2, 0, 4, 1),
        5: w(4, 0, 5, 0), 6: w(3, 1, 6, 0), 7: w(4, 1, 7, 0),
    },
)  # fmt: skip
THREE = Snapshot(  # a macro of a macro of macros, plus a gate of its own
    {
        1: ("IN", "x", 0.0, 0.0, {}),
        2: ("IN", "y", 0.0, 100.0, {}),
        3: ("macro:two", "", 100.0, 0.0, {}),
        4: ("NAND", "", 300.0, 0.0, {}),
        5: ("OUT", "q", 500.0, 0.0, {}),
        6: ("OUT", "s", 500.0, 100.0, {}),
    },
    {
        1: w(1, 0, 3, 0), 2: w(2, 0, 3, 1), 3: w(3, 0, 4, 0), 4: w(3, 1, 4, 1),
        5: w(4, 0, 5, 0), 6: w(3, 0, 6, 0, bends=[(250.0, 50.0)]),
    },
)  # fmt: skip


def f(x):
    return repr(x)


def check_tables(ed, step):
    t = getattr(ed, "part_table", None)
    if t is None:
        return
    dots = ed.world.buffer(_C_DOT, ed.layers.pins)
    for part, v in ed.part_views.items():
        slots = [p.slot for p in part.pins]
        if t.pin_slots(v.row).tolist() != slots:
            problems.append((step, "pins", part.uid))
        for p, d in zip(part.pins, v.dots):
            want = np.array(v.pin_pos(p), np.float32)
            if not np.array_equal(dots.f["center"][d], want) and not v.lifted:
                problems.append((step, "dot", part.uid, p.index, p.is_input))
    for wire, v in ed.wire_views.items():
        for end, at in ((wire.src, v.src), (wire.dst, v.dst)):
            if isinstance(end, Pin) and not v.lifted:
                pv = ed.part_views[end.part]
                if pv.pin_pos(end) != at and not pv.lifted:
                    problems.append((step, "wire end", wire.uid))


def frame_hash(ed):
    ed.switch_to()
    # the board only: the HUD shows live timings
    ed.clear()
    ed.grid.draw(ed, ed.camera, snapping=ed.snap_step)
    ed.view = ed.camera.matrix()
    ed.world.draw()
    from pyglet.math import Mat4

    ed.view = Mat4()
    from pyglet import gl

    gl.glFinish()
    img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    arr = np.frombuffer(img.get_data("RGBA", img.width * 4), np.uint8)
    if os.environ.get("MEQ_DUMP") and _step[0] in os.environ["MEQ_DUMP"].split(","):
        np.save(os.environ["MEQ_OUT"] + "_" + _step[0].replace(" ", "_").replace("/", "_") + ".npy", arr.reshape(img.height, img.width, 4))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


_step = [""]


def digest(ed, step):
    _step[0] = step
    d = {"step": step}
    snap = capture(ed)
    d["board"] = hashlib.sha1(
        f((sorted(snap.parts.items()), sorted(snap.wires.items()), sorted(snap.wire_colors.items()))).encode()
    ).hexdigest()[:12]
    d["n"] = (len(snap.parts), len(snap.wires), len(ed.circuit.hidden_parts), len(ed.selection.parts))
    sim = [
        (p.uid, [int(q.state) for q in p.pins]) for p in sorted(ed.part_views, key=lambda p: p.uid)
    ]
    d["sim"] = hashlib.sha1(f(sim).encode()).hexdigest()[:12]
    d["outs"] = [
        (p.uid, p.label, int(p.inputs[0].state))
        for p in sorted(ed.part_views, key=lambda p: p.uid)
        if p.kind == "OUT"
    ][:6]
    pv = []
    for part, v in sorted(ed.part_views.items(), key=lambda kv: kv[0].uid):
        pv.append(
            (
                part.uid, part.kind, v.x, v.y, v.w, v.h, v.opacity, bool(v.lifted), bool(v.selected),
                v.pin_labels_shown, [v.pin_pos(p) for p in part.pins], v.title,
            )
        )
    d["views"] = hashlib.sha1(f(pv).encode()).hexdigest()[:12]
    wv = []
    for wire, v in sorted(ed.wire_views.items(), key=lambda kv: kv[0].uid):
        wv.append((wire.uid, list(v.points), v.color, f(list(v.stops)), bool(v.selected), bool(v.lifted)))
    d["wires"] = hashlib.sha1(f(wv).encode()).hexdigest()[:12]
    h = ed.history.current
    d["history_in_sync"] = (f(sorted(snap.parts.items())), f(sorted(snap.wires.items()))) == (
        f(sorted(h.parts.items())), f(sorted(h.wires.items())))
    d["tagged"] = sorted(p.uid for p, v in ed.part_views.items() if v.pin_labels_shown)
    blind_parts = sorted(
        (p.kind, p.label, v.x, v.y, [int(q.state) for q in p.pins], v.pin_labels_shown, bool(v.selected))
        for p, v in ed.part_views.items()
    )
    blind_wires = sorted((f(list(v.points)), v.color, bool(v.selected)) for v in ed.wire_views.values())
    d["blind"] = hashlib.sha1(f((blind_parts, blind_wires)).encode()).hexdigest()[:12]
    d["frame"] = frame_hash(ed)
    check_tables(ed, step)
    out.append(d)


def frames(ed, n=4):
    for _ in range(n):
        B.frame(ed)


ed = B.fresh_editor()
from pijl.ui.sdf_shapes import DOT as _C_DOT  # noqa: E402

if not os.environ.get("MEQ_SETTLE"):
    ed.circuit.settle_ticks = 0
for id, snap in (("ha", HALF_ADDER), ("two", TWO), ("three", THREE)):
    ed.store.save(id, snap, ed.catalog, id)
ed.picker.refresh() if hasattr(ed.picker, "refresh") else None

x = ed.add_part("IN", 100, 100)
y = ed.add_part("IN", 100, 260)
m3 = ed.add_part("macro:three", 300, 140)
ha = ed.add_part("macro:ha", 300, 420)
g = ed.add_part("AND", 640, 300)
q = ed.add_part("OUT", 820, 160)
r = ed.add_part("OUT", 820, 420)
s = ed.add_part("OUT", 820, 560)
q.part.label, r.part.label = "Q", "R"
q.refresh_name(), r.refresh_name()
ed.connect(x.part.outputs[0], m3.part.inputs[0])
ed.connect(y.part.outputs[0], m3.part.inputs[1], bends=[(220, 280), (220, 180)])
ed.connect(x.part.outputs[0], ha.part.inputs[0])
ed.connect(y.part.outputs[0], ha.part.inputs[1])
w_q = ed.connect(m3.part.outputs[0], q.part.inputs[0], bends=[(700, 160)])
ed.connect(m3.part.outputs[1], g.part.inputs[0])
ed.connect(ha.part.outputs[0], g.part.inputs[1])
ed.connect(g.part.outputs[0], r.part.inputs[0])
qv = ed.wire_views[w_q]
(ax, ay), (bx, by) = qv.points[0], qv.points[1]
jx, jy = (ax + bx) / 2, (ay + by) / 2
ed.connect(w_q, s.part.inputs[0], bends=[(jx, 560)], a_pos=(jx, jy))  # junction off a macro output
ed._record()
ed.circuit.click(x.part)
frames(ed, 8)
ed._fit_camera()
frames(ed)
digest(ed, "built")

for i in range(3):
    B.press(ed, key.TAB, 0)
    frames(ed)
    digest(ed, f"tab {i + 1}")  # hidden, always (macros have tags), back to hover

ed.dispatch_event("on_mouse_motion", *ed.camera.world_to_screen(m3.x + 5, m3.y + 5), 0, 0)
frames(ed)
digest(ed, "hover macro")
ed.circuit.click(y.part)
frames(ed, 8)
digest(ed, "toggle y")

B.press(ed, key.A, key.MOD_CTRL)
B.drag(ed, 0)
digest(ed, "lifted")
B.drag(ed, 1)
frames(ed)
digest(ed, "dropped")

B.press(ed, key.ESCAPE, 0)
m3v = ed.part_views[ed.circuit.part_by_uid[m3.part.uid]]
hav = ed.part_views[ed.circuit.part_by_uid[ha.part.uid]]
ed.selection.set([m3v, hav], B.internal_wires(ed, [m3v, hav]))
B.press(ed, key.D, key.MOD_CTRL)
frames(ed)
B.press(ed, key.D, key.MOD_CTRL)
frames(ed)
digest(ed, "ctrl+d macros x2")
ed._space_tiling(1)
frames(ed)
digest(ed, "ctrl+scroll")
B.press(ed, key.TAB, 0)  # always: tags on every macro copy
frames(ed)
digest(ed, "tags on copies")
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "undo")
B.press(ed, key.Y, key.MOD_CTRL)
frames(ed)
digest(ed, "redo")

B.press(ed, key.ESCAPE, 0)
m3v = ed.part_views[ed.circuit.part_by_uid[m3.part.uid]]
ed.selection.set([m3v], [])
B.press(ed, key.DELETE, 0)
frames(ed)
digest(ed, "delete macro")
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed, 8)
digest(ed, "undo delete")

w_q2 = ed.circuit.wire_by_uid[w_q.uid]
wv = ed.wire_views[w_q2]
px, py = wv.points[-2]
ed.cut_wire(wv, (px + 3, py))
ed._record()
frames(ed)
digest(ed, "cut at junction")

B.press(ed, key.A, key.MOD_CTRL)
B.drag(ed, 0)
B.drag(ed, 1)
frames(ed)
digest(ed, "drag all w/ tags")
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "undo drag")

ed.store.save("board", capture(ed), ed.catalog, "board")
ed._load("board")
if not os.environ.get("MEQ_SETTLE"):
    ed.circuit.settle_ticks = 0
frames(ed, 8)
digest(ed, "save+open")

B.press(ed, key.ESCAPE, 0)
B.press(ed, key.A, key.MOD_CTRL)
B.press(ed, key.C, key.MOD_CTRL)
B.press(ed, key.V, key.MOD_CTRL)
ed.dispatch_event("on_mouse_motion", 400, 300, 0, 0)
frames(ed)
digest(ed, "paste ghosts")
ed.dispatch_event("on_mouse_press", 400, 300, mouse.LEFT, 0)
ed.dispatch_event("on_mouse_release", 400, 300, mouse.LEFT, 0)
frames(ed, 8)
digest(ed, "placed")
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "undo paste")

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"{len(out)} steps; last n = {out[-1]['n']}; outs {out[-1]['outs']}")
print("table problems:", problems[:10] if problems else "none")
ed.close()
