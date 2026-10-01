"""Run a varied board through many edits; after each step, digest every instance buffer
(field by field), the spatial indexes, the views' own data and whether the undo
history matches the board. For refactors that must not change a single byte: compare
a run of HEAD with one of the working tree.

    uv run python tools/equiv_buffers.py out.json

To compare against another commit, run it from an extracted copy of that commit's src:

    git archive <commit> src | tar -x -C <dir>
    PYTHONPATH=<dir>/src uv run python tools/equiv_buffers.py before.json
    uv run python tools/equiv_buffers.py after.json
    uv run python tools/equiv_cmp.py before.json after.json

EQ_JUNK=n allocates junk first, shifting where objects land in memory: sets of objects
hashed by address reorder with it, so a difference then means something depends on
that order. EQ_SHOW=1 prints where the history and the board disagree.
EQ_HASH_BY_UID=1 (with PYTHONHASHSEED=0) hashes views by uid: for commits from before
Ctrl+D became deterministic."""
import hashlib
import json
import os
import sys
import tempfile

os.environ["PIJL_DATA"] = tempfile.mkdtemp(prefix="pijl-eq-")
import numpy as np
from pyglet.window import key, mouse

sys.path.insert(0, "tools")
import bench_grids as B  # noqa: E402
from pijl.ui.views import PartView, WireView  # noqa: E402

if os.environ.get("EQ_HASH_BY_UID"):  # (HEAD before the order fix needs it)
    # Sets of views iterate by hash; by default that's the address, which varies run to
    # run (and with it Ctrl+D's order). Hash by uid so both versions see the same order.
    from pijl.sim.circuit import Part, Wire  # noqa: E402
    
    PartView.__hash__ = lambda self: 4 * self.part.uid
    WireView.__hash__ = lambda self: 4 * self.wire.uid + 1
    Part.__hash__ = lambda self: 4 * self.uid + 2
    Wire.__hash__ = lambda self: 4 * self.uid + 3

# EQ_JUNK=n: shift where objects land in memory (address-hashed sets reorder with it)
_junk = []
if os.environ.get("EQ_JUNK"):
    import random

    rng = random.Random(int(os.environ["EQ_JUNK"]))
    _junk = [[None] * rng.randint(1, 40) for _ in range(rng.randint(1000, 50000))]
    del _junk[:: rng.randint(2, 5)]  # (and some holes)

out = []


def h(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def okey(o):
    if hasattr(o, "part"):
        return ("p", o.part.uid)
    if hasattr(o, "wire"):
        return ("w", o.wire.uid)
    return ("?", repr(o))


def digest(ed, step):
    d = {"step": step}
    for k, buf in sorted(ed.world._buffers.items(), key=lambda kv: kv[0]):
        used = np.flatnonzero(buf.used[: buf.end])
        e = {"top": buf.top, "used": h(used)}
        for name in buf.dtype.names:
            e[name] = h(buf.data[name][used])
        e["state"] = h(buf.state[used]) if hasattr(buf, "state") else h(buf.data["flags"][used, 0]) if "flags" in buf.dtype.names else "-"
        for name in ("pin_src", "wire_src"):  # (None = never set: all -1; any int width)
            src = getattr(buf, name)
            e[name] = h(np.full(len(used), -1, np.int64) if src is None else src[used].astype(np.int64))
        d[str(k)] = e
    d["echoes"] = sorted(str(k) for k in ed.world._echoes)
    for name in ("part_index", "wire_index"):
        ix = getattr(ed, name)
        rows = sorted(
            (okey(o), [list(map(float, ix._cols[:, r])) for r in ([rs] if type(rs) is int else rs)])
            for o, rs in ix.where.items()
        )
        d[name] = hashlib.sha1(repr(rows).encode()).hexdigest()[:12]
        d[name + "_n"] = len(rows)
    pv = []
    for part, v in sorted(ed.part_views.items(), key=lambda kv: kv[0].uid):
        pv.append(
            (
                part.uid, part.kind, part.label, sorted(part.props.items()),
                v.x, v.y, v.w, v.h, v.opacity, bool(v.lifted), bool(v.selected),
                repr(v.pin_tints), repr(v.body_tint), v.pin_labels_shown,
            )
        )
    wv = []
    for wire, v in sorted(ed.wire_views.items(), key=lambda kv: kv[0].uid):
        wv.append(
            (
                wire.uid, v.src, v.dst, list(v.bends), v.color, repr(list(v.stops)),
                bool(v.selected), bool(v.lifted), list(v.points),
            )
        )
    d["parts"] = hashlib.sha1(repr(pv).encode()).hexdigest()[:12]
    d["wires"] = hashlib.sha1(repr(wv).encode()).hexdigest()[:12]
    d["n"] = (len(pv), len(wv), len(ed.selection.parts), len(ed.selection.wires))
    d["pins"] = h(ed.circuit._pins.states[: ed.circuit.pin_count])
    from pijl.ui.document import capture
    board, hist = capture(ed), ed.history.current
    # (after a record: the history's board must match the real one; repr catches 200 vs 200.0)
    d["history_in_sync"] = (repr(sorted(board.parts.items())), repr(sorted(board.wires.items()))) == (
        repr(sorted(hist.parts.items())), repr(sorted(hist.wires.items())))
    if not d["history_in_sync"] and os.environ.get("EQ_SHOW"):
        for sec in ("parts", "wires"):
            bb, hh = getattr(board, sec), getattr(hist, sec)
            for u in sorted(bb.keys() | hh.keys()):
                if repr(bb.get(u)) != repr(hh.get(u)):
                    print(step, sec, u, "board:", bb.get(u), "history:", hh.get(u))
    d["tagged"] = sorted(p.uid for p, v in ed.part_views.items() if v.pin_labels_shown)
    d["hover"] = ed.hover_view.part.uid if ed.hover_view is not None else None
    d["mouse"] = list(ed.mouse)
    d["labels"] = sorted((p.uid, p.label) for p in ed.part_views if p.label)
    out.append(d)


def frames(ed, n=3):
    for _ in range(n):
        B.frame(ed)


ed = B.fresh_editor()
ed.circuit.settle_ticks = 0
a = ed.add_part("IN", 200, 360)
b = ed.add_part("IN", 200, 220)
g = ed.add_part("NAND", 380, 280)
n = ed.add_part("NOT", 560, 280)
o = ed.add_part("OUT", 740, 280)
o2 = ed.add_part("OUT", 740, 440)
a.part.label, o.part.label = "A", "Q"
a.refresh_name(), o.refresh_name()
ed._set_part_color(o, "red")
w1 = ed.connect(a.part.outputs[0], g.part.inputs[0], bends=[(320, 380), (320, 320)])
ed.connect(b.part.outputs[0], g.part.inputs[1])
ed.connect(g.part.outputs[0], n.part.inputs[0])
w4 = ed.connect(n.part.outputs[0], o.part.inputs[0], color="blue")
v1 = ed.wire_views[w1]
jx, jy = v1.points[1][0], (v1.points[1][1] + v1.points[2][1]) / 2
ed.connect(w1, o2.part.inputs[0], bends=[(jx, 460)], a_pos=(jx, jy))
ed._record()
ed.circuit.click(a.part)
frames(ed, 6)
digest(ed, "built")

B.press(ed, key.A, key.MOD_CTRL)
digest(ed, "select all")
B.drag(ed, 0)
digest(ed, "lifted")
B.drag(ed, 1)
frames(ed)
digest(ed, "dropped")

B.press(ed, key.D, key.MOD_CTRL)
frames(ed)
B.press(ed, key.D, key.MOD_CTRL)
frames(ed)
digest(ed, "ctrl+d x2")

B.press(ed, key.TAB, 0)
frames(ed)
digest(ed, "tab 1")
B.press(ed, key.TAB, 0)
frames(ed)
digest(ed, "tab 2")
B.press(ed, key.TAB, 0)
frames(ed)

g.part.label = "gate"
g.refresh_name()
ed._set_part_color(a, "green")
ed.wire_views[w4].color = "yellow"
ed._record()
frames(ed)
digest(ed, "relabel+recolor")

ed.dispatch_event("on_mouse_motion", *ed.camera.world_to_screen(g.x + 5, g.y + 5), 0, 0)
frames(ed)
digest(ed, "hover")

B.press(ed, key.ESCAPE, 0)
B.press(ed, key.A, key.MOD_CTRL)
B.press(ed, key.DELETE, 0)
frames(ed)
digest(ed, "delete all")
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "undo")
B.press(ed, key.Y, key.MOD_CTRL)
frames(ed)
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "redo+undo")

# cut a wire that has a junction on it: splices
w1v = ed.wire_views[ed.circuit.wire_by_uid[w1.uid]]
if w1v is not None:
    px, py = w1v.points[-2]
    ed.cut_wire(w1v, (px, py - 5))
    ed._record()
    frames(ed)
    digest(ed, "cut")

# edit a label in place (a part that had none: its label is made now), and a wire edit
nv = ed.part_views[ed.circuit.part_by_uid[n.part.uid]]
ed._start_edit(nv)
ed.dispatch_event("on_text", "inv")
frames(ed)
digest(ed, "editing label")
ed._finish_edit(True)
ed._record()
frames(ed)
digest(ed, "label done")
ed._start_wire_edit(ed.wire_views[ed.circuit.wire_by_uid[w4.uid]])
frames(ed)
digest(ed, "wire edit")
B.press(ed, key.ESCAPE, 0)
frames(ed)
digest(ed, "wire edit done")

# a colorless unit (two gates and the wire between them), duplicated
B.press(ed, key.ESCAPE, 0)
gv = ed.part_views[ed.circuit.part_by_uid[g.part.uid]]
nv = ed.part_views[ed.circuit.part_by_uid[n.part.uid]]
ed.selection.set([gv, nv], B.internal_wires(ed, [gv, nv]))
B.press(ed, key.D, key.MOD_CTRL)
frames(ed)
B.press(ed, key.D, key.MOD_CTRL)
frames(ed)
digest(ed, "plain ctrl+d")
ed._space_tiling(1)  # Ctrl+scroll: the block spreads out (amended into one step)
ed._space_tiling(1)
frames(ed)
digest(ed, "ctrl+scroll x2")

# the whole board dragged, then undone and redone (a move step)
B.press(ed, key.ESCAPE, 0)
B.press(ed, key.A, key.MOD_CTRL)
B.drag(ed, 0)
B.drag(ed, 1)
frames(ed)
digest(ed, "drag all")
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "undo drag")
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "undo ctrl+scroll")
B.press(ed, key.Y, key.MOD_CTRL)
B.press(ed, key.Y, key.MOD_CTRL)
frames(ed)
digest(ed, "redo both")

# ghosts: copy + paste follow the cursor
B.press(ed, key.ESCAPE, 0)
B.press(ed, key.A, key.MOD_CTRL)
B.press(ed, key.C, key.MOD_CTRL)
B.press(ed, key.V, key.MOD_CTRL)
ed.dispatch_event("on_mouse_motion", 300, 300, 0, 0)
frames(ed)
digest(ed, "paste ghosts")
ed.dispatch_event("on_mouse_press", 300, 300, mouse.LEFT, 0)
ed.dispatch_event("on_mouse_release", 300, 300, mouse.LEFT, 0)
frames(ed)
digest(ed, "placed")
B.press(ed, key.Z, key.MOD_CTRL)
B.press(ed, key.Z, key.MOD_CTRL)
frames(ed)
digest(ed, "undo x2")

json.dump(out, open(sys.argv[1], "w"), indent=0)
print(f"{len(out)} steps; last n = {out[-1]['n']}")
ed.close()
