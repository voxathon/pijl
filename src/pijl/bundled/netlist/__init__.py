"""Netlists: wire a box up by typing what connects to what (see lang.py for the language).

In the console (`), a line that isn't a command is a netlist line, run in the
console's box (the one box selected, else the box under the cursor, else the whole
board). A box can also keep a script (on the box itself, Editor.set_box_data: it's
saved, copied and undone with it). Two commands for those:

    run [BOX]           runs the script of the console's box, or of the box labeled BOX
    edit [BOX]          opens it in your text editor, through a scratch file: save it
                        there, and the next run (or edit) takes it in

A box's right-click menu gets a Netlist submenu too: the console for that box, and
Run / Edit for its script. A box with no script whose label has an old
<project>/netlists/<label>.txt takes that file in the first time.

A name without a box/ in front means a part in the box, or anywhere on the board if
nothing in the box has that name. A run is all
or nothing: if any line can't be done, nothing is wired and the status line says why.
What it wires is one undo step, and comes out selected.

Wires are routed around parts and other wires (router.py): a pin feeding several
inputs gets a trunk with branches. An input the router can't reach gets a plain Z
wire instead, colored orange so it stands out. A bus gets its spine first: one
straight free-floating wire, laid on the free line nearest the middle of its pins.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from pijl.mods import after_import
from pijl.sim import FREE, Wire

from . import router
from .lang import NetlistError, Scope, pin_name, resolve

log = logging.getLogger(__name__)

TEMPLATE = """\
# Netlist for the box "{label}". Save, then: run (console) or Netlist > Run script.
#
#   CLK.q -> R*.clk              one pin to many
#   A*.q -> B*.d                 pairwise, in name order (R2 before R10)
#   ALU.a*, ALU.b* -> regs/R*.q  lists; regs/ is the box labeled "regs"
#   <NAND>*.a -> X.q             <kind>: parts by kind
#   chain ADD*: cout -> cin      ADD0.cout -> ADD1.cin, ADD1.cout -> ADD2.cin, ...
#   net DATA: R*.q, ALU.a        all one net (outputs and inputs, any mix)
#   bus DATA: R*.q, ALU.a        the same, drawn as a straight spine with taps
#
# A part's name is its label, or its kind if it has none. Names look in this box
# first, then on the whole board. Pins go by name, or in1, in2, ... out1, ...
# counted from the top. Case doesn't matter. Quote names with spaces.

"""


def script_file(project_path: Path, label: str) -> Path:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label).strip(" .") or "_"
    return project_path / "netlists" / f"{safe}.txt"


def run(editor, box, script: str, say=None) -> bool:
    """Resolve `script` in `box` (None: the whole board) and wire it. Returns False on
    a problem. `say(text, kind)` tells how it went (kind "out" or "error"); without
    it, the status line does."""
    if say is None:
        def say(text, kind="out"):
            (editor._report if kind == "error" else editor._notice)(f"netlist: {text}")
    try:
        plan = resolve(script, _scope(editor, box))
        jobs = _jobs(editor, plan)
    except NetlistError as e:
        say(str(e), "error")
        return False
    if not any(j.sinks for j in jobs):
        say("nothing to wire")
        return True
    results = _route(editor, jobs)
    wires, unrouted = [], 0
    with editor.wire_batch():
        for job, res in zip(jobs, results):
            made, failed = _make(editor, job, res)
            wires += made
            unrouted += failed
    editor.selection.set(wires=[editor.wire_views[w] for w in wires if w in editor.wire_views])
    said = [
        f"{n} {what}{'s' * (n != 1)}"
        for n, what in ((len(plan.pairs), "connection"), (len(plan.nets), "net"))
        if n
    ]
    if unrouted:
        said.append(f"{unrouted} not routed (orange)")
    msg = ", ".join(said)
    say(msg)
    log.info("%s: %s", f"box {box.label!r}" if box is not None else "board", msg)
    return True


@dataclass
class _Job:
    """One router net and the pins / wires behind it."""

    sinks: list  # pins to wire
    root: object = None  # a pin, or None
    seeds: list = field(default_factory=list)  # wires the net has already
    spine: bool = False
    width: int = 1


def _jobs(editor, plan) -> list[_Job]:
    c = editor.circuit
    jobs = []
    by_output: dict = {}  # (pairs: one net per output pin)
    for a, b in plan.pairs:
        by_output.setdefault(a, []).append(b)
    for a, bs in by_output.items():
        jobs.append(_Job(bs, root=a, seeds=_tree(c, [a]), width=a.width))
    for net in plan.nets:
        wired = [p for p in net.pins if c.ends_on(p)]
        trees: list[tuple[set, list]] = []  # (wired pins on it, its wires)
        for p in wired:
            if not any(p in t[0] for t in trees):
                wires = _tree(c, [p])
                on = {q for q in wired if any(w in wires for w in c.ends_on(q))}
                trees.append((on, wires))
        if len(trees) > 1:
            a, b = (next(iter(t[0])) for t in trees[:2])
            raise NetlistError(
                net.line,
                f"{pin_name(a)} and {pin_name(b)} are on different wires already: "
                "join those by hand",
            )
        sinks = [p for p in net.pins if p not in wired]
        seeds = trees[0][1] if trees else []
        # pins all of one kind can't be wired to each other: they need a spine
        alike = len({p.is_input for p in net.pins}) == 1
        jobs.append(
            _Job(
                sinks,
                seeds=seeds,
                spine=net.kind == "bus" or (alike and not seeds),
                width=net.pins[0].width,
            )
        )
    return jobs


def _tree(c, pins) -> list:
    """The wires on these pins, and every wire joined to those, in uid order."""
    seen: set = set()
    todo = [w for p in pins for w in c.ends_on(p)]
    while todo:
        w = todo.pop()
        if w in seen:
            continue
        seen.add(w)
        todo += c.attachments(w)
        todo += [e for e in (w.src, w.dst) if isinstance(e, Wire) and e is not w]
    return sorted(seen, key=lambda w: w.uid)


def _make(editor, job: _Job, res) -> tuple[list, int]:
    """The wires of one routed net. Returns (wires made, how many went unrouted)."""
    made, unrouted = [], 0
    spine = None
    if res.spine is not None:
        a, b = res.spine
        spine = editor.connect(FREE, FREE, [], _world(a), _world(b), width=job.width)
        if spine is not None:
            made.append(spine)
    root = job.sinks[res.rooted] if res.rooted is not None else job.root
    wire_of: list = []  # per route: its wire
    for r in res.routes:
        sink = job.sinks[r.sink]
        wire = None
        if r.path is not None:
            bends = [_world(c) for c in router.corners(r.path)]
            at = _world(r.path[0])
            if r.seed == router.SPINE:
                wire = editor.connect(spine, sink, bends, a_pos=at) if spine else None
            elif r.seed is not None:
                wire = editor.connect(r.seed, sink, bends, a_pos=at)
            elif r.joins is not None:
                if wire_of[r.joins] is not None:
                    wire = editor.connect(wire_of[r.joins], sink, bends, a_pos=at)
            elif root is not None:
                wire = editor.connect(root, sink, bends)
        if wire is None:
            unrouted += 1
            wire = _fallback(editor, job, root, spine, res, sink)
        if wire is not None:
            made.append(wire)
        wire_of.append(wire)
    return made, unrouted


def _fallback(editor, job: _Job, root, spine, res, sink):
    """An unroutable sink still gets its wire: orange, straight-ish, to wherever it can
    join the net."""
    if root is not None and root.is_input != sink.is_input:
        return editor.connect(root, sink, _z(editor, root, sink), color=UNROUTED)
    p = editor.pin_pos(sink)
    if spine is not None:
        ends = [_world(c) for c in res.spine]
        at = min(ends, key=lambda e: abs(e[0] - p[0]) + abs(e[1] - p[1]))
        return editor.connect(spine, sink, [], a_pos=at, color=UNROUTED)
    for w in job.seeds:
        if w in editor.wire_views:
            return editor.connect(w, sink, [], a_pos=editor.end_pos(w, p), color=UNROUTED)
    log.warning("couldn't wire %s into its net", sink)
    return None


UNROUTED = "orange"


def _grid() -> float:
    from pijl.ui.theme import GRID  # (pins are on it)

    return GRID


def _cell(p) -> tuple[int, int]:
    g = _grid()
    return round(p[0] / g), round(p[1] / g)


def _world(c) -> tuple[float, float]:
    g = _grid()
    return c[0] * g, c[1] * g


def _route(editor, jobs: list[_Job]) -> list:
    """Routes for every job (router.NetResult each): obstacles are the parts and wires
    near the pins."""
    g = _grid()
    pins = [p for j in jobs for p in (*j.sinks, *([j.root] if j.root else []))]
    pts = [editor.pin_pos(p) for p in pins]
    pad = router.MARGIN * 4 * g
    x0, y0 = min(p[0] for p in pts) - pad, min(p[1] for p in pts) - pad
    x1, y1 = max(p[0] for p in pts) + pad, max(p[1] for p in pts) + pad
    board = router.Board()
    for v in editor.part_index.query(x0, y0, x1, y1):
        if v in editor.placing_views:
            continue
        board.add_rect(
            math.floor(v.x / g), math.floor(v.y / g),
            math.ceil((v.x + v.w) / g), math.ceil((v.y + v.h) / g),
        )
        for pin in (*v.part.inputs, *v.part.outputs):
            board.add_stub(_pin(editor, pin))
    seeds = {w for j in jobs for w in j.seeds}  # (the router lays those down itself)
    for v in editor.wire_index.query(x0, y0, x1, y1):
        if v.wire not in seeds:
            for run in _on_grid(v.points):
                board.add_wire(v.wire.uid, run)
    nets = [
        router.Net(
            [_pin(editor, p) for p in j.sinks],
            root=_pin(editor, j.root) if j.root is not None else None,
            seeds=[
                (w, run)
                for w in j.seeds
                if w in editor.wire_views
                for run in _on_grid(editor.wire_views[w].points)
            ],
            spine=j.spine,
        )
        for j in jobs
        if j.sinks
    ]
    results = iter(router.route(board, nets))
    return [next(results) if j.sinks else router.NetResult([]) for j in jobs]


def _pin(editor, pin) -> router.Pin:
    return router.Pin(_cell(editor.pin_pos(pin)), -1 if pin.is_input else 1)


def _on_grid(points) -> list[list[tuple[int, int]]]:
    """A wire's points as cells, split where a point isn't on the grid."""
    g = _grid()
    runs, run = [], []
    for x, y in points:
        cx, cy = x / g, y / g
        if abs(cx - round(cx)) < 1e-6 and abs(cy - round(cy)) < 1e-6:
            run.append((round(cx), round(cy)))
        else:
            if len(run) > 1:
                runs.append(run)
            run = []
    if len(run) > 1:
        runs.append(run)
    return runs


def _scope(editor, box) -> Scope:
    views = editor.part_views

    def boxes(label: str) -> list:
        found = [b for b in editor.box_views.values() if b.label == label]
        if not found:
            raise KeyError(f"no box labeled {label!r}")
        if len(found) > 1:
            raise KeyError(f"{len(found)} boxes are labeled {label!r}")
        return [v.part for v in editor.box_contents(found[0])[0]]

    board = [p for p, v in views.items() if v not in editor.placing_views]
    return Scope(
        parts=[v.part for v in editor.box_contents(box)[0]] if box is not None else board,
        boxes=boxes,
        pos=lambda p: (views[p].x, views[p].y),
        wired=lambda pin: bool(editor.circuit.ends_on(pin)),
        board=lambda: board,
    )


def _z(editor, a, b) -> list[tuple[float, float]]:
    """Bends from a to b: straight if they line up, else a Z turning halfway across."""
    (x0, y0), (x1, y1) = editor.pin_pos(a), editor.pin_pos(b)
    if y0 == y1:
        return []
    from pijl.ui.theme import GRID as step  # (pins are on it)

    mx = round((x0 + x1) / 2 / step) * step
    return [(mx, y0), (mx, y1)]


def _open_in_editor(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


KEY = "netlist"  # where the script is kept on a box (Editor.set_box_data)

# box uid -> (scratch file, its mtime when written): scripts out in a text editor
_editing: dict[int, tuple[Path, float]] = {}


def script_of(box) -> str | None:
    return (box.mod_data.get(KEY) or {}).get("script")


def set_script(editor, box, script: str) -> None:
    editor.set_box_data(box, KEY, {"script": script} if script.strip() else None)


def _scratch(editor, box) -> Path:
    doc = re.sub(r"[^\w-]", "_", editor.doc or "untitled")
    return Path(tempfile.gettempdir()) / "pijl-netlists" / f"{doc}-box{box.uid}.txt"


def sync(editor, box, say) -> None:
    """Take in what was saved in the box's scratch file since it was opened (edit), or
    an old netlists/<label>.txt for a box with no script yet."""
    if box.uid in _editing:
        path, written = _editing[box.uid]
        try:
            mtime = path.stat().st_mtime
            if mtime != written:
                set_script(editor, box, path.read_text(encoding="utf-8"))
                _editing[box.uid] = (path, mtime)
                say("took in the edited script")
        except OSError as e:
            say(f"couldn't read {path}: {e}", "error")
        return
    if script_of(box) is None and box.label:
        old = script_file(editor.project.path, box.label)
        if old.is_file():
            try:
                set_script(editor, box, old.read_text(encoding="utf-8"))
                say(f"took in {old} (the script lives on the box now)")
            except OSError as e:
                say(f"couldn't read {old}: {e}", "error")


def run_box(editor, box, say) -> None:
    """Run the script kept on `box` (after taking in edits)."""
    sync(editor, box, say)
    script = script_of(box)
    if script is None:
        say("this box has no script yet (edit makes one)", "error")
    else:
        run(editor, box, script, say)


def edit_box(editor, box, say) -> None:
    """Open the box's script in the text editor, through a scratch file that run (or
    edit again) reads back."""
    sync(editor, box, say)
    path = _scratch(editor, box)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script_of(box) or TEMPLATE.format(label=box.label or "this box"), encoding="utf-8")
        _editing[box.uid] = (path, path.stat().st_mtime)
        _open_in_editor(path)
        say(f"editing in {path}: save it, then run")
    except OSError as e:
        say(f"couldn't open {path}: {e}", "error")


def _box_labeled(editor, label: str):
    found = [b for b in editor.box_views.values() if b.label == label]
    return found[0] if len(found) == 1 else None


@after_import("pijl.ui.console")
def _(con):
    def bare(ctx, line) -> bool:
        run(ctx.editor, ctx.box, line, ctx.say)
        return True

    con.fallback(bare)

    def box_for(ctx, rest: str):
        label = rest.strip().strip('"')
        if not label:
            if ctx.box is None:
                ctx.error("which box? (a label, or select one / point at one)")
            return ctx.box
        box = _box_labeled(ctx.editor, label)
        if box is None:
            n = sum(b.label == label for b in ctx.editor.box_views.values())
            ctx.error(f"{n} boxes are labeled {label!r}" if n else f"no box labeled {label!r}")
        return box

    @con.command("run", "run [BOX]: run the script kept on a box (the console's, or the one labeled BOX)")
    def _run(ctx, rest):
        if (box := box_for(ctx, rest)) is not None:
            run_box(ctx.editor, box, ctx.say)

    @con.command("edit", "edit [BOX]: edit a box's script in your text editor (run takes it in)")
    def _edit(ctx, rest):
        if (box := box_for(ctx, rest)) is not None:
            edit_box(ctx.editor, box, ctx.say)


@after_import("pijl.ui.menus")
def _(menus):
    from pijl.ui.menu import MenuItem

    def status(editor):
        def say(text, kind="out"):
            (editor._report if kind == "error" else editor._notice)(f"netlist: {text}")

        return say

    def console_for(editor, box) -> None:
        editor.selection.set(boxes=[box])  # (so it's the console's box)
        editor.console.set_open(True)

    @menus.items("box")
    def rows(editor, box):
        return [
            MenuItem(
                "Netlist",
                submenu=[
                    MenuItem("Type lines (console)", lambda: console_for(editor, box)),
                    MenuItem("Run script", lambda: run_box(editor, box, status(editor))),
                    MenuItem("Edit script", lambda: edit_box(editor, box, status(editor))),
                ],
            )
        ]
