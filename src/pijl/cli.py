"""What `pijl` (and main.py, and `python -m pijl`) does with its arguments.

    pijl                          the launcher (see launcher.py; --tui: in the terminal)
    pijl gui                      the editor, skipping the launcher
    pijl prefs [KEY=VALUE...]     show or change the preferences (see prefs.py)
    pijl list [--projects]        the project's macros and their pins
    pijl bench bogobips           the shift register benchmark (see bogobips.py)
    pijl run MACRO [VALUES...]    run a macro headless (see below)

Global: --project NAME|PATH (default: the one the editor had open last),
--data DIR (the data root, like PIJL_DATA) and --engine OPTIONS (the engine's code
paths, like PIJL_ENGINE: "dirty=off"; see pijl/sim/config.py). The engine options
come from --engine, else PIJL_ENGINE, else the preferences. Headless commands
never import pyglet.

`pijl run` drives the macro's inputs from outside and prints its outputs:

    pijl run "half adder" a=1 b=0      one shot -> sum=1 carry=0
    pijl run "half adder" 10           all inputs in pin order, as bits
    pijl run "half adder" --table      every 0/1 combination
    pijl run counter -                 a stream: one line in, one line out
    pijl run counter --raw             a raw stream, for speed (see _raw)
    pijl run counter --bin             the binary pipe, for programs (see pijl.pipe)

A stream reads stdin line by line, keeping the circuit's state from line to line,
and flushes every answer, so another program can hold the macro as a co-process.
A line is "a=1 b=X" (only those inputs change), a string of bits for all of them,
a JSON object ({"a": 1, "b": "Z"}, answered in JSON), "step N" to just let N ticks
pass, or "#..." / blank (ignored). After each line the circuit runs until it's
stable (or --ticks N, exactly N ticks) and the outputs are printed.

--bin speaks framed binary on stdin / stdout instead: batches of packed vectors
in, packed samples out (see pijl.pipe, and HEADLESS.md for all of headless pijl).

Levels are 0 / 1 / X / Z. In JSON, 0 and 1 are numbers and X and Z strings.
Exit status: 0 fine, 1 couldn't run, 2 bad arguments or input, 3 something
didn't settle (within --max-ticks; the outputs are printed anyway).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO

if TYPE_CHECKING:
    from .engine import Harness
    from .logic import Level

COMMANDS = ("launch", "gui", "prefs", "run", "list", "bench")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(argv if _has_command(argv) else ["launch", *argv])
    if args.data:
        os.environ["PIJL_DATA"] = args.data
    from . import prefs
    from .sim.config import EngineConfig, set_default

    if args.engine:
        try:
            EngineConfig.parse(args.engine)
        except ValueError as e:
            print(f"pijl: --engine: {e}", file=sys.stderr)
            return 2
        # (in the environment, so whatever's started from here gets it too)
        os.environ["PIJL_ENGINE"] = ",".join(
            t for t in (os.environ.get("PIJL_ENGINE", ""), args.engine) if t
        )
    if args.command == "prefs":
        return _prefs(args)
    if args.command == "launch":
        start = _launch(args)
        return 0 if start is None else main(start)
    values, problems = prefs.load()
    for problem in problems:
        print(f"pijl: warning: {problem}", file=sys.stderr)
    try:
        set_default(prefs.engine_config(values))
    except ValueError as e:
        print(f"pijl: PIJL_ENGINE: {e}", file=sys.stderr)
        return 2
    if args.command == "gui":
        from .ui import theme

        theme.UI_SCALE = values["ui.scale"]  # (before the editor's modules read it)
        from .ui.editor import run  # (pyglet: only now)

        if run(args.project, settle_ticks=values["editor.noise"]):
            from .launcher import relaunch

            return relaunch()
        return 0
    if args.command == "bench":
        from .bogobips import main as bogobips

        return bogobips(args)
    from .engine import Engine

    try:
        engine = Engine(args.project)
    except (OSError, ValueError) as e:
        return _fail(f"can't open the project: {e}")
    for problem in engine.problems:
        print(f"pijl: warning: {problem}", file=sys.stderr)
    if args.command == "list":
        return _list(engine, args)
    return _run(engine, args)


def _launch(args) -> list[str] | None:
    """Show the launcher (a window, else the terminal one); what to start, or None."""
    from .launcher import STREAM_HELP, Launcher, has_console, terminal

    try:
        launcher = Launcher()
        if args.project:
            launcher.pick(args.project)
    except (OSError, ValueError) as e:
        _fail(str(e))
        return None
    start = None
    if args.tui:
        start = terminal(launcher)
    else:
        from .ui.launcher import NoWindow, run_window

        try:
            start = run_window(launcher)
        except NoWindow as e:
            if not has_console():
                raise
            print(f"pijl: no window ({e}); here's the terminal launcher", file=sys.stderr)
            start = terminal(launcher)
        else:
            if start and start[0] == "run" and start[2] == "-":
                print(STREAM_HELP, file=sys.stderr)
    return start


def _prefs(args) -> int:
    from . import prefs

    values, problems = prefs.load()
    for problem in problems:
        print(f"pijl: warning: {problem}", file=sys.stderr)
    if args.reset:
        values = prefs.defaults()
    try:
        for item in args.set:
            key, eq, text = item.partition("=")
            if not eq:
                return _fail(f"{item!r}: expected KEY=VALUE", 2)
            values[key.strip()] = prefs.parse_text(key.strip(), text)
    except ValueError as e:
        return _fail(str(e), 2)
    if args.reset or args.set:
        try:
            prefs.save(values)
        except OSError as e:
            return _fail(f"can't save settings.json: {e}")
    width = max(len(k) for k in prefs.PREFS)
    for heading, group in prefs.SECTIONS:
        print(f"# {heading}")
        for key, setting in group.items():
            mark = "" if values[key] == setting.initial else "  (changed)"
            print(f"{key:<{width}}  {setting.show(values[key])}{mark}")
    return 0


def _has_command(argv: list[str]) -> bool:
    """Is there a command among the arguments? (No command means the launcher.)"""
    for a in argv:
        if a in ("-h", "--help") or a in COMMANDS:
            return True
    return False


def _parser() -> argparse.ArgumentParser:
    def common(default: Any) -> argparse.ArgumentParser:
        # (before or after the command; after it, SUPPRESS keeps the subcommand from
        # resetting what was given before it)
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument("-p", "--project", default=default, help="project name or folder (default: the last one open)")
        c.add_argument("--data", default=default, help="data root folder (default: PIJL_DATA, else the per-user one)")
        c.add_argument("--engine", default=default, help='engine options, e.g. "dirty=off" (default: PIJL_ENGINE, else the fastest)')
        return c

    p = argparse.ArgumentParser(
        prog="pijl",
        description="A visual logic circuit editor and simulator. No command: the launcher.",
        parents=[common(None)],
    )
    common = common(argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch", parents=[common], help="the launcher (the default)")
    launch.add_argument("--tui", action="store_true", help="in the terminal, not a window")
    sub.add_parser("gui", parents=[common], help="the editor, skipping the launcher")
    pr = sub.add_parser("prefs", parents=[common], help="show or change the preferences")
    pr.add_argument("set", nargs="*", metavar="KEY=VALUE", help='e.g. ui.scale=1.5 "engine.dirty=off"')
    pr.add_argument("--reset", action="store_true", help="everything back to its default (before any KEY=VALUE)")
    ls = sub.add_parser("list", parents=[common], help="the project's macros and their pins")
    ls.add_argument("--projects", action="store_true", help="list the projects instead")
    run = sub.add_parser(
        "run",
        parents=[common],
        help="run a macro headless",
        description="Run a macro with its inputs driven from outside. See `pydoc pijl.cli`.",
    )
    run.add_argument("macro", help="title or id, or a path to a macro .json file")
    run.add_argument(
        "values",
        nargs="*",
        help="inputs: name=level pairs, or one string of bits for all of them; - for a stream on stdin",
    )
    run.add_argument("--table", action="store_true", help="print the truth table (0/1 inputs)")
    run.add_argument("--json", action="store_true", help="answer in JSON lines")
    run.add_argument("--ticks", type=int, help="run exactly this many ticks per input (default: until stable)")
    run.add_argument("--max-ticks", type=int, default=10_000, help="give up settling after this many (default 10000)")
    run.add_argument("--noise", type=int, default=64, help="power-on settling ticks, 0 for none (default 64, as the editor)")
    run.add_argument("--seed", type=int, default=0, help="seed for the power-on noise")
    run.add_argument("--raw", action="store_true", help="a raw stream on stdin: vectors of level characters, ? to read (see pydoc pijl.cli)")
    run.add_argument("--bin", action="store_true", help="the binary pipe on stdin / stdout: framed, packed, batched (see pydoc pijl.pipe)")
    bench = sub.add_parser(
        "bench",
        parents=[common],
        help="internal benchmarks",
        description="bogobips: BIt-shifts Per Second through generated shift registers. See `pydoc pijl.bogobips`.",
    )
    bench.add_argument("name", choices=["bogobips"])
    bench.add_argument("--kind", default="sipo,piso,counter,lfsr,tree,decoder,adder", help="any of sipo, piso (shift: both), counter, lfsr (loop: both), tree-and, tree-xor (tree: both), decoder, adder (default: all)")
    bench.add_argument("--depth", help="depths, comma separated (default: each kind's own; see pydoc pijl.bogobips)")
    bench.add_argument("--layers", default="engine,settle,pipe,bin", help="which of engine, settle, pipe, bin")
    bench.add_argument("--seconds", type=float, default=0.5, help="time spent per measurement (default 0.5)")
    bench.add_argument("--seed", type=int, default=0, help="seed for the random stimulus")
    bench.add_argument("--flips", type=int, default=1, help="inputs flipped per vector, for trees, decoders and adders (default 1)")
    bench.add_argument("--nest", action="store_true", help="build trees and adders from nested macros")
    bench.add_argument("--engines", default="", help='engine configs to compare, ";" between them, e.g. "dirty=off;dirty=adaptive" (default: the --engine one)')
    return p


# ---- list --------------------------------------------------------------------------


def _list(engine, args) -> int:
    from .project import project_names

    if args.projects:
        for name in project_names():
            print(("* " if name == engine.project.name else "  ") + name)
        return 0
    for id, title in engine.macros().items():
        try:
            t = engine.catalog.get("macro:" + id)
        except KeyError as e:
            print(f"{title}  (can't load: {e.args[0] if e.args else e})")
            continue
        name = title if title == id else f"{title} [{id}]"
        print(f"{name}  ({', '.join(t.ins) or '-'}) -> ({', '.join(t.outs) or '-'})")
    return 0


# ---- run ---------------------------------------------------------------------------


class _BadInput(ValueError):
    pass


def _run(engine, args) -> int:
    try:
        h = engine.harness(args.macro, settle_ticks=args.noise, seed=args.seed)
    except (KeyError, OSError, ValueError) as e:
        return _fail(str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e))
    for problem in engine.problems:
        print(f"pijl: warning: {problem}", file=sys.stderr)
    engine.problems.clear()
    unsettled = False
    reported: set[str] = set()

    def run() -> None:
        nonlocal unsettled
        if args.ticks is None:
            if not h.settle(args.max_ticks):
                unsettled = True
                print(f"pijl: warning: not stable after {args.max_ticks} ticks", file=sys.stderr)
        else:
            h.step(args.ticks)
        for problem in h.problems:
            if problem not in reported:
                reported.add(problem)
                print(f"pijl: warning: part script {problem} (that kind is off now)", file=sys.stderr)

    out = sys.stdout
    try:
        if args.bin:
            if args.values not in ([], ["-"]) or args.table or args.raw or args.json or args.ticks is not None:
                raise _BadInput("--bin takes no input values, --table, --raw, --json or --ticks (frames say all that)")
            from .pipe import serve

            first = [h]

            def make(noise: int, seed: int) -> Harness:
                return first.pop() if first else engine.harness(args.macro, settle_ticks=noise, seed=seed)

            return serve(make, args.noise, args.seed, sys.stdin.buffer, sys.stdout.buffer, args.max_ticks)
        if args.table:
            if args.values:
                raise _BadInput("--table takes no input values")
            _table(h, run, args.json, out)
        elif args.raw:
            if args.values not in ([], ["-"]):
                raise _BadInput("--raw reads stdin; it takes no input values")
            return _raw(h, run, sys.stdin.buffer, sys.stdout.buffer) or (3 if unsettled else 0)
        elif args.values == ["-"]:
            return _stream(h, run, args.json, sys.stdin, out) or (3 if unsettled else 0)
        else:
            _apply(h, " ".join(args.values))
            run()
            _answer(h.read(), args.json, out)
    except _BadInput as e:
        return _fail(str(e), 2)
    return 3 if unsettled else 0


def _stream(h: Harness, run, as_json: bool, inp: TextIO, out: TextIO) -> int:
    """One answer per line in. Bad lines get an error on stderr (and an
    {"error": ...} answer to a JSON line) and change nothing; returns 2 if any."""
    status = 0
    for n, line in enumerate(inp, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        is_json = line.startswith("{")
        try:
            words = line.split()
            if not is_json and words[0] in ("step", "tick", "ticks") and len(words) <= 2:
                count = words[1] if len(words) == 2 else "1"
                if not count.isdigit():
                    raise _BadInput(f"step {count!r}: not a number of ticks")
                h.step(int(count))
            else:
                _apply(h, line)
                run()
        except _BadInput as e:
            status = 2
            print(f"pijl: line {n}: {e}", file=sys.stderr)
            if is_json:
                print(json.dumps({"error": str(e)}), file=out, flush=True)
            continue
        _answer(h.read(), as_json or is_json, out)
    return status


# a level run, a read, a run-now, or an addressed level (@7=1: input 7, counting from 1)
_RAW_TOKEN = re.compile(rb"([01xXzZ]+)|(\?)|(;)|@([0-9]+)=([01xXzZ])")
_RAW_PARTIAL = re.compile(rb"@[0-9]*=?")  # an addressed level cut off by the chunk's end


def _raw(h: Harness, run, inp: BinaryIO, out: BinaryIO) -> int:
    """The --raw stream: built for speed, so no lines and no names. Each level
    character (0 1 X Z) drives the next input in pin order; once every input has
    one, that's a vector: it's applied and run. ";" applies a vector early (only
    the inputs given so far change; on its own: run, change nothing). "@n=L" drives
    input n (counting from 1) right away, without running: "@7=1@9=0;" changes two
    inputs, then runs. "?" answers with every output's level, one character each,
    no newline. Whitespace is ignored. Answers are flushed whenever the input there
    is has been worked off. Read a chunk at a time: a long vector costs a few slice
    copies, not a Python step per character."""
    n = len(h.inputs)
    vec = bytearray()
    answer = bytearray()
    pending = b""

    def apply() -> None:
        h.set_bits(bytes(vec))
        vec.clear()
        run()

    def stop(msg: str) -> int:
        out.write(answer)
        out.flush()
        return _fail(f"--raw: {msg}", 2)

    while chunk := (inp.read1(65536) if hasattr(inp, "read1") else inp.read(65536)):
        data = pending + chunk.translate(None, b" \t\r\n")
        pending, pos = b"", 0
        while pos < len(data):
            m = _RAW_TOKEN.match(data, pos)
            if m is None:
                if _RAW_PARTIAL.fullmatch(data, pos):
                    pending = data[pos:]
                    break
                return stop(f"unexpected {chr(data[pos])!r} in the stream")
            pos = m.end()
            levels, read, now, at, level = m.groups()
            try:
                if levels is not None:
                    if not n:
                        return stop(f"{h.macro.title} has no inputs")
                    while levels:
                        take = n - len(vec)
                        vec += levels[:take]
                        levels = levels[take:]
                        if len(vec) == n:
                            apply()
                elif read is not None:
                    answer += h.bits().encode()
                elif now is not None:
                    apply()
                else:
                    h.set_bits(level, int(at) - 1)
            except ValueError as e:
                return stop(str(e))
        if answer:
            out.write(answer)
            out.flush()
            answer.clear()
    if pending:
        return stop(f"the stream ends in the middle of {pending.decode()!r}")
    if vec:
        apply()
    return 0


def _apply(h: Harness, text: str) -> None:
    """Drive inputs from one line of input (see the module docstring)."""
    text = text.strip()
    try:
        if text.startswith("{"):
            try:
                values = json.loads(text)
            except json.JSONDecodeError as e:
                raise _BadInput(f"bad JSON ({e.msg})") from None
            if not isinstance(values, dict):
                raise _BadInput("a JSON line must be an object")
            h.set(values)
        elif text and "=" not in text and all(c in "01xXzZ" for c in text):
            h.set_all(text)
        else:
            values: dict[str, Any] = {}
            for word in _words(text):
                name, eq, value = word.rpartition("=")
                if not eq or not name:
                    raise _BadInput(f"{word!r}: expected name=level")
                values[name] = value
            h.set(values)
    except (KeyError, ValueError) as e:
        if isinstance(e, _BadInput):
            raise
        raise _BadInput(e.args[0] if e.args else str(e)) from None


def _words(text: str) -> list[str]:
    """Split on spaces, but a name with spaces can be quoted: "carry in"=1."""
    import shlex

    try:
        return shlex.split(text)
    except ValueError as e:
        raise _BadInput(str(e)) from None


def _table(h: Harness, run, as_json: bool, out: TextIO) -> None:
    n = len(h.inputs)
    if n > 16:
        raise _BadInput(f"{n} inputs is {2**n} rows; that's too many for a table")
    if not as_json:
        print(" ".join(h.inputs) + " | " + " ".join(h.outputs), file=out)
    for row in range(2**n):
        h.set_all([(row >> (n - 1 - i)) & 1 for i in range(n)])
        run()
        ins, outs = h.driven(), h.read()
        if as_json:
            print(json.dumps({"in": _json_levels(ins), "out": _json_levels(outs)}), file=out, flush=True)
        else:
            left = " ".join(str(v).rjust(len(k)) for k, v in ins.items())
            right = " ".join(str(v).rjust(len(k)) for k, v in outs.items())
            print(f"{left} | {right}", file=out, flush=True)


def _answer(values: dict[str, Level], as_json: bool, out: TextIO) -> None:
    if as_json:
        line = json.dumps(_json_levels(values))
    else:
        line = " ".join(f"{_quote(k)}={v}" for k, v in values.items())
    print(line, file=out, flush=True)


def _json_levels(values: dict[str, Level]) -> dict[str, int | str]:
    return {k: int(bool(v)) if str(v) in "01" else str(v) for k, v in values.items()}


def _quote(name: str) -> str:
    return f'"{name}"' if not name or any(c in name for c in " =\"'") else name


def _fail(msg: str, status: int = 1) -> int:
    print(f"pijl: {msg}", file=sys.stderr)
    return status
