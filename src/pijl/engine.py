"""Headless pijl: run macros with no window, no pyglet, nothing on screen.

An Engine is a project opened read-only: its part scripts and macros, the same
catalog the editor uses. A Harness is one macro placed on an empty board of its
own, with an IN switch wired to each of its inputs and an OUT to each of its
outputs: whoever holds the harness *is* the outside world. Set inputs, run,
read outputs:

    from pijl.engine import Engine

    eng = Engine()                     # the project the editor had open last
    ha = eng.harness("half adder")     # by title or id
    ha.set(a=1, b=0)                   # or ha.set({"a": 1, "b": "X"})
    ha.settle()                        # step until nothing changes
    ha.read()                          # {"sum": Level.ONE, "carry": Level.ZERO}
    ha.apply({"a": 1, "b": 1})         # set + settle + read in one go

The harness keeps its state between calls, so sequential macros (latches,
counters) work: drive a clock input 0 then 1 and read what it latched.

Pins are named like the macro's pins (the ports' labels, unlabeled ones numbered,
see macros.py), in pin order. Values in are 0 / 1 / X / Z: Levels, bools, ints or
strings. IN switches only ever show 0 / 1 in the editor; here a pin can be driven
X or Z too, which is what a test bench wants.

Nothing here writes to the project: no settings.json, no remembered macro.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .logic import CODE, Level, Logic
from .macros import Catalog, MacroType
from .parts import TEMPLATES, Registry
from .parts import load as load_parts
from .project import Project, last_project, project_names, projects_dir
from .sim import Circuit
from .sim.config import EngineConfig
from .storage import MacroStore, load_file

log_files = logging.getLogger("pijl.files")
log_sim = logging.getLogger("pijl.sim")

SETTLE_TICKS = 64  # power-on noise, like the editor's (see sim/circuit.py): latches pick a side
MAX_TICKS = 10_000  # run-to-stable gives up after this many steps that change something

_LEVELS = {
    "0": Level.ZERO,
    "1": Level.ONE,
    "x": Level.X,
    "z": Level.Z,
    "false": Level.ZERO,
    "true": Level.ONE,
}


# bit strings <-> logic codes (see logic.py): "Z01X"[code]
_CODE_OF = np.full(256, 255, CODE)
for _i, _c in enumerate("Z01X"):
    _CODE_OF[ord(_c)] = _CODE_OF[ord(_c.lower())] = _i
_CHAR_OF = np.frombuffer(b"Z01X", np.uint8)


def _drive(pin, value: Any) -> None:
    """Pin.state = value, with level()'s spellings for a one-lane pin. A bus takes a
    number, a Level or Logic, or text: "X" / "Z" (every lane), or a number ("200",
    "0xC8", "0b11001000")."""
    if pin.width == 1:
        pin.state = level(value)
        return
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("x", "z"):
            value = _LEVELS[text]
        else:
            try:
                value = int(text, 0)
            except ValueError:
                raise ValueError(f"{value!r} isn't a number or X / Z (for a {pin.width}-lane bus)") from None
    elif isinstance(value, bool):
        raise ValueError(f"{value!r} for a {pin.width}-lane bus: give a number")
    pin.state = value


def _lane_names(names: tuple[str, ...], widths: list[int]) -> tuple[str, ...]:
    return tuple(n if w == 1 else f"{n}[{i}]" for n, w in zip(names, widths) for i in range(w))


def _lane_slots(pins: list) -> np.ndarray:
    if not pins:
        return np.empty(0, np.intp)
    return np.concatenate([np.arange(p.slot, p.slot + p.width) for p in pins]).astype(np.intp)


def level(value: Any) -> Level:
    """A Level from anything a caller might write: Level, bool, 0/1, or "0" "1" "X" "Z"."""
    if isinstance(value, Level):
        return value
    if isinstance(value, bool) or value in (0, 1):
        return Level.ONE if value else Level.ZERO
    if isinstance(value, str) and value.strip().lower() in _LEVELS:
        return _LEVELS[value.strip().lower()]
    raise ValueError(f"{value!r} isn't a logic level (0, 1, X or Z)")


class Engine:
    """A project's part scripts and macros, for running headless.

    `project`: a project's name (under the data root, see project.py) or a path to a
    project folder; None for the one the editor had open last. Unlike the editor,
    a project that doesn't exist is an error, not created."""

    def __init__(self, project: str | Path | None = None) -> None:
        if project is None:
            path = projects_dir() / last_project()
        elif isinstance(project, Path) or any(c in str(project) for c in "/\\"):
            path = Path(project)
        else:
            path = projects_dir() / project
        if not (path / "project.json").is_file():
            known = ", ".join(project_names()) or "none"
            raise FileNotFoundError(f"no project at {path} (projects: {known})")
        self.project = Project(path)
        log_files.info("project %s (%s)", self.project.name, path)
        # (no parts folder: a project the editor never opened, which would copy the
        # shipped templates into it; read those from the package instead)
        parts = self.project.parts_dir
        self.parts: Registry = load_parts(parts if parts.is_dir() else TEMPLATES)
        self.problems: list[str] = self.project.mod_problems()
        self.problems += [f"part script {m}" for m in self.parts.errors]
        self.store = MacroStore(self.project.macros_dir)
        self.catalog = Catalog(
            self.parts,
            lambda name: self._load(name),
            self.store.title,
            self.store.uses,
        )

    def _load(self, name: str):
        loaded = self.store.load(name, self.catalog)
        self.problems += [f"macro {name!r}: {w}" for w in loaded.warnings]
        return loaded.snapshot

    def macros(self) -> dict[str, str]:
        """id -> title of every macro in the project, in title order."""
        return self.store.titles()

    def find(self, name: str) -> str:
        """A macro's id from its title or id (ignoring case). KeyError if there's none."""
        id = self.store.find(name) or self.store.find_id(name)
        if id is None:
            raise KeyError(f"no macro {name!r} in project {self.project.name!r}")
        return id

    def macro(self, name: str | Path) -> MacroType:
        """A macro by title or id; or a macro file anywhere on disk (a Path, or a
        string ending in .json), whose own nested macros come from this project.
        KeyError if it can't be built (missing, or contains itself)."""
        if isinstance(name, Path) or str(name).lower().endswith(".json"):
            path = Path(name)
            loaded = load_file(path, self.catalog)
            self.problems += [f"{path.name}: {w}" for w in loaded.warnings]
            return MacroType(path.stem, loaded.snapshot, self.catalog, loaded.title)
        return self.catalog.get("macro:" + self.find(name))

    def harness(
        self,
        name: str | Path | MacroType,
        settle_ticks: int = SETTLE_TICKS,
        seed: int = 0,
        config: EngineConfig | None = None,
    ) -> Harness:
        """The macro on a board of its own, its pins brought out (see Harness).
        `config`: the engine's code paths (default: as decided at startup)."""
        t = name if isinstance(name, MacroType) else self.macro(name)
        return Harness(t, self.catalog, settle_ticks, seed, config)


class Harness:
    """One macro with its pins brought out: an IN driving each input, an OUT on each
    output. Stays live between calls (the circuit keeps its state)."""

    def __init__(
        self,
        macro: MacroType,
        catalog: Catalog,
        settle_ticks: int = SETTLE_TICKS,
        seed: int = 0,
        config: EngineConfig | None = None,
    ) -> None:
        self.macro = macro
        self.circuit = c = Circuit(catalog, settle_ticks=settle_ticks, seed=seed, config=config)
        self.inputs: tuple[str, ...] = macro.ins
        self.outputs: tuple[str, ...] = macro.outs
        part = c.add_parts([macro], [None])[0]
        # IN / OUT as wide as the pins they're on (buses: props["width"])
        widths = [p.width for p in part.pins]
        n_in = len(self.inputs)
        props = [{"width": w} if w > 1 else None for w in widths]
        self._ins = c.add_parts([catalog.get("IN")] * n_in, [None] * n_in, props=props[:n_in])
        self._outs = c.add_parts([catalog.get("OUT")] * len(self.outputs), [None] * len(self.outputs), props=props[n_in:])
        self.wide = any(w > 1 for w in widths)  # has a bus
        # One bit per *lane* (set_bits, bits, truth tables, the binary pipe): a bus is
        # its lanes, as if each were a pin of its own -- "d[0]" ... "d[7]", lane 0
        # first. Without buses these are just the pin names.
        self.in_lanes = _lane_names(self.inputs, widths[:n_in])
        self.out_lanes = _lane_names(self.outputs, widths[n_in:])
        for i, sw in enumerate(self._ins):
            c.connect(sw.outputs[0], part.inputs[i])
        for i, led in enumerate(self._outs):
            c.connect(part.outputs[i], led.inputs[0])
        self.part = part
        log_sim.debug(
            "harness for %s: %d in, %d out, %s", macro.title, len(self.inputs), len(self.outputs), c.config
        )
        self.last_ticks: int | None = 0  # what the last settle() took (None: gave up)
        # lane slots, for set_bits / bits: no handles or Levels per pin on the fast path
        self._in_slots = _lane_slots([sw.outputs[0] for sw in self._ins])
        self._out_slots = _lane_slots([led.inputs[0] for led in self._outs])

    # ---- pins by name -------------------------------------------------------------

    def _index(self, names: tuple[str, ...], key: str | int, side: str) -> int:
        """A pin's index from its name, or its position (an int, or "#3": 1-based)."""
        if isinstance(key, int):
            if 0 <= key < len(names):
                return key
        elif key in names:
            return names.index(key)
        elif key.startswith("#") and key[1:].isdigit() and 1 <= int(key[1:]) <= len(names):
            return int(key[1:]) - 1
        raise KeyError(f"{self.macro.title} has no {side} {key!r} (it has: {', '.join(names) or 'none'})")

    def set(self, values: Mapping[str | int, Any] | None = None, /, **named: Any) -> None:
        """Drive inputs. Not run yet: settle() or step() for that. A bus takes a number
        (its bits), a Level (every lane) or a Logic array of its lanes."""
        for key, value in {**(values or {}), **named}.items():
            i = self._index(self.inputs, key, "input")
            _drive(self._ins[i].outputs[0], value)

    def set_all(self, values: Iterable[Any]) -> None:
        """Drive every input, in pin order (e.g. a string of bits, "1010")."""
        values = list(values)
        if len(values) != len(self.inputs):
            raise ValueError(f"{self.macro.title} has {len(self.inputs)} inputs, got {len(values)} values")
        for sw, value in zip(self._ins, values):
            _drive(sw.outputs[0], value)

    def set_bits(self, bits: str | bytes, start: int = 0) -> None:
        """Drive inputs start, start + 1, ... from a string of 0 / 1 / X / Z, one per
        pin (per lane of a bus: see in_lanes): the fast way (no Levels made). Inputs
        past the string keep their values."""
        raw = np.frombuffer(bits.encode() if isinstance(bits, str) else bytes(bits), np.uint8)
        codes = _CODE_OF[raw]
        if start < 0 or start + len(codes) > len(self._in_slots):
            raise ValueError(
                f"{self.macro.title} has {len(self._in_slots)} {'input lanes' if self.wide else 'inputs'}; "
                f"can't set {len(codes)} from #{start + 1}"
            )
        if (codes == 255).any():
            raise ValueError(f"{bytes(raw).decode(errors='replace')!r}: levels are 0, 1, X and Z")
        self.circuit.write_pins(self._in_slots[start : start + len(codes)], codes)

    def bits(self) -> str:
        """Every output as one string of 0 / 1 / X / Z, in pin order, a bus lane by lane
        (out_lanes): the fast read."""
        return _CHAR_OF[self.circuit._pins.states[self._out_slots]].tobytes().decode()

    def get(self, name: str | int) -> Level | Logic:
        """One output's level (as of the last step); a bus's is a Logic array of its
        lanes (pijl.logic.ints makes it a number)."""
        return self._outs[self._index(self.outputs, name, "output")].inputs[0].state

    def read(self) -> dict[str, Level | Logic]:
        """Every output, by name. (Pins sharing a name: the first one wins; use get("#n").)"""
        out: dict[str, Level] = {}
        for name, led in zip(self.outputs, self._outs):
            out.setdefault(name, led.inputs[0].state)
        return out

    def driven(self) -> dict[str, Level | Logic]:
        """What the inputs are being driven with right now."""
        return {name: sw.outputs[0].state for name, sw in zip(self.inputs, self._ins)}

    def lanes_driven(self) -> dict[str, Level]:
        """driven(), a bus lane by lane (in_lanes)."""
        codes = self.circuit._pins.states[self._in_slots].tolist()
        return {name: Level(c) for name, c in zip(self.in_lanes, codes)}

    def lanes_read(self) -> dict[str, Level]:
        """read(), a bus lane by lane (out_lanes)."""
        out: dict[str, Level] = {}
        for name, c in zip(self.out_lanes, self.circuit._pins.states[self._out_slots].tolist()):
            out.setdefault(name, Level(c))
        return out

    # ---- time ---------------------------------------------------------------------

    @property
    def tick(self) -> int:
        return self.circuit.tick

    def step(self, n: int = 1) -> None:
        """Exactly n ticks (one gate delay each)."""
        for _ in range(n):
            self.circuit.step()

    def settle(self, limit: int = MAX_TICKS) -> bool:
        """Step until nothing changes (see Circuit.run_until_stable). False if it
        didn't within `limit` steps: it oscillates, or it's very deep."""
        self.last_ticks = self.circuit.run_until_stable(limit)
        return self.last_ticks is not None

    def frame(self) -> None:
        """Run the parts' frame hooks once, as the editor does every frame."""
        self.circuit.frame()

    def apply(
        self, values: Mapping[str | int, Any] | None = None, ticks: int | None = None
    ) -> dict[str, Level]:
        """Set inputs, run (`ticks` exactly, or to stable), read the outputs."""
        self.set(values or {})
        if ticks is None:
            self.settle()
        else:
            self.step(ticks)
        return self.read()

    def truth_table(self, ticks: int | None = None) -> Iterator[tuple[dict[str, Level], dict[str, Level]]]:
        """Every 0/1 combination of the inputs (first input = most significant bit),
        in counting order, with the outputs each settles to. The state carries over
        from row to row, as it would on a real bench. A bus counts as its lanes
        (in_lanes, out_lanes)."""
        n = len(self.in_lanes)
        if n > 20:
            raise ValueError(f"{n} inputs is {2**n} rows: too many for a truth table")
        for row in range(2**n):
            self.set_bits(format(row, f"0{n}b") if n else "")
            if ticks is None:
                self.settle()
            else:
                self.step(ticks)
            yield self.lanes_driven(), self.lanes_read()

    @property
    def problems(self) -> list[str]:
        """Part scripts that raised while running (their kind got disabled)."""
        return [f"{kind}: {why}" for kind, why in self.circuit.faults.items()]
