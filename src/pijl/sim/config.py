"""Engine settings: which code paths a Circuit runs.

A Circuit picks its code paths once, when it's built (Circuit.__init__ binds the
stepper the config names), so the hot loop never asks which mode it's in. Every
option's first choice is the default: the fastest one proven so far. The others
stay, as references to test against and as rows to benchmark against.

The process-wide default is decided at startup: PIJL_ENGINE in the environment, or
`pijl --engine ...` (which calls set_default before any circuit is made). Text form:
"dirty=off" -- comma separated name=choice pairs; anything left out keeps its default.

Options:
  dirty   adaptive: run only the pure parts whose inputs changed, and carry only the
                    nets whose drivers did, when that's cheaper than everything
                    (sim/dirty.py)
          off:      every part runs and every net is carried, every tick (sim/plain.py)
  eval    lut:      pure kinds with up to 4 inputs become lookup tables, and every
                    part of those kinds is looked up at once (sim/lut.py); the rest
                    run as batches
          batches:  each kind's eval runs on arrays of its instances (sim/batches.py)
  compile off:      macros are flattened into their gates, a tick each
          mixed:    every macro on the board runs as a program: logic without loops
                    in one tick, a tick per trip round a loop (sim/compile.py)
          zero:     the same, but loops run until they stop changing within the
                    tick (capped; what doesn't settle goes X)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

OPTIONS: dict[str, tuple[str, ...]] = {
    "dirty": ("adaptive", "off"),
    "eval": ("lut", "batches"),
    "compile": ("off", "mixed", "zero"),
}


@dataclass(frozen=True)
class EngineConfig:
    dirty: str = OPTIONS["dirty"][0]
    eval: str = OPTIONS["eval"][0]
    compile: str = OPTIONS["compile"][0]

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value not in OPTIONS[f.name]:
                raise ValueError(f"engine option {f.name}={value!r}: pick from {', '.join(OPTIONS[f.name])}")

    @classmethod
    def parse(cls, text: str, base: EngineConfig | None = None) -> EngineConfig:
        """'dirty=off,...' on top of `base` (default: the defaults). Blank: just base."""
        changes: dict[str, str] = {}
        for item in (t.strip() for t in text.split(",")):
            if not item:
                continue
            name, eq, value = item.partition("=")
            name, value = name.strip(), value.strip()
            if not eq or name not in OPTIONS:
                known = ", ".join(f"{n}={'|'.join(c)}" for n, c in OPTIONS.items())
                raise ValueError(f"engine option {item!r}: expected name=choice, one of {known}")
            changes[name] = value
        return replace(base or cls(), **changes)

    def __str__(self) -> str:
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


def _from_env() -> EngineConfig:
    try:
        return EngineConfig.parse(os.environ.get("PIJL_ENGINE", ""))
    except ValueError as e:
        raise SystemExit(f"pijl: PIJL_ENGINE: {e}") from None


_default = _from_env()


def default() -> EngineConfig:
    """What a Circuit built without a config gets."""
    return _default


def set_default(config: EngineConfig) -> None:
    """At startup, before any circuit is made: what the ones built without a config get."""
    global _default
    _default = config
