"""Preferences: pijl's app-wide settings, kept in settings.json in the data root.

They're edited in the launcher (see launcher.py) or with `pijl prefs`, and take
effect when something is started from there, before pyglet or the engine is set
up. So nothing here needs to change while it runs: every setting is read once.

Who wins, for the engine options: `--engine` on the command line, then
PIJL_ENGINE, then these, then the built-in defaults (see sim/config.py).

The settings use the same schema as part scripts' settings (parts/settings.py):
values are checked and clamped on load and on every write, and a bad value in
the file falls back to its default with a warning. No pyglet in here.
"""

from __future__ import annotations

import json
import os

from .parts.settings import Choice, Number, Setting, Toggle
from .project import data_root, write_atomic
from .sim.config import OPTIONS, EngineConfig

# Under this key in settings.json (the file also remembers the last project).
KEY = "prefs"

_ENGINE_LABELS = {"dirty": "Dirty-set stepping", "eval": "Evaluator"}
_ENGINE_HINTS = {
    "dirty": "adaptive: only run what changed, when that's cheaper. off: run everything",
    "eval": "lut: small gates become lookup tables. batches: each kind runs as arrays",
}

# Shown in this order, under these headings.
SECTIONS: list[tuple[str, dict[str, Setting]]] = [
    (
        "Editor",
        {
            "ui.scale": Number(
                1.2,
                0.8,
                2.0,
                step=0.05,
                label="UI scale",
                hint="how big the picker, menus and status bar are",
            ),
            "editor.noise": Number(
                64,
                0,
                1024,
                unit="ticks",
                label="Power-on noise",
                hint="random settling ticks for new parts, so latches pick a side",
            ),
        },
    ),
    (
        "Engine",
        {
            f"engine.{name}": Choice(
                choices,
                choices[0],
                label=_ENGINE_LABELS.get(name, ""),
                hint=_ENGINE_HINTS.get(name, ""),
            )
            for name, choices in OPTIONS.items()
        },
    ),
]
PREFS: dict[str, Setting] = {k: s for _, group in SECTIONS for k, s in group.items()}

for _s in PREFS.values():
    _s.check()


def _file():
    return data_root() / "settings.json"


def _read() -> dict:
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def defaults() -> dict:
    return {k: s.initial for k, s in PREFS.items()}


def load() -> tuple[dict, list[str]]:
    """Every preference's value (defaults for what's missing), and what was wrong
    with the file, if anything."""
    values, problems = defaults(), []
    try:
        raw = json.loads(_file().read_text(encoding="utf-8")).get(KEY, {})
    except FileNotFoundError:
        return values, problems
    except (OSError, ValueError, AttributeError) as e:
        return values, [f"settings.json: can't read it ({e}); using the defaults"]
    if not isinstance(raw, dict):
        return values, [f"settings.json: {KEY!r} isn't an object; using the defaults"]
    for k, v in raw.items():
        setting = PREFS.get(k)
        if setting is None:
            problems.append(f"settings.json: unknown setting {k!r} (ignored)")
            continue
        try:
            values[k] = setting.parse(v)
        except ValueError as e:
            problems.append(f"settings.json: {k}: {e}; using {setting.show(values[k])}")
    return values, problems


def save(values: dict) -> None:
    """Write the preferences that differ from their defaults (the rest stay out of
    the file, so a later pijl's better defaults reach them)."""
    data = _read()
    data[KEY] = {
        k: values[k] for k, s in PREFS.items() if k in values and values[k] != s.initial
    }
    _file().parent.mkdir(parents=True, exist_ok=True)
    write_atomic(_file(), json.dumps(data, indent=2) + "\n")


def parse_text(key: str, text: str):
    """A value typed on the command line or into the launcher -> the value to store
    (ValueError if it isn't one)."""
    setting = PREFS.get(key)
    if setting is None:
        raise ValueError(f"no setting {key!r} (settings: {', '.join(PREFS)})")
    text = text.strip()
    if isinstance(setting, Number):
        return setting.parse_text(text)
    if isinstance(setting, Toggle):
        word = text.lower()
        if word in ("on", "true", "yes", "1"):
            return True
        if word in ("off", "false", "no", "0"):
            return False
        raise ValueError(f"{text!r}: say on or off")
    if isinstance(setting, Choice):
        for v in setting.values:
            if text.casefold() in (str(v).casefold(), setting.show(v).casefold()):
                return v
        raise ValueError(
            f"{text!r}: pick from {', '.join(str(v) for v in setting.values)}"
        )
    return setting.parse(text)


def engine_config(values: dict) -> EngineConfig:
    """The engine options these preferences pick, with PIJL_ENGINE on top."""
    base = EngineConfig(
        **{
            name: values[f"engine.{name}"]
            for name in OPTIONS
            if f"engine.{name}" in values
        }
    )
    return EngineConfig.parse(os.environ.get("PIJL_ENGINE", ""), base)
