"""Manuscript: a bundled mod that writes everything pijl logs into files.

pijl logs to the "pijl.*" loggers but attaches no handlers (see pijl/__init__.py).
Manuscript attaches one: a file per run in logs/, next to this file, plus
uncaught exceptions with their tracebacks. Its settings are in config.json, also
next to this file, which it writes with the defaults the first time it runs.

    levels      logger name -> debug / info / warning / error / off. Each one also
                covers the loggers under it: "pijl" is all of pijl, "pijl.edit"
                only the edits. Others mods log under "pijl_mods.<name>".
                (pijl's: app, mods, files, edit, ui, sim)
    keep        how many log files to keep; the oldest go first
    crashes     log uncaught exceptions (true / false)
    stderr      also copy whatever's printed to stderr into the log (true / false)

A key missing from config.json takes its default; config.json is never rewritten.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
CONFIG = HERE / "config.json"
LOGS = HERE / "logs"

DEFAULTS = {
    "levels": {
        "pijl": "info",
        "pijl.edit": "info",
        "pijl.sim": "info",
        "pijl_mods": "info",
        "py.warnings": "warning",
    },
    "keep": 30,
    "crashes": True,
    "stderr": False,
}
OFF = logging.CRITICAL + 10
LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "off": OFF,
}
FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger(__name__)  # (pijl_mods.manuscript)


def read_config() -> tuple[dict, list[str]]:
    """The settings (defaults filled in), and what was wrong with config.json."""
    config = json.loads(json.dumps(DEFAULTS))  # (a deep copy)
    problems = []
    if not CONFIG.exists():
        try:
            CONFIG.write_text(json.dumps(DEFAULTS, indent=4) + "\n", encoding="utf-8")
        except OSError as e:
            problems.append(f"can't write {CONFIG.name}: {e}")
        return config, problems
    try:
        mine = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        return config, [f"{CONFIG.name}: {e}; using the defaults"]
    if not isinstance(mine, dict):
        return config, [f"{CONFIG.name}: not a JSON object; using the defaults"]
    for key, value in mine.items():
        if key not in DEFAULTS:
            problems.append(f"{CONFIG.name}: unknown setting {key!r}")
        elif key == "levels":
            if isinstance(value, dict):
                config["levels"].update(value)
            else:
                problems.append(f"{CONFIG.name}: levels should be an object")
        elif type(value) is not type(DEFAULTS[key]):
            problems.append(f"{CONFIG.name}: {key} should be like {DEFAULTS[key]!r}")
        else:
            config[key] = value
    return config, problems


def prune(keep: int) -> None:
    """Leave room for one more log: the newest keep - 1 stay. (A file another pijl
    still has open can't go on Windows: it's skipped.)"""
    try:
        old = sorted(LOGS.glob("*.log"))
    except OSError:
        return
    for path in old[: max(0, len(old) - max(1, keep) + 1)]:
        try:
            path.unlink()
        except OSError:
            pass


class _Tee:
    """sys.stderr, but every whole line written to it is logged too."""

    def __init__(self, inner, logger: logging.Logger) -> None:
        self._inner, self._log = inner, logger
        self._line = ""
        self._busy = threading.local()

    def write(self, text: str) -> int:
        if self._inner is not None:
            self._inner.write(text)
        if not getattr(self._busy, "on", False):  # (logging's own errors go to stderr)
            self._busy.on = True
            try:
                *lines, self._line = (self._line + text).split("\n")
                for line in lines:
                    if line.strip():
                        self._log.info("%s", line.rstrip("\r"))
            finally:
                self._busy.on = False
        return len(text)

    def flush(self) -> None:
        if self._inner is not None:
            self._inner.flush()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _hook_crashes() -> None:
    old_sys, old_thread = sys.excepthook, threading.excepthook

    def excepthook(kind, value, tb):
        if not issubclass(kind, KeyboardInterrupt):
            log.critical("uncaught exception", exc_info=(kind, value, tb))
        old_sys(kind, value, tb)

    def thread_excepthook(args):
        if args.exc_type is not SystemExit:
            name = args.thread.name if args.thread else "?"
            exc = (args.exc_type, args.exc_value, args.exc_traceback)
            log.critical("uncaught exception in thread %s", name, exc_info=exc)
        old_thread(args)

    sys.excepthook, threading.excepthook = excepthook, thread_excepthook


def start() -> Path | None:
    config, problems = read_config()
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        prune(int(config["keep"]))
        path = LOGS / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{os.getpid()}.log"
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as e:
        print(f"pijl: mod manuscript: can't write a log: {e}", file=sys.stderr)
        return None
    handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    names = set(config["levels"]) | {log.name}
    for name in names:
        logging.getLogger(name).addHandler(handler)
    # (a logger under another one would hand its records up to it as well: twice)
    for name in names:
        if any(name.startswith(other + ".") for other in names):
            logging.getLogger(name).propagate = False
    for name, level in config["levels"].items():
        if str(level).lower() not in LEVELS:
            problems.append(f"{CONFIG.name}: {name}: {level!r} isn't one of {', '.join(LEVELS)}")
            continue
        logging.getLogger(name).setLevel(LEVELS[str(level).lower()])
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)
    logging.captureWarnings(True)
    if config["crashes"]:
        _hook_crashes()
    if config["stderr"]:
        sys.stderr = _Tee(sys.stderr, logging.getLogger(f"{log.name}.stderr"))
    log.info("logging to %s (pid %d)", path, os.getpid())
    for problem in problems:
        log.warning("%s", problem)
        print(f"pijl: mod manuscript: {problem}", file=sys.stderr)
    atexit.register(log.info, "exit")
    return path


PATH = start()
