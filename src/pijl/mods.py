"""Mods: Python modules in <data root>/mods/ that pijl imports at startup.

A mod is arbitrary code; its "API" is monkey-patching pijl. The one helper is
`after_import`, which runs a function as soon as a pijl module has been imported
(or right away if it already has), so a patch lands before anything copies the
patched name. Patch methods on classes rather than module-level functions: a
`from x import f` elsewhere keeps the old f. (See MODDING.md.)

    <data root>/mods/
        loadorder.txt       the mods that load, in this order (one name per line)
        disabled.txt        the mods that don't; new mods are added here
        foo.py              a single-file mod, `foo`
        bar/__init__.py     a folder mod, `bar` (bar/lib/, if any, goes on sys.path)
        bar/mod.json        its manifest (or MOD = {...} in the entry point)

(PIJL_MODS: another mods folder.)

Official mods ship inside pijl (OFFICIAL) and are copied into the mods folder when
it's planned, like any mod dropped in by hand: so they start out disabled. A name
that's in either list but not on disk was deleted by the user and stays deleted.
When pijl ships a newer version (mod.json's), its files are copied over the old
ones; anything else in the mod's folder (its config, its logs) is left alone.

The two lists: one name per line, any mix of LF / CRLF / CR, blank lines and
"#..." lines ignored, names compared without case. A mod on disk that's in neither
list is new and gets appended to disabled.txt. A name in both is disabled. A name
with nothing on disk is warned about and left in the list (pijl only ever appends).

Mods are imported as `pijl_mods.<name>`, so they can import each other. A mod that
fails to import is reported and skipped; whatever it patched before failing stays.

The manifest is read without running the mod (JSON, or the literal assigned to MOD
at the entry point's top level): name, version, description, author, requires (mod
names) and pijl (a version range like ">=0.2,<0.3"). requires and pijl only warn.

A crash while starting leaves a marker file (see SENTINEL) that the next start
sees; `settled()` removes it once pijl is up. `--safe` / PIJL_SAFE=1: no mods.
"""

from __future__ import annotations

import ast
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from . import __version__

log = logging.getLogger("pijl.mods")

PACKAGE = "pijl_mods"
LOADORDER = "loadorder.txt"
DISABLED = "disabled.txt"
MANIFEST = "mod.json"
SENTINEL = "mods.loading"
OFFICIAL = (
    Path(__file__).parent / "official_mods"
)  # in the data root, while mods load and pijl starts

LOADORDER_HEAD = (
    "# Mods that load, in this order: one folder or script name per line.\n"
    "# Move a name here from disabled.txt to turn it on.\n"
)
DISABLED_HEAD = "# Mods that don't load. New mods are added here.\n"


def mods_dir() -> Path:
    """<data root>/mods, unless PIJL_MODS says otherwise (main() sets it, so a pijl
    started from this one with another --data still gets these mods)."""
    from .project import data_root

    if override := os.environ.get("PIJL_MODS"):
        return Path(override)
    return data_root() / "mods"


def safe_mode() -> bool:
    return os.environ.get("PIJL_SAFE", "") not in ("", "0")


# ---- the mods on disk ----------------------------------------------------------------


@dataclass
class Mod:
    name: str  # as on disk: the module name (foo for foo.py and foo/)
    path: Path  # foo.py, or the folder
    manifest: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)  # why it can't load
    warnings: list[str] = field(default_factory=list)  # what's off, but it can

    @property
    def entry(self) -> Path:
        return self.path / "__init__.py" if self.path.is_dir() else self.path

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", ""))

    @property
    def title(self) -> str:
        """`name version`, as it's shown and recorded."""
        return f"{self.name} {self.version}".rstrip()

    @property
    def usable(self) -> bool:
        return not self.problems


def discover(folder: Path) -> dict[str, Mod]:
    """Every mod in `folder`, by casefolded name, in name order. Folders without an
    __init__.py and names starting with "_" or "." aren't mods."""
    found: dict[str, Mod] = {}
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.casefold())
    except OSError:
        return found
    for p in entries:
        if p.name.startswith(("_", ".")):
            continue
        if p.is_dir():
            if not (p / "__init__.py").is_file():
                continue
            name = p.name
        elif p.suffix == ".py" and p.is_file():
            name = p.stem
        else:
            continue
        mod = Mod(name, p)
        key = name.casefold()
        if key in found:  # foo/ and foo.py: Python imports the folder
            if p.is_dir():
                found[key], mod = mod, found[key]
            found[key].warnings.append(
                f"{mod.path.name} has the same name; it's ignored"
            )
            continue
        if not name.isidentifier():
            mod.problems.append(
                f"{name!r} isn't a valid module name (letters, digits, _)"
            )
        mod.manifest, mod.warnings = read_manifest(mod)
        found[key] = mod
    return found


def read_manifest(mod: Mod) -> tuple[dict, list[str]]:
    """The mod's manifest, without running it; and what's wrong with it. A bad
    manifest is a warning, not a reason not to load."""
    m: object = {}
    try:
        if mod.path.is_dir() and (mod.path / MANIFEST).is_file():
            m = json.loads((mod.path / MANIFEST).read_text(encoding="utf-8-sig"))
        else:
            tree = ast.parse(mod.entry.read_bytes(), str(mod.entry))
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "MOD"
                ):
                    m = ast.literal_eval(node.value)
    except SyntaxError:
        return {}, []  # (importing it will say so)
    except (OSError, ValueError) as e:
        return {}, [f"bad manifest: {e}"]
    if not isinstance(m, dict):
        return {}, ["bad manifest: not a JSON object / dict"]
    return m, []


def read_list(path: Path) -> list[str]:
    """The names in a list file (see the module docstring); [] if there's none."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        return []
    names = []
    for line in text.splitlines():  # (\n, \r\n, \r, mixed)
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def append_list(path: Path, names: list[str], head: str = "") -> None:
    """Add lines at the end of a list file, keeping what's there byte for byte (and
    its line endings)."""
    from .project import write_atomic_bytes

    try:
        old = path.read_bytes()
    except FileNotFoundError:
        old = b""
    nl = b"\r\n" if b"\r\n" in old or (not old and sys.platform == "win32") else b"\n"
    if not old and head:
        old = head.replace("\n", nl.decode()).encode()
    if old and not old.endswith((b"\n", b"\r")):
        old += nl
    write_atomic_bytes(path, old + b"".join(n.encode() + nl for n in names))


def write_list(path: Path, names: list[str], head: str = "") -> None:
    """Make a list file hold `names`, in order: comment and blank lines stay where
    they are, name lines are filled in from `names` in turn (the rest go at the end,
    the leftover lines go). Keeps the file's line endings."""
    from .project import write_atomic_bytes

    try:
        old = path.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        old = head
    nl = "\r\n" if "\r\n" in old or (not old and sys.platform == "win32") else "\n"
    rest = iter(names)
    out = []
    for line in old.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            out.append(line)
        elif (name := next(rest, None)) is not None:
            out.append(name)
    out += list(rest)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_bytes(path, "".join(line + nl for line in out).encode())


def _dedup(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for n in names:
        if n.casefold() not in seen:
            seen.add(n.casefold())
            out.append(n)
    return out


def _without(names: list[str], name: str) -> list[str]:
    return [n for n in names if n.casefold() != name.casefold()]


def enable(name: str, folder: Path | None = None) -> None:
    """Turn a mod on: last in the load order, out of disabled.txt."""
    folder = mods_dir() if folder is None else folder
    found = discover(folder).get(name.casefold())
    name = found.name if found else name  # (spelled as on disk)
    order = _without(_dedup(read_list(folder / LOADORDER)), name) + [name]
    write_list(folder / LOADORDER, order, LOADORDER_HEAD)
    write_list(
        folder / DISABLED, _without(read_list(folder / DISABLED), name), DISABLED_HEAD
    )


def disable(name: str, folder: Path | None = None) -> None:
    """Turn a mod off: out of the load order, into disabled.txt (if it's on disk;
    a missing one is just forgotten)."""
    folder = mods_dir() if folder is None else folder
    write_list(
        folder / LOADORDER,
        _without(read_list(folder / LOADORDER), name),
        LOADORDER_HEAD,
    )
    off = _without(read_list(folder / DISABLED), name)
    if name.casefold() in discover(folder):
        off.append(name)
    write_list(folder / DISABLED, off, DISABLED_HEAD)


def move(name: str, by: int, folder: Path | None = None) -> None:
    """Move a mod `by` places in the load order (negative: earlier)."""
    folder = mods_dir() if folder is None else folder
    order = _dedup(read_list(folder / LOADORDER))
    keys = [n.casefold() for n in order]
    if name.casefold() not in keys:
        raise ValueError(f"{name} isn't in the load order")
    i = keys.index(name.casefold())
    j = max(0, min(len(order) - 1, i + by))
    order.insert(j, order.pop(i))
    write_list(folder / LOADORDER, order, LOADORDER_HEAD)


def crashed() -> bool:
    """Did the last start die while loading mods (or before pijl was up)?"""
    from .project import data_root

    return (data_root() / SENTINEL).exists()


# ---- what loads, in what order ---------------------------------------------------------


@dataclass
class Plan:
    """The two lists reconciled with what's on disk."""

    folder: Path
    mods: dict[str, Mod]  # every mod on disk, by casefolded name
    enabled: list[Mod]  # in load order
    disabled: list[Mod]
    new: list[Mod]  # (also in disabled: they've just been added there)
    missing: list[str]  # names in loadorder.txt with no mod on disk
    both: list[Mod]  # in both lists (so disabled)


def plan(folder: Path | None = None, write: bool = True) -> Plan:
    """Read the lists; with `write`, create the folder and the lists if they're missing
    and append new mods to disabled.txt."""
    folder = mods_dir() if folder is None else folder
    if write:
        install_official(folder)
    mods = discover(folder)
    order, off = read_list(folder / LOADORDER), read_list(folder / DISABLED)
    off_keys = {n.casefold() for n in off}
    enabled, missing, both, seen = [], [], [], set()
    for name in order:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        if key not in mods:
            if key not in off_keys:
                missing.append(name)
        elif key not in off_keys:
            enabled.append(mods[key])
        else:
            both.append(mods[key])
    listed = seen | off_keys
    new = [m for k, m in mods.items() if k not in listed]
    disabled = [m for k, m in mods.items() if k in off_keys] + new
    if write:
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / LOADORDER).exists():
            append_list(folder / LOADORDER, [], LOADORDER_HEAD)
        if new or not (folder / DISABLED).exists():
            append_list(folder / DISABLED, [m.name for m in new], DISABLED_HEAD)
    return Plan(folder, mods, enabled, disabled, new, missing, both)


def install_official(folder: Path) -> list[str]:
    """Copy the official mods into `folder`: the ones not there yet (unless a list
    names them: then they were deleted), and newer versions over older ones (only
    the shipped files). Returns the names copied."""
    try:
        shipped = [
            p for p in sorted(OFFICIAL.iterdir()) if (p / "__init__.py").is_file()
        ]
    except OSError:
        return []
    listed = {
        n.casefold()
        for n in read_list(folder / LOADORDER) + read_list(folder / DISABLED)
    }
    copied = []
    for src in shipped:
        dst = folder / src.name
        if dst.exists():
            have = Mod(dst.name, dst)
            have.manifest, _ = read_manifest(have)
            new = Mod(src.name, src)
            new.manifest, _ = read_manifest(new)
            try:
                if parse_version(new.version) <= parse_version(have.version):
                    continue
            except ValueError:
                continue  # (no version on one side: leave theirs be)
        elif src.name.casefold() in listed:
            continue
        try:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        except OSError as e:
            log.warning("can't install the official mod %s: %s", src.name, e)
            continue
        copied.append(src.name)
    return copied


def notes(p: Plan) -> dict[str, list[str]]:
    """What's wrong with each mod (by casefolded name), as load() would report it:
    why it can't load, manifest warnings, and for enabled mods version / requires."""
    order = {m.name.casefold(): i for i, m in enumerate(p.enabled)}
    out = {}
    for key, mod in p.mods.items():
        msgs = mod.problems + mod.warnings
        if key in order:
            msgs += [m.split(": ", 1)[1] for m in _check(mod, order[key], order)]
        out[key] = msgs
    return out


# ---- loading --------------------------------------------------------------------------


@dataclass
class Report:
    """What happened at startup. `loaded` is what's recorded in saves."""

    safe: bool = False
    crashed: bool = False  # the last start didn't get as far as settled()
    loaded: list[Mod] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One line for people: which mods are in."""
        if self.safe:
            return "mods: off (--safe)"
        return "mods: " + (", ".join(m.title for m in self.loaded) or "none")


_report: Report | None = None


def report() -> Report:
    """This process's mods (an empty report if none were loaded)."""
    return _report or Report()


def active() -> list[str]:
    """The loaded mods as "name@version" (or "name"), for save files."""
    return [f"{m.name}@{m.version}" if m.version else m.name for m in report().loaded]


def recorded_problems(recorded: list[str]) -> list[str]:
    """What's off between the mods a save recorded ("name@version", see active())
    and the ones loaded now: missing mods (one line), other versions (one each)."""
    have = {m.name.casefold(): m for m in report().loaded}
    missing, out = [], []
    for entry in recorded:
        name, _, version = entry.partition("@")
        mod = have.get(name.casefold())
        if mod is None:
            missing.append(f"{name} {version}".rstrip())
        elif version and mod.version and version != mod.version:
            out.append(f"saved with mod {name} {version}, this is {mod.version}")
    if missing:
        why = " (--safe)" if report().safe else ""
        s = "s" if len(missing) > 1 else ""
        out.insert(0, f"saved with mod{s} {', '.join(missing)}, not loaded{why}")
    return out


def load(safe: bool | None = None, folder: Path | None = None) -> Report:
    """Import the enabled mods, once per process; later calls return the first report."""
    global _report, _current, _loading
    if _report is not None:
        return _report
    from .project import data_root

    safe = safe_mode() if safe is None else safe
    rep = _report = Report(safe=safe)
    if safe:
        return rep
    sentinel = data_root() / SENTINEL
    rep.crashed = sentinel.exists()
    if rep.crashed:
        rep.problems.append(
            "the last start stopped before pijl was up (a mod?); --safe starts without mods"
        )
    try:
        p = plan(folder)
    except OSError as e:
        rep.problems.append(f"can't read the mods folder: {e}")
        return rep
    for name in p.missing:
        rep.problems.append(f"{name} is in {LOADORDER} but isn't in the mods folder")
    for mod in p.both:
        rep.problems.append(f"{mod.name} is in both lists; disabled wins")
    if p.new:
        names = ", ".join(m.name for m in p.new)
        rep.problems.append(f"new: {names} (disabled; see {DISABLED})")
    if not p.enabled:
        return rep
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            ", ".join(m.name for m in p.enabled) + "\n", encoding="utf-8"
        )
    except OSError:
        pass
    package = _package(p.folder)
    _loading = True
    order = {m.name.casefold(): i for i, m in enumerate(p.enabled)}
    for i, mod in enumerate(p.enabled):
        if not mod.usable:
            rep.problems += [f"{mod.name}: {msg}; not loaded" for msg in mod.problems]
            continue
        rep.problems += [f"{mod.name}: {msg}" for msg in mod.warnings]
        rep.problems += _check(mod, i, order)
        lib = mod.path / "lib"
        if mod.path.is_dir() and lib.is_dir() and str(lib) not in sys.path:
            sys.path.append(str(lib))
        _current = mod.name
        try:
            importlib.import_module(f"{package.__name__}.{mod.name}")
        except BaseException as e:  # (SystemExit too: a mod doesn't get to quit pijl)
            if isinstance(e, KeyboardInterrupt):
                raise
            rep.problems.append(f"{mod.name}: {_last_line(e)}; not loaded")
            _tracebacks[mod.name] = traceback.format_exc()
            dotted = f"{package.__name__}.{mod.name}"
            for k in [
                k for k in sys.modules if k == dotted or k.startswith(dotted + ".")
            ]:
                del sys.modules[k]
            continue
        finally:
            _current = None
        rep.loaded.append(mod)
    _loading = False
    # (only now: a mod that logs, like manuscript, has had the chance to listen)
    log.info("%s", rep.summary())
    for problem in rep.problems:
        log.warning("%s", problem)
    return rep


def settled() -> None:
    """pijl is up: the mods didn't crash it. Clears the marker load() left."""
    from .project import data_root

    try:
        (data_root() / SENTINEL).unlink(missing_ok=True)
    except OSError:
        pass


def traceback_of(name: str) -> str:
    """The full traceback of a mod that failed (to import, or in a hook), or ""."""
    return _tracebacks.get(name, "")


def _package(folder: Path) -> ModuleType:
    spec = importlib.machinery.ModuleSpec(PACKAGE, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(folder)]
    sys.modules[PACKAGE] = module
    return module


def _check(mod: Mod, index: int, order: dict[str, int]) -> list[str]:
    """Warnings from the manifest: pijl version, requirements."""
    out = []
    m = mod.manifest
    spec = m.get("pijl")
    if isinstance(spec, str) and spec.strip():
        ok = version_matches(__version__, spec)
        if ok is None:
            out.append(f"{mod.name}: can't read its pijl version range {spec!r}")
        elif not ok:
            out.append(f"{mod.name}: made for pijl {spec}, this is {__version__}")
    requires = m.get("requires", [])
    if isinstance(requires, str):
        requires = [requires]
    for need in requires if isinstance(requires, list) else ():
        at = order.get(str(need).casefold())
        if at is None:
            out.append(f"{mod.name}: needs {need}, which isn't enabled")
        elif at > index:
            out.append(f"{mod.name}: needs {need}, which loads after it")
    return out


def _last_line(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(e)).strip().splitlines()[-1]


# ---- versions --------------------------------------------------------------------------

_CLAUSE = re.compile(r"^\s*(>=|<=|==|!=|>|<)?\s*([0-9][0-9A-Za-z.\-+]*)\s*$")


def parse_version(text: str) -> tuple[int, ...]:
    """The release numbers of a version: "0.2.3rc1" -> (0, 2, 3)."""
    release = re.match(r"[0-9]+(?:\.[0-9]+)*", text.strip())
    if not release:
        raise ValueError(text)
    return tuple(int(n) for n in release.group().split("."))


def version_matches(version: str, spec: str) -> bool | None:
    """Does `version` satisfy `spec` (comma-separated clauses like ">=0.2", "<0.3",
    "0.2.3")? None if either can't be read."""
    try:
        have = parse_version(version)
        for clause in spec.split(","):
            m = _CLAUSE.match(clause)
            if not m:
                return None
            op, want = m.group(1) or "==", parse_version(m.group(2))
            n = max(len(have), len(want))
            a, b = have + (0,) * (n - len(have)), want + (0,) * (n - len(want))
            ok = {
                ">=": a >= b,
                "<=": a <= b,
                ">": a > b,
                "<": a < b,
                "==": a == b,
                "!=": a != b,
            }[op]
            if not ok:
                return False
    except ValueError:
        return None
    return True


# ---- after_import ----------------------------------------------------------------------

_hooks: dict[str, list[tuple[Callable[[ModuleType], object], str | None]]] = {}
_current: str | None = None  # the mod being imported (whose hooks these are)
_loading = False  # (load()'s caller reports what goes wrong meanwhile; later, print it)
_tracebacks: dict[str, str] = {}


def after_import(module: str, fn: Callable[[ModuleType], object] | None = None):
    """Call fn(module) once `module` has been imported: now, if it already has been.
    Use as a decorator, @after_import("pijl.ui.editor"), or call it directly. Hooks
    run in the order they were added. A hook that raises is reported against the
    mod that added it; the import itself goes on."""

    def add(fn):
        owner = _current
        done = sys.modules.get(module)
        if done is not None and not _importing(done):
            _call(fn, done, owner)
        else:
            _install()
            _hooks.setdefault(module, []).append((fn, owner))
        return fn

    return add if fn is None else add(fn)


def _importing(module: ModuleType) -> bool:
    spec = getattr(module, "__spec__", None)
    return bool(getattr(spec, "_initializing", False))


def _call(fn, module: ModuleType, owner: str | None) -> None:
    try:
        fn(module)
    except Exception as e:  # noqa: BLE001 (any bug in a mod's hook)
        who = owner or getattr(fn, "__module__", "?")
        msg = f"{who}: after_import({module.__name__!r}): {_last_line(e)}"
        report().problems.append(msg)
        _tracebacks.setdefault(who, traceback.format_exc())
        if not _loading:
            print(f"pijl: mod {msg}", file=sys.stderr)


class _Loader(importlib.abc.Loader):
    """Wraps a module's real loader to run its hooks once it has executed."""

    def __init__(self, inner, name: str) -> None:
        self._inner, self._name = inner, name

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        for fn, owner in _hooks.pop(self._name, ()):
            _call(fn, module, owner)

    def __getattr__(self, attr):  # (get_source, is_package, resource readers...)
        return getattr(self._inner, attr)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _hooks:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _Loader(spec.loader, fullname)
        return spec


_finder = _Finder()


def _install() -> None:
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
