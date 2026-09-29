"""The registry of part types, and the loader that fills it from a folder of scripts.

Loading a folder:
  - every .py file under it is imported, recursively, in path order; files and
    folders whose name starts with "_" are private (helpers) and not imported
    directly -- scripts can still import them, relative imports included
  - a script that defines `register` is a part script: its `API` must match and
    `register(reg)` is called; a script without `register` is just a helper
  - the folder becomes a package under a unique made-up name, so two folders
    that both have a utils.py don't collide
  - a script that fails in any way (import error, wrong API, a bad or duplicate
    part) is skipped as a whole and its error is kept in `errors`; everything
    else still loads
"""

from __future__ import annotations

import copy
import importlib
import importlib.machinery
import importlib.util
import itertools
import sys
import traceback
from pathlib import Path

from .contract import API, LABEL_SIDES, Look, PartType
from .ports import PORTS

_package_ids = itertools.count(1)


class Registry:
    def __init__(self) -> None:
        self.types: dict[str, PartType] = {}  # kind -> type, in load order
        self.errors: list[str] = []           # one message per script that didn't load
        self._pending: list[PartType] | None = None  # adds from the script being loaded
        for t in PORTS:
            self._check(t, engine=True)
            self.types[t.kind] = t

    def __iter__(self):
        return iter(self.types.values())

    def __contains__(self, kind: str) -> bool:
        return kind in self.types

    def get(self, kind: str) -> PartType:
        return self.types[kind]

    def add(self, t: PartType | type[PartType]) -> None:
        """Called by part scripts. Accepts a PartType or a PartType subclass."""
        if isinstance(t, type):
            t = t()
        self._check(t)
        if self._pending is None:
            self.types[t.kind] = t
        else:
            self._pending.append(t)

    def _check(self, t: PartType, engine: bool = False) -> None:
        if not isinstance(t, PartType):
            raise TypeError(f"not a PartType: {t!r}")
        if not isinstance(t.kind, str) or not t.kind.strip():
            raise ValueError(f"{type(t).__name__} has no kind")
        name = t.kind
        if ":" in name:
            raise ValueError(f"{name}: ':' isn't allowed in part names")  # "macro:..." is taken
        taken = set(self.types) | {p.kind for p in self._pending or ()}
        if name in taken:
            raise ValueError(f"{name}: there already is a part with that name")
        t.ins, t.outs = tuple(t.ins), tuple(t.outs)
        if not all(isinstance(p, str) for p in t.ins + t.outs):
            raise TypeError(f"{name}: pin names must be strings")
        if t.port is not None and not engine:
            raise ValueError(f"{name}: only the engine defines ports (IN/OUT)")
        if t.pure and not t.outs:
            raise ValueError(f"{name}: a pure part without outputs can't do anything")
        if t.pure and not t.has("eval"):
            raise ValueError(f"{name}: a pure part needs an eval")
        if not isinstance(t.props, dict):
            raise TypeError(f"{name}: props must be a dict")
        if not isinstance(t.look, Look) or t.look.label not in LABEL_SIDES:
            raise ValueError(f"{name}: bad look {t.look!r}")

    # ---- loading -----------------------------------------------------------

    def load_folder(self, root: Path) -> None:
        """Import every part script under `root` (see the module docstring)."""
        root = Path(root)
        if not root.is_dir():
            return
        package = f"_pijl_parts_{next(_package_ids)}"
        spec = importlib.machinery.ModuleSpec(package, None, is_package=True)
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(root)]
        sys.modules[package] = module
        # The folder is the user's (a project's parts/): don't litter it with __pycache__.
        was, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            for path in sorted(root.rglob("*.py")):
                rel = path.relative_to(root)
                if any(p.startswith("_") for p in rel.parts):
                    continue
                self._load_script(path, ".".join((package, *rel.with_suffix("").parts)))
        finally:
            sys.dont_write_bytecode = was

    def _load_script(self, path: Path, module_name: str) -> None:
        self._pending = []
        try:
            module = importlib.import_module(module_name)
            register = getattr(module, "register", None)
            if register is None:
                return  # a helper
            if getattr(module, "API", None) != API:
                raise ValueError(f"API = {getattr(module, 'API', None)!r}, this pijl has API = {API}")
            register(self)
            for t in self._pending:
                self.types[t.kind] = t
        except Exception:
            detail = traceback.format_exc(limit=-1).strip().splitlines()[-1]
            self.errors.append(f"{path.name}: {detail}")
        finally:
            self._pending = None


def load(*folders: Path) -> Registry:
    """A registry with the engine's ports plus every part script in `folders`."""
    reg = Registry()
    for folder in folders:
        reg.load_folder(folder)
    return reg


def fresh_props(t: PartType) -> dict:
    """A new instance's own copy of its type's default props."""
    return copy.deepcopy(t.props)
