"""The registry of part types, and the loader that fills it from a folder of scripts.

Loading a folder:
  - every .py file under it is imported, recursively, in path order; files and
    folders whose name starts with "_" are private (helpers) and not imported
    directly -- scripts can still import them, relative imports included
  - a script that defines `register` is a part script: its `API` must be one this
    pijl supports (see SUPPORTED_APIS) and `register(reg)` is called; a script
    without `register` is just a helper
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

from .contract import API, GRID_STEP, LABEL_SIDES, SUPPORTED_APIS, Look, Mark, PartType
from .ports import PORTS
from .settings import Action, Setting

RESERVED_PROPS = frozenset({"color"})  # props the editor keeps on every part (Recolor)

_package_ids = itertools.count(1)


class Registry:
    def __init__(self) -> None:
        self.types: dict[str, PartType] = {}  # kind -> type, in load order
        self.errors: list[str] = []  # one message per script that didn't load
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
            raise ValueError(
                f"{name}: ':' isn't allowed in part names"
            )  # "macro:..." is taken
        taken = set(self.types) | {p.kind for p in self._pending or ()}
        if name in taken:
            raise ValueError(f"{name}: there already is a part with that name")
        t.ins, t.outs, t.weak = tuple(t.ins), tuple(t.outs), tuple(t.weak)
        if not all(isinstance(p, str) for p in t.ins + t.outs):
            raise TypeError(f"{name}: pin names must be strings")
        if not set(t.weak) <= set(t.outs):
            raise ValueError(f"{name}: weak pins must be outputs")
        t.joins = tuple(tuple(group) for group in t.joins)
        joined = [p for group in t.joins for p in group]
        if not set(joined) <= set(t.ins + t.outs) or len(joined) != len(set(joined)):
            raise ValueError(f"{name}: joins must name pins, each at most once")
        if len(set(t.ins + t.outs)) != len(t.ins + t.outs) and joined:
            raise ValueError(f"{name}: joined pins need unique names")
        if t.port is not None and not engine:
            raise ValueError(f"{name}: only the engine defines ports (IN/OUT)")
        if t.pure and not t.outs:
            raise ValueError(f"{name}: a pure part without outputs can't do anything")
        if t.pure and not t.has("eval"):
            raise ValueError(f"{name}: a pure part needs an eval")
        if not isinstance(t.props, dict):
            raise TypeError(f"{name}: props must be a dict")
        if hasattr(t, "choices"):
            raise ValueError(
                f"{name}: `choices` was replaced by `settings` (see pijl/parts/settings.py)"
            )
        if not isinstance(t.settings, dict) or not isinstance(t.actions, dict):
            raise TypeError(f"{name}: settings and actions must be dicts")
        t.settings, t.actions = dict(t.settings), dict(t.actions)
        for key, s in t.settings.items():
            where = f"{name}: settings[{key!r}]"
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError(f"{where}: keys must be identifiers")
            if not isinstance(s, Setting):
                raise TypeError(f"{where}: not a Setting")
            if key in t.props:
                raise ValueError(
                    f"{where}: its default lives in the setting, not also in props"
                )
            try:
                s.check()
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
        taken = RESERVED_PROPS & (t.props.keys() | t.settings.keys())
        if taken and not engine:
            raise ValueError(
                f"{name}: {', '.join(sorted(taken))} is reserved for the editor"
            )
        for action_name, a in t.actions.items():
            if not isinstance(action_name, str) or not action_name.isidentifier():
                raise ValueError(
                    f"{name}: actions[{action_name!r}]: names must be identifiers"
                )
            if not isinstance(a, Action):
                raise TypeError(f"{name}: actions[{action_name!r}]: not an Action")
        if t.actions and not t.has("action"):
            raise ValueError(f"{name}: has actions but no action() hook")
        if not isinstance(t.look, Look) or t.look.label not in LABEL_SIDES:
            raise ValueError(f"{name}: bad look {t.look!r}")
        _check_face(t)

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
            api = getattr(module, "API", None)
            if api not in SUPPORTED_APIS:
                raise ValueError(f"API = {api!r}, this pijl has API = {API}")
            register(self)
            for t in self._pending:
                t.api = api
                self.types[t.kind] = t
        except Exception:
            detail = traceback.format_exc(limit=-1).strip().splitlines()[-1]
            self.errors.append(f"{path.name}: {detail}")
        finally:
            self._pending = None


def _check_face(t: PartType) -> None:
    """Look.size and Look.face (see contract.Mark)."""
    name, look = t.kind, t.look
    if look.size is not None:
        size = look.size
        if (
            not isinstance(size, tuple)
            or len(size) != 2
            or not all(type(v) is int and v > 0 and v % GRID_STEP == 0 for v in size)
        ):
            raise ValueError(
                f"{name}: Look.size must be (width, height), multiples of {GRID_STEP}"
            )
    if not isinstance(look.face, tuple):
        raise TypeError(f"{name}: Look.face must be a tuple of Marks")
    if not (
        isinstance(look.face_colors, tuple)
        and len(look.face_colors) == 2
        and all(isinstance(c, str) for c in look.face_colors)
    ):
        raise ValueError(f"{name}: Look.face_colors must be two theme color names")
    pins = set(t.ins + t.outs)
    hooked = 0
    for i, m in enumerate(look.face):
        where = f"{name}: Look.face[{i}]"
        if not isinstance(m, Mark):
            raise TypeError(f"{where}: not a Mark")
        points = (m.a,) if m.b is None else (m.a, m.b)
        if not all(
            isinstance(p, tuple)
            and len(p) == 2
            and all(isinstance(v, (int, float)) for v in p)
            for p in points
        ) or not (isinstance(m.radius, (int, float)) and m.radius > 0):
            raise ValueError(
                f"{where}: points must be (x, y) tuples, the radius a number > 0"
            )
        if m.pin is None:
            hooked += 1
        elif m.pin not in pins:
            raise ValueError(f"{where}: no pin called {m.pin!r}")
    if hooked and not t.has("face"):
        raise ValueError(f"{name}: face marks without a pin need a face() hook")
    if t.has("face") and not hooked:
        raise ValueError(f"{name}: has a face() hook but no face marks without a pin")


def load(*folders: Path) -> Registry:
    """A registry with the engine's ports plus every part script in `folders`."""
    reg = Registry()
    for folder in folders:
        reg.load_folder(folder)
    return reg


def fresh_props(t: PartType) -> dict:
    """A new instance's own copy of its type's default props, settings included."""
    props = copy.deepcopy(t.props)
    for key, s in t.settings.items():
        props[key] = s.initial
    return props


def copy_props(props: dict) -> dict:
    """An instance's own copy of these props: a plain copy when every value is
    immutable (the usual case, and much cheaper than a deep copy)."""
    return dict(props) if flat(props) else copy.deepcopy(props)


def flat(props: dict) -> bool:
    """Every value immutable: a plain copy of these props is as good as a deep one."""
    return all(
        v is None or isinstance(v, (bool, int, float, str, bytes))
        for v in props.values()
    )


def check_props(t: PartType, props: dict) -> tuple[dict, list[str]]:
    """Saved props -> what an instance gets: defaults for what's missing, every setting's
    value through its parse(). A value that doesn't parse is reset to the default; the
    second item says which ones were. Keys the type doesn't know are kept (the script
    may have dropped a setting; its value isn't thrown away on load)."""
    out = {**fresh_props(t), **props}
    bad = []
    for key, s in t.settings.items():
        try:
            out[key] = s.parse(out[key])
        except ValueError as e:
            out[key] = s.initial
            bad.append(f"{key}: {e}, reset to {s.show(s.initial)}")
    return out, bad
