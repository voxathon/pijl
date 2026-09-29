"""Macro files: reading and writing boards on disk. Pure data, no pyglet.

Every saved board is a macro: projects/<name>/macros/<macro name>.json. The file
name *is* the macro's name, so there's exactly one place a name lives.

Format (version 1). One part / wire per line and a fixed order, so files diff
nicely in git:

    {
      "pijl": 1,
      "parts": [
        {"uid": 1, "kind": "IN", "label": "a", "pos": [200, 360]},
        {"uid": 3, "kind": "NAND", "pos": [380, 280]}
      ],
      "wires": [
        {"uid": 1, "from": {"part": 1, "out": 0}, "to": {"part": 3, "in": 0}, "bends": [[320, 380]]},
        {"uid": 2, "from": {"wire": 1, "at": [320, 350]}, "to": {"part": 3, "in": 1}}
      ]
    }

  - "kind" names a part type (pijl.parts). Macros placed inside other macros will
    get their own field ("macro": name) instead, so the two can never collide.
  - "label", "props" and "bends" are left out when empty.
  - A wire end is a pin ({"part", "in"/"out"}) or a point along another wire
    ({"wire", "at"}). A wire only attaches to wires with a smaller uid.

Loading is forgiving: parts of unknown kinds and wires whose ends don't exist
(any more) are dropped, with a warning each, instead of refusing the file.
Only a file from a *newer* pijl, or one that isn't a pijl file at all, is refused.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parts import Registry
from .project import write_atomic
from .snapshot import EndRef, Point, Snapshot

FORMAT = 1
NAME_MAX = 40
_FORBIDDEN = '<>:"/\\|?*'
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class FormatError(ValueError):
    """Not a macro file this pijl can read."""


@dataclass
class Loaded:
    snapshot: Snapshot
    warnings: list[str] = field(default_factory=list)


# ---- names ---------------------------------------------------------------------


def check_name(name: str) -> str:
    """The cleaned-up macro name, or ValueError saying what's wrong with it.
    Names are file names, so Windows' rules apply."""
    name = name.strip()
    if not name:
        raise ValueError("type a name")
    if len(name) > NAME_MAX:
        raise ValueError(f"at most {NAME_MAX} characters")
    for c in name:
        if c in _FORBIDDEN or not c.isprintable():
            raise ValueError(f"can't contain {c!r}")
    if name.endswith("."):
        raise ValueError("can't end with a dot")
    if name.split(".")[0].upper() in _RESERVED:
        raise ValueError(f"{name!r} is reserved by Windows")
    return name


# ---- snapshot <-> dict -----------------------------------------------------------


def encode(snap: Snapshot) -> dict[str, Any]:
    parts = []
    for uid in sorted(snap.parts):
        kind, label, x, y, props = snap.parts[uid]
        d: dict[str, Any] = {"uid": uid, "kind": kind}
        if label:
            d["label"] = label
        d["pos"] = [_num(x), _num(y)]
        if props:
            d["props"] = props
        parts.append(d)
    wires = []
    for uid in sorted(snap.wires):
        src, dst, bends, src_pt, dst_pt = snap.wires[uid]
        d = {"uid": uid, "from": _encode_end(src, src_pt), "to": _encode_end(dst, dst_pt)}
        if bends:
            d["bends"] = [_point(p) for p in bends]
        wires.append(d)
    return {"pijl": FORMAT, "parts": parts, "wires": wires}


def decode(data: Any, registry: Registry) -> Loaded:
    """A dict from a macro file -> the Snapshot to build, plus what had to be dropped."""
    if not isinstance(data, dict) or "pijl" not in data:
        raise FormatError("not a pijl macro file")
    version = data["pijl"]
    if not isinstance(version, int) or version < 1:
        raise FormatError(f"unknown format version {version!r}")
    if version > FORMAT:
        raise FormatError(f"made by a newer pijl (format {version}; this one reads up to {FORMAT})")
    out = Loaded(Snapshot({}, {}))
    warn = out.warnings.append

    kinds: dict[int, Any] = {}  # part uid -> its PartType, for checking pin indices
    for d in _list(data, "parts"):
        try:
            uid = _int(d["uid"])
            if uid in out.snapshot.parts:
                raise ValueError(f"duplicate uid {uid}")
            kind = d["kind"]
            if kind not in registry:
                warn(f"part {uid}: unknown kind {kind!r}, dropped")
                continue
            label = d.get("label", "")
            props = d.get("props", {})
            if not isinstance(label, str) or not isinstance(props, dict):
                raise ValueError("bad label or props")
            t = registry.get(kind)
            x, y = _pair(d["pos"])
            out.snapshot.parts[uid] = (kind, label, x, y, {**t.props, **props})
            kinds[uid] = t
        except (KeyError, TypeError, ValueError) as e:
            warn(f"a part was unreadable ({_why(e)}), dropped")

    wired_inputs: set[tuple[int, int]] = set()  # an input pin takes one wire
    lost = 0  # wires whose ends are gone: counted, not listed one by one
    for d in sorted(_list(data, "wires"), key=lambda d: d.get("uid", 0) if isinstance(d, dict) else 0):
        try:
            uid = _int(d["uid"])
            if uid in out.snapshot.wires:
                raise ValueError(f"duplicate uid {uid}")
            ends = [_decode_end(d[k], uid, kinds, out.snapshot) for k in ("from", "to")]
            if any(ref is None for ref, _ in ends):
                lost += 1
                continue
            (src, src_pt), (dst, dst_pt) = ends
            pins = [ref for ref, _ in ends if ref[0] == "p"]
            if len(pins) == 2 and (pins[0][2] == pins[1][2] or pins[0][1] == pins[1][1]):
                raise ValueError("connects two inputs, two outputs, or a part to itself")
            inputs = {(ref[1], ref[3]) for ref in pins if ref[2]}
            if inputs & wired_inputs:
                raise ValueError("a second wire into an input pin")
            wired_inputs |= inputs
            bends = tuple(_pair(p) for p in d.get("bends", ()))
            out.snapshot.wires[uid] = (src, dst, bends, src_pt, dst_pt)
        except (KeyError, TypeError, ValueError) as e:
            warn(f"a wire was unreadable ({_why(e)}), dropped")
    if lost:
        warn(f"{lost} wire(s) lost an end (a dropped part or wire), dropped")
    return out


def dumps(data: dict[str, Any]) -> str:
    """The on-disk text: stable order, one part / wire per line."""
    def block(key: str) -> str:
        rows = data[key]
        if not rows:
            return f'  "{key}": []'
        return f'  "{key}": [\n' + ",\n".join("    " + json.dumps(r, ensure_ascii=False) for r in rows) + "\n  ]"
    return '{\n  "pijl": ' + json.dumps(data["pijl"]) + ",\n" + block("parts") + ",\n" + block("wires") + "\n}\n"


# ---- files -----------------------------------------------------------------------


class MacroStore:
    """The macros of one project: one .json file per macro, named after it."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    def names(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted((p.stem for p in self.folder.glob("*.json")), key=str.casefold)

    def find(self, name: str) -> str | None:
        """The saved macro whose name matches `name` ignoring case (file names on
        Windows do), or None."""
        key = name.strip().casefold()
        return next((n for n in self.names() if n.casefold() == key), None)

    def path(self, name: str) -> Path:
        return self.folder / f"{check_name(name)}.json"

    def save(self, name: str, snap: Snapshot) -> None:
        name = check_name(name)
        old = self.find(name)
        if old is not None and old != name:
            self.path(old).unlink()  # same name, new capitalization: rename, don't keep the old spelling
        self.folder.mkdir(parents=True, exist_ok=True)
        write_atomic(self.path(name), dumps(encode(snap)))

    def load(self, name: str, registry: Registry) -> Loaded:
        """FormatError / OSError if the file can't be used at all."""
        try:
            data = json.loads(self.path(name).read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise FormatError(f"not valid JSON ({e.msg}, line {e.lineno})") from None
        return decode(data, registry)


# ---- helpers ----------------------------------------------------------------------


def _encode_end(ref: EndRef, at: Point | None) -> dict[str, Any]:
    if ref[0] == "p":
        _, part, is_input, index = ref
        return {"part": part, "in" if is_input else "out": index}
    return {"wire": ref[1], "at": _point(at)}


def _decode_end(d: Any, wire_uid: int, kinds: dict[int, Any], snap: Snapshot) -> tuple[EndRef | None, Point | None]:
    """(ref, junction point); ref is None when the thing it points at is gone."""
    if not isinstance(d, dict):
        raise TypeError("a wire end isn't an object")
    if "wire" in d:
        target = _int(d["wire"])
        if target >= wire_uid:
            raise ValueError("attaches to a newer wire")
        if target not in snap.wires:
            return None, None
        return ("w", target), _pair(d["at"])
    part = _int(d["part"])
    is_input = "in" in d
    index = _int(d["in"] if is_input else d["out"])
    t = kinds.get(part)
    if t is None:
        return None, None
    if not 0 <= index < len(t.ins if is_input else t.outs):
        return None, None  # the part type has fewer pins now
    return ("p", part, is_input, index), None


def _list(data: dict, key: str) -> list:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise FormatError(f'"{key}" isn\'t a list')
    return value


def _int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{v!r} isn't a whole number")
    return v


def _pair(v: Any) -> Point:
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                                                         for c in v)) or not all(map(math.isfinite, v)):
        raise TypeError(f"{v!r} isn't a point")
    return float(v[0]), float(v[1])


def _num(v: float) -> int | float:
    """Whole numbers without the '.0', the rest rounded: positions stay readable."""
    v = round(float(v), 4)
    return int(v) if v.is_integer() else v


def _point(p: Point) -> list:
    return [_num(p[0]), _num(p[1])]


def _why(e: Exception) -> str:
    return f"missing {e}" if isinstance(e, KeyError) else str(e)
