"""Macro files: reading and writing boards on disk. Pure data, no pyglet.

Every saved board is a macro: projects/<name>/macros/<id>.json. The file name is
the macro's *id*: what other boards (and library.json) refer to it by, which never
changes. What the user sees is its *title*, kept inside the file, so renaming a
macro rewrites one line of one file and nothing that uses it. A new macro's id is
its first title (numbered if that file is taken); a file without a title shows
its id. Titles are unique in a project, ignoring case.

Format (version 1). One part / wire per line and a fixed order, so files diff
nicely in git:

    {
      "pijl": 1,
      "title": "half adder",
      "parts": [
        {"uid": 1, "kind": "IN", "label": "a", "pos": [200, 360]},
        {"uid": 3, "kind": "NAND", "pos": [380, 280]}
      ],
      "wires": [
        {"uid": 1, "from": {"part": 1, "out": 0}, "to": {"part": 3, "in": 0}, "bends": [[320, 380]]},
        {"uid": 2, "from": {"wire": 1, "at": [320, 350]}, "to": {"part": 3, "in": 1}}
      ]
    }

  - "kind" names a part type (pijl.parts); a placed macro has "macro": its name
    instead, so the two can never collide.
  - "label", "props", "bends" and a wire's "color" (a name, see ui/theme.py
    WIRE_COLORS) and "layer" (0 to 15, see snapshot.py) are left out when empty /
    default.
  - A wire end is a pin ({"part", "in"/"out": index}), a macro's pin
    ({"part", "pin": uid of the IN/OUT inside the macro that it comes from}, so
    it survives the macro's ports being moved around), or a point along another
    wire ({"wire", "at"}). A wire only attaches to wires with a smaller uid --
    or to itself: that end is free (attached to nothing), and "at" is where it is.

Both directions need to know the part types (pin counts, macro pins): `types`
is a Registry or a macros.Catalog.

Loading is forgiving: parts of unknown kinds and wires whose ends don't exist
(any more) are dropped, with a warning each, instead of refusing the file.
Only a file from a *newer* pijl, or one that isn't a pijl file at all, is refused.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parts import Registry, check_props
from .project import write_atomic
from .snapshot import LAYER_COUNT, MACRO, EndRef, Point, Snapshot

FORMAT = 1
NAME_MAX = 40
_FORBIDDEN = '<>:"/\\|?*'
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class FormatError(ValueError):
    """Not a macro file this pijl can read."""


@dataclass
class Loaded:
    snapshot: Snapshot
    warnings: list[str] = field(default_factory=list)
    title: str | None = None  # None: the file doesn't have one


# ---- names ---------------------------------------------------------------------


def check_name(name: str) -> str:
    """The cleaned-up macro title (or project name), or ValueError saying what's
    wrong with it. New macros' ids are made from titles and ids are file names, so
    Windows' rules apply."""
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


def encode(
    snap: Snapshot, types: Registry | None = None, title: str | None = None
) -> dict[str, Any]:
    """`types` is needed only if the board has macros on it (for their pin uids)."""
    parts = []
    for uid in sorted(snap.parts):
        kind, label, x, y, props = snap.parts[uid]
        d: dict[str, Any] = {"uid": uid}
        if kind.startswith(MACRO):
            d["macro"] = kind[len(MACRO) :]
        else:
            d["kind"] = kind
        if label:
            d["label"] = label
        d["pos"] = [_num(x), _num(y)]
        if props:
            d["props"] = props
        parts.append(d)
    wires = []
    for uid in sorted(snap.wires):
        src, dst, bends, src_pt, dst_pt = snap.wires[uid]
        d = {
            "uid": uid,
            "from": _encode_end(src, src_pt, snap, types),
            "to": _encode_end(dst, dst_pt, snap, types),
        }
        if bends:
            d["bends"] = [_point(p) for p in bends]
        if uid in snap.wire_colors:
            d["color"] = snap.wire_colors[uid]
        if snap.wire_layers.get(uid):
            d["layer"] = snap.wire_layers[uid]
        wires.append(d)
    head: dict[str, Any] = {"pijl": FORMAT}
    if title is not None:
        head["title"] = title
    return {**head, "parts": parts, "wires": wires}


def decode(data: Any, types: Registry) -> Loaded:
    """A dict from a macro file -> the Snapshot to build, plus what had to be dropped."""
    if not isinstance(data, dict) or "pijl" not in data:
        raise FormatError("not a pijl macro file")
    version = data["pijl"]
    if not isinstance(version, int) or version < 1:
        raise FormatError(f"unknown format version {version!r}")
    if version > FORMAT:
        raise FormatError(
            f"made by a newer pijl (format {version}; this one reads up to {FORMAT})"
        )
    out = Loaded(Snapshot({}, {}), title=_title(data))
    warn = out.warnings.append

    kinds: dict[int, Any] = {}  # part uid -> its PartType, for checking pin indices
    for d in _list(data, "parts"):
        try:
            uid = _int(d["uid"])
            if uid in out.snapshot.parts:
                raise ValueError(f"duplicate uid {uid}")
            kind = MACRO + d["macro"] if "macro" in d else d["kind"]
            if not isinstance(kind, str):
                raise TypeError("bad kind")
            try:
                t = types.get(kind)
            except KeyError as e:
                what = (
                    f"can't use macro {kind[len(MACRO) :]!r} ({e.args[0] if e.args else 'missing'})"
                    if kind.startswith(MACRO)
                    else f"unknown kind {kind!r}"
                )
                warn(f"part {uid}: {what}, dropped")
                continue
            label = d.get("label", "")
            props = d.get("props", {})
            if not isinstance(label, str) or not isinstance(props, dict):
                raise ValueError("bad label or props")
            x, y = _pair(d["pos"])
            props, bad = check_props(t, props)
            for why in bad:
                warn(f"part {uid} ({kind}): {why}")
            out.snapshot.parts[uid] = (kind, label, x, y, props)
            kinds[uid] = t
        except (KeyError, TypeError, ValueError) as e:
            warn(f"a part was unreadable ({_why(e)}), dropped")

    wired_inputs: set[tuple[int, int]] = set()  # an input pin takes one wire
    lost = 0  # wires whose ends are gone: counted, not listed one by one
    for d in sorted(
        _list(data, "wires"),
        key=lambda d: d.get("uid", 0) if isinstance(d, dict) else 0,
    ):
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
            if len(pins) == 2 and (
                pins[0][2] == pins[1][2] or pins[0][1] == pins[1][1]
            ):
                raise ValueError(
                    "connects two inputs, two outputs, or a part to itself"
                )
            inputs = {(ref[1], ref[3]) for ref in pins if ref[2]}
            if inputs & wired_inputs:
                raise ValueError("a second wire into an input pin")
            wired_inputs |= inputs
            bends = tuple(_pair(p) for p in d.get("bends", ()))
            color = d.get("color")
            if color is not None and not isinstance(color, str):
                raise ValueError("bad color")
            layer = _int(d.get("layer", 0))
            if not 0 <= layer < LAYER_COUNT:
                raise ValueError(f"no layer {layer}")
            out.snapshot.wires[uid] = (src, dst, bends, src_pt, dst_pt)
            if color:
                out.snapshot.wire_colors[uid] = (
                    color  # names the UI doesn't know draw as default
                )
            if layer:
                out.snapshot.wire_layers[uid] = layer
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
        return (
            f'  "{key}": [\n'
            + ",\n".join("    " + json.dumps(r, ensure_ascii=False) for r in rows)
            + "\n  ]"
        )

    return (
        '{\n  "pijl": '
        + json.dumps(data["pijl"])
        + ",\n"
        + (
            f'  "title": {json.dumps(data["title"], ensure_ascii=False)},\n'
            if "title" in data
            else ""
        )
        + block("parts")
        + ",\n"
        + block("wires")
        + "\n}\n"
    )


# ---- files -----------------------------------------------------------------------


class MacroStore:
    """The macros of one project: one .json file per macro, named after its id.
    Methods take ids, except find() and new_id(), which take titles."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        # id -> (the file's mtime, its title, the macros it uses directly): read once
        # per change to the file, not every time the picker asks (see _peek)
        self._titles: dict[str, tuple[int, str, frozenset[str]]] = {}

    def ids(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted((p.stem for p in self.folder.glob("*.json")), key=str.casefold)

    def title(self, id: str) -> str:
        """What the macro is called (its id if the file has no title, or can't be read)."""
        return self._info(id)[0]

    def uses(self, id: str) -> frozenset[str]:
        """The macros (ids) placed directly on macro `id`'s board, without loading it:
        enough to answer "does A contain B" (macros.MacroBook.contains). Empty if the
        file can't be read."""
        return self._info(id)[1]

    def _info(self, id: str) -> tuple[str, frozenset[str]]:
        try:
            mtime = self.path(id).stat().st_mtime_ns
        except (OSError, ValueError):
            return id, frozenset()
        cached = self._titles.get(id)
        if cached is not None and cached[0] == mtime:
            return cached[1], cached[2]
        title, uses = _peek(self.path(id))
        self._titles[id] = (mtime, title or id, uses)
        return title or id, uses

    def titles(self) -> dict[str, str]:
        """id -> title for every macro, in title order."""
        out = {id: self.title(id) for id in self.ids()}
        return dict(sorted(out.items(), key=lambda item: item[1].casefold()))

    def find(self, title: str) -> str | None:
        """The id of the macro titled `title` (ignoring case), or None."""
        key = title.strip().casefold()
        return next(
            (id for id, t in self.titles().items() if t.casefold() == key), None
        )

    def find_id(self, id: str) -> str | None:
        """The id as spelled on disk of the macro `id` (ignoring case, like file names
        on Windows), or None if there's no such file."""
        key = id.casefold()
        return next((i for i in self.ids() if i.casefold() == key), None)

    def new_id(self, title: str) -> str:
        """An id for a new macro titled `title`: the title, numbered if that file is taken."""
        title = check_name(title)
        taken = {i.casefold() for i in self.ids()}
        id, n = title, 1
        while id.casefold() in taken:
            n += 1
            suffix = f" ({n})"
            id = title[: NAME_MAX - len(suffix)].rstrip(" .") + suffix
        return id

    def path(self, id: str) -> Path:
        return self.folder / f"{check_name(id)}.json"

    def save(
        self,
        id: str,
        snap: Snapshot,
        types: Registry | None = None,
        title: str | None = None,
    ) -> None:
        """Write macro `id`; `title` None keeps the one it has (a new one: its id)."""
        id = check_name(id)
        title = check_name(title) if title is not None else self.title(id)
        old = self.find_id(id)
        if old is not None and old != id:
            self.path(
                old
            ).unlink()  # same id, new capitalization: rename, don't keep the old spelling
        self.folder.mkdir(parents=True, exist_ok=True)
        write_atomic(self.path(id), dumps(encode(snap, types, title)))
        self._titles.pop(id, None)

    def retitle(self, id: str, title: str) -> None:
        """Rename macro `id`: only the title in its file changes. FormatError / OSError
        if the file can't be read or written."""
        title = check_name(title)
        data = _read(self.path(id))
        if not isinstance(data, dict) or "pijl" not in data:
            raise FormatError("not a pijl macro file")
        data["title"] = title
        for key in ("parts", "wires"):
            data.setdefault(key, [])
        write_atomic(self.path(id), dumps(data))
        self._titles.pop(id, None)

    def remove(self, id: str, trash: Path | None) -> Path | None:
        """Delete a macro's file: moved into the folder `trash` (made if needed; a
        numbered name if that's taken), or gone for good if `trash` is None.
        Returns where it went. OSError if it can't."""
        src = self.path(id)
        if trash is None:
            src.unlink()
            return None
        trash.mkdir(parents=True, exist_ok=True)
        dest, n = trash / src.name, 1
        while dest.exists():
            n += 1
            dest = trash / f"{src.stem} ({n}){src.suffix}"
        os.replace(src, dest)
        return dest

    def put_back(self, id: str, trashed: Path) -> None:
        """Undo remove(): the file at `trashed` becomes macro `id` again.
        FileExistsError if that id or its title is taken now; OSError if it can't."""
        title = _read_title(trashed) or id
        taken = self.find(title)
        if taken is not None:
            raise FileExistsError(f"there's a macro called {self.title(taken)} now")
        if self.find_id(id) is not None:
            raise FileExistsError(f"there's a macro with the file {id}.json now")
        self.folder.mkdir(parents=True, exist_ok=True)
        os.replace(trashed, self.path(id))

    def load(self, id: str, types: Registry) -> Loaded:
        """FormatError / OSError if the file can't be used at all."""
        return load_file(self.path(id), types)


def load_file(path: Path, types: Registry) -> Loaded:
    """A macro file anywhere on disk. FormatError / OSError if it can't be used at all."""
    return decode(_read(path), types)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise FormatError(f"not valid JSON ({e.msg}, line {e.lineno})") from None


_TITLE = re.compile(r'"title"\s*:\s*("(?:[^"\\]|\\.)*")')
_USES = re.compile(r'"macro"\s*:\s*("(?:[^"\\]|\\.)*")')


def _peek(path: Path) -> tuple[str | None, frozenset[str]]:
    """A file's title and the macros it uses, without decoding the whole board (a
    big one is hundreds of MB as Python objects). Neither key can appear unescaped
    inside a string, so a regex finds them; only "title" needs a check that it's
    the file's own (written above "parts", see encode), else the file is parsed."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, frozenset()
    uses = frozenset(
        u for m in {m.group(1) for m in _USES.finditer(text)} if (u := _string(m))
    )
    m = _TITLE.search(text)
    if m is None:
        return None, uses
    parts = text.find('"parts"')
    if parts == -1 or m.start() < parts:
        try:
            return check_name(_string(m.group(1))), uses
        except (ValueError, TypeError):
            return None, uses
    try:  # a "title" further down (a hand-edited file, or a part's props)
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, uses
    return (_title(data) if isinstance(data, dict) else None), uses


def _string(literal: str) -> str | None:
    try:
        return json.loads(literal)
    except ValueError:  # (a bad escape)
        return None


def _read_title(path: Path) -> str | None:
    try:
        data = _read(path)
    except (OSError, FormatError):
        return None
    return _title(data) if isinstance(data, dict) else None


def _title(data: dict) -> str | None:
    """A file's title, if it has a usable one."""
    try:
        return check_name(data["title"])
    except (KeyError, TypeError, AttributeError, ValueError):
        return None


# ---- helpers ----------------------------------------------------------------------


def _encode_end(
    ref: EndRef, at: Point | None, snap: Snapshot, types: Registry | None
) -> dict[str, Any]:
    if ref[0] == "p":
        _, part, is_input, index = ref
        kind = snap.parts[part][0]
        if kind.startswith(MACRO):
            t = types.get(kind)
            return {"part": part, "pin": (t.in_ids if is_input else t.out_ids)[index]}
        return {"part": part, "in" if is_input else "out": index}
    return {"wire": ref[1], "at": _point(at)}


def _decode_end(
    d: Any, wire_uid: int, kinds: dict[int, Any], snap: Snapshot
) -> tuple[EndRef | None, Point | None]:
    """(ref, junction point); ref is None when the thing it points at is gone."""
    if not isinstance(d, dict):
        raise TypeError("a wire end isn't an object")
    if "wire" in d:
        target = _int(d["wire"])
        if target > wire_uid:
            raise ValueError("attaches to a newer wire")
        if target == wire_uid:  # itself: a free end
            return ("w", target), _pair(d["at"])
        if target not in snap.wires:
            return None, None
        return ("w", target), _pair(d["at"])
    part = _int(d["part"])
    t = kinds.get(part)
    if t is None:
        return None, None
    if "pin" in d:  # a macro's pin, by the uid of its port
        pin = _int(d["pin"])
        in_ids, out_ids = getattr(t, "in_ids", ()), getattr(t, "out_ids", ())
        if pin in in_ids:
            return ("p", part, True, in_ids.index(pin)), None
        if pin in out_ids:
            return ("p", part, False, out_ids.index(pin)), None
        return None, None  # the macro doesn't have that pin (any more)
    is_input = "in" in d
    index = _int(d["in"] if is_input else d["out"])
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
    if not (
        isinstance(v, list)
        and len(v) == 2
        and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v)
    ) or not all(map(math.isfinite, v)):
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
