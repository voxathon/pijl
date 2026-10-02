"""Where saved data lives. Pure filesystem logic, no pyglet.

Everything the user makes is grouped into *projects*, each a folder under the
data root. "default" is created on first run; more can be made from the status
bar's cogwheel menu. The one open last is reopened on start (settings.json).

    <data root>/                  %APPDATA%\\pijl on Windows (PIJL_DATA overrides)
      settings.json               app-wide: which project was open last
      .trashbin/<project>/        deleted macros (not Shift+deleted ones), to fish back by hand
      projects/
        default/
          project.json            marks the folder as a project; format version
          library.json            picker collections (later)
          macros/                 one .json file per macro (later)
          parts/                  part scripts: a copy of the shipped templates, made when
                                  the project is created; the project's own from then on

The data root is outside the program's own folder on purpose: a Nuitka onefile
build unpacks to a temp folder that's deleted on exit, and `uv run` works from
the source tree. Neither is a place to keep user data.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .parts import TEMPLATES

APP = "pijl"
DEFAULT_PROJECT = "default"
FORMAT = 1  # bump when the project layout changes; see Project.open


def data_root() -> Path:
    """The per-user folder all of pijl's saved data goes in."""
    if override := os.environ.get("PIJL_DATA"):
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP


def projects_dir() -> Path:
    return data_root() / "projects"


def project_names() -> list[str]:
    """Every project on disk (a folder under projects/ with a project.json), by name."""
    try:
        found = [
            p.name for p in projects_dir().iterdir() if (p / "project.json").is_file()
        ]
    except OSError:
        return []
    return sorted(found, key=str.casefold)


def _settings_file() -> Path:
    return data_root() / "settings.json"


def last_project() -> str:
    """The project that was open last time (DEFAULT_PROJECT if none, or it's gone)."""
    try:
        name = json.loads(_settings_file().read_text(encoding="utf-8")).get("project")
    except (OSError, ValueError, AttributeError):
        return DEFAULT_PROJECT
    if isinstance(name, str) and (projects_dir() / name / "project.json").is_file():
        return name
    return DEFAULT_PROJECT


def remember_project(name: str) -> None:
    try:
        settings = json.loads(_settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    if not isinstance(settings, dict):
        settings = {}
    if settings.get("project") != name:
        settings["project"] = name
        _settings_file().parent.mkdir(parents=True, exist_ok=True)
        write_atomic(_settings_file(), json.dumps(settings, indent=2) + "\n")


def rename_project(old: str, new: str) -> str:
    """Rename a project's folder (and its trash folder), keeping it the last one open
    if it was. Returns the new name; ValueError if it can't be called that."""
    from .storage import check_name

    new = check_name(new)
    src, dst = projects_dir() / old, projects_dir() / new
    if not (src / "project.json").is_file():
        raise ValueError(f"there's no project {old!r}")
    if new == old:
        return new
    if new.casefold() != old.casefold() and dst.exists():
        raise ValueError(f"there's already a project called {new!r}")
    was_last = last_project() == old
    # (via a temporary name, so changing only the case works on Windows too)
    def move(src: Path, dst: Path) -> None:
        tmp = src.with_name(src.name + ".renaming")
        os.replace(src, tmp)
        os.replace(tmp, dst)

    move(src, dst)
    trash = data_root() / ".trashbin"
    if (trash / old).is_dir() and (
        new.casefold() == old.casefold() or not (trash / new).exists()
    ):
        move(trash / old, trash / new)
    if was_last:
        remember_project(new)
    return new


@dataclass(frozen=True)
class Project:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def macros_dir(self) -> Path:
        return self.path / "macros"

    @property
    def parts_dir(self) -> Path:
        return self.path / "parts"

    @property
    def library_file(self) -> Path:
        return self.path / "library.json"

    @property
    def trash_dir(self) -> Path:
        """Where this project's deleted macros go (see MacroStore.remove)."""
        return self.path.parent.parent / ".trashbin" / self.name

    @property
    def meta_file(self) -> Path:
        return self.path / "project.json"

    def last_open(self) -> str | None:
        """The macro that was open when pijl last saved or opened one (reopened on start)."""
        try:
            value = json.loads(self.meta_file.read_text(encoding="utf-8")).get("open")
        except (OSError, ValueError):
            return None
        return value if isinstance(value, str) else None

    def remember_open(self, name: str | None) -> None:
        meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        if meta.get("open") != name:
            meta["open"] = name
            write_atomic(self.meta_file, json.dumps(meta, indent=2) + "\n")

    @classmethod
    def open(cls, name: str = DEFAULT_PROJECT) -> Project:
        """Open the named project, creating it (and the data root) if it doesn't exist yet."""
        project = cls(projects_dir() / name)
        project.macros_dir.mkdir(parents=True, exist_ok=True)
        if project.meta_file.exists():
            version = json.loads(project.meta_file.read_text(encoding="utf-8")).get(
                "pijl"
            )
            if not isinstance(version, int) or version > FORMAT:
                raise ValueError(
                    f"project {name!r} has format {version!r}; this pijl reads up to {FORMAT}"
                )
        else:
            write_atomic(
                project.meta_file, json.dumps({"pijl": FORMAT}, indent=2) + "\n"
            )
        if not project.parts_dir.exists():
            # The project gets its own copy of the built-in parts, so it keeps behaving
            # the same whatever later pijl versions ship (and can edit them).
            shutil.copytree(
                TEMPLATES,
                project.parts_dir,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        return project


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a crash mid-save never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
