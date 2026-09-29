"""Where saved data lives. Pure filesystem logic, no pyglet.

Everything the user makes is grouped into *projects*, each a folder under the
data root. For now there's only ever one, "default", created on first run.

    <data root>/                  %APPDATA%\\pijl on Windows (PIJL_DATA overrides)
      projects/
        default/
          project.json            marks the folder as a project; format version
          library.json            picker collections (later)
          macros/                 one .json file per macro (later)

The data root is outside the program's own folder on purpose: a Nuitka onefile
build unpacks to a temp folder that's deleted on exit, and `uv run` works from
the source tree. Neither is a place to keep user data.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
    def library_file(self) -> Path:
        return self.path / "library.json"

    @property
    def meta_file(self) -> Path:
        return self.path / "project.json"

    @classmethod
    def open(cls, name: str = DEFAULT_PROJECT) -> Project:
        """Open the named project, creating it (and the data root) if it doesn't exist yet."""
        project = cls(projects_dir() / name)
        project.macros_dir.mkdir(parents=True, exist_ok=True)
        if project.meta_file.exists():
            version = json.loads(project.meta_file.read_text(encoding="utf-8")).get("pijl")
            if not isinstance(version, int) or version > FORMAT:
                raise ValueError(f"project {name!r} has format {version!r}; this pijl reads up to {FORMAT}")
        else:
            write_atomic(project.meta_file, json.dumps({"pijl": FORMAT}, indent=2) + "\n")
        return project


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a crash mid-save never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
