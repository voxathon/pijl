"""Part types: what kinds of parts exist and how each one behaves.

Everything except IN/OUT (see ports.py) comes from part scripts. The ones that
ship with pijl live in templates/ and are copied into every new project; the
project's copy is what gets loaded (see project.py). No pyglet in here.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .contract import API, Ctx, Look, PartType, part
from .registry import Registry, fresh_props, load

TEMPLATES = Path(__file__).parent / "templates"


@cache
def builtin_registry() -> Registry:
    """The ports plus the shipped templates, loaded straight from the package.
    For tests and anything else that has no project."""
    return load(TEMPLATES)


__all__ = ["API", "Ctx", "Look", "PartType", "Registry", "TEMPLATES", "builtin_registry",
           "fresh_props", "load", "part"]
