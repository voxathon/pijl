"""The launcher: what `pijl` shows before anything starts. It's where projects are
picked, made and renamed, where the preferences (prefs.py) are changed and mods
(mods.py) turned on and off and ordered, and it starts either the editor or a
macro run headless -- with or without mods ("safe start", which is on by itself
after a start that died while loading mods).

There are two faces on one model (Launcher): a window (ui/launcher.py) and a text
menu in the terminal (terminal() below, for `pijl --tui`, or when no window can
be opened). Either one ends with what to start, as `pijl` arguments
(["gui", "-p", "default"], ["run", "adder", "-", ...]), which cli.main then
runs as if they'd been typed; None means quit.

No pyglet in here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import TextIO

from . import mods, prefs
from .project import (
    DEFAULT_PROJECT,
    Project,
    last_project,
    project_names,
    projects_dir,
    remember_project,
    rename_project,
)
from .storage import MacroStore, check_name


def version() -> str:
    from . import __version__

    return __version__


class Launcher:
    """What both launchers show and change. Every change is saved right away."""

    def __init__(self) -> None:
        self.prefs, self.problems = prefs.load()
        if not project_names():
            Project.open(DEFAULT_PROJECT)  # first run: there's always one to start
        self.project = last_project()
        self.safe = mods.safe_mode()  # start without mods (--safe)
        self.crashed = mods.crashed()
        if self.crashed and not self.safe:
            self.safe = True
            self.problems.append(
                "the last start died while loading mods: safe start is on (no mods)"
            )
        mods.settled()  # (asked about; the next start makes a new marker)
        try:
            self.new_mods = {m.name for m in mods.plan().new}  # (now in disabled.txt)
        except OSError as e:
            self.new_mods = set()
            self.problems.append(f"can't read the mods folder: {e}")

    # ---- projects ----------------------------------------------------------------

    def projects(self) -> list[str]:
        return project_names()

    def pick(self, name: str) -> None:
        if name not in project_names():
            raise ValueError(f"there's no project {name!r}")
        self.project = name
        remember_project(name)

    def new_project(self, text: str) -> str:
        name = check_name(text)
        if (
            any(n.casefold() == name.casefold() for n in project_names())
            or (projects_dir() / name).exists()
        ):
            raise ValueError(f"there's already a project called {name!r}")
        Project.open(name)
        self.pick(name)
        return name

    def rename(self, old: str, text: str) -> str:
        new = rename_project(old, text)
        if self.project == old:
            self.project = new
        return new

    def macros(self) -> dict[str, str]:
        """id -> title of the picked project's macros, in title order."""
        return MacroStore(projects_dir() / self.project / "macros").titles()

    # ---- preferences -------------------------------------------------------------

    def set_pref(self, key: str, value) -> None:
        self.prefs[key] = prefs.PREFS[key].parse(value)
        prefs.save(self.prefs)

    def reset_prefs(self) -> None:
        self.prefs = prefs.defaults()
        prefs.save(self.prefs)

    # ---- mods --------------------------------------------------------------------

    def mods(self) -> tuple[list[ModRow], list[ModRow]]:
        """The enabled mods in load order (missing ones too), and the disabled ones."""
        p = mods.plan()
        notes = mods.notes(p)

        def row(m: mods.Mod, on: bool) -> ModRow:
            key = m.name.casefold()
            return ModRow(m.name, on, m, m.name in self.new_mods, notes.get(key, []))

        on, keys = [], {m.name.casefold() for m in p.enabled}
        for name in mods._dedup(mods.read_list(p.folder / mods.LOADORDER)):
            key = name.casefold()
            if key in keys:
                on.append(row(p.mods[key], True))
            elif name in p.missing:
                on.append(ModRow(name, True, None, False, ["not in the mods folder"]))
        return on, [row(m, False) for m in p.disabled]

    def mod_on(self, name: str) -> None:
        mods.enable(name)

    def mod_off(self, name: str) -> None:
        mods.disable(name)

    def mod_move(self, name: str, by: int) -> None:
        mods.move(name, by)

    def mods_folder(self) -> str:
        return str(mods.mods_dir())

    # ---- what to start -----------------------------------------------------------

    def _flags(self) -> list[str]:
        return ["--safe"] if self.safe else []

    def editor(self) -> list[str]:
        return ["gui", "-p", self.project, *self._flags()]

    def headless(self, macro: str, table: bool = False) -> list[str]:
        """`macro`: an id. Interactive: a stream on stdin; table: the truth table."""
        return ["run", macro, "--table" if table else "-", "-p", self.project, *self._flags()]


@dataclass
class ModRow:
    """A mod as the launcher lists it."""

    name: str
    on: bool
    mod: mods.Mod | None  # None: listed in loadorder.txt, not on disk
    new: bool
    notes: list[str]  # what's wrong with it

    @property
    def title(self) -> str:
        return self.mod.title if self.mod else self.name

    @property
    def description(self) -> str:
        return str(self.mod.manifest.get("description", "")) if self.mod else ""

    @property
    def tags(self) -> str:
        """(new), (missing), (!): shown after the title."""
        if self.mod is None:
            return "(missing)"
        return " ".join(t for t in ("(new)" * self.new, "(!)" * bool(self.notes)) if t)


STREAM_HELP = (
    "Type input values, one line at a time: a=1 b=0, or all of them as bits (10).\n"
    "'step N' lets N ticks pass. End with Ctrl+Z, Enter (Ctrl+D outside Windows).\n"
)


# ---- starting pijl again -----------------------------------------------------------


def own_command(console: bool = False) -> list[str]:
    """How to run this pijl again, as a fresh process. `console`: one that can talk
    in a console (not pythonw, which pijl-gui runs on)."""
    if "__compiled__" in globals():  # a Nuitka build: the program itself
        return [sys.argv[0]]
    python = sys.executable
    base = os.path.basename(python).lower()
    if console and base.startswith("pythonw"):
        console = os.path.join(
            os.path.dirname(python), base.replace("pythonw", "python", 1)
        )
        if os.path.exists(console):
            python = console
    return [python, "-m", "pijl"]


def has_console() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def open_console(args: list[str]) -> str | None:
    """Start `pijl args` in a console window of its own, which stays open after it
    ends. Returns why it couldn't, or None."""
    if sys.platform != "win32":
        return "start pijl from a terminal to run macros headless"
    cmd = subprocess.list2cmdline(own_command(console=True) + args)
    try:
        subprocess.Popen(f'cmd /k "{cmd}"', creationflags=subprocess.CREATE_NEW_CONSOLE)
    except OSError as e:
        return f"couldn't open a console: {e}"
    return None


def relaunch() -> int:
    """Replace this pijl with a fresh one showing the launcher. Returns the exit
    status to end with (where the process can't simply be replaced)."""
    cmd = own_command()
    if sys.platform != "win32":
        os.execv(cmd[0], cmd)
    if has_console():
        # (Windows has no real exec: wait for it, so the terminal waits too)
        return subprocess.call(cmd)
    subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS)
    return 0


# ---- the terminal launcher ---------------------------------------------------------


def terminal(
    launcher: Launcher, inp: TextIO | None = None, out: TextIO | None = None
) -> list[str] | None:
    """The launcher as a text menu. Returns what to start, or None to quit."""
    inp = inp or sys.stdin
    out = out or sys.stdout

    def say(text: str = "") -> None:
        print(text, file=out, flush=True)

    def ask(prompt: str) -> str | None:
        print(prompt, end="", file=out, flush=True)
        line = inp.readline()
        return None if not line else line.strip()

    for problem in launcher.problems:
        say(f"warning: {problem}")
    while True:
        say()
        say(f"pijl {version()}".rstrip())
        say(f"project: {launcher.project}")
        if launcher.safe:
            say("safe start: no mods")
        say()
        say("  Enter  start the editor")
        say("  l      load project      n  new project")
        say("  h      headless          s  settings")
        say("  m      mods              q  quit")
        answer = ask("> ")
        if answer is None or answer.lower() in ("q", "quit", "exit"):
            return None
        choice = answer.lower()
        if choice == "":
            return launcher.editor()
        if choice == "l":
            _load_page(launcher, say, ask)
        elif choice == "n":
            name = ask("new project's name (Enter: cancel): ")
            if name:
                try:
                    launcher.new_project(name)
                except (OSError, ValueError) as e:
                    say(f"can't: {e}")
        elif choice == "h":
            start = _headless_page(launcher, say, ask)
            if start is not None:
                if start[2] == "-":
                    say()
                    say(STREAM_HELP)
                return start
        elif choice == "s":
            _settings_page(launcher, say, ask)
        elif choice == "m":
            _mods_page(launcher, say, ask)
        else:
            say(f"{answer!r}? pick one of the letters")


def _number(text: str, count: int) -> int | None:
    if text.isdigit() and 1 <= int(text) <= count:
        return int(text) - 1
    return None


def _load_page(launcher: Launcher, say, ask) -> None:
    while True:
        names = launcher.projects()
        say()
        for i, name in enumerate(names, 1):
            say(f"  {i:>2}{'*' if name == launcher.project else ' '} {name}")
        answer = ask("number: open it, r + number: rename it, Enter: back > ")
        if not answer:
            return
        rename = answer[:1].lower() == "r"
        i = _number(answer[1:].strip() if rename else answer, len(names))
        if i is None:
            say(f"{answer!r}? no such project")
            continue
        try:
            if rename:
                text = ask(f"rename {names[i]!r} to (Enter: cancel): ")
                if text:
                    launcher.rename(names[i], text)
            else:
                launcher.pick(names[i])
                return
        except (OSError, ValueError) as e:
            say(f"can't: {e}")


def _settings_page(launcher: Launcher, say, ask) -> None:
    keys = list(prefs.PREFS)
    while True:
        say()
        n = 0
        for heading, group in prefs.SECTIONS:
            say(f"  {heading}")
            for key, setting in group.items():
                n += 1
                value = setting.show(launcher.prefs[key])
                say(f"  {n:>2}  {setting.title(key):<22} {value}")
        answer = ask("number: change it, d: all to defaults, Enter: back > ")
        if not answer:
            return
        if answer.lower() == "d":
            launcher.reset_prefs()
            continue
        i = _number(answer, len(keys))
        if i is None:
            say(f"{answer!r}? no such setting")
            continue
        key = keys[i]
        setting = prefs.PREFS[key]
        if setting.hint:
            say(f"      {setting.hint}")
        if isinstance(setting, prefs.Choice):
            say(f"      one of: {', '.join(str(v) for v in setting.values)}")
        elif isinstance(setting, prefs.Number):
            say(f"      {setting.show(setting.min)} to {setting.show(setting.max)}")
        text = ask(
            f"{setting.title(key)} (Enter: keep {setting.show(launcher.prefs[key])}): "
        )
        if text:
            try:
                launcher.set_pref(key, prefs.parse_text(key, text))
            except (OSError, ValueError) as e:
                say(f"can't: {e}")


def _mods_page(launcher: Launcher, say, ask) -> None:
    while True:
        try:
            on, off = launcher.mods()
        except OSError as e:
            say(f"can't read the mods folder: {e}")
            return
        rows = on + off
        say()
        say(f"  mods in {launcher.mods_folder()}")
        say(f"  safe start (no mods): {'on' if launcher.safe else 'off'}")
        for i, r in enumerate(rows, 1):
            if i == 1 and on:
                say("  LOAD ORDER")
            if i == len(on) + 1:
                say("  DISABLED")
            line = f"  {i:>2}  {r.title} {r.tags}".rstrip()
            if r.description:
                line += f"  - {r.description}"
            say(line[:100])
            for note in r.notes:
                say(f"        {note}")
        if not rows:
            say("  (none: put a mod's .py file or folder in there)")
        answer = ask("number: on/off, u/d + number: move up/down, s: safe start, Enter: back > ")
        if not answer:
            return
        a = answer.lower()
        if a == "s":
            launcher.safe = not launcher.safe
            continue
        by = {"u": -1, "d": 1}.get(a[:1], 0)
        i = _number(a[1:].strip() if by else a, len(rows))
        if i is None:
            say(f"{answer!r}? no such mod")
            continue
        r = rows[i]
        try:
            if by:
                if not r.on:
                    say(f"{r.name} isn't on")
                else:
                    launcher.mod_move(r.name, by)
            elif r.on:
                launcher.mod_off(r.name)
            else:
                launcher.mod_on(r.name)
        except (OSError, ValueError) as e:
            say(f"can't: {e}")


def _headless_page(launcher: Launcher, say, ask) -> list[str] | None:
    macros = list(launcher.macros().items())
    if not macros:
        say(f"there are no macros in {launcher.project!r} yet")
        return None
    say()
    for i, (_, title) in enumerate(macros, 1):
        say(f"  {i:>2}  {title}")
    while True:
        answer = ask(
            "number: run it interactively, t + number: truth table, Enter: back > "
        )
        if not answer:
            return None
        table = answer[:1].lower() == "t"
        i = _number(answer[1:].strip() if table else answer, len(macros))
        if i is None:
            say(f"{answer!r}? no such macro")
            continue
        return launcher.headless(macros[i][0], table)
