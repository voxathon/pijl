# Modding pijl

A mod is a Python module that pijl imports at startup. There is no mod API: a mod
monkey-patches whatever it wants. pijl does two things for it: it loads the mods
in the order you choose, and `after_import` gives a patch the right moment to run.

Mods run with full access to your machine, like any Python script. Only install
mods you trust.

## Where mods go

```
<data root>/mods/          %APPDATA%\pijl\mods on Windows (PIJL_DATA moves the root,
    loadorder.txt          PIJL_MODS moves just this folder)
    disabled.txt
    foo.py                 a single-file mod called foo
    bar/__init__.py        a folder mod called bar
    bar/mod.json           its manifest (optional)
    bar/lib/               optional: added to sys.path (vendored dependencies)
```

A mod's name is its file or folder name, and it must be a valid Python identifier
(`my_mod`, not `my-mod`). Names that start with `_` or `.`, and folders without an
`__init__.py`, are not mods.

## Turning mods on and off

`loadorder.txt` lists the mods that load, in that order. `disabled.txt` lists the
ones that don't. Both have one name per line. Line endings can be LF, CRLF or any
mix, blank lines and `#` comments are fine, and case doesn't matter.

- A mod pijl hasn't seen before is added to the end of `disabled.txt`. To turn it
  on, move its line to `loadorder.txt`.
- A name in both files is disabled.
- A name with nothing on disk gets a warning and stays in the list. Loading never
  changes the lists, except to add new mods to the end of `disabled.txt`.

You can do all of this from the launcher's MODS page (`m` in `pijl --tui`): turn
mods on and off, move them up and down the load order, see what's wrong with each
one, and switch safe start on or off. When the page rewrites a list, comment and
blank lines stay where they are.

## Bundled mods

Some mods ship with pijl. The first time pijl sees the mods folder without one,
it copies the mod in. It's new, so it starts out disabled, like any other. If you
delete it while its name is still in one of the lists, it stays deleted. When a
pijl update brings a newer version (the `version` in its `mod.json`), the shipped
files are copied over the old ones, and anything else in its folder stays.

- **manuscript** writes pijl's log to files: one per run in `manuscript/logs/`, with
  starts, loaded mods, projects, opens, saves, deletes and renames, every edit,
  undo and redo, problems, and uncaught exceptions with their tracebacks. Its
  settings are in `manuscript/config.json`, which it writes with the defaults on
  its first run: `levels` (logger name -> `debug`, `info`, `warning`, `error` or
  `off`), `keep` (how many log files), `crashes` and `stderr` (also copy
  everything printed to stderr).
- **displays** ("Displays & Stuff") adds a DISPLAYS collection to the part picker:
  `7SEG`, a seven-segment digit with a pin per segment (a..g, clockwise from the
  top, g in the middle, and dp); `HEX`, a digit that decodes four bits (8, 4, 2,
  1) into 0..F; and `BAR`, eight LEDs in a column. Recolor them from the
  right-click menu. Boards that use them need the mod to open with the displays in
  place. The mod draws nothing itself: its parts use `Look.size`, `Look.face` and
  the `face()` hook from the part contract (`pijl/parts/contract.py`), which part
  scripts can use too.

## Logging

pijl logs to Python's `logging`, under `pijl.app`, `pijl.mods`, `pijl.files`,
`pijl.edit`, `pijl.ui` and `pijl.sim`, and attaches no handlers: without a mod
that listens, nothing is written anywhere. To log from your own mod, use
`logging.getLogger(__name__)` (that's `pijl_mods.<your mod>`). manuscript picks it up.

## Manifest

The manifest is optional. Write it as `mod.json` in a folder mod, or as
`MOD = {...}` at the top level of the entry point. pijl reads it without running
the mod, so it must be a literal:

```python
MOD = {
    "name": "Fancy wires",
    "version": "1.2",
    "description": "...",
    "author": "...",
    "requires": ["other_mod"],     # warns if it isn't enabled earlier in the order
    "pijl": ">=0.2,<0.3",          # warns if this pijl isn't in range
}
```

`requires` and `pijl` only produce warnings. pijl never reorders your mods.

## Loading

Every command except `launch` and `prefs` loads the enabled mods first: the
editor, `run`, `list` and `bench`. `bench` prints which mods were loaded.
`--safe` (or `PIJL_SAFE=1`) starts pijl without mods. pijl processes started from
pijl, such as bench's pipe children, get the same mods (or none, with `--safe`).

Mods are imported as `pijl_mods.<name>`, so one mod can
`from pijl_mods.other import thing`. If a mod raises while it's imported, it's
reported and skipped, and the next mod still loads. Any patches it made before
failing stay in place. Problems show up on stderr and in the editor's status line.

If pijl dies while starting with mods enabled, the next launcher comes up with
safe start on and says why. Turn it off on the MODS page once the mod is fixed.
Without the launcher (`pijl gui`, `pijl run`), you get a warning instead, and
`--safe` is the way back in.

## Mods in save files

When you save a macro, pijl writes the mods that were loaded into the file
(`"mods": ["foo@1.2", "bar"]`) and into the project's `project.json`. With no
mods loaded, nothing is written and the files look as they did before. When you
open a macro or a project saved with a mod that isn't loaded now, or with
another version of it, you get a warning. Nothing is refused.

## Patching

```python
from pijl.mods import after_import

@after_import("pijl.ui.editor")
def _(m):
    old = m.Editor.on_draw

    def on_draw(self):
        old(self)
        ...  # draw your thing

    m.Editor.on_draw = on_draw
```

`after_import(name)` calls your function once `name` has been imported, or
immediately if it already has been. Mods load before the editor is imported, so
hooking `pijl.ui.*` modules always works. If your hook raises, the error is
reported against your mod and pijl carries on.

**Patch methods on classes, not module-level functions.** When a module does
`from .parts import load as load_parts`, it keeps its own reference to `load`, so
replacing `pijl.parts.load` later doesn't reach it. A class attribute is looked up
on every call, so a patched method reaches every caller.

Many UI modules copy theme values when they're imported (`S = T.UI_SCALE`). To
change those, patch `pijl.ui.theme` in an `after_import("pijl.ui.theme")` hook.
That hook runs before the modules that copy the values are imported.

## Patching other mods

Mods are patched the same way as pijl itself. How you reach the other mod depends
on where it is in the load order.

A mod that loads **earlier** has already been imported, so import it and patch it:

```python
import pijl_mods.other

old = pijl_mods.other.Thing.run

def run(self):
    ...
    return old(self)

pijl_mods.other.Thing.run = run
```

A mod that loads **later** hasn't been imported yet. Hook it with `after_import`.
The hook runs right after that mod's own code, before the next mod loads:

```python
@after_import("pijl_mods.later")
def _(m):
    m.Thing.run = ...
```

Don't `import pijl_mods.later` from an earlier mod. The import runs the later mod
on the spot, ahead of its place in the load order, and any errors from its
`after_import` hooks are reported against your mod.

When several mods wrap the same method, the last one in the load order wraps all
the others, so its code runs first. The load order is the only thing that decides
this.

If the other mod is disabled, missing or fails to import, an `after_import` hook
on it never runs, and nothing reports it. For a mod that loads earlier, put it in
`requires` and you get a warning when it's not enabled. Don't do that for a mod
that loads later: `requires` would warn that it loads after yours. The rule about
module-level functions applies here too: patch the class, not a name another
module has already copied.

## Adding part types from a mod

You can, but it's discouraged: projects then depend on a mod that isn't saved with
them. Prefer part scripts in the project's `parts/` folder. If you do it anyway,
add your parts after the project's own scripts, so a part the project defines
wins over yours:

```python
@after_import("pijl.parts.registry")
def _(m):
    old = m.Registry.load_folder

    def load_folder(self, root):
        old(self, root)
        try:
            self.add(MyPart)
        except ValueError:
            pass  # the project already has a part with that name

    m.Registry.load_folder = load_folder
```

## Pin layouts (buses, SPLIT)

A part type's pins are plain data. `PartType.widths` is a dict (`{"d": 8}`, or
`{"d": "width"}` to read an instance's prop), and each registered type gets its
own copy, so editing one type's dict in place doesn't touch any other type. The
engine reads an instance's pins only through `PartType.layout(props)`. That
returns a dict with `"ins"`, `"outs"`, `"widths"` and `"joins"`; any key left out
means the class attribute. Override it to give a part pins that depend on its
settings, as SPLIT does (`parts/templates/wiring/split.py`). Joins may name lane
ranges: `"bus[0:4]"`. Three things to know:

- Pins are read when an instance is made. Patching a type afterwards doesn't
  reshape instances that already exist. A settings edit in the editor that gives
  a part other pins rebuilds it.
- The registry checks the layout an instance with the default props gets. If
  `layout()` later raises or returns something impossible (a width outside 1 to
  64, a join of unknown or unequally wide lanes), that kind is disabled in the
  circuit like a hook that raised, and gets its class attribute pins, one lane each.
- A part with an `eval` joins whole outputs only (its eval value drives them). A
  part without one, like SPLIT, may join any lanes.
