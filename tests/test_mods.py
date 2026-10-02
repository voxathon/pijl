"""Mods (mods.py): the two lists, discovery, manifests, loading, after_import, the
crash marker, --safe, and pijl's own version."""

import os
import sys
import tomllib
import types
from pathlib import Path

import pytest

import pijl
from pijl import mods
from pijl.cli import main
from pijl.project import Project, data_root


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("PIJL_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("PIJL_MODS", "")
    monkeypatch.setenv("PIJL_SAFE", "")
    monkeypatch.setenv("PIJL_ENGINE", "")
    trace = types.ModuleType("modtrace")
    trace.seen = []
    monkeypatch.setitem(sys.modules, "modtrace", trace)
    yield
    mods._report = None
    mods._hooks.clear()
    mods._tracebacks.clear()
    mods._loading = False
    for name in [n for n in sys.modules if n == mods.PACKAGE or n.startswith(mods.PACKAGE + ".")]:
        del sys.modules[name]
    if mods._finder in sys.meta_path:
        sys.meta_path.remove(mods._finder)


def folder() -> Path:
    f = data_root() / "mods"
    f.mkdir(parents=True, exist_ok=True)
    return f


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())
    return path


def traced(name: str, extra: str = "") -> str:
    return f"import modtrace\nmodtrace.seen.append({name!r})\n{extra}"


def enable(*names: str) -> None:
    write(folder() / mods.LOADORDER, "\n".join(names) + "\n")


def seen() -> list[str]:
    return sys.modules["modtrace"].seen


# ---- the lists ---------------------------------------------------------------------


def test_lists_take_any_line_endings_a_bom_comments_and_blanks(tmp_path):
    p = tmp_path / "l.txt"
    p.write_bytes(b"\xef\xbb\xbf# order\r\nfoo\r\n\r\n  bar  \nbaz\rqux")
    assert mods.read_list(p) == ["foo", "bar", "baz", "qux"]
    assert mods.read_list(tmp_path / "none.txt") == []


def test_new_mods_are_appended_to_disabled_and_the_lists_are_made():
    write(folder() / "b.py", "")
    write(folder() / "a" / "__init__.py", "")
    plan = mods.plan()
    assert [m.name for m in plan.new] == ["a", "b"]
    assert plan.enabled == []
    assert mods.read_list(folder() / mods.DISABLED) == ["a", "b"]
    assert (folder() / mods.LOADORDER).is_file()
    assert mods.plan().new == []  # (now they're listed)


def test_appending_keeps_the_file_byte_for_byte_and_its_line_endings():
    write(folder() / mods.DISABLED, "# mine\r\nold")  # (no newline at the end)
    write(folder() / "new.py", "")
    mods.plan()
    assert (folder() / mods.DISABLED).read_bytes() == b"# mine\r\nold\r\nnew\r\n"


def test_order_is_loadorder_names_match_without_case_and_disabled_wins():
    for n in ("one", "two", "three"):
        write(folder() / f"{n}.py", "")
    write(folder() / mods.LOADORDER, "THREE\r\none\nTwo\none\ngone\n")
    write(folder() / mods.DISABLED, "two\n")
    plan = mods.plan()
    assert [m.name for m in plan.enabled] == ["three", "one"]
    assert [m.name for m in plan.disabled] == ["two"]
    assert plan.missing == ["gone"]
    assert [m.name for m in plan.both] == ["two"]
    assert "gone" in (folder() / mods.LOADORDER).read_text()  # (never removed)


def test_what_isnt_a_mod():
    write(folder() / "_helper.py", "")
    write(folder() / ".hidden.py", "")
    write(folder() / "notes.txt", "")
    write(folder() / "stuff" / "main.py", "")  # (no __init__.py)
    write(folder() / "bad-name.py", "")
    found = mods.discover(folder())
    assert list(found) == ["bad-name"]
    assert found["bad-name"].problems  # (listed, so it can be seen; can't load)


def test_a_folder_beats_a_script_of_the_same_name():
    write(folder() / "dup.py", "")
    write(folder() / "dup" / "__init__.py", "")
    m = mods.discover(folder())["dup"]
    assert m.path.is_dir() and m.usable and m.warnings


# ---- manifests ---------------------------------------------------------------------


def test_manifest_from_json_or_from_the_entry_point_without_running_it():
    write(folder() / "j" / "__init__.py", "raise SystemExit")
    write(folder() / "j" / "mod.json", '{"name": "Jay", "version": "1.2"}')
    write(folder() / "p.py", 'MOD = {"version": "0.1", "requires": ["j"]}\nraise SystemExit\n')
    found = mods.discover(folder())
    assert found["j"].manifest == {"name": "Jay", "version": "1.2"}
    assert found["j"].title == "j 1.2"
    assert found["p"].manifest["requires"] == ["j"]


def test_a_bad_manifest_is_a_warning():
    write(folder() / "j" / "__init__.py", "")
    write(folder() / "j" / "mod.json", "{nope")
    write(folder() / "p.py", "MOD = {'v': len('x')}\n")
    found = mods.discover(folder())
    assert found["j"].usable and found["j"].warnings
    assert found["p"].usable and found["p"].warnings


# ---- loading -----------------------------------------------------------------------


def test_mods_load_in_order_as_a_package_and_can_import_each_other():
    write(folder() / "base.py", traced("base", "VALUE = 41\n"))
    write(folder() / "top" / "__init__.py", traced("top", "from .inner import x\n"))
    write(folder() / "top" / "inner.py", "from pijl_mods.base import VALUE\nx = VALUE + 1\n")
    write(folder() / "off.py", traced("off"))
    enable("base", "top")
    write(folder() / mods.DISABLED, "off\n")
    rep = mods.load()
    assert seen() == ["base", "top"]
    assert sys.modules["pijl_mods.top"].x == 42
    assert [m.name for m in rep.loaded] == ["base", "top"]
    assert mods.active() == ["base", "top"]
    assert rep.summary() == "mods: base, top"
    assert mods.load() is rep  # (once per process)


def test_a_broken_mod_is_skipped_and_the_rest_still_load():
    write(folder() / "a.py", traced("a", "raise RuntimeError('boom')\n"))
    write(folder() / "b.py", "import sys\nsys.exit(3)\n")
    write(folder() / "c.py", traced("c", "MOD = {'version': '2'}\n"))
    enable("a", "b", "c")
    rep = mods.load()
    assert [m.name for m in rep.loaded] == ["c"]
    assert any("a: RuntimeError: boom" in p for p in rep.problems)
    assert any(p.startswith("b: SystemExit") for p in rep.problems)
    assert "boom" in mods.traceback_of("a")
    assert "pijl_mods.a" not in sys.modules
    assert mods.active() == ["c@2"]


def test_manifest_warnings_requires_and_pijl_version():
    write(folder() / "early.py", "MOD = {'requires': ['late', 'absent']}\n")
    write(folder() / "late.py", "MOD = {'pijl': '<0.1'}\n")
    write(folder() / "odd.py", "MOD = {'pijl': 'whenever'}\n")
    enable("early", "late", "odd")
    problems = mods.load().problems
    assert "early: needs late, which loads after it" in problems
    assert "early: needs absent, which isn't enabled" in problems
    assert f"late: made for pijl <0.1, this is {pijl.__version__}" in problems
    assert any(p.startswith("odd: can't read") for p in problems)


def test_a_mod_with_an_invalid_name_is_not_loaded():
    write(folder() / "bad-name.py", traced("bad"))
    enable("bad-name")
    rep = mods.load()
    assert rep.loaded == [] and seen() == []
    assert any("not loaded" in p for p in rep.problems)


def test_new_and_missing_mods_are_reported():
    write(folder() / "fresh.py", "")
    enable("gone")
    problems = mods.load().problems
    assert any("gone" in p for p in problems)
    assert any("new: fresh" in p for p in problems)


def test_a_folder_mods_lib_goes_on_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    write(folder() / "withlib" / "__init__.py", "import vendored\n")
    write(folder() / "withlib" / "lib" / "vendored.py", "")
    enable("withlib")
    assert [m.name for m in mods.load().loaded] == ["withlib"]
    sys.modules.pop("vendored", None)


def test_safe_mode_loads_nothing_and_doesnt_touch_the_folder(monkeypatch):
    write(folder() / "a.py", traced("a"))
    enable("a")
    monkeypatch.setenv("PIJL_SAFE", "1")
    rep = mods.load()
    assert rep.safe and rep.loaded == [] and seen() == []
    assert rep.summary() == "mods: off (--safe)"
    assert not (folder() / mods.DISABLED).exists()


def test_the_crash_marker():
    write(folder() / "a.py", "")
    enable("a")
    rep = mods.load()
    assert not rep.crashed
    marker = data_root() / mods.SENTINEL
    assert marker.is_file()  # (until pijl is up)
    mods._report = None
    del sys.modules["pijl_mods.a"]
    assert mods.load().crashed  # (it never got as far as settled())
    mods.settled()
    assert not marker.exists()


def test_mods_folder_override(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere"
    write(other / "x.py", traced("x"))
    write(other / mods.LOADORDER, "x\n")
    monkeypatch.setenv("PIJL_MODS", str(other))
    assert [m.name for m in mods.load().loaded] == ["x"]


# ---- after_import ------------------------------------------------------------------


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A module, not yet imported, that can be."""
    monkeypatch.setattr(sys, "path", [str(tmp_path / "lib"), *sys.path])
    write(tmp_path / "lib" / "fake_target.py", "def hello():\n    return 'hi'\n")
    yield "fake_target"
    sys.modules.pop("fake_target", None)


def test_after_import_runs_once_the_module_has_run(fake):
    calls = []

    @mods.after_import(fake)
    def patch(m):
        calls.append(m.hello())
        m.hello = lambda: "patched"

    assert calls == []
    import fake_target

    assert calls == ["hi"]
    assert fake_target.hello() == "patched"
    del sys.modules[fake]
    import fake_target

    assert calls == ["hi"]


def test_after_import_of_an_imported_module_runs_now():
    calls = []
    mods.after_import("pijl.project", lambda m: calls.append(m.__name__))
    assert calls == ["pijl.project"]


def test_after_import_keeps_the_real_loader_usable(fake):
    import inspect

    mods.after_import(fake, lambda m: None)
    import fake_target

    assert "return 'hi'" in inspect.getsource(fake_target)


def test_a_failing_hook_is_reported_against_its_mod_and_the_import_goes_on(fake):
    write(
        folder() / "hooky.py",
        f"from pijl.mods import after_import\n"
        f"@after_import({fake!r})\n"
        f"def _(m):\n    raise ValueError('nope')\n",
    )
    enable("hooky")
    mods.load()
    import fake_target

    assert fake_target.hello() == "hi"
    assert any(p.startswith("hooky: after_import") and "nope" in p for p in mods.report().problems)
    assert "nope" in mods.traceback_of("hooky")


# ---- through the command line ------------------------------------------------------


def test_commands_load_mods_and_safe_skips_them(capsys, monkeypatch):
    write(folder() / "loud.py", "import sys\nprint('loud is in', file=sys.stderr)\n")
    enable("loud")
    Project.open()
    assert main(["--safe", "list", "--projects"]) == 0
    assert "loud is in" not in capsys.readouterr().err
    assert os.environ["PIJL_SAFE"] == "1"  # (for what it starts)
    monkeypatch.setenv("PIJL_SAFE", "")
    mods._report = None
    assert main(["list", "--projects"]) == 0
    assert "loud is in" in capsys.readouterr().err
    assert not (data_root() / mods.SENTINEL).exists()  # (headless: up once loaded)


# ---- versions ----------------------------------------------------------------------


def test_version_matches_pyproject():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    assert pijl.__version__ == tomllib.loads(pyproject.read_text())["project"]["version"]


@pytest.mark.parametrize(
    ("version", "spec", "ok"),
    [
        ("0.2.3", ">=0.2", True),
        ("0.2.3", ">=0.2,<0.3", True),
        ("0.3", ">=0.2,<0.3", False),
        ("0.2", "0.2.0", True),
        ("0.2.3rc1", "==0.2.3", True),
        ("1.0", "!=1", False),
        ("0.2.3", "~=0.2", None),
        ("0.2.3", "soon", None),
    ],
)
def test_version_ranges(version, spec, ok):
    assert mods.version_matches(version, spec) is ok


# ---- mods recorded in saves --------------------------------------------------------


def pretend_loaded(*mods_: tuple[str, str], safe: bool = False) -> None:
    mods._report = mods.Report(
        safe=safe,
        loaded=[mods.Mod(n, Path(f"{n}.py"), {"version": v} if v else {}) for n, v in mods_],
    )


def board():
    from pijl.snapshot import Snapshot

    return Snapshot({1: ("NOT", "", 0, 0, {})}, {})


def test_a_macro_records_the_mods_loaded_when_it_was_saved():
    from pijl.engine import Engine
    from pijl.storage import MacroStore

    project = Project.open()
    store = MacroStore(project.macros_dir)
    store.save("plain", board())
    assert '"mods"' not in store.path("plain").read_text()  # (vanilla: unchanged format)
    pretend_loaded(("foo", "1.2"), ("bar", ""))
    store.save("modded", board())
    text = store.path("modded").read_text()
    assert '  "mods": ["foo@1.2", "bar"],\n' in text
    store.retitle("modded", "Modded")
    assert '"mods": ["foo@1.2", "bar"]' in store.path("modded").read_text()
    loaded = store.load("modded", Engine().catalog)
    assert loaded.mods == ["foo@1.2", "bar"] and loaded.warnings == []


def test_loading_warns_about_missing_mods_and_other_versions_first():
    from pijl.engine import Engine
    from pijl.storage import MacroStore

    project = Project.open()
    store = MacroStore(project.macros_dir)
    store.save("m", board(), mods=["foo@1.2", "bar@1", "baz"])
    pretend_loaded(("foo", "1.3"))
    warnings = store.load("m", Engine().catalog).warnings
    assert warnings == ["saved with mods bar 1, baz, not loaded", "saved with mod foo 1.2, this is 1.3"]
    pretend_loaded(safe=True)
    assert store.load("m", Engine().catalog).warnings[0].endswith("not loaded (--safe)")


def test_a_bad_mods_list_is_a_warning():
    from pijl.engine import Engine
    from pijl.storage import decode

    Project.open()
    loaded = decode({"pijl": 1, "mods": "foo", "parts": [], "wires": []}, Engine().catalog)
    assert loaded.mods == [] and loaded.warnings == ["unreadable list of mods, ignored"]


def test_the_project_records_its_mods_and_warns_on_open():
    from pijl.engine import Engine

    project = Project.open()
    pretend_loaded(("foo", "1.2"))
    project.remember_mods(mods.active())
    assert project.recorded_mods() == ["foo@1.2"]
    assert project.mod_problems() == []
    pretend_loaded()
    assert Engine().problems[0] == "project: saved with mod foo 1.2, not loaded"
    project.remember_mods([])
    assert '"mods"' not in project.meta_file.read_text()


# ---- turning mods on and off, ordering (the launcher's MODS page) ------------------


def test_write_list_keeps_comments_in_place_and_line_endings(tmp_path):
    p = write(tmp_path / "l.txt", "# top\r\na\r\n\r\n# mid\r\nb\r\nc\r\n")
    mods.write_list(p, ["c", "a"])
    assert p.read_bytes() == b"# top\r\nc\r\n\r\n# mid\r\na\r\n"
    mods.write_list(p, ["c", "a", "x", "y"])
    assert p.read_bytes() == b"# top\r\nc\r\n\r\n# mid\r\na\r\nx\r\ny\r\n"


def test_enable_disable_and_move():
    for n in ("a", "b", "c"):
        write(folder() / f"{n}.py", "")
    mods.plan()  # (all three new: disabled)
    mods.enable("b")
    mods.enable("A")  # (any case)
    mods.enable("c")
    assert [m.name for m in mods.plan().enabled] == ["b", "a", "c"]
    assert mods.read_list(folder() / mods.DISABLED) == []
    mods.move("c", -1)
    mods.move("b", 5)  # (clamped)
    assert [m.name for m in mods.plan().enabled] == ["c", "a", "b"]
    assert mods.read_list(folder() / mods.LOADORDER) == ["c", "a", "b"]  # (as on disk)
    mods.disable("c")
    p = mods.plan()
    assert [m.name for m in p.enabled] == ["a", "b"]
    assert [m.name for m in p.disabled] == ["c"]
    with pytest.raises(ValueError):
        mods.move("c", 1)


def test_disabling_a_missing_mod_forgets_it():
    enable("gone")
    mods.disable("gone")
    assert mods.read_list(folder() / mods.LOADORDER) == []
    assert mods.read_list(folder() / mods.DISABLED) == []


def test_notes_are_what_load_would_say():
    write(folder() / "a.py", "MOD = {'requires': ['b']}\n")
    write(folder() / "b.py", "")
    write(folder() / "bad-name.py", "")
    enable("a", "b")
    notes = mods.notes(mods.plan())
    assert notes["a"] == ["needs b, which loads after it"]
    assert notes["b"] == []
    assert notes["bad-name"]


def test_the_launcher_lists_mods_and_starts_with_or_without_them():
    from pijl.launcher import Launcher

    write(folder() / "a.py", "MOD = {'version': '1', 'description': 'does a'}\n")
    write(folder() / "b.py", "")
    enable("a", "gone")
    la = Launcher()
    assert la.new_mods == {"b"} and not la.safe
    on, off = la.mods()
    assert [(r.name, r.title, r.tags, r.description) for r in on] == [
        ("a", "a 1", "", "does a"),
        ("gone", "gone", "(missing)", ""),
    ]
    assert [(r.name, r.tags) for r in off] == [("b", "(new)")]
    la.mod_on("b")
    la.mod_move("b", -2)
    assert [r.name for r in la.mods()[0]] == ["b", "a", "gone"]
    assert "--safe" not in la.editor()
    la.safe = True
    assert la.editor()[-1] == "--safe" and la.headless("m")[-1] == "--safe"


def test_after_a_crash_the_launcher_starts_safe_and_clears_the_marker():
    from pijl.launcher import Launcher

    write(data_root() / mods.SENTINEL, "a\n")
    la = Launcher()
    assert la.safe and la.crashed
    assert any("safe start is on" in p for p in la.problems)
    assert not mods.crashed()  # (answered: the next start makes its own)


def test_the_terminal_mods_page():
    import io

    from pijl.launcher import Launcher, terminal

    write(folder() / "a.py", "")
    write(folder() / "b.py", "")
    la = Launcher()
    out = io.StringIO()
    # m: mods page; 1, 2: turn a and b on; u2: b up; s: safe start; back; Enter: start
    start = terminal(la, io.StringIO("m\n1\n2\nu2\ns\n\n\n"), out)
    assert [m.name for m in mods.plan().enabled] == ["b", "a"]
    assert start == ["gui", "-p", "default", "--safe"]
    assert "LOAD ORDER" in out.getvalue() and "safe start: no mods" in out.getvalue()
