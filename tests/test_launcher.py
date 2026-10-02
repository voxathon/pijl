"""Preferences (prefs.py), the launcher's model and its terminal face (launcher.py),
project renaming, and the `pijl` arguments around them (cli.py). No windows."""

import io
import json

import pytest

from pijl import prefs
from pijl.cli import main
from pijl.launcher import Launcher, terminal
from pijl.project import (
    Project,
    data_root,
    last_project,
    project_names,
    remember_project,
    rename_project,
)
from pijl.sim import config


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PIJL_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("PIJL_ENGINE", "")  # (recorded, so what main() sets is undone)
    before = config.default()
    yield tmp_path / "data"
    config.set_default(before)


def settings() -> dict:
    return json.loads((data_root() / "settings.json").read_text(encoding="utf-8"))


# ---- prefs -------------------------------------------------------------------------


def test_every_default_is_a_value_its_setting_allows():
    for key, setting in prefs.PREFS.items():
        assert setting.parse(setting.initial) == setting.initial, key


def test_no_file_means_the_defaults():
    assert prefs.load() == (prefs.defaults(), [])


def test_only_changed_prefs_are_saved_and_the_project_is_kept():
    remember_project("default")
    values = prefs.defaults() | {"ui.scale": 1.5}
    prefs.save(values)
    assert settings() == {"project": "default", "prefs": {"ui.scale": 1.5}}
    assert prefs.load() == (values, [])
    remember_project("other")  # (writing the project keeps the prefs)
    assert settings()["prefs"] == {"ui.scale": 1.5}


def test_bad_values_fall_back_with_a_warning():
    path = data_root() / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"prefs": {"ui.scale": 9, "engine.dirty": "sometimes", "nope": 1}}
        )
    )
    values, problems = prefs.load()
    assert values["ui.scale"] == 2.0  # clamped
    assert values["engine.dirty"] == "adaptive"
    assert len(problems) == 2  # the bad choice and the unknown key


def test_unreadable_file_means_the_defaults():
    path = data_root() / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    values, problems = prefs.load()
    assert values == prefs.defaults() and problems


def test_typed_values():
    assert prefs.parse_text("ui.scale", "1.25") == 1.25
    assert prefs.parse_text("editor.noise", "100 ticks") == 100
    assert prefs.parse_text("engine.eval", "BATCHES") == "batches"
    with pytest.raises(ValueError):
        prefs.parse_text("engine.eval", "fast")
    with pytest.raises(ValueError):
        prefs.parse_text("nope", "1")


def test_engine_env_beats_prefs(monkeypatch):
    values = prefs.defaults() | {"engine.dirty": "off", "engine.eval": "batches"}
    assert str(prefs.engine_config(values)) == "dirty=off,eval=batches"
    monkeypatch.setenv("PIJL_ENGINE", "eval=lut")
    assert str(prefs.engine_config(values)) == "dirty=off,eval=lut"


# ---- renaming projects -------------------------------------------------------------


def test_rename_moves_the_folder_trash_and_last_project(data_dir):
    p = Project.open("old")
    (p.macros_dir / "a.json").write_text("{}")
    p.trash_dir.mkdir(parents=True)
    remember_project("old")
    assert rename_project("old", " new ") == "new"
    assert project_names() == ["new"]
    assert (data_dir / "projects" / "new" / "macros" / "a.json").exists()
    assert (data_dir / ".trashbin" / "new").is_dir()
    assert last_project() == "new"


def test_rename_only_the_case():
    Project.open("thing")
    rename_project("thing", "Thing")
    assert project_names() == ["Thing"]


def test_rename_refusals():
    Project.open("a")
    Project.open("b")
    for old, new in (("a", "b"), ("a", "B"), ("a", "x/y"), ("a", ""), ("zz", "c")):
        with pytest.raises(ValueError):
            rename_project(old, new)
    assert project_names() == ["a", "b"]


# ---- the launcher ------------------------------------------------------------------


def test_first_run_makes_the_default_project():
    la = Launcher()
    assert la.project == "default" and la.projects() == ["default"]


def test_new_project_is_picked_and_remembered():
    la = Launcher()
    la.new_project("adders")
    assert la.project == "adders" and last_project() == "adders"
    with pytest.raises(ValueError):
        la.new_project("ADDERS")


def test_renaming_the_picked_project_follows_it():
    la = Launcher()
    la.rename("default", "main")
    assert la.project == "main" and Launcher().project == "main"


def test_what_it_starts():
    la = Launcher()
    assert la.editor() == ["gui", "-p", "default"]
    assert la.headless("adder") == ["run", "adder", "-", "-p", "default"]
    assert la.headless("adder", table=True)[2] == "--table"


def tui(*lines: str):
    out = io.StringIO()
    start = terminal(Launcher(), io.StringIO("".join(f"{s}\n" for s in lines)), out)
    return start, out.getvalue()


def test_terminal_enter_starts_the_editor():
    assert tui("")[0] == ["gui", "-p", "default"]


def test_terminal_quits_on_q_or_end_of_input():
    assert tui("q")[0] is None
    assert terminal(Launcher(), io.StringIO(""), io.StringIO()) is None


def test_terminal_new_load_and_rename():
    start, _ = tui("n", "second", "l", "r1", "first", "1", "")
    # n: made "second"; l, r1: renamed "default" (sorted first) to "first"; 1: picked it
    assert sorted(project_names()) == ["first", "second"]
    assert start == ["gui", "-p", "first"]


def test_terminal_settings():
    start, out = tui("s", "1", "1.5", "4", "batches", "3", "nonsense", "", "q")
    assert start is None
    values, _ = prefs.load()
    assert values["ui.scale"] == 1.5 and values["engine.eval"] == "batches"
    assert "can't" in out  # the bad dirty value
    assert values["engine.dirty"] == "adaptive"


def test_terminal_headless_with_no_macros():
    start, out = tui("h", "q")
    assert start is None and "no macros" in out


# ---- cli ---------------------------------------------------------------------------


def test_prefs_command_sets_shows_and_resets(capsys):
    assert main(["prefs", "ui.scale=1.4", "engine.dirty=off"]) == 0
    assert "1.4  (changed)" in capsys.readouterr().out
    assert prefs.load()[0]["engine.dirty"] == "off"
    assert main(["prefs", "ui.scale=huge"]) == 2
    assert main(["prefs", "--reset"]) == 0
    assert prefs.load()[0] == prefs.defaults()


def test_engine_prefs_reach_headless_commands():
    Project.open()
    prefs.save(prefs.defaults() | {"engine.eval": "batches"})
    assert main(["list"]) == 0
    assert str(config.default()) == "dirty=adaptive,eval=batches"
    assert main(["list", "--engine", "dirty=off"]) == 0
    assert str(config.default()) == "dirty=off,eval=batches"


def test_tui_launch_runs_what_was_picked(monkeypatch, capsys):
    Project.open()
    monkeypatch.setattr("sys.stdin", io.StringIO("l\n1\nq\n"))
    assert main(["--tui"]) == 0
    assert "project: default" in capsys.readouterr().out
