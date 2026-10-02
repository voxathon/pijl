"""Official mods (mods.OFFICIAL): installed into the mods folder, disabled, upgraded;
and manuscript, the logging one."""

import json
import logging
import sys
import threading
from pathlib import Path

import pytest

from pijl import mods
from pijl.project import data_root

SHIPPED = Path(mods.__file__).parent / "official_mods"
LOGGERS = ("pijl", "pijl.edit", "pijl.sim", "pijl_mods", "pijl_mods.manuscript", "py.warnings")


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("PIJL_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("PIJL_MODS", "")
    monkeypatch.setenv("PIJL_SAFE", "")
    monkeypatch.setattr(mods, "OFFICIAL", SHIPPED)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    yield
    mods._report = None
    mods._hooks.clear()
    mods._tracebacks.clear()
    for name in [n for n in sys.modules if n == mods.PACKAGE or n.startswith(mods.PACKAGE + ".")]:
        del sys.modules[name]
    for name in LOGGERS:
        logger = logging.getLogger(name)
        for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
            logger.removeHandler(h)
            h.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
    logging.captureWarnings(False)


def folder() -> Path:
    return data_root() / "mods"


def test_official_mods_are_installed_disabled():
    p = mods.plan()
    assert "manuscript" in [m.name for m in p.new]
    assert (folder() / "manuscript" / "__init__.py").is_file()
    assert "manuscript" in mods.read_list(folder() / mods.DISABLED)
    assert mods.load().loaded == []


def test_a_deleted_official_mod_stays_deleted():
    mods.plan()
    for f in (folder() / "manuscript").iterdir():
        f.unlink()
    (folder() / "manuscript").rmdir()
    mods.plan()
    assert not (folder() / "manuscript").exists()


def test_a_newer_official_mod_is_copied_over_keeping_config_and_logs():
    mods.plan()
    mine = folder() / "manuscript"
    (mine / "mod.json").write_text(json.dumps({"version": "0.1"}))
    (mine / "__init__.py").write_text("# old\n")
    (mine / "config.json").write_text('{"keep": 3}')
    (mine / "logs").mkdir()
    (mine / "logs" / "old.log").write_text("x")
    mods.plan()
    assert (mine / "__init__.py").read_text() == (SHIPPED / "manuscript" / "__init__.py").read_text()
    assert (mine / "config.json").read_text() == '{"keep": 3}'
    assert (mine / "logs" / "old.log").exists()


def test_the_same_version_is_left_alone():
    mods.plan()
    edited = folder() / "manuscript" / "__init__.py"
    edited.write_text("# mine\n")
    mods.plan()
    assert edited.read_text() == "# mine\n"


def enable_manuscript() -> Path:
    mods.plan()
    mods.enable("manuscript")
    return folder() / "manuscript"


def log_text(mine: Path) -> str:
    for h in logging.getLogger("pijl").handlers:
        h.flush()
    (path,) = (mine / "logs").glob("*.log")
    return path.read_text(encoding="utf-8")


def test_manuscript_logs_pijl_and_writes_its_config():
    mine = enable_manuscript()
    rep = mods.load()
    assert [m.name for m in rep.loaded] == ["manuscript"], rep.problems
    assert json.loads((mine / "config.json").read_text())["keep"] == 30
    logging.getLogger("pijl.edit").info("edit: +1 part")
    logging.getLogger("pijl.files").debug("not at info")
    text = log_text(mine)
    assert "pijl.mods: mods: manuscript 1.0" in text
    assert "pijl.edit: edit: +1 part" in text
    assert "not at info" not in text
    assert text.count("edit: +1 part") == 1  # (pijl.edit and pijl both listen: once)


def test_manuscript_levels_and_crashes(monkeypatch):
    mine = enable_manuscript()
    (mine / "config.json").write_text(
        json.dumps({"levels": {"pijl.edit": "off", "pijl.files": "debug"}, "bogus": 1})
    )
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: seen.append(exc[0]))
    mods.load()
    logging.getLogger("pijl.edit").warning("hidden")
    logging.getLogger("pijl.files").debug("shown")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    text = log_text(mine)
    assert "hidden" not in text and "pijl.files: shown" in text
    assert "unknown setting 'bogus'" in text
    assert "uncaught exception" in text and "RuntimeError: boom" in text
    assert seen == [RuntimeError]  # (and the old hook still ran)


def test_manuscript_keeps_the_newest_logs():
    mine = enable_manuscript()
    (mine / "config.json").write_text('{"keep": 2}')
    (mine / "logs").mkdir()
    for n in ("2000-01-01_000000_1.log", "2000-01-02_000000_1.log"):
        (mine / "logs" / n).write_text("")
    mods.load()
    names = sorted(p.name for p in (mine / "logs").glob("*.log"))
    assert len(names) == 2 and names[0] == "2000-01-02_000000_1.log"
