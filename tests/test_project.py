import json

import pytest

from pijl.project import FORMAT, Project, data_root, projects_dir


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PIJL_DATA", str(tmp_path / "data"))
    return tmp_path / "data"


def test_override_sets_the_data_root(data_dir):
    assert data_root() == data_dir
    assert projects_dir() == data_dir / "projects"


def test_default_project_is_created_on_first_open(data_dir):
    p = Project.open()
    assert p.name == "default"
    assert p.path == data_dir / "projects" / "default"
    assert p.macros_dir.is_dir()
    assert json.loads(p.meta_file.read_text()) == {"pijl": FORMAT}
    assert not list(p.path.glob("*.tmp"))


def test_reopening_keeps_what_is_there():
    p = Project.open()
    (p.macros_dir / "adder.json").write_text("{}")
    assert (Project.open().macros_dir / "adder.json").exists()


def test_newer_project_format_is_refused():
    p = Project.open()
    p.meta_file.write_text(json.dumps({"pijl": FORMAT + 1}))
    with pytest.raises(ValueError):
        Project.open()
