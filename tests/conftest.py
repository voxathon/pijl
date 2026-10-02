import pytest

from pijl import mods


@pytest.fixture(autouse=True)
def no_official_mods(tmp_path_factory, monkeypatch):
    """Tests see only the mods they write (test_official_mods.py points this back)."""
    monkeypatch.setattr(mods, "OFFICIAL", tmp_path_factory.mktemp("no_official"))
