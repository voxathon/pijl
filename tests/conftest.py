import pytest

from pijl import mods


@pytest.fixture(autouse=True)
def no_bundled_mods(tmp_path_factory, monkeypatch):
    """Tests see only the mods they write (test_bundled_mods.py points this back)."""
    monkeypatch.setattr(mods, "BUNDLED", tmp_path_factory.mktemp("no_bundled"))
