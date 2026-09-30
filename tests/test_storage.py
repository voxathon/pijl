import json

import pytest

from pijl.parts import builtin_registry
from pijl.snapshot import Snapshot
from pijl.storage import (
    FORMAT,
    FormatError,
    MacroStore,
    check_name,
    decode,
    dumps,
    encode,
)

REG = builtin_registry()


def board() -> Snapshot:
    """a, b -> NAND -> LED, with a branch off the first wire into a NOT."""
    parts = {
        1: ("IN", "a", 200.0, 360.0, {}),
        2: ("IN", "b", 200.0, 220.0, {}),
        3: ("NAND", "", 380.0, 280.0, {}),
        4: ("OUT", "q", 580.5, 290.25, {}),
        5: ("NOT", "", 380.0, 500.0, {}),
    }
    wires = {
        1: (
            ("p", 1, False, 0),
            ("p", 3, True, 0),
            ((320.0, 380.0), (320.0, 320.0)),
            None,
            None,
        ),
        2: (("p", 2, False, 0), ("p", 3, True, 1), (), None, None),
        3: (("p", 3, False, 0), ("p", 4, True, 0), (), None, None),
        4: (
            ("w", 1),
            ("p", 5, True, 0),
            ((320.0, 520.0),),
            (320.0, 350.0),
            None,
        ),  # branch off wire 1
    }
    return Snapshot(parts, wires)


def roundtrip(snap: Snapshot) -> Snapshot:
    loaded = decode(json.loads(dumps(encode(snap))), REG)
    assert loaded.warnings == []
    return loaded.snapshot


# ---- format ------------------------------------------------------------------------


def test_roundtrip_is_exact():
    assert roundtrip(board()) == board()


def test_empty_board():
    assert roundtrip(Snapshot({}, {})) == Snapshot({}, {})


def test_props_roundtrip():
    snap = Snapshot({1: ("NOT", "", 0.0, 0.0, {"color": "red"})}, {})
    assert roundtrip(snap) == snap


def test_wire_colors_roundtrip():
    b = board()
    snap = Snapshot(b.parts, b.wires, {1: "blue", 4: "blue", 3: "pink"})
    assert roundtrip(snap) == snap
    assert '"color"' not in dumps(encode(board()))  # default color: not written at all


def test_text_is_stable_and_one_line_per_item():
    text = dumps(encode(board()))
    assert text == dumps(encode(roundtrip(board())))
    lines = text.splitlines()
    assert lines[:2] == ["{", f'  "pijl": {FORMAT},']
    assert (
        sum('"uid"' in line for line in lines) == 9
    )  # 5 parts + 4 wires, one per line
    assert (
        '"pos": [200, 360]' in text and '"pos": [580.5, 290.25]' in text
    )  # whole numbers without .0
    assert (
        '"label": ""' not in text
        and '"bends": []' not in text
        and '"props"' not in text
    )
    assert json.loads(text) == encode(board())


# ---- forgiving loads -----------------------------------------------------------------


def test_unknown_kind_drops_the_part_and_every_wire_that_needed_it():
    data = encode(board())
    data["parts"][2]["kind"] = "FLUX_CAPACITOR"  # the NAND
    loaded = decode(data, REG)
    assert set(loaded.snapshot.parts) == {1, 2, 4, 5}
    assert loaded.snapshot.wires == {}  # 1-3 touched the NAND, 4 hung off wire 1
    assert any("FLUX_CAPACITOR" in w for w in loaded.warnings)
    assert any("4 wire(s)" in w for w in loaded.warnings)


def test_pin_that_no_longer_exists_drops_the_wire():
    data = encode(board())
    data["wires"][1]["to"] = {"part": 3, "in": 7}  # NAND has 2 inputs
    loaded = decode(data, REG)
    assert set(loaded.snapshot.wires) == {1, 3, 4}


def test_invalid_connections_are_dropped():
    data = encode(board())
    data["wires"].append(
        {"uid": 9, "from": {"part": 2, "out": 0}, "to": {"part": 4, "in": 0}}
    )  # LED already driven
    data["wires"].append(
        {"uid": 10, "from": {"part": 1, "out": 0}, "to": {"part": 2, "out": 0}}
    )  # out -> out
    data["wires"].append(
        {"uid": 11, "from": {"wire": 12, "at": [0, 0]}, "to": {"part": 5, "in": 0}}
    )  # newer wire
    loaded = decode(data, REG)
    assert set(loaded.snapshot.wires) == {1, 2, 3, 4}
    assert len(loaded.warnings) == 3


def test_garbage_entries_are_skipped_not_fatal():
    data = encode(board())
    data["parts"].append({"uid": "seven", "kind": "NOT", "pos": [0, 0]})
    data["parts"].append({"uid": 8, "kind": "NOT", "pos": [0]})
    data["parts"].append({"uid": 1, "kind": "NOT", "pos": [0, 0]})  # duplicate uid
    data["wires"].append("nope")
    loaded = decode(data, REG)
    assert set(loaded.snapshot.parts) == {1, 2, 3, 4, 5}
    assert loaded.snapshot.parts[1][0] == "IN"  # the first one wins
    assert len(loaded.warnings) == 4


@pytest.mark.parametrize(
    "data", [[], {}, {"pijl": "one"}, {"pijl": 0}, {"pijl": FORMAT, "parts": {}}]
)
def test_not_a_macro_file(data):
    with pytest.raises(FormatError):
        decode(data, REG)


def test_newer_format_is_refused():
    with pytest.raises(FormatError, match="newer"):
        decode({"pijl": FORMAT + 1}, REG)


# ---- names and files ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["", "   ", "a/b", "what?", "x" * 41, "trailing.", "CON", "nul.txt", "tab\there"],
)
def test_bad_names(name):
    with pytest.raises(ValueError):
        check_name(name)


def test_good_names_are_trimmed():
    assert check_name("  half adder  ") == "half adder"
    assert check_name("ALU (v2) #3") == "ALU (v2) #3"


def test_store_saves_lists_and_loads(tmp_path):
    store = MacroStore(tmp_path / "macros")
    assert store.ids() == []
    store.save("half adder", board())
    store.save("Alu", Snapshot({}, {}))
    assert store.ids() == ["Alu", "half adder"]
    assert store.load("half adder", REG).snapshot == board()
    assert not list((tmp_path / "macros").glob("*.tmp"))


def test_names_match_ignoring_case_and_resaving_renames(tmp_path):
    store = MacroStore(tmp_path)
    store.save("adder", board())
    assert store.find("ADDER") == "adder" and store.find("subtractor") is None
    store.save("Adder", Snapshot({}, {}))
    assert store.ids() == ["Adder"]
    assert store.load("Adder", REG).snapshot == Snapshot({}, {})


def test_remove_moves_to_the_trash_without_clobbering(tmp_path):
    store, trash = MacroStore(tmp_path / "macros"), tmp_path / "trash"
    store.save("adder", board())
    assert store.remove("adder", trash) == trash / "adder.json"
    store.save("adder", Snapshot({}, {}))
    assert store.remove("adder", trash) == trash / "adder (2).json"
    assert store.ids() == []
    assert MacroStore(trash).load("adder", REG).snapshot == board()


def test_put_back_undoes_remove_unless_the_name_is_taken(tmp_path):
    store, trash = MacroStore(tmp_path / "macros"), tmp_path / "trash"
    store.save("adder", board())
    trashed = store.remove("adder", trash)
    store.put_back("adder", trashed)
    assert store.load("adder", REG).snapshot == board() and not trashed.exists()
    trashed = store.remove("adder", trash)
    store.save("ADDER", Snapshot({}, {}))
    with pytest.raises(FileExistsError):
        store.put_back("adder", trashed)
    assert trashed.exists()


def test_remove_without_a_trash_deletes_for_good(tmp_path):
    store = MacroStore(tmp_path)
    store.save("adder", board())
    assert store.remove("adder", None) is None
    assert list(tmp_path.iterdir()) == []


def test_broken_file_raises_format_error(tmp_path):
    store = MacroStore(tmp_path)
    (tmp_path / "bad.json").write_text("{ not json")
    with pytest.raises(FormatError, match="JSON"):
        store.load("bad", REG)


def test_titles_live_in_the_file_and_default_to_the_id(tmp_path):
    store = MacroStore(tmp_path)
    store.save("adder", board(), title="Half Adder")
    (tmp_path / "old.json").write_text('{"pijl": 1, "parts": [], "wires": []}')
    assert store.titles() == {"adder": "Half Adder", "old": "old"}
    assert store.find("half adder") == "adder" and store.find("adder") is None
    assert '"title": "Half Adder"' in (tmp_path / "adder.json").read_text()
    store.save("adder", Snapshot({}, {}))  # no title: keeps the one it has
    assert store.title("adder") == "Half Adder"


def test_retitle_changes_only_the_title(tmp_path):
    store = MacroStore(tmp_path)
    store.save("adder", board())
    store.retitle("adder", "sum")
    assert store.ids() == ["adder"] and store.title("adder") == "sum"
    loaded = store.load("adder", REG)
    assert loaded.snapshot == board() and loaded.title == "sum"


def test_new_ids_are_numbered_when_the_file_is_taken(tmp_path):
    store = MacroStore(tmp_path)
    assert store.new_id("adder") == "adder"
    store.save("adder", board(), title="sum")
    assert store.new_id("Adder") == "Adder (2)"
    assert len(store.new_id("x" * 40)) == 40


def test_put_back_refuses_a_title_thats_taken_now(tmp_path):
    store, trash = MacroStore(tmp_path / "macros"), tmp_path / "trash"
    store.save("adder", board(), title="sum")
    trashed = store.remove("adder", trash)
    store.save("other", Snapshot({}, {}), title="SUM")
    with pytest.raises(FileExistsError):
        store.put_back("adder", trashed)
