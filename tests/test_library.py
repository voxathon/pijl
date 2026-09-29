from pijl.sim.circuit import BUILTINS
from pijl.ui.library import Library


def everything(lib: Library) -> list[str]:
    return [p for c in lib.collections for p in c.parts] + lib.loose


def test_defaults_hold_every_builtin_once():
    lib = Library()
    assert [c.name for c in lib.collections] == ["I/O", "GATES"]
    assert all(c.builtin for c in lib.collections)
    assert sorted(everything(lib)) == sorted(BUILTINS)


def test_move_part_between_collections_and_out():
    lib = Library()
    io, gates = lib.collections
    lib.move_part("NOT", io, 1)
    assert io.parts == ["IN", "NOT", "OUT"] and "NOT" not in gates.parts
    lib.move_part("NOT", None)
    assert lib.where("NOT") is None and lib.loose == ["NOT"]
    assert sorted(everything(lib)) == sorted(BUILTINS)


def test_move_part_within_a_list_uses_marker_positions():
    lib = Library()
    gates = lib.collections[1]  # NAND AND OR NOT
    lib.move_part("NAND", gates, 3)  # marker between OR and NOT
    assert gates.parts == ["AND", "OR", "NAND", "NOT"]
    lib.move_part("NOT", gates, 0)
    assert gates.parts == ["NOT", "AND", "OR", "NAND"]


def test_new_collections_get_unique_names_and_delete_frees_parts():
    lib = Library()
    a, b = lib.new_collection(), lib.new_collection()
    assert a.name != b.name
    lib.move_part("AND", a)
    lib.delete_collection(a)
    assert a not in lib.collections and lib.loose == ["AND"]
    lib.delete_collection(lib.collections[0])  # builtin: refused
    assert lib.collections[0].name == "I/O"


def test_move_collection_and_rename():
    lib = Library()
    io, gates = lib.collections
    lib.move_collection(io, 2)
    assert lib.collections == [gates, io]
    lib.rename(io, "  ")
    assert io.name == "I/O"
    lib.rename(io, " PINS ")
    assert io.name == "PINS"
