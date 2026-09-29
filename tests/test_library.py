from pijl.ui.library import Library

PARTS = [("IN", "I/O"), ("OUT", "I/O"), ("NAND", "GATES"), ("AND", "GATES"), ("OR", "GATES"),
         ("NOT", "GATES"), ("XOR", "")]
KINDS = [kind for kind, _ in PARTS]


def everything(lib: Library) -> list[str]:
    return [p for c in lib.collections for p in c.parts] + lib.loose


def test_defaults_hold_every_builtin_once():
    lib = Library(PARTS)
    assert [c.name for c in lib.collections] == ["I/O", "GATES"]
    assert all(c.builtin for c in lib.collections)
    assert lib.loose == ["XOR"]  # no category: loose
    assert sorted(everything(lib)) == sorted(KINDS)


def test_move_part_between_collections_and_out():
    lib = Library(PARTS)
    io, gates = lib.collections
    lib.move_part("NOT", io, 1)
    assert io.parts == ["IN", "NOT", "OUT"] and "NOT" not in gates.parts
    lib.move_part("NOT", None)
    assert lib.where("NOT") is None and lib.loose == ["XOR", "NOT"]
    assert sorted(everything(lib)) == sorted(KINDS)


def test_move_part_within_a_list_uses_final_positions():
    lib = Library(PARTS)
    gates = lib.collections[1]  # NAND AND OR NOT
    lib.move_part("NAND", gates, 2)
    assert gates.parts == ["AND", "OR", "NAND", "NOT"]
    lib.move_part("NOT", gates, 0)
    assert gates.parts == ["NOT", "AND", "OR", "NAND"]


def test_new_collections_get_unique_names_and_delete_frees_parts():
    lib = Library(PARTS)
    a, b = lib.new_collection(), lib.new_collection()
    assert a.name != b.name
    lib.move_part("AND", a)
    lib.delete_collection(a)
    assert a not in lib.collections and lib.loose == ["XOR", "AND"]
    lib.delete_collection(lib.collections[0])  # builtin: refused
    assert lib.collections[0].name == "I/O"


def test_move_collection_and_rename():
    lib = Library(PARTS)
    io, gates = lib.collections
    lib.move_collection(io, 1)
    assert lib.collections == [gates, io]
    lib.rename(io, "  ")
    assert io.name == "I/O"
    lib.rename(io, " PINS ")
    assert io.name == "PINS"


# ---- saving / syncing --------------------------------------------------------------


def test_sync_drops_what_is_gone_and_adds_what_is_new_where_the_user_left_things():
    lib = Library(PARTS)
    io, gates = lib.collections
    lib.move_part("NAND", io)
    lib.sync([p for p in PARTS if p[0] != "OR"] + [("macro:adder", "MACROS")])
    assert "OR" not in gates.parts
    assert lib.where("NAND") is io  # stays where the user put it, not back in GATES
    assert [c.name for c in lib.collections] == ["I/O", "GATES", "MACROS"]
    assert lib.collections[2].parts == ["macro:adder"] and lib.collections[2].builtin


def test_to_dict_from_dict_roundtrip():
    lib = Library(PARTS)
    c = lib.new_collection()
    lib.rename(c, "MINE")
    lib.move_part("XOR", c)
    lib.collections[0].open = False
    again = Library.from_dict(lib.to_dict(), PARTS)
    assert again.to_dict() == lib.to_dict()


def test_from_dict_forgives_junk_and_duplicates():
    data = {"collections": [{"name": "A", "parts": ["NAND", "NAND", 5, "GHOST"]}, "junk", {"parts": []},
                            {"name": "A", "parts": ["OR"]}],
            "loose": "not a list"}
    lib = Library.from_dict(data, PARTS)
    assert lib.collections[0].name == "A" and lib.collections[0].parts == ["NAND"]
    assert sorted(everything(lib)) == sorted(KINDS)  # every part exactly once
    assert lib.where("OR").name == "GATES"  # the duplicate "A" was skipped, so OR got its default
