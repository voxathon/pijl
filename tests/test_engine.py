import io
import subprocess
import sys

import pytest
from test_macros import half_adder, w

from pijl.cli import main
from pijl.engine import Engine, level
from pijl.logic import ONE, ZERO, X, Z
from pijl.project import Project
from pijl.snapshot import MACRO, Snapshot
from pijl.storage import MacroStore, dumps, encode


def sr_latch() -> Snapshot:
    """Active-low set / reset NAND latch: inputs s (top), r; outputs q, nq."""
    parts = {
        1: ("IN", "s", 0.0, 100.0, {}),
        2: ("IN", "r", 0.0, 0.0, {}),
        3: ("NAND", "", 100.0, 100.0, {}),
        4: ("NAND", "", 100.0, 0.0, {}),
        5: ("OUT", "q", 200.0, 100.0, {}),
        6: ("OUT", "nq", 200.0, 0.0, {}),
    }
    wires = {
        1: w(1, 0, 3, 0),
        2: w(2, 0, 4, 1),
        3: w(3, 0, 4, 0),
        4: w(4, 0, 3, 1),
        5: w(3, 0, 5, 0),
        6: w(4, 0, 6, 0),
    }
    return Snapshot(parts, wires)


def ring() -> Snapshot:
    """Three NOTs in a ring: never stable."""
    parts = {i: ("NOT", "", 100.0 * i, 0.0, {}) for i in (1, 2, 3)}
    parts[4] = ("OUT", "y", 400.0, 0.0, {})
    return Snapshot(parts, {1: w(1, 0, 2, 0), 2: w(2, 0, 3, 0), 3: w(3, 0, 1, 0), 4: w(3, 0, 4, 0)})


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("PIJL_DATA", str(tmp_path / "data"))
    p = Project.open("bench")
    store = MacroStore(p.macros_dir)
    store.save("ha", half_adder(), title="half adder")
    store.save("latch", sr_latch())
    store.save("ring", ring())
    # a macro using a macro (pins by port uid in the file)
    wrap = Snapshot(
        {
            1: ("IN", "x", 0.0, 100.0, {}),
            2: ("IN", "y", 0.0, 0.0, {}),
            3: (MACRO + "ha", "", 100.0, 50.0, {}),
            4: ("OUT", "and", 200.0, 50.0, {}),
        },
        {1: w(1, 0, 3, 0), 2: w(2, 0, 3, 1), 3: w(3, 1, 4, 0)},
    )
    eng = Engine("bench")
    store.save("wrap", wrap, eng.catalog)
    return p


def test_levels_from_anything():
    assert [level(v) for v in (0, 1, True, "x", "Z", " 1 ", ONE)] == [ZERO, ONE, ONE, X, Z, ONE, ONE]
    with pytest.raises(ValueError):
        level("2")


def test_a_project_that_does_not_exist_is_not_made():
    with pytest.raises(FileNotFoundError):
        Engine("nope")


def test_macros_by_title_or_id_and_the_harness_computes():
    eng = Engine("bench")
    assert eng.macros()["ha"] == "half adder"
    h = eng.harness("Half Adder")
    assert h.inputs == ("a", "2") and h.outputs == ("sum", "carry")
    for a in (0, 1):
        for b in (0, 1):
            assert h.apply({"a": a, "#2": b}) == {"sum": level(a != b), "carry": level(a and b)}
    assert eng.harness("ha").macro is h.macro


def test_x_in_x_out_and_truth_table_order():
    h = Engine("bench").harness("ha")
    assert h.apply({"a": "X", "2": 0})["sum"] is X
    rows = [(tuple(i.values()), tuple(o.values())) for i, o in h.truth_table()]
    assert rows == [
        ((ZERO, ZERO), (ZERO, ZERO)),
        ((ZERO, ONE), (ONE, ZERO)),
        ((ONE, ZERO), (ONE, ZERO)),
        ((ONE, ONE), (ZERO, ONE)),
    ]


def test_state_carries_over_between_calls():
    h = Engine("bench").harness("latch")
    assert h.apply({"s": 0, "r": 1}) == {"q": ONE, "nq": ZERO}
    assert h.apply({"s": 1, "r": 1}) == {"q": ONE, "nq": ZERO}  # holds
    assert h.apply({"s": 1, "r": 0}) == {"q": ZERO, "nq": ONE}
    assert h.apply({"s": 1, "r": 1}) == {"q": ZERO, "nq": ONE}


def test_nested_macros_and_files_outside_the_project(project, tmp_path):
    eng = Engine("bench")
    assert eng.harness("wrap").apply({"x": 1, "y": 1}) == {"and": ONE}
    loose = tmp_path / "loose.json"
    loose.write_text(dumps(encode(half_adder(), title="loose one")), encoding="utf-8")
    h = eng.harness(str(loose))
    assert h.macro.title == "loose one" and h.apply({"a": 1}) ["sum"] in (ONE, X)


def test_oscillation_gives_up():
    quiet = Engine("bench").harness("ring", settle_ticks=0)
    assert quiet.settle() and quiet.read() == {"y": X}  # no noise: X forever, which is stable
    h = Engine("bench").harness("ring")  # noise picks 0s and 1s: then it runs around
    assert not h.settle(limit=200) and h.last_ticks is None
    h.step(3)  # exact ticks still work
    assert h.tick > 200


def test_unknown_pins_say_what_there_is():
    h = Engine("bench").harness("ha")
    with pytest.raises(KeyError, match="a, 2"):
        h.set(c=1)
    with pytest.raises(ValueError):
        h.set_all("101")


# ---- the command line --------------------------------------------------------------


def cli(capsys, monkeypatch, *argv, stdin=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    status = main(["-p", "bench", *argv])
    out = capsys.readouterr()
    return status, out.out, out.err


def test_cli_one_shot(capsys, monkeypatch):
    assert cli(capsys, monkeypatch, "run", "half adder", "a=1", "2=1") == (0, "sum=0 carry=1\n", "")
    assert cli(capsys, monkeypatch, "run", "ha", "10", "--json")[1] == '{"sum": 1, "carry": 0}\n'


def test_cli_table(capsys, monkeypatch):
    status, out, _ = cli(capsys, monkeypatch, "run", "ha", "--table")
    assert status == 0
    assert out.splitlines() == ["a 2 | sum carry", "0 0 |   0     0", "0 1 |   1     0", "1 0 |   1     0", "1 1 |   0     1"]


def test_cli_stream_keeps_state_and_mixes_formats(capsys, monkeypatch):
    lines = "# set it\ns=0 r=1\n\n11\n{\"r\": 0}\nr=2\nstep 3\n"
    status, out, err = cli(capsys, monkeypatch, "run", "latch", "-", stdin=lines)
    assert out.splitlines() == [
        "q=1 nq=0",
        "q=1 nq=0",
        '{"q": 0, "nq": 1}',
        "q=0 nq=1",  # (step)
    ]
    assert status == 2 and "line 6" in err


def test_cli_unstable_exit_status(capsys, monkeypatch):
    status, out, err = cli(capsys, monkeypatch, "run", "ring", "--max-ticks", "200")
    assert status == 3 and "not stable" in err and out.startswith("y=")


def test_cli_list_and_errors(capsys, monkeypatch):
    status, out, _ = cli(capsys, monkeypatch, "list")
    assert status == 0 and "half adder [ha]  (a, 2) -> (sum, carry)" in out
    assert cli(capsys, monkeypatch, "run", "nope")[0] == 1
    assert cli(capsys, monkeypatch, "run", "ha", "q=1")[0] == 2


def test_headless_never_imports_pyglet(project):
    code = "import sys; from pijl.cli import main; main(['-p', 'bench', 'run', 'ha', '11']); assert 'pyglet' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "sum=0 carry=1\n"


# ---- fast bits and the raw stream ----------------------------------------------------


def test_bits_in_and_out():
    h = Engine("bench").harness("ha")
    h.set_bits("11")
    h.settle()
    assert h.bits() == "01"
    h.set_bits("0", start=1)  # only b
    h.settle()
    assert h.bits() == "10" and h.driven() == {"a": ONE, "2": ZERO}
    with pytest.raises(ValueError):
        h.set_bits("111")
    with pytest.raises(ValueError):
        h.set_bits("2")


def raw(text: bytes, macro="latch", ticks=None):
    from pijl import cli as C

    h = Engine("bench").harness(macro)

    def run():
        h.settle() if ticks is None else h.step(ticks)

    out = io.BytesIO()
    return C._raw(h, run, io.BytesIO(text), out), out.getvalue()


def test_raw_stream_vectors_partial_vectors_and_reads():
    # s r: set (q=1), hold, reset (q=0); "1;" changes only s; ";" alone just runs
    assert raw(b"01? 11?\n10?1;?;?") == (0, b"1010010101")  # 10 10 01 01 01


def test_raw_stream_refuses_junk(capsys):
    status, out = raw(b"01?k11?")
    assert status == 2 and out == b"10" and "'k'" in capsys.readouterr().err


# ---- bogobips --------------------------------------------------------------------------


SMALL = {"sipo": 5, "piso": 5, "tree-and": 4, "tree-xor": 4, "decoder": 5, "adder": 6}


@pytest.mark.parametrize("nest", [False, True])
def test_bogobips_every_kind_computes_what_its_script_says(tmp_path, nest):
    import numpy as np

    from pijl import bogobips as B

    plan = list(SMALL.items())
    eng = B.make_project(tmp_path / "bogo-data", plan, nest)
    for kind, n in plan:
        k = B.KINDS[kind]
        for fixed in (False, True):  # settle, and the longest path's ticks
            if fixed and k.path is None:
                continue
            h = eng.harness(f"{kind} {n}", settle_ticks=0)
            steps = k.script(n, np.random.default_rng(3), 2)
            reads = 0
            while reads < 60:  # every read checked, not every 64th
                sets, read, expect = next(steps)
                for start, bits in sets:
                    h.set_bits(bits, start)
                if fixed:
                    h.step(k.path(n))
                else:
                    assert h.settle(10_000)
                    assert k.path is None or h.last_ticks <= k.path(n), (kind, h.last_ticks)
                if read:
                    assert h.bits() == expect(), (kind, n, fixed, reads)
                    reads += 1


def test_bogobips_steps_survive_the_raw_stream(tmp_path):
    import numpy as np

    from pijl import bogobips as B
    from pijl import cli as C

    eng = B.make_project(tmp_path / "bogo-data", [("adder", 4), ("piso", 3)])
    for macro, kind, n in (("adder 4", "adder", 4), ("piso 3", "piso", 3)):
        h = eng.harness(macro, settle_ticks=0)
        steps = B.KINDS[kind].script(n, np.random.default_rng(5), 2)
        stream, want = bytearray(), bytearray()
        for _ in range(40):
            sets, read, expect = next(steps)
            stream += B.encode(sets, len(h.inputs)) + (b"?" if read else b"")
            want += expect().encode() if read else b""
        out = io.BytesIO()
        assert C._raw(h, h.settle, io.BytesIO(bytes(stream)), out) == 0
        assert out.getvalue() == bytes(want)


def test_raw_addresses_and_chunk_edges(capsys):
    from pijl import cli as C

    class Dribble(io.BytesIO):  # one byte per read: every token cut somewhere
        def read1(self, n=-1):
            return self.read(1)

    h = Engine("bench").harness("latch")
    out = io.BytesIO()
    assert C._raw(h, h.settle, Dribble(b"01?@1=1;?@2=0;?@12=1;"), out) == 2
    assert out.getvalue() == b"101001"  # 10 10 01
    assert "12" in capsys.readouterr().err
