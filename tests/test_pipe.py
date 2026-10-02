import io
import struct

import numpy as np
import pytest
from test_engine import project  # noqa: F401 (the bench project, for every test)

from pijl import pipe as P
from pijl.engine import Engine


def frame(op, body=b"", flags=0, port=0) -> bytes:
    return P.HEADER.pack(op, flags, port, len(body)) + body


def run_frame(vectors, *, port=0, out=0, sample=P.EVERY, ticks=0, limit=0, flags=0, mask=b"", width=2) -> bytes:
    codes, one = P._vectors(vectors, width)
    one = one and not flags & 0x80  # (0x80: force 2 bits; not a real flag)
    flags = (flags & ~0x80) | (P.IN_1BIT if one else 0)
    body = P.RUN_HEAD.pack(len(codes), out, sample, ticks, limit) + mask + P.pack(codes, one)[0]
    return frame(P.RUN, body, flags, port)


def talk(macro, data: bytes, noise=64, seed=0, max_ticks=10_000):
    eng = Engine("bench")
    out, warnings = io.BytesIO(), []
    status = P.serve(
        lambda n, s: eng.harness(macro, settle_ticks=n, seed=s), noise, seed, io.BytesIO(data), out, max_ticks, warnings.append
    )
    got, buf, pos = [], out.getvalue(), 0
    while pos < len(buf):
        op, flags, port, length = P.HEADER.unpack_from(buf, pos)
        pos += P.HEADER.size
        got.append((op, flags, port, buf[pos : pos + length]))
        pos += length
    assert got[0][0] == P.HELLO
    return status, got[1:]


def result(answer, width):
    """A RESULT's (head fields, per-vector ticks or None, samples as strings)."""
    op, flags, _, body = answer
    assert op == P.RESULT, body
    head = P.RESULT_HEAD.unpack_from(body)
    pos = P.RESULT_HEAD.size
    ticks = None
    if flags & P.TICKS:
        ticks = np.frombuffer(body, "<u4", head[1], pos).tolist()
        pos += 4 * head[1]
    codes = P.unpack(body[pos:], head[2], width, bool(flags & P.OUT_1BIT))
    text = P._CHAR_OF[codes].tobytes().decode()
    return head, ticks, [text[i * width : (i + 1) * width] for i in range(head[2])]


def test_packing_round_trips():
    rng = np.random.default_rng(1)
    for width in (0, 1, 3, 4, 5, 8, 9, 31):
        codes = rng.integers(0, 4, (7, width)).astype(np.uint8)
        data, lossy = P.pack(codes, False)
        assert len(data) == 7 * P.stride(width, False) and not lossy
        assert (P.unpack(data, 7, width, False) == codes).all()
        bits = rng.integers(1, 3, (7, width)).astype(np.uint8)
        data, lossy = P.pack(bits, True)
        assert len(data) == 7 * P.stride(width, True) and not lossy
        assert (P.unpack(data, 7, width, True) == bits).all()
    # pin 0 is the low bit: a 1-bit vector is the value
    assert P.pack(P._vectors([0b1011], 4)[0], True)[0] == b"\x0b"
    assert P.pack(P._vectors(["10ZX"], 4)[0], False)[0] == bytes([0b11_00_01_10])
    assert P.pack(np.array([[3, 1]], np.uint8), True) == (b"\x00", True)


def test_hello_names_the_pins():
    eng = Engine("bench")
    out = io.BytesIO()
    assert P.serve(lambda n, s: eng.harness("ha"), 64, 0, io.BytesIO(b""), out) == 0
    op, _, _, length = P.HEADER.unpack_from(out.getvalue())
    body = out.getvalue()[8:]
    assert op == P.HELLO and length == len(body)
    assert P.HELLO_HEAD.unpack_from(body) == (b"PIJL", 1, 0, 2, 2)
    names = body[P.HELLO_HEAD.size :].split(b"\0")
    assert names[0] == b"half adder" and names[2:] == [b"a", b"2", b"sum", b"carry", b""]


def test_a_batch_is_the_same_as_its_vectors_one_by_one():
    vectors = ["01", "11", "10", "11", "X1", "01", "ZZ", "10", "11"]  # set, hold, reset, hold, ...
    status, batch = talk("latch", run_frame(vectors, flags=P.TICKS))
    assert status == 0 and len(batch) == 1
    head, ticks, samples = result(batch[0], 2)
    assert head[:3] == (0, 9, 9) and len(ticks) == 9
    _, singles = talk("latch", b"".join(run_frame([v], flags=P.TICKS) for v in vectors))
    assert [result(a, 2)[2][0] for a in singles] == samples
    assert [result(a, 2)[1][0] for a in singles] == ticks
    assert [result(a, 2)[0][0] for a in singles] == list(range(9))  # frame numbers
    h = Engine("bench").harness("latch")  # and the same as the harness in-process
    want = []
    for v in vectors:
        h.set_bits(v)
        h.settle()
        want.append(h.bits())
    assert samples == want
    _, two_bit = talk("latch", run_frame(vectors[:4], flags=0x80))  # 0/1 vectors sent 2 bits a pin
    assert result(two_bit[0], 2)[2] == want[:4]


def test_ports_ints_and_one_bit_answers():
    ins = frame(P.PORT, P.PORT_HEAD.pack(P.IN, 1) + struct.pack("<I", 1), port=7)  # just "2"
    outs = frame(P.PORT, P.PORT_HEAD.pack(P.OUT, 2) + struct.pack("<II", 1, 0), port=8)  # carry, sum
    data = ins + outs + run_frame([0b11, 0b01, 0b10]) + run_frame([1, 0], port=7, out=8, flags=P.OUT_1BIT, width=1)
    status, got = talk("ha", data)
    assert status == 0
    assert result(got[0], 2)[2] == ["01", "10", "10"]  # sum, carry
    _, _, samples = result(got[1], 2)  # a stays 0 (last driven by "10" = a=0, 2=1)
    assert got[1][1] & P.OUT_1BIT and got[1][2] == 8
    assert samples == ["01", "00"]  # 2=1: carry 0, sum 1; 2=0: both 0


def test_sample_modes_and_a_mask():
    vectors = ["01", "11", "10", "11"]
    data = (
        run_frame(vectors, sample=P.NONE)
        + run_frame(vectors, sample=P.LAST)
        + run_frame(vectors, sample=P.MASK, mask=bytes([0b0101]))
        + run_frame([], sample=P.LAST)
    )
    status, got = talk("latch", data)
    assert status == 0
    assert result(got[0], 2)[0][1:3] == (4, 0)
    assert result(got[1], 2)[2] == ["01"]
    assert result(got[2], 2)[2] == ["10", "01"]
    assert result(got[3], 2)[0][1:3] == (0, 0)


def test_unsettled_vectors_are_counted_and_fixed_ticks_still_run():
    status, got = talk("ring", run_frame([], sample=P.NONE, width=0) + run_frame(["", ""], limit=50, flags=P.TICKS, width=0))
    assert status == 3
    head, ticks, _ = result(got[1], 0)
    assert head[3] == 2 and ticks == [P.UNSETTLED] * 2
    status, got = talk("ring", run_frame(["", ""], ticks=3, flags=P.TICKS, width=0))
    assert status == 0 and result(got[0], 0)[1] == [3, 3]


def test_x_in_one_bit_is_flagged_lossy():
    _, got = talk("latch", run_frame(["11"]) + frame(P.READ, flags=P.OUT_1BIT), noise=0)
    assert result(got[0], 2)[2] == ["XX"]  # no power-on noise: the latch never picks a side
    assert got[1][1] & P.LOSSY and result(got[1], 2)[2] == ["00"]


def test_bad_frames_get_an_error_and_the_stream_goes_on():
    data = (
        frame(0x42)  # 0: unknown op
        + frame(P.READ, flags=P.TICKS)  # 1: a flag READ doesn't take
        + frame(P.STEP, b"\0")  # 2: wrong size
        + run_frame(["01"], out=5)  # 3: no such port
        + frame(P.PORT, P.PORT_HEAD.pack(P.IN, 1) + struct.pack("<I", 9), port=1)  # 4: no pin 9
        + frame(P.PORT, P.PORT_HEAD.pack(P.IN, 1) + struct.pack("<I", 0), port=1)  # 5: fine
        + run_frame(["01"], out=1)  # 6: an input port as outputs
        + frame(P.RUN, P.RUN_HEAD.pack(1, 0, P.EVERY, 0, 0))  # 7: one vector, but no bytes for it
        + run_frame(["01"])  # 8: fine
    )
    status, got = talk("latch", data)
    assert status == 2
    errors = [(P.ERROR_HEAD.unpack_from(b)[0], b[4:].decode()) for op, _, _, b in got if op == P.ERROR]
    assert [f for f, _ in errors] == [0, 1, 2, 3, 4, 6, 7]
    assert "0x42" in errors[0][1] and "no port 5" in errors[3][1] and "input port" in errors[5][1]
    assert result(got[-1], 2)[0][0] == 8 and result(got[-1], 2)[2] == ["10"]


def test_a_stream_cut_off_mid_frame_ends_it():
    status, got = talk("latch", run_frame(["01"])[:-1])
    assert status == 2 and got[-1][0] == P.ERROR and b"ends inside a frame" in got[-1][3]
    status, got = talk("latch", b"\x05\x00\x00")
    assert status == 2 and b"frame header" in got[-1][3]
    status, got = talk("latch", P.HEADER.pack(P.RUN, 0, 0, 0xFFFFFFFF))
    assert status == 2 and b"isn't a frame stream" in got[-1][3]


def test_drive_step_read_and_reset():
    reset = frame(P.RESET, P.RESET_BODY.pack(5, 64))
    one = frame(P.DRIVE, b"\x02", P.IN_1BIT) + frame(P.STEP, P.STEP_BODY.pack(0, 0)) + frame(P.READ)
    status, got = talk("latch", one + reset + frame(P.READ) + one, seed=5)
    assert status == 0
    assert result(got[1], 2)[2] == ["10"]  # s=0 r=1: set
    head, _, _ = result(got[0], 2)
    assert head[1:3] == (1, 0)
    assert result(got[2], 2)[2] == result(talk("latch", frame(P.READ), seed=5)[1][0], 2)[2]  # as it powered on
    assert result(got[4], 2)[2] == ["10"]


def test_the_client_end_to_end(project):  # noqa: F811
    with P.Client("half adder", project="bench") as p:
        assert (p.title, p.inputs, p.outputs) == ("half adder", ("a", "2"), ("sum", "carry"))
        r = p.run(["00", "01", "10", "11"], want_ticks=True)
        assert r.strings() == ["00", "10", "10", "01"] and r.unsettled == 0 and len(r.ticks) == 4
        both = p.port(["a", "#2"])
        carry = p.port(["carry"], side="out")
        assert p.run([0, 1, 2, 3], port=both, out=carry, out_1bit=True).ints() == [0, 0, 0, 1]
        f = p.submit(np.array([[1, 1], [0, 1]]))
        g = p.submit(["1X"], sample="last")
        assert p.receive().frame == f
        assert p.receive().strings() == ["XX"]
        assert g == f + 1
        with pytest.raises(ValueError):
            p.run(["11"], out=both)  # caught by the client: an input port
        p._send(P.PORT, 0, 0, P.PORT_HEAD.pack(P.IN, 0))  # the child refuses: port 0 is fixed
        with pytest.raises(P.PipeError, match="port 0") as e:
            p.read()
        assert e.value.frame == p.frames - 2
        assert p.receive().strings() == ["XX"]  # (the read's own answer, still to come)
        p.drive("11")
        assert p.read().strings() == ["XX"]  # not run yet: still what "1X" gave
        assert p.step().n == 1 and p.read().strings() == ["01"]
        with pytest.raises(ValueError):
            p.run([4], port=both)
    assert p.close() == 2  # a frame failed (the PORT)
