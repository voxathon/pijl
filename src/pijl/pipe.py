"""The binary pipe: a macro held as a co-process, spoken to in framed, packed,
batched binary messages. `pijl run MACRO --bin` is the child's end (serve);
Client is a Python parent's end. The whole spec, with byte-level examples, is in
HEADLESS.md; this is the short of it.

Every message is a frame: an 8-byte header, then `length` bytes of payload. All
numbers are little-endian.

    op u8 | flags u8 | port u16 | length u32 | payload ...

The child speaks first (HELLO: its pins), then answers frames in the order they
come. Frames from the client are numbered from 0, every op counted; RESULT and
ERROR name the frame they answer. A frame that can't be done gets an ERROR and
changes nothing, and the stream goes on; only a stream cut off mid-frame (or a
header no frame could have) ends it.

    client -> pijl                          pijl -> client
    PORT  0x01 name a group of pins         HELLO  0x80 title, engine, pin names
    RUN   0x02 a batch of vectors           RESULT 0x81 what RUN/STEP/READ got
    DRIVE 0x03 set inputs, no run           ERROR  0x82 a frame that failed
    STEP  0x04 run without driving
    READ  0x05 sample outputs
    RESET 0x06 power on again

Levels go 2 bits a pin as the engine's own codes (Z=0, 0=1, 1=2, X=3: bit 0 is
"could be 0", bit 1 "could be 1"), four pins a byte; or, with the 1-bit flags,
1 bit a pin (0/1 only), eight a byte. Pin i of a port is byte i // per-byte, low
bits first, so a 1-bit vector is the port's value as a little-endian integer,
pin 0 its least significant bit. Each vector starts on a byte of its own.

Port 0 is every pin of whichever side a frame wants (inputs to drive, outputs to
sample); PORT names others, by pin index, on one side.

A RUN of n vectors does exactly what n RUNs of one vector each would: drive,
run (exactly `ticks`, or until stable), sample if asked. Only the answers are
batched.
"""

from __future__ import annotations

import queue
import struct
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np

from .engine import _CHAR_OF, _CODE_OF

if TYPE_CHECKING:
    from .engine import Harness

VERSION = 1
MAGIC = b"PIJL"

HEADER = struct.Struct("<BBHI")  # op, flags, port, payload length
MAX_FRAME = 1 << 30  # a longer "frame" is a stream that isn't one

# client -> pijl
PORT, RUN, DRIVE, STEP, READ, RESET = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06
# pijl -> client
HELLO, RESULT, ERROR = 0x80, 0x81, 0x82

# flags
IN_1BIT = 0x01  # RUN, DRIVE: vectors in are 1 bit a pin (0 / 1 only)
OUT_1BIT = 0x02  # RUN, READ: samples out are 1 bit a pin; echoed on RESULT
TICKS = 0x04  # RUN: RESULT lists the ticks each vector took; echoed on that RESULT
LOSSY = 0x08  # RESULT: a 1-bit sample held an X or a Z (sent as 0)

# RUN: which vectors' outputs are sampled
NONE, EVERY, LAST, MASK = 0, 1, 2, 3

UNSETTLED = 0xFFFFFFFF  # a vector's ticks, when it didn't settle within the limit

HELLO_HEAD = struct.Struct("<4sHHII")  # magic, version, reserved, inputs, outputs
PORT_HEAD = struct.Struct("<BxxxI")  # side (0 in, 1 out), pin count; then u32 pin indices
RUN_HEAD = struct.Struct("<IHBxII")  # count, out port, sample mode, ticks (0: settle), limit (0: default)
STEP_BODY = struct.Struct("<II")  # ticks (0: settle), limit (0: default)
RESET_BODY = struct.Struct("<QI")  # seed, power-on noise ticks
RESULT_HEAD = struct.Struct("<IIIII")  # frame, vectors run, samples, unsettled, most ticks
ERROR_HEAD = struct.Struct("<I")  # frame; then a UTF-8 message

IN, OUT = 0, 1
_FLAGS = {PORT: 0, RUN: IN_1BIT | OUT_1BIT | TICKS, DRIVE: IN_1BIT, STEP: 0, READ: OUT_1BIT, RESET: 0}
_NAMES = {PORT: "PORT", RUN: "RUN", DRIVE: "DRIVE", STEP: "STEP", READ: "READ", RESET: "RESET"}


# ---- packing ---------------------------------------------------------------------------


def stride(width: int, one_bit: bool) -> int:
    """Bytes a vector of `width` pins takes."""
    return (width + 7) // 8 if one_bit else (width + 3) // 4


def pack(codes: np.ndarray, one_bit: bool) -> tuple[bytes, bool]:
    """(n, width) level codes -> n packed vectors; and, for 1 bit, whether an X or a
    Z had to go out as 0."""
    n, width = codes.shape
    if one_bit:
        lossy = bool(((codes != 1) & (codes != 2)).any())
        return np.packbits(codes == 2, axis=1, bitorder="little").tobytes(), lossy
    s = stride(width, False)
    q = np.zeros((n, s * 4), np.uint8)
    q[:, :width] = codes
    q = q.reshape(n, s, 4)
    return (q[..., 0] | q[..., 1] << 2 | q[..., 2] << 4 | q[..., 3] << 6).tobytes(), False


def unpack(data: bytes | memoryview, n: int, width: int, one_bit: bool) -> np.ndarray:
    """n packed vectors -> (n, width) level codes."""
    s = stride(width, one_bit)
    if not n * s:
        return np.zeros((n, width), np.uint8)
    raw = np.frombuffer(data, np.uint8, n * s).reshape(n, s)
    if one_bit:
        return np.unpackbits(raw, axis=1, count=width, bitorder="little") + np.uint8(1)
    q = np.stack((raw & 3, raw >> 2 & 3, raw >> 4 & 3, raw >> 6), axis=2).reshape(n, s * 4)
    return q[:, :width]


# ---- pijl's end ------------------------------------------------------------------------


class _Bad(Exception):
    """A frame that can't be done: answered with an ERROR, nothing changed."""


class _Server:
    def __init__(
        self,
        make: Callable[[int, int], Harness],
        noise: int,
        seed: int,
        out: BinaryIO,
        max_ticks: int,
        warn: Callable[[str], None],
    ) -> None:
        self.make = make
        self.h = make(noise, seed)
        self.out = out
        self.max_ticks = max_ticks
        self.warn = warn
        self.ports: dict[int, tuple[int, np.ndarray]] = {}  # id -> side, pin indices
        self.frame = 0
        self.unsettled = False
        self.reported: set[str] = set()

    def send(self, op: int, flags: int, port: int, *parts: bytes) -> None:
        self.out.write(HEADER.pack(op, flags, port, sum(map(len, parts))))
        for p in parts:
            self.out.write(p)
        self.out.flush()

    def hello(self) -> None:
        h = self.h
        names = [h.macro.title, str(h.circuit.config), *h.inputs, *h.outputs]
        body = b"".join(n.encode() + b"\0" for n in names)
        self.send(HELLO, 0, 0, HELLO_HEAD.pack(MAGIC, VERSION, 0, len(h.inputs), len(h.outputs)), body)

    def error(self, msg: str) -> None:
        self.send(ERROR, 0, 0, ERROR_HEAD.pack(self.frame), msg.encode())

    def notice(self) -> None:
        for problem in self.h.problems:
            if problem not in self.reported:
                self.reported.add(problem)
                self.warn(f"part script {problem} (that kind is off now)")

    def slots(self, port: int, side: int) -> np.ndarray:
        """A port's pin slots in the circuit (they change on RESET; indices don't)."""
        h = self.h
        every = h._in_slots if side == IN else h._out_slots
        if port == 0:
            return every
        if port not in self.ports:
            raise _Bad(f"no port {port}")
        has, idx = self.ports[port]
        if has != side:
            raise _Bad(f"port {port} is an {('input', 'output')[has]} port; this needs an {('input', 'output')[side]} one")
        return every[idx]

    def do(self, op: int, flags: int, port: int, body: bytes) -> None:
        if op not in _FLAGS:
            raise _Bad(f"unknown op 0x{op:02x}")
        if flags & ~_FLAGS[op]:
            raise _Bad(f"{_NAMES[op]} takes no flags 0x{flags & ~_FLAGS[op]:02x}")
        getattr(self, _NAMES[op].lower())(flags, port, body)

    def _size(self, what: str, body: bytes, want: int) -> None:
        if len(body) != want:
            raise _Bad(f"{what}: {len(body)} payload bytes, expected {want}")

    def port(self, flags: int, port: int, body: bytes) -> None:
        if port == 0:
            raise _Bad("port 0 is every pin; it can't be redefined")
        if len(body) < PORT_HEAD.size:
            raise _Bad(f"PORT: {len(body)} payload bytes, expected at least {PORT_HEAD.size}")
        side, count = PORT_HEAD.unpack_from(body)
        self._size("PORT", body, PORT_HEAD.size + 4 * count)
        if side not in (IN, OUT):
            raise _Bad(f"PORT: side {side} (0 is inputs, 1 outputs)")
        idx = np.frombuffer(body, "<u4", count, PORT_HEAD.size).astype(np.intp)
        n = len(self.h.inputs if side == IN else self.h.outputs)
        if (idx >= n).any():
            raise _Bad(f"PORT: pin {int(idx.max())}, but there are {n} {('inputs', 'outputs')[side]} (from 0)")
        self.ports[port] = (side, idx)

    def run(self, flags: int, port: int, body: bytes) -> None:
        if len(body) < RUN_HEAD.size:
            raise _Bad(f"RUN: {len(body)} payload bytes, expected at least {RUN_HEAD.size}")
        count, out_port, sample, ticks, limit = RUN_HEAD.unpack_from(body)
        ins, outs = self.slots(port, IN), self.slots(out_port, OUT)
        pos = RUN_HEAD.size
        take = np.zeros(count, bool)
        if sample == MASK:
            m = (count + 7) // 8
            if len(body) < pos + m:
                raise _Bad(f"RUN: {len(body)} payload bytes; the mask alone ends at {pos + m}")
            take[:] = np.unpackbits(np.frombuffer(body, np.uint8, m, pos), count=count, bitorder="little")
            pos += m
        elif sample == EVERY:
            take[:] = True
        elif sample == LAST:
            take[-1:] = True
        elif sample != NONE:
            raise _Bad(f"RUN: sample mode {sample} (0 none, 1 every, 2 last, 3 mask)")
        one = bool(flags & IN_1BIT)
        s = stride(len(ins), one)
        self._size(f"RUN of {count} vectors, {s} bytes each", body, pos + count * s)
        codes = unpack(memoryview(body)[pos:], count, len(ins), one)
        samples = np.empty((int(take.sum()), len(outs)), np.uint8)
        per = np.empty(count, "<u4") if flags & TICKS else None
        c = self.h.circuit
        write, step, settle = c.write_pins, c.step, c.run_until_stable
        limit = limit or self.max_ticks
        unsettled = most = k = 0
        # only the pins a vector changes are written: a write is a poke, and the
        # dirty stepper runs whatever a poked pin feeds, changed or not
        before = np.vstack((c._pins.states[ins][None], codes[:-1]))
        for i, sampled in enumerate(take.tolist()):
            moved = np.flatnonzero(codes[i] != before[i])
            if moved.size:
                write(ins[moved], codes[i][moved])
            if ticks:
                for _ in range(ticks):
                    step()
                t = ticks
            else:
                t = settle(limit)
                if t is None:
                    unsettled += 1
                    t = UNSETTLED
            if t != UNSETTLED and t > most:
                most = t
            if per is not None:
                per[i] = t
            if sampled:
                samples[k] = c._pins.states[outs]
                k += 1
        self.unsettled |= bool(unsettled)
        self.result(flags & OUT_1BIT, out_port, count, samples, unsettled, most, per)

    def result(self, out_1bit: int, port: int, n: int, samples: np.ndarray, unsettled: int, most: int, per=None) -> None:
        packed, lossy = pack(samples, bool(out_1bit))
        head = RESULT_HEAD.pack(self.frame, n, len(samples), unsettled, most)
        ticks = per.tobytes() if per is not None else b""
        flags = out_1bit | (LOSSY if lossy else 0) | (TICKS if per is not None else 0)
        self.send(RESULT, flags, port, head, ticks, packed)

    def drive(self, flags: int, port: int, body: bytes) -> None:
        ins = self.slots(port, IN)
        one = bool(flags & IN_1BIT)
        self._size("DRIVE", body, stride(len(ins), one))
        c = self.h.circuit
        codes = unpack(body, 1, len(ins), one)[0]
        moved = np.flatnonzero(codes != c._pins.states[ins])
        c.write_pins(ins[moved], codes[moved])

    def step(self, flags: int, port: int, body: bytes) -> None:
        self._size("STEP", body, STEP_BODY.size)
        ticks, limit = STEP_BODY.unpack(body)
        c = self.h.circuit
        if ticks:
            for _ in range(ticks):
                c.step()
            t = ticks
        else:
            t = c.run_until_stable(limit or self.max_ticks)
        self.unsettled |= t is None
        self.result(0, 0, 1, np.zeros((0, 0), np.uint8), int(t is None), t or 0)

    def read(self, flags: int, port: int, body: bytes) -> None:
        self._size("READ", body, 0)
        outs = self.slots(port, OUT)
        self.result(flags & OUT_1BIT, port, 0, self.h.circuit._pins.states[outs][None], 0, 0)

    def reset(self, flags: int, port: int, body: bytes) -> None:
        self._size("RESET", body, RESET_BODY.size)
        seed, noise = RESET_BODY.unpack(body)
        self.h = self.make(noise, seed)


def _read(inp: BinaryIO, n: int) -> bytes:
    """n bytes, or fewer only at the end of the stream."""
    data = inp.read(n)
    if data is None or len(data) == n or not data:
        return data or b""
    parts = [data]
    got = len(data)
    while got < n and (more := inp.read(n - got)):
        parts.append(more)
        got += len(more)
    return b"".join(parts)


def serve(
    make: Callable[[int, int], Harness],
    noise: int,
    seed: int,
    inp: BinaryIO,
    out: BinaryIO,
    max_ticks: int = 10_000,
    warn: Callable[[str], None] = lambda msg: print(f"pijl: warning: {msg}", file=sys.stderr),
) -> int:
    """`pijl run MACRO --bin`: answer frames on `inp` until it ends. `make(noise,
    seed)` builds the harness (again, on RESET). Returns the exit status: 2 if any
    frame failed or the stream broke off, else 3 if anything didn't settle, else 0."""
    sv = _Server(make, noise, seed, out, max_ticks, warn)
    sv.hello()
    status = 0
    while head := _read(inp, HEADER.size):
        if len(head) < HEADER.size:
            sv.error(f"the stream ends inside a frame header ({len(head)} of {HEADER.size} bytes)")
            return 2
        op, flags, port, length = HEADER.unpack(head)
        if length > MAX_FRAME:
            sv.error(f"a frame of {length} bytes: this isn't a frame stream")
            return 2
        body = _read(inp, length)
        if len(body) < length:
            sv.error(f"the stream ends inside a frame ({len(body)} of {length} payload bytes)")
            return 2
        try:
            sv.do(op, flags, port, body)
        except _Bad as e:
            status = 2
            sv.error(str(e))
        sv.notice()
        sv.frame += 1
    return status or (3 if sv.unsettled else 0)


# ---- the client's end ------------------------------------------------------------------


class PipeError(Exception):
    """The child answered a frame with an ERROR (or went away)."""

    def __init__(self, frame: int | None, message: str) -> None:
        super().__init__(message if frame is None else f"frame {frame}: {message}")
        self.frame = frame
        self.message = message


@dataclass
class Result:
    """What a RUN, STEP or READ got back."""

    frame: int  # the frame it answers
    port: int  # the output port sampled
    n: int  # vectors run (STEP: 1, READ: 0)
    codes: np.ndarray  # (samples, width) level codes: Z=0, 0=1, 1=2, X=3
    unsettled: int  # how many of the n didn't settle
    most: int  # the most ticks any settled one took
    ticks: np.ndarray | None  # each vector's ticks (UNSETTLED: didn't), if asked
    lossy: bool  # 1-bit samples, and an X or Z went out as 0

    def strings(self) -> list[str]:
        """Each sample as a string of 0 / 1 / X / Z, pin 0 first."""
        w = self.codes.shape[1]
        text = _CHAR_OF[self.codes].tobytes().decode()
        return [text[i : i + w] for i in range(0, len(text), w)] if w else [""] * len(self.codes)

    def bits(self) -> np.ndarray:
        """(samples, width) of 0 / 1. ValueError if there's an X or a Z."""
        if ((self.codes != 1) & (self.codes != 2)).any():
            raise ValueError("a sample holds X or Z; use .strings() or .codes")
        return self.codes - np.uint8(1)

    def ints(self) -> list[int]:
        """Each sample as an integer, pin 0 the least significant bit."""
        packed = np.packbits(self.bits(), axis=1, bitorder="little")
        return [int.from_bytes(row.tobytes(), "little") for row in packed]


def command() -> list[str]:
    """How to start pijl again: this interpreter -m pijl, or a frozen build itself."""
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "pijl"]


def _vectors(values: Any, width: int) -> tuple[np.ndarray, bool]:
    """Vectors as (n, width) level codes, and whether they're all 0 / 1. Takes:
    strings of 0 / 1 / X / Z; ints (the port's value, pin 0 the low bit); or a 2-D
    array (or list of lists) of 0s and 1s."""
    if not isinstance(values, np.ndarray):
        values = list(values)
        if not values:
            return np.zeros((0, width), np.uint8), True
    first = values[0] if len(values) else None
    if isinstance(first, str):
        if any(len(v) != width for v in values):
            raise ValueError(f"a vector string must have {width} levels")
        codes = _CODE_OF[np.frombuffer("".join(values).encode(), np.uint8)]
        if (codes == 255).any():
            raise ValueError("levels are 0, 1, X and Z")
        codes = codes.reshape(len(values), width)
    elif isinstance(first, (int, np.integer)) and not isinstance(first, bool) and np.ndim(values) == 1:
        s = (width + 7) // 8
        try:
            raw = b"".join(int(v).to_bytes(s, "little") for v in values)
        except OverflowError:
            raise ValueError(f"a value doesn't fit {width} pins") from None
        if any(v < 0 for v in values):
            raise ValueError("port values can't be negative")
        bits = np.unpackbits(np.frombuffer(raw, np.uint8).reshape(len(values), s), axis=1, count=width, bitorder="little")
        if s * 8 > width and any(int(v) >> width for v in values):
            raise ValueError(f"a value doesn't fit {width} pins")
        return bits + np.uint8(1), True
    else:
        bits = np.asarray(values)
        if bits.ndim != 2 or bits.shape[1] != width:
            raise ValueError(f"expected vectors of {width} pins, got shape {bits.shape}")
        if bits.size and (bits.min() < 0 or bits.max() > 1):
            raise ValueError("a bit array holds 0s and 1s (strings can carry X and Z)")
        return bits.astype(np.uint8) + np.uint8(1), True
    return codes, bool(((codes == 1) | (codes == 2)).all())


class Client:
    """A macro in a child `pijl run MACRO --bin`, from Python:

        with Client("half adder") as p:
            p.run(["00", "01", "10", "11"]).strings()   # ["00", "10", "10", "01"]
            carry = p.port(["carry"], side="out")
            p.run([0, 1, 2, 3], out=carry, out_1bit=True).ints()   # [0, 0, 0, 1]

    run() sends a batch and waits for its answer; submit() / receive() split the
    two so batches can be kept in flight (a thread reads the child's answers, so
    writing ahead never deadlocks). Answers come in the order frames went out; an
    ERROR comes back as PipeError from the receive() that meets it."""

    def __init__(
        self,
        macro: str,
        *,
        project: str | None = None,
        data: str | None = None,
        engine: str | None = None,
        noise: int = 64,
        seed: int = 0,
        max_ticks: int | None = None,
        stderr: Any = None,
        cmd: Sequence[str] | None = None,
    ) -> None:
        args = list(cmd or command())
        for flag, value in (("--data", data), ("-p", project), ("--engine", engine)):
            if value is not None:
                args += [flag, value]
        args += ["run", macro, "--bin", "--noise", str(noise), "--seed", str(seed)]
        if max_ticks is not None:
            args += ["--max-ticks", str(max_ticks)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        self.frames = 0  # sent
        self._answers: queue.Queue = queue.Queue()
        self._widths: dict[int, tuple[int, int]] = {}  # port -> side, width
        self._next_port = 1
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        op, _, _, body = self._get()
        if op != HELLO or len(body) < HELLO_HEAD.size:
            raise PipeError(None, "the child didn't say HELLO")
        magic, version, _, n_in, n_out = HELLO_HEAD.unpack_from(body)
        if magic != MAGIC or version != VERSION:
            raise PipeError(None, f"the child speaks {magic!r} version {version}, not {MAGIC!r} {VERSION}")
        names = body[HELLO_HEAD.size :].split(b"\0")[:-1]
        if len(names) != 2 + n_in + n_out:
            raise PipeError(None, "a garbled HELLO")
        text = [n.decode() for n in names]
        self.title, self.engine = text[0], text[1]
        self.inputs: tuple[str, ...] = tuple(text[2 : 2 + n_in])
        self.outputs: tuple[str, ...] = tuple(text[2 + n_in :])

    def _read(self) -> None:
        out = self.proc.stdout
        while True:
            head = _read(out, HEADER.size)
            if len(head) < HEADER.size:
                break
            op, flags, port, length = HEADER.unpack(head)
            body = _read(out, length)
            if len(body) < length:
                break
            self._answers.put((op, flags, port, body))
        self._answers.put(None)

    def _get(self) -> tuple[int, int, int, bytes]:
        got = self._answers.get()
        if got is None:
            self._answers.put(None)  # (the next one gets it too)
            raise PipeError(None, f"the child went away (exit status {self.proc.wait()})")
        return got

    def _send(self, op: int, flags: int, port: int, *parts: bytes) -> int:
        frame = self.frames
        self.proc.stdin.write(HEADER.pack(op, flags, port, sum(map(len, parts))) + b"".join(parts))
        self.proc.stdin.flush()
        self.frames += 1
        return frame

    def _width(self, port: int, side: int) -> int:
        if port == 0:
            return len(self.inputs if side == IN else self.outputs)
        if port not in self._widths:
            raise KeyError(f"no port {port}")
        has, width = self._widths[port]
        if has != side:
            raise ValueError(f"port {port} is an {('input', 'output')[has]} port")
        return width

    def pin(self, key: str | int, side: str = "in") -> int:
        """A pin's index: its name, "#n" (from 1), or an index (from 0)."""
        names = self.inputs if side == "in" else self.outputs
        if isinstance(key, int) and 0 <= key < len(names):
            return key
        if isinstance(key, str):
            if key in names:
                return names.index(key)
            if key.startswith("#") and key[1:].isdigit() and 1 <= int(key[1:]) <= len(names):
                return int(key[1:]) - 1
        raise KeyError(f"{self.title} has no {'input' if side == 'in' else 'output'} {key!r}")

    def port(self, pins: Iterable[str | int], side: str = "in") -> int:
        """Name a group of pins (names, "#n" or indices; the first is pin 0 of the
        port, the low bit of its value). Returns its number."""
        if side not in ("in", "out"):
            raise ValueError("side is 'in' or 'out'")
        idx = [self.pin(p, side) for p in pins]
        s = IN if side == "in" else OUT
        id = self._next_port
        self._next_port += 1
        self._send(PORT, 0, id, PORT_HEAD.pack(s, len(idx)), np.asarray(idx, "<u4").tobytes())
        self._widths[id] = (s, len(idx))
        return id

    def submit(
        self,
        vectors: Any,
        *,
        port: int = 0,
        out: int = 0,
        ticks: int | None = None,
        limit: int | None = None,
        sample: str | None | Sequence[bool] | np.ndarray = "every",
        out_1bit: bool = False,
        want_ticks: bool = False,
    ) -> int:
        """Send a RUN; its answer comes from a later receive(). Returns the frame's
        number. `ticks`: exactly that many a vector (None: until stable, giving up
        after `limit`). `sample`: "every", "last", None, or a mask, one per vector."""
        if ticks is not None and ticks < 1:
            raise ValueError("ticks: at least 1 (drive() changes inputs without running)")
        codes, one = _vectors(vectors, self._width(port, IN))
        self._width(out, OUT)
        n = len(codes)
        mask = b""
        if isinstance(sample, str) or sample is None:
            modes = {"every": EVERY, "last": LAST, None: NONE, "none": NONE}
            if sample not in modes:
                raise ValueError(f"sample: {sample!r} (every, last, none, or a mask)")
            mode = modes[sample]
        else:
            m = np.asarray(sample, bool)
            if m.shape != (n,):
                raise ValueError(f"a sample mask needs one entry per vector ({n})")
            mode, mask = MASK, np.packbits(m, bitorder="little").tobytes()
        packed, _ = pack(codes, one)
        flags = (IN_1BIT if one else 0) | (OUT_1BIT if out_1bit else 0) | (TICKS if want_ticks else 0)
        head = RUN_HEAD.pack(n, out, mode, ticks or 0, limit or 0)
        return self._send(RUN, flags, port, head, mask, packed)

    def receive(self) -> Result:
        """The next answer (a RESULT); PipeError if it's an ERROR."""
        op, flags, port, body = self._get()
        if op == ERROR:
            (frame,) = ERROR_HEAD.unpack_from(body)
            raise PipeError(frame, body[ERROR_HEAD.size :].decode(errors="replace"))
        if op != RESULT:
            raise PipeError(None, f"unexpected op 0x{op:02x} from the child")
        frame, n, samples, unsettled, most = RESULT_HEAD.unpack_from(body)
        pos = RESULT_HEAD.size
        ticks = None
        if flags & TICKS:
            ticks = np.frombuffer(body, "<u4", n, pos).copy()
            pos += 4 * n
        width, one = self._width(port, OUT), bool(flags & OUT_1BIT)
        codes = unpack(memoryview(body)[pos:], samples, width, one).copy()
        return Result(frame, port, n, codes, unsettled, most, ticks, bool(flags & LOSSY))

    def run(self, vectors: Any, **kw: Any) -> Result:
        """submit() and wait for its answer. Takes what submit() takes."""
        self.submit(vectors, **kw)
        return self.receive()

    def drive(self, vector: Any, port: int = 0) -> None:
        """Set inputs (one vector, as run() takes them), without running."""
        codes, one = _vectors([vector] if isinstance(vector, (str, int)) else [list(vector)], self._width(port, IN))
        self._send(DRIVE, IN_1BIT if one else 0, port, pack(codes, one)[0])

    def step(self, ticks: int | None = None, limit: int | None = None) -> Result:
        """Run without driving anything: exactly `ticks`, or until stable."""
        self._send(STEP, 0, 0, STEP_BODY.pack(ticks or 0, limit or 0))
        return self.receive()

    def read(self, out: int = 0, out_1bit: bool = False) -> Result:
        """Sample outputs now, without running."""
        self._width(out, OUT)
        self._send(READ, OUT_1BIT if out_1bit else 0, out)
        return self.receive()

    def reset(self, seed: int = 0, noise: int = 64) -> None:
        """Power on again: a fresh circuit, as the child started with. Ports stay."""
        self._send(RESET, 0, 0, RESET_BODY.pack(seed, noise))

    def close(self) -> int:
        """Hang up; the child's exit status."""
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        status = self.proc.wait()
        self._reader.join(5)
        return status

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
