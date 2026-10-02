# Headless pijl

pijl runs macros with no window: from Python (`pijl.engine`), from the command
line (`pijl run`), or as a co-process another program drives over a pipe (text
lines, a raw character stream, or framed binary). None of it imports pyglet, and
none of it writes to the project.

- [Python: Engine and Harness](#python-engine-and-harness)
- [The command line](#the-command-line)
- [Streams: lines and raw](#streams-lines-and-raw)
- [The binary pipe](#the-binary-pipe)
- [The Python client](#the-python-client)
- [Benchmarks](#benchmarks)

Levels are always one of four: `0`, `1`, `X` (unknown, or fighting drivers) and
`Z` (undriven).

## Python: Engine and Harness

```python
from pijl.engine import Engine

eng = Engine()                     # the project the editor had open last
ha = eng.harness("half adder")     # a macro by title or id, or a path to a .json
ha.set(a=1, b=0)                   # drive inputs (by name, "#n" from 1, or index)
ha.settle()                        # step until nothing changes; False if it gave up
ha.read()                          # {"sum": Level.ONE, "carry": Level.ZERO}
ha.apply({"a": 1, "b": 1})         # set + settle + read
```

`Engine(project)` opens a project read-only: a name under the data root, or a
path to a project folder; a project that doesn't exist is an error, not created.
`eng.macros()` lists id -> title; `eng.problems` collects warnings (part scripts
that failed to load, macros with parts missing).

A **Harness** is one macro on a board of its own, with an IN switch driving each
input and an OUT on each output. It keeps its state between calls, so latches
and counters work. Its methods:

| | |
|---|---|
| `set(...)`, `set_all(values)` | drive inputs (not run yet) |
| `set_bits("01XZ", start=0)` | drive inputs from a string, the fast way |
| `step(n)` | exactly n ticks (one gate delay each) |
| `settle(limit)` | step until stable; `last_ticks` is what it took (None: gave up) |
| `read()`, `get(name)`, `bits()` | outputs: by name, one, or a string in pin order |
| `driven()` | what the inputs are driven with |
| `apply(values, ticks=None)` | set, run, read |
| `truth_table()` | every 0/1 input combination, in counting order |
| `frame()` | run the parts' frame hooks once, as the editor does every frame |
| `problems` | part scripts that raised while running (that kind is off now) |

`eng.harness(name, settle_ticks=64, seed=0, config=None)`: `settle_ticks` is the
power-on noise (as in the editor: latches pick a side; 0 for none), `seed` seeds
it, `config` is the engine's code paths (see `pijl/sim/config.py`).

Pins are named like the macro's pins (the ports' labels; unlabeled ones are
numbered), in pin order.

## The command line

```bash
pijl list                          # the project's macros and their pins
pijl list --projects
pijl run "half adder" a=1 b=0      # one shot -> sum=1 carry=0
pijl run "half adder" 10           # all inputs in pin order, as bits
pijl run "half adder" --table      # every 0/1 combination
pijl run counter -                 # a line stream (below)
pijl run counter --raw             # a raw stream (below)
pijl run counter --bin             # the binary pipe (below)
```

Global options, before or after the command: `-p/--project NAME|PATH`,
`--data DIR` (the data root, like `PIJL_DATA`), `--engine OPTIONS` (like
`PIJL_ENGINE`, e.g. `dirty=off`).

`run` options: `--json` (answers as JSON), `--ticks N` (exactly N ticks per input
instead of until stable), `--max-ticks N` (give up settling after N, default
10000), `--noise N` (power-on noise ticks, default 64), `--seed N`.

Exit status: 0 fine, 1 couldn't run, 2 bad arguments or input, 3 something
didn't settle (the outputs are printed anyway).

## Streams: lines and raw

Both read stdin and answer on stdout, flushing every answer, and keep the
circuit's state from one input to the next.

**Lines** (`pijl run MACRO -`): one answer per line. A line is `a=1 b=X` (only
those inputs change), a string of bits for all of them, a JSON object
(`{"a": 1, "b": "Z"}`, answered in JSON), `step N` (just let N ticks pass), or
blank / `#...` (ignored). After each line the circuit runs until stable (or
`--ticks`), then the outputs are printed. A bad line gets an error on stderr and
changes nothing; the exit status is then 2.

**Raw** (`pijl run MACRO --raw`): no lines and no names. Each level character
(`0 1 X Z`) drives the next input in pin order; when every input has one, the
vector is applied and run. `;` applies a partial vector early (alone: just run).
`@n=L` drives input n (from 1) without running. `?` answers every output's level,
one character each, no newline. Whitespace is ignored. The first bad character
ends the stream (status 2).

## The binary pipe

`pijl run MACRO --bin` speaks framed binary on stdin / stdout. It takes no input
values, `--table`, `--raw`, `--json` or `--ticks` (frames carry all of that);
`--noise`, `--seed` and `--max-ticks` still apply.

What it adds over the streams:

- **Batches**: one RUN frame carries any number of input vectors and gets one
  RESULT back. A RUN of n vectors does exactly what n RUNs of one vector would:
  drive, run, sample. Only the answers are batched.
- **Packing**: 2 bits a pin (any level), or 1 bit a pin (0 and 1 only).
- **Ports**: named groups of pins; a 1-bit vector on a port is the port's value
  as an integer.
- **Sampling**: which vectors' outputs come back (every one, the last, none, or
  a mask), and from which pins.
- **Status**: how many vectors didn't settle, and optionally each vector's ticks.
- **Errors as frames**: a frame that can't be done is answered with an ERROR
  and changes nothing; the stream goes on.

### Frames

Every message, both ways, is a frame: an 8-byte header and a payload. All
numbers are little-endian, all structures packed (no padding but what's listed).

| offset | size | field |
|---:|---:|---|
| 0 | 1 | `op` |
| 1 | 1 | `flags` |
| 2 | 2 | `port` |
| 4 | 4 | `length`: payload bytes that follow |

The child speaks first (HELLO), then answers frames in the order they arrive.
Frames from the client are numbered from 0, counting every op; RESULT and ERROR
carry the number of the frame they answer. PORT, DRIVE and RESET get no answer
unless they fail.

| op | name | direction | answer |
|---|---|---|---|
| `0x01` | PORT | client -> pijl | none |
| `0x02` | RUN | client -> pijl | RESULT |
| `0x03` | DRIVE | client -> pijl | none |
| `0x04` | STEP | client -> pijl | RESULT |
| `0x05` | READ | client -> pijl | RESULT |
| `0x06` | RESET | client -> pijl | none |
| `0x80` | HELLO | pijl -> client | (first, unasked) |
| `0x81` | RESULT | pijl -> client | |
| `0x82` | ERROR | pijl -> client | |

Flags:

| bit | name | on | meaning |
|---|---|---|---|
| `0x01` | IN_1BIT | RUN, DRIVE | input vectors are 1 bit a pin |
| `0x02` | OUT_1BIT | RUN, READ, RESULT | samples are 1 bit a pin |
| `0x04` | TICKS | RUN, RESULT | RESULT lists each vector's ticks |
| `0x08` | LOSSY | RESULT | a 1-bit sample held an X or Z, sent as 0 |

A flag an op doesn't take is an error.

### Levels and vectors

**2 bits a pin** uses the engine's own codes: `Z = 0b00`, `0 = 0b01`,
`1 = 0b10`, `X = 0b11` (bit 0: "could be 0", bit 1: "could be 1"). Four pins a
byte; pin i is in byte `i // 4`, bits `2 * (i % 4)` and up.

**1 bit a pin**: `0` or `1`, eight pins a byte; pin i is byte `i // 8`, bit
`i % 8`. So a 1-bit vector is the port's value as a little-endian integer, pin 0
the least significant bit.

Each vector starts on a byte of its own: a vector of w pins takes
`ceil(w / 4)` bytes (2-bit) or `ceil(w / 8)` (1-bit); unused high bits are
ignored on the way in and 0 on the way out.

A 1-bit sample can't hold X or Z: those go out as 0, and the RESULT has LOSSY set.

### Ports

Port 0 is every pin of whichever side a frame needs: all inputs to drive, all
outputs to sample. PORT defines others, each on one side. Port numbers are the
client's to choose (1 to 65535); defining one again replaces it. Ports survive
RESET.

### Ops

**HELLO** (pijl -> client, first frame):

| size | field |
|---:|---|
| 4 | magic `PIJL` |
| 2 | version (1) |
| 2 | reserved (0) |
| 4 | number of inputs |
| 4 | number of outputs |
| ... | NUL-terminated UTF-8: the macro's title, the engine config, each input's name, each output's name |

**PORT** (`port` = the port's number, not 0):

| size | field |
|---:|---|
| 1 | side: 0 inputs, 1 outputs |
| 3 | reserved |
| 4 | pin count |
| 4 each | pin indices (from 0) on that side; pin 0 of the port first |

**RUN** (`port` = the input port to drive; flags IN_1BIT, OUT_1BIT, TICKS):

| size | field |
|---:|---|
| 4 | vector count n |
| 2 | output port to sample |
| 1 | sample mode: 0 none, 1 every vector, 2 the last, 3 a mask |
| 1 | reserved |
| 4 | ticks: 0 runs each vector until stable, else exactly this many |
| 4 | limit: settling gives up after this many ticks; 0 is `--max-ticks` |
| `ceil(n / 8)` | (mode 3 only) the mask: vector i is sampled if bit `i % 8` of byte `i // 8` is set |
| n x stride | the vectors |

Each vector: its pins are driven (inputs outside the port keep their levels),
then the circuit runs, then the output port is read if the vector is sampled.

**RESULT** (`port` = the output port sampled; flags OUT_1BIT, TICKS, LOSSY):

| size | field |
|---:|---|
| 4 | the frame this answers |
| 4 | vectors run (STEP: 1, READ: 0) |
| 4 | samples s |
| 4 | how many of the vectors didn't settle |
| 4 | the most ticks a vector that settled took |
| 4 each | (TICKS only) each vector's ticks; `0xFFFFFFFF`: didn't settle |
| s x stride | the samples |

**DRIVE** (`port` = input port; flag IN_1BIT): one vector. Drives without running.

**STEP**: `ticks u32`, `limit u32` as in RUN. Runs once without driving; the
RESULT has n = 1 and no samples.

**READ** (`port` = output port; flag OUT_1BIT): no payload. Samples without
running; the RESULT has n = 0 and one sample.

**RESET**: `seed u64`, `noise u32`. Powers on again: a fresh circuit, as built at
start with that power-on noise and seed.

**ERROR**: `frame u32`, then a UTF-8 message.

### Errors and the end

A frame that can't be done (unknown op, a flag it doesn't take, the wrong payload
size, no such port, a port on the wrong side, a pin out of range) is answered
with an ERROR and changes nothing; the next frame is read as usual. A stream that
ends inside a frame, or a header whose length is over 2^30 bytes, gets an ERROR
and ends the child.

The child exits when stdin ends: status 2 if any frame failed, else 3 if anything
didn't settle, else 0. Part-script failures and other warnings go to stderr.

### An example, byte by byte

`pijl run "half adder" --bin --noise 0` (inputs `a`, `2`; outputs `sum`, `carry`).

The child's HELLO:

```
80 00 0000 41000000                        HELLO, 65 bytes
50 49 4a 4c  0100  0000                    "PIJL", version 1
02000000 02000000                          2 inputs, 2 outputs
"half adder\0" "dirty=adaptive,eval=lut\0" "a\0" "2\0" "sum\0" "carry\0"
```

Four vectors at 1 bit a pin (values 0, 1, 2, 3 on port 0: `a` is bit 0),
settling each, sampling every one:

```
02 01 0000 14000000                        RUN, IN_1BIT, port 0, 20 bytes
04000000 0000 01 00 00000000 00000000      4 vectors, out port 0, every, settle, default limit
00 01 02 03                                the vectors
```

The answer, samples 2 bits a pin:

```
81 00 0000 18000000                        RESULT, port 0, 24 bytes
00000000 04000000 04000000                 frame 0, 4 run, 4 samples
00000000 04000000                          0 unsettled, at most 4 ticks
05 06 06 09                                sum,carry: 00, 10, 10, 01
```

(`0x06` = `0b0110`: pin 0 `sum` is `0b10` = 1, pin 1 `carry` is `0b01` = 0.)

A port of just `carry`, then two vectors at 2 bits a pin (`11`, then `1Z`),
sampling only the second, 1-bit answers, with ticks:

```
01 00 0100 0c000000                        PORT 1, 12 bytes
01 000000 01000000 01000000                outputs, 1 pin: output 1 (carry)

02 06 0000 13000000                        RUN, OUT_1BIT|TICKS, port 0, 19 bytes
02000000 0100 03 00 00000000 00000000      2 vectors, out port 1, mask, settle
02                                         mask: vector 1 only
0a 02                                      "11", "1Z"
```

```
81 0e 0100 1d000000                        RESULT, OUT_1BIT|TICKS|LOSSY, port 1
02000000 02000000 01000000                 frame 2, 2 run, 1 sample
00000000 04000000                          0 unsettled, at most 4
00000000 04000000                          ticks: 0 (already there), 4
00                                         carry = AND(1, Z) = X, sent as 0: LOSSY
```

An unknown op `0x42` as frame 3:

```
82 00 0000 13000000                        ERROR, 19 bytes
03000000 "unknown op 0x42"
```

## The Python client

`pijl.pipe.Client` starts `pijl run MACRO --bin` as a child process and speaks the
protocol:

```python
from pijl.pipe import Client

with Client("half adder") as p:
    p.inputs, p.outputs                          # ("a", "2"), ("sum", "carry"): from HELLO
    p.run(["00", "01", "10", "11"]).strings()    # ["00", "10", "10", "01"]
    carry = p.port(["carry"], side="out")        # names, "#n" or indices
    p.run([0, 1, 2, 3], out=carry, out_1bit=True).ints()   # [0, 0, 0, 1]
```

- Vectors can be strings of `0 1 X Z`, integers (the port's value, pin 0 the
  low bit), or a 2-D array of 0s and 1s. Vectors that are all 0 / 1 go 1 bit a
  pin, others 2 bits.
- `run(vectors, port=0, out=0, ticks=None, limit=None, sample="every",
  out_1bit=False, want_ticks=False)` sends a RUN and waits. `sample` is
  `"every"`, `"last"`, `None`, or a mask (one bool per vector).
- `submit(...)` sends without waiting and returns the frame's number;
  `receive()` takes the next answer. A thread reads the child's answers as they
  come, so any number of frames can be in flight without deadlocking.
- `drive(vector, port=0)`, `step(ticks=None)`, `read(out=0)`,
  `reset(seed=0, noise=64)`.
- A **Result** has `codes` (samples x pins, level codes), `strings()`, `bits()`,
  `ints()`, `n`, `unsettled`, `most`, `ticks` (with `want_ticks`), `lossy`,
  `frame`.
- An ERROR from the child is raised as `PipeError` (with `.frame`) by the
  `receive()` (or `run`, `step`, `read`) that meets it.
- `close()` ends stdin and returns the child's exit status.

`Client(macro, project=None, data=None, engine=None, noise=64, seed=0,
max_ticks=None, stderr=None)`: the child's `--project`, `--data`, `--engine`,
`--noise`, `--seed`, `--max-ticks`; `stderr` is passed to `subprocess.Popen`.

## Benchmarks

`pijl bench bogobips` drives generated circuits through each way in: `engine`
(Harness in-process, fixed ticks), `settle` (Harness, until stable), `pipe` (a
`--raw` child) and `bin` (a `--bin` child through `Client`, 256 vectors a RUN,
reads picked out by a mask). See `pydoc pijl.bogobips` and [BENCH.md](BENCH.md).
