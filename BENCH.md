Current BogoBIPS numbers:

### shift registers

| kind | depth |  build | ticks/clock |  engine |  settle |    pipe |
| ---- | ----: | -----: | ----------: | ------: | ------: | ------: |
| sipo |     4 |  340ms |          10 |   7.20k |   5.44k |   7.12k |
| sipo |    32 |    5ms |          10 |  50.15k |  43.57k |  52.69k |
| sipo |   256 |   25ms |          10 | 267.22k | 215.01k | 259.93k |
| sipo |  1024 |  154ms |          10 | 481.16k | 411.63k | 484.51k |
| sipo |  4096 |  373ms |          10 | 616.53k | 521.69k | 599.48k |
| sipo | 16384 | 1508ms |          10 | 553.06k | 427.95k | 559.28k |
| sipo | 65536 | 6377ms |          10 | 404.79k | 350.44k | 489.24k |
| piso |     4 |    4ms |          14 |   4.83k |   4.67k |   4.93k |
| piso |    32 |    7ms |          14 |  34.53k |  36.19k |  34.58k |
| piso |   256 |   40ms |          14 | 130.74k | 129.56k | 135.04k |
| piso |  1024 |  291ms |          14 | 248.98k | 233.62k | 252.46k |
| piso |  4096 |  467ms |          14 | 260.47k | 262.59k | 261.03k |
| piso | 16384 | 2024ms |          14 | 266.47k | 249.52k | 245.80k |
| piso | 65536 | 8523ms |          14 | 251.82k | 228.16k | 236.75k |

**Peak:** 616.53k BogoBIPS — `sipo`, depth `4096`, `engine`

Ran with: `uv run pijl bench bogobips --kind sipo,piso --depth 4,32,256,1024,4096,16384,65536 --seconds 3`

### trees, decoder, adder

| kind     | depth |  gates |  build | ticks/vector |  engine |  settle |    pipe |
| -------- | ----: | -----: | -----: | -----------: | ------: | ------: | ------: |
| tree-and |     4 |     15 |  264ms |            5 |  67.70k |  94.09k |  59.88k |
| tree-and |     8 |    255 |   13ms |            9 | 660.91k |   1.58M | 581.56k |
| tree-and |    12 |  4.09k |  164ms |           13 |   2.22M |  10.13M |   1.94M |
| tree-and |    16 | 65.53k | 3080ms |           17 |   1.85M |  10.39M |   2.10M |
| tree-xor |     4 |     15 |    3ms |            5 |  71.19k |  53.00k |  70.22k |
| tree-xor |     8 |    255 |   12ms |            9 | 610.91k | 468.19k | 620.57k |
| tree-xor |    12 |  4.09k |  249ms |           13 |   2.50M |   1.95M |   2.29M |
| tree-xor |    16 | 65.53k | 3129ms |           17 |   2.00M |   1.81M |   1.88M |
| decoder  |     4 |     28 |    6ms |            4 | 112.70k |  75.37k | 104.83k |
| decoder  |     8 |    312 |   15ms |            5 | 911.50k | 659.42k | 934.84k |
| decoder  |    12 |  4.28k |  198ms |            6 |   3.50M |   3.13M |   3.83M |
| decoder  |    16 | 66.16k | 4231ms |            6 |   5.28M |   4.32M |   5.88M |
| adder    |     8 |     40 |    3ms |           19 |  23.62k |  69.23k |  22.80k |
| adder    |    64 |    320 |   12ms |          131 |  25.02k | 484.87k |  25.39k |
| adder    |   512 |  2.56k |   90ms |         1027 |  17.02k |   2.25M |  16.06k |

**Peaks:** tree-and 10.39M (depth `16`, `settle`) · tree-xor 2.50M (depth `12`, `engine`) · decoder 5.88M (depth `16`, `pipe`) · adder 2.25M (depth `512`, `settle`)

Ran with: `uv run pijl bench bogobips --kind tree,decoder,adder --seconds 3` (default depths, 1 flip per vector, flat gates)

All on an Intel i5-9600K overclocked to 4.28 GHz, on Windows 10.

---

## bogobips, explained simply

bogobips is a **Bogus** speed test measuring **BI**t-shifts **P**er **S**econd (**BIPS**), built for pijl's simulator. we push random bits through circuits built from plain gates and count how fast the circuit gets its work done.

BIPS = clocks (or input changes) per second × the work each one does. higher is better. what counts as work depends on the kind, so **only compare numbers within one kind**:
- shift registers: every cell passing its bit along is one bit-shift. work = depth.
- everything else: every gate passing its value on is one unit. work = the gate count.

the kinds, and what depth means for each:
- SIPO (one bit in, every cell read out; depth = cells): tests how fast we can *read* a big circuit.
- PISO (load every cell at once, shift one bit out; depth = cells): tests how fast we can *drive* a big circuit. each cell has extra logic in front of it, so it's about half SIPO's speed.
- tree-and / tree-xor (depth = levels, so 2^depth inputs merging into one output): each step flips one random input. in an AND tree a single 0 decides a gate, so most changes die out after a level or two: the quiet case. in an XOR tree every change runs all the way to the top: the busy case. together they bracket how busy a real circuit is.
- decoder (depth = inputs, 2^depth outputs, exactly one of them on): every input fans out to lots of gates. tests spreading changes out.
- adder (depth = bits, ripple carry): a carry can run the whole width, but with random inputs it usually stops after a few bits. tests timing that depends on the data.

the three columns:
- engine: the simulator alone, at full speed, running a fixed number of steps every time. that number has to cover the worst case (the ticks column), even when nothing is moving.
- settle: the same, but the simulator works out for itself when the circuit has stopped changing. costs a little per step to check, and saves everything after the circuit goes quiet.
- pipe: a separate pijl process driven through text in and out, the way an outside program would use it.

how to read the gaps:
- settle way ahead of engine (tree-and, adder): most of the circuit is idle most of the time. the adder at 512 bits is the extreme: engine pays for a carry across all 512 bits on every step, settle only for the few bits that actually rippled, so it's over 100× faster.
- settle behind engine (tree-xor, decoder, shift registers): there's nothing idle to skip, so the checking is pure cost.
- pipe about level with engine: talking to pijl from outside costs almost nothing.
- the engine column flattens out past a few thousand gates (or cells): that's the simulator's real per-gate speed limit. the planned "only re-run gates whose inputs changed" work should lift it a lot for tree-and and the adder, and barely at all for tree-xor. if tree-xor jumps too, something's off.

the output is checked against what it should be: a shift register gives back what went in, just later; a tree is an AND or a parity of its inputs; a decoder lights exactly one output; an adder adds. if something breaks, the cell says FAIL instead of showing a number. a FAIL means your engine is cooked.

things to keep in mind:
- the build column is setup time, not speed.
- the same run can vary by about ±25%, so run it a few times before trusting a small change.
- this is one synthetic workload, not a measure of how fast your circuits run; use it to compare pijl with an earlier pijl, nothing else.
