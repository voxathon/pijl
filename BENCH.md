Current BogoBIPS numbers, on the default engine (`dirty=adaptive,eval=lut`; see "engine settings" below):

### shift registers

| kind | depth |  build | ticks/clock |  engine |  settle |    pipe |
| ---- | ----: | -----: | ----------: | ------: | ------: | ------: |
| sipo |     4 |  784ms |          10 |  12.01k |   8.81k |  10.62k |
| sipo |    32 |    6ms |          10 |  76.59k |  67.47k |  68.98k |
| sipo |   256 |   24ms |          10 | 336.36k | 284.49k | 325.02k |
| sipo |  1024 |  111ms |          10 | 548.82k | 484.93k | 518.23k |
| sipo |  4096 |  351ms |          10 | 569.80k | 548.17k | 556.76k |
| sipo | 16384 | 1689ms |          10 | 465.60k | 438.78k | 465.37k |
| sipo | 65536 | 6102ms |          10 | 442.22k | 452.97k | 448.86k |
| piso |     4 |    3ms |          14 |   9.06k |   8.97k |   8.24k |
| piso |    32 |    6ms |          14 |  54.04k |  54.21k |  49.03k |
| piso |   256 |  148ms |          14 | 202.77k | 200.47k | 195.69k |
| piso |  1024 |  112ms |          14 | 347.49k | 360.43k | 347.95k |
| piso |  4096 |  511ms |          14 | 374.29k | 331.48k | 343.89k |
| piso | 16384 | 2033ms |          14 | 288.59k | 289.26k | 288.59k |
| piso | 65536 | 8618ms |          14 | 275.98k | 290.95k | 268.90k |

**Peaks:** sipo 569.80k (depth `4096`, `engine`) · piso 374.29k (depth `4096`, `engine`)

Ran with: `uv run pijl bench bogobips --kind sipo,piso --depth 4,32,256,1024,4096,16384,65536 --seconds 3`

### trees, decoder, adder

| kind     | depth |  gates |  build | ticks/vector |  engine |  settle |    pipe |
| -------- | ----: | -----: | -----: | -----------: | ------: | ------: | ------: |
| tree-and |     4 |     15 |  219ms |            5 |  83.37k |  96.13k |  77.15k |
| tree-and |     8 |    255 |   11ms |            9 | 938.26k |   1.40M |   1.00M |
| tree-and |    12 |  4.09k |  167ms |           13 |  12.95M |  17.34M |  12.84M |
| tree-and |    16 | 65.53k | 2745ms |           17 | 188.31M | 311.99M | 179.47M |
| tree-xor |     4 |     15 |    3ms |            5 | 114.77k |  70.55k | 117.70k |
| tree-xor |     8 |    255 |   11ms |            9 | 926.40k | 718.79k | 934.95k |
| tree-xor |    12 |  4.09k |  230ms |           13 |   3.49M |   3.56M |   3.49M |
| tree-xor |    16 | 65.53k | 2949ms |           17 |  43.05M |  42.98M |  40.36M |
| decoder  |     4 |     28 |    3ms |            4 | 257.76k | 141.51k | 240.75k |
| decoder  |     8 |    312 |   15ms |            5 |   1.87M |   1.25M |   1.89M |
| decoder  |    12 |  4.28k |  182ms |            6 |   6.86M |   6.76M |   6.68M |
| decoder  |    16 | 66.16k | 3548ms |            6 |  66.85M |  66.48M |  47.11M |
| adder    |     8 |     40 |    4ms |           19 | 117.33k | 189.52k | 118.96k |
| adder    |    64 |    320 |   12ms |          131 | 220.75k |   1.22M | 223.94k |
| adder    |   512 |  2.56k |   88ms |         1027 | 257.39k |   5.05M | 258.21k |

**Peaks:** tree-and 311.99M (depth `16`, `settle`) · tree-xor 43.05M (depth `16`, `engine`) · decoder 66.85M (depth `16`, `engine`) · adder 5.05M (depth `512`, `settle`)

Ran with: `uv run pijl bench bogobips --kind tree,decoder,adder --seconds 3` (default depths, 1 flip per vector, flat gates)

### counter, lfsr

| kind    | depth | build | ticks/clock |  engine |  settle |    pipe |
| ------- | ----: | ----: | ----------: | ------: | ------: | ------: |
| counter |    64 |  45ms |         140 |  28.20k |  97.84k |  31.45k |
| counter |   512 |  98ms |        1036 |  45.67k | 301.84k |  45.20k |
| counter |  4096 | 774ms |        8204 |  51.64k | 701.16k |  47.88k |
| lfsr    |    64 |   9ms |          14 | 112.53k | 113.59k |  83.88k |
| lfsr    |   512 |  72ms |          14 | 383.32k | 387.75k | 363.34k |
| lfsr    |  4096 | 430ms |          12 | 709.86k | 748.51k | 683.97k |

**Peaks:** counter 701.16k (depth `4096`, `settle`) · lfsr 748.51k (depth `4096`, `settle`)

Ran with: `uv run pijl bench bogobips --kind counter,lfsr --depth 64,512,4096 --seconds 3`

### engine settings compared (settle column)

| kind     | depth | off, batches | off, lut | adaptive, batches | adaptive, lut |
| -------- | ----: | -----------: | -------: | ----------------: | ------------: |
| sipo     |    64 |       58.62k |  165.11k |            52.56k |       115.88k |
| sipo     |  4096 |      521.56k |  489.67k |           585.77k |       534.47k |
| piso     |   256 |      151.94k |  261.55k |           135.04k |       195.19k |
| counter  |   512 |      146.46k |  281.73k |           181.91k |       287.57k |
| lfsr     |   256 |      122.13k |  328.84k |           112.15k |       253.72k |
| tree-and |    16 |       12.85M |   13.13M |           260.27M |       320.99M |
| tree-xor |    16 |        2.20M |    2.17M |            37.02M |        44.30M |
| decoder  |     8 |      756.12k |    2.03M |           487.38k |         1.22M |
| decoder  |    16 |        5.53M |    5.70M |            60.49M |        69.31M |
| adder    |   512 |        2.62M |    5.96M |             3.00M |         5.11M |

Ran with: `uv run pijl bench bogobips --kind <kind> --depth <depth> --seconds 3 --layers settle --engines "dirty=off,eval=batches;dirty=off,eval=lut;dirty=adaptive,eval=batches;dirty=adaptive,eval=lut"`, one run per row group (so rows from different runs vary a bit: compare across a row, not down a column)

All on an Intel i5-9600K overclocked to 4.28 GHz, on Windows 10.

---

## bogobips, explained simply

bogobips is a **Bogus** speed test measuring **BI**t-shifts **P**er **S**econd (**BIPS**), built for pijl's simulator. we push random bits through circuits built from plain gates and count how fast the circuit gets its work done.

BIPS = clocks (or input changes) per second × the work each one does. higher is better. what counts as work depends on the kind, so **only compare numbers within one kind**:
- shift registers: every cell passing its bit along is one bit-shift. work = depth.
- everything else: every gate passing its value on is one unit. work = the gate count.

the kinds, and what depth means for each:
- SIPO (one bit in, every cell read out; depth = cells): tests how fast we can *read* a big circuit.
- PISO (load every cell at once, shift one bit out; depth = cells): tests how fast we can *drive* a big circuit. each cell has extra logic in front of it, so it's slower than SIPO (60-66% of it from 256 cells up, in the table above).
- counter (depth = bits): a counter that adds 1 on clocks where its enable input (random) is on. each bit's next value is worked out from the bits themselves, so the logic feeds back into the flip-flops: the shape of most real circuits with memory. few bits change per clock, but a carry can run the whole width, and the engine column has to leave room for that every clock.
- lfsr (depth = stages): a shift register whose first stage gets the XNOR of its last two stages. the feedback loops through every stage, and every stage is busy.
- tree-and / tree-xor (depth = levels, so 2^depth inputs merging into one output): each step flips one random input. in an AND tree a single 0 decides a gate, so most changes die out after a level or two: the quiet case. in an XOR tree every change runs all the way to the top, one gate per level. together they bracket how far a change travels in a real circuit.
- decoder (depth = inputs, 2^depth outputs, exactly one of them on): every input fans out to lots of gates. tests spreading changes out.
- adder (depth = bits, ripple carry): a carry can run the whole width, but with random inputs it usually stops after a few bits. tests timing that depends on the data.

the three columns:
- engine: the simulator alone, at full speed, running a fixed number of steps every time. that number has to cover the worst case (the ticks column), even when nothing is moving.
- settle: the same, but the simulator works out for itself when the circuit has stopped changing. costs a little per step to check, and saves everything after the circuit goes quiet.
- pipe: a separate pijl process driven through text in and out, the way an outside program would use it.

how to read the gaps:
- settle way ahead of engine (adder, counter, tree-and): engine runs enough steps for the worst case every time, settle stops when the circuit goes quiet. at 512 bits the adder's settle is about 20× its engine, the 4096-bit counter's about 14×.
- settle behind engine (small decoders and XOR trees, small shift registers): checking for "quiet" costs something every step, and these circuits have few quiet steps to skip. on the big ones the two columns come out level.
- pipe about level with engine: talking to pijl from outside costs almost nothing. the exception is decoder 16, where every vector reads 65,536 outputs back as text (47M against 67M).
- shift registers peak around 4,096 cells and drop after that (sipo 570k at 4,096, 442k at 65,536). trees and the decoder keep climbing up to depth 16, because each step only runs the gates whose inputs changed (see "engine settings").

the output is checked against what it should be: a shift register gives back what went in, just later; a counter counts; an lfsr gives the sequence its feedback rule says; a tree is an AND or a parity of its inputs; a decoder lights exactly one output; an adder adds. if something breaks, the cell says FAIL instead of showing a number. a FAIL means your engine is cooked.

things to keep in mind:
- the build column is setup time, not speed.
- the same run can vary by about ±25%, so run it a few times before trusting a small change.
- this is one synthetic workload, not a measure of how fast your circuits run; use it to compare pijl with an earlier pijl, nothing else.

## engine settings

pijl's simulator has two settings. every combination gives exactly the same results, step for step; only the speed differs.
- dirty: `adaptive` (the default) only runs the gates whose inputs changed, when picking those out is cheaper than running them all. `off` runs every gate on every step.
- eval: `lut` (the default) turns every gate type with up to 4 inputs into a lookup table, and looks all the gates up at once. `batches` runs each gate type's own code, one type at a time.

pick one with `--engine dirty=off,eval=batches` (or the `PIJL_ENGINE` environment variable); `--engines "a;b"` runs every circuit once per setting, a row each.

what the comparison table shows:
- lut against batches: ahead on most rows, by 1.7-2.8× on the small and middling ones. on sipo 4096 batches came out ahead in this run, by less than the run-to-run noise.
- adaptive against off: 12-25× on the big trees and the decoder, where most gates are idle on any step.
- small circuits that are busy every step (sipo 64, piso 256, lfsr 256, decoder 8): `dirty=off,eval=lut` is the fastest. checking what changed costs more than it saves there.
- the default, `adaptive,lut`, is the fastest or close to it everywhere except those small busy circuits.
