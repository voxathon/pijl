Current BogoBIPS numbers:

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

Ran with: `uv run pijl bench bogobips --depth 4,32,256,1024,4096,16384,65536 --seconds 3` on an Intel i5-9600K overclocked to 4.28 GHz.

---

## bogobips, explained simply

bogobips is a **Bogus** speed test measuring **BI**t-shifts **P**er **S**econd (**BIPS**), built for pijl's simulator. we push random bits through a shift register (a chain of memory cells that passes each bit one step along per clock) and count how fast they move.

BIPS = clocks per second × register length. every cell passing its bit along counts as one bit-shift. higher is better.

The two register types:
- SIPO (one bit in, every cell read out): tests how fast we can *read* a big circuit.
- PISO (load every cell at once, shift one bit out): tests how fast we can *drive* a big circuit. each cell has extra logic in front of it, so it's about half SIPO's speed.

the three columns:
- engine: the simulator alone, at full speed.
- settle: the same, but the simulator works out for itself when the circuit has stopped changing. 
- pipe: a separate pijl process driven through text in and out, the way an outside program would use it. 

the output is checked against the input: a shift register should give back exactly what went in, just later. if something breaks, the cell says FAIL instead of showing a number. a FAIL means your engine is cooked.

things to keep in mind:
- the build column is setup time, not speed.
- this is one synthetic workload, not a measure of how fast your circuits run; use it to compare pijl with an earlier pijl, nothing else.
