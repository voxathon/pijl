"""Compare two runs of equiv_buffers.py or equiv_macros.py, step by step:

    uv run python tools/equiv_cmp.py before.json after.json
"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for x, y in zip(a, b):
    diffs = [k for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
    if diffs:
        bad += 1
        print(f"{x['step']:20s} DIFFERS: " + ", ".join(f"{k} {x.get(k)} != {y.get(k)}" if not isinstance(x.get(k), dict) else f"{k}[{','.join(f for f in sorted(set(x.get(k) or {}) | set(y.get(k) or {})) if (x.get(k) or {}).get(f) != (y.get(k) or {}).get(f))}]" for k in diffs))
    else:
        print(f"{x['step']:20s} same   n={x['n']}")
if len(a) != len(b):
    print(f"step counts differ: {len(a)} vs {len(b)}")
    bad += 1
print("ALL SAME" if not bad else f"{bad} steps differ")
