from rbo_v2 import GRIDS
import itertools

total = 0
for k, v in GRIDS.items():
    p = v['params']
    keys, values = zip(*p.items())
    combos = [dict(zip(keys, c)) for c in itertools.product(*values)]
    total += len(combos)

print(f"Total Combinations Generated: {total}")
