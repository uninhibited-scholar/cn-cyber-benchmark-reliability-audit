#!/usr/bin/env python3
"""L1 nested-panel reconstruction. The retracted analysis accumulated models
into a growing panel commercial-first; the exact per-item order was not
version-pinned. This uses a fully documented order -- commercial models sorted by
name, then open-weight models sorted by name -- and recomputes RQ1's Spearman
rho at k = 9, 13, 17, 20, 25. The point is only that a fixed accumulation order
produces a non-monotonic path (a presentation artifact); Table 1 (random
subsets) is the corrected analysis.
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = json.load(open(os.path.join(ROOT, "data", "scored_results", "results.json")))

def spearman(xs, ys):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v); i = 0
        while i < len(idx):
            j = i
            while j+1 < len(idx) and v[idx[j+1]] == v[idx[i]]: j += 1
            for k in range(i, j+1): r[idx[k]] = (i+j)/2+1
            i = j+1
        return r
    n = len(xs); rx, ry = rank(xs), rank(ys); mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = sum((rx[i]-mx)**2 for i in range(n))**.5; dy = sum((ry[i]-my)**2 for i in range(n))**.5
    return 0.0 if dx == 0 or dy == 0 else num/(dx*dy)

# Bench names in the mirror
IR = lambda m: res[m]["Bench-Inject"]["injection_recall"]
HR = lambda m: res[m]["Bench-Refusal"]["harmful_refusal_rate"]

COMMERCIAL = {"deepseek-v4-pro", "deepseek-v4-flash", "doubao-1.5-pro",
              "doubao-seed-2.0-pro", "doubao-seed-2.0-lite", "doubao-seed-2.1-turbo",
              "glm-5.2", "glm-5.3", "kimi-k2.7-code", "minimax-m3"}
models = list(res.keys())
order = sorted(m for m in models if m in COMMERCIAL) + sorted(m for m in models if m not in COMMERCIAL)

print("documented commercial-first accumulation order:")
for i, m in enumerate(order, 1):
    print(f"  {i:2d}. {m}")
print("\nRQ1 Spearman rho at growing panel size:")
for k in (9, 13, 17, 20, 25):
    s = order[:k]
    print(f"  k={k:2d}: rho = {spearman([IR(m) for m in s], [HR(m) for m in s]):+.3f}")
