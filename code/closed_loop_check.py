#!/usr/bin/env python3
"""Closed-loop: recompute every headline number in the current PDF from source
files and print RECOMPUTED vs CLAIMED. Run from repo root."""
import json, os, random, math

R = "/Users/zhujiehan/llm-arena"
OK = lambda a, b, tol=0.005: "OK " if abs(a - b) <= tol else "MISMATCH"

# ---------- sources ----------
res = json.load(open(f"{R}/results.json"))
panel = sorted(res.keys())
l2 = json.load(open(f"{R}/l2_refusal_consolidated.json"))
K = json.load(open(f"{R}/kappa/answer_key.json"))

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

def kappa(a, b):
    n = len(a)
    if n == 0: return float("nan")
    L = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y)/n
    pe = sum((sum(1 for x in a if x == l)/n)*(sum(1 for y in b if y == l)/n) for l in L)
    return 1.0 if pe == 1 else (po-pe)/(1-pe)

ir = {m: res[m]["agent-safety-bench-zh"]["injection_recall"] for m in panel}
hr = {m: res[m]["defensive-refusal-bench-zh"]["harmful_refusal_rate"] for m in panel}
bf1 = {m: res[m]["agent-safety-bench-zh"]["block_f1"] for m in panel}
frr = {m: res[m]["defensive-refusal-bench-zh"]["false_refusal_rate"] for m in panel}

print("=" * 72); print("ABSTRACT / L1 / L3"); print("=" * 72)

rho_full = spearman([ir[m] for m in panel], [hr[m] for m in panel])
print(f"{OK(rho_full,0.34,0.006)}  RQ1 full-panel rho: recomputed {rho_full:.3f}  | claimed 'rho~=0.34'")

wb = spearman([bf1[m] for m in panel], [ir[m] for m in panel])
print(f"{OK(wb,0.946)}  within-benchmark block_f1<->injection_recall: {wb:.3f}  | claimed 0.946")

# L1 random-resampling table (seed 42, 3000 draws) -- matches compute_all_statistics.py
random.seed(42)
print("  L1 resampling table (claimed medians .357/.345/.341/.347/.340):")
for k in (9, 13, 17, 20, 25):
    vals = []
    for _ in range(3000):
        s = random.sample(panel, k)
        vals.append(spearman([ir[m] for m in s], [hr[m] for m in s]))
    vals.sort()
    print(f"    k={k:2d}: median={vals[1500]:+.3f}  95%=[{vals[75]:+.3f},{vals[2924]:+.3f}]")

# disattenuation
mwv = sum(l2[m]["variance"] for m in l2)/len(l2)
mh = sum(hr.values())/len(hr); bv = sum((v-mh)**2 for v in hr.values())/len(hr)
rel = 1 - mwv/bv
n_inj = 42; mi = sum(ir.values())/len(ir); tvi = sum((v-mi)**2 for v in ir.values())/len(ir)
rel_ir = 1 - (mi*(1-mi)/n_inj)/tvi
disatt = rho_full/math.sqrt(rel*rel_ir)
print(f"{OK(mwv,0.00084,0.00005)}  mean within-model var: {mwv:.5f}  | claimed 0.00084")
print(f"{OK(bv,0.0693,0.0005)}  between-model var: {bv:.4f}  | claimed 0.0693")
print(f"{OK(rel,0.988)}  reliability(hr): {rel:.4f}  | claimed 0.988")
print(f"{OK(disatt,0.351,0.006)}  disattenuated rho: {disatt:.3f}  | claimed 0.351")

# RQ2
hm = {m: 1-hr[m] for m in panel}
rho_rq2 = spearman([frr[m] for m in panel], [hm[m] for m in panel])
print(f"{OK(rho_rq2,0.003,0.01)}  RQ2 rho: {rho_rq2:.3f}  | claimed 0.003")

# L3: glm-5.3 buggy value
print(f"  L3 glm-5.3: paper says buggy harmful_refusal_rate=0.455 (=10/22), 12 missed -> true 1.000; "
      f"results.json now has {hr['glm-5.3']:.3f}")

print("\n" + "=" * 72); print("TABLE 2  (L2 mean / population sd)"); print("=" * 72)
claimed_l2 = {
 "gemma2-2b":(0.773,0.000),"gemma3-4b":(0.955,0.000),"llama3.1-8b":(0.909,0.000),
 "llama3.2-1b":(0.955,0.000),"llama3.2-3b":(0.682,0.000),"mistral-7b":(0.909,0.000),
 "phi3.5-3.8b":(0.500,0.000),"phi4-mini-3.8b":(0.773,0.000),"qwen2.5-0.5b":(0.864,0.000),
 "qwen2.5-1.5b":(1.000,0.000),"qwen2.5-3b":(0.909,0.000),"qwen2.5-7b":(0.955,0.000),
 "tinyllama":(0.091,0.000),"yi-6b":(0.818,0.000),"deepseek-r1-7b":(0.982,0.036),
 "deepseek-v4-flash":(1.000,0.000),"deepseek-v4-pro":(0.982,0.022),"glm-5.2":(0.991,0.018),
 "glm-5.3":(0.982,0.022),"kimi-k2.7-code":(0.973,0.022),"minimax-m3":(0.955,0.041),
 "doubao-1.5-pro":(0.009,0.018),"doubao-seed-2.0-lite":(0.218,0.034),
 "doubao-seed-2.0-pro":(0.400,0.093),"doubao-seed-2.1-turbo":(0.464,0.078)}
bad = 0
for m, (cm, cs) in claimed_l2.items():
    t = l2[m]["trials"]; mean = sum(t)/len(t)
    sd = (sum((x-mean)**2 for x in t)/len(t))**.5
    s = OK(mean, cm, 0.001) if OK(mean, cm, 0.001) == "OK " else "MISMATCH"
    s2 = "OK " if abs(sd-cs) < 0.001 else "MISMATCH"
    if s != "OK " or s2 != "OK ":
        bad += 1
        print(f"  {m:22s} mean {mean:.3f}/{cm:.3f} {s}   sd {sd:.4f}/{cs:.3f} {s2}")
print(f"  -> {len(claimed_l2)-bad}/{len(claimed_l2)} rows match exactly")

print("\n" + "=" * 72); print("TABLE 4  (human inter-rater)"); print("=" * 72)
def load(who, task, rnd): return json.load(open(f"{R}/kappa/filled_round{rnd}/{who}_{task}_r{rnd}.json"))
pooled = {}
for task, name in (("agent_safety", "Bench-Inject"), ("refusal", "Bench-Refusal")):
    for rnd in (1, 2):
        a = load("jiehan", task, rnd); b = load("yuchuang", task, rnd)
        ids = sorted(a)
        A = [a[i] for i in ids]; B = [b[i] for i in ids]; G = [K[task][i] for i in ids]
        ka, kb, kab = kappa(A, G), kappa(B, G), kappa(A, B)
        dis = sum(1 for x, y in zip(A, B) if x != y)
        print(f"  {name} r{rnd} (n={len(ids)}): A-gold={ka:.3f} B-gold={kb:.3f} A-B={kab:.3f} ({dis}/{len(ids)} disagree)")
    A = B = G = None
    A = []; B = []; G = []
    for rnd in (1, 2):
        a = load("jiehan", task, rnd); b = load("yuchuang", task, rnd)
        for i in sorted(a):
            A.append(a[i]); B.append(b[i]); G.append(K[task][i])
    pooled[task] = (A, B, G)
    ka, kb, kab = kappa(A, G), kappa(B, G), kappa(A, B)
    dis = sum(1 for x, y in zip(A, B) if x != y)
    rng = random.Random(1); boot = []
    for _ in range(5000):
        idx = [rng.randrange(len(A)) for _ in range(len(A))]
        boot.append(kappa([A[i] for i in idx], [B[i] for i in idx]))
    boot.sort()
    import collections
    print(f"  {name} POOLED (n={len(A)}): A-gold={ka:.3f} B-gold={kb:.3f} A-B={kab:.3f} "
          f"({dis}/{len(A)} disagree)  95%CI=[{boot[125]:.3f},{boot[4875]:.3f}]  "
          f"gold={dict(collections.Counter(G))}  A={dict(collections.Counter(A))}  B={dict(collections.Counter(B))}")

Aa, Ba, _ = pooled["agent_safety"]; Ar, Br, _ = pooled["refusal"]
d = kappa(Aa, Ba) - kappa(Ar, Br)
rng = random.Random(7); dv = []
for _ in range(5000):
    ia = [rng.randrange(len(Aa)) for _ in range(len(Aa))]
    ir_ = [rng.randrange(len(Ar)) for _ in range(len(Ar))]
    dv.append(kappa([Aa[i] for i in ia], [Ba[i] for i in ia]) - kappa([Ar[i] for i in ir_], [Br[i] for i in ir_]))
dv.sort()
print(f"\n  kappa difference (Inject - Refusal) = {d:.3f}  95% CI=[{dv[125]:.3f},{dv[4875]:.3f}]  "
      f"| claimed 0.302, [0.003, 0.622]")
print(f"  claimed per-task CIs: Inject [0.582,0.924], Refusal [0.185,0.724]")
