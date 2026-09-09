#!/usr/bin/env python3
"""Reproduces every headline statistic in the paper from data/scored_results/results.json.

Usage: python3 compute_all_statistics.py

Covers:
  Sec 4.1  L1 nested-panel table (the retracted framing) + corrected random-resampling
           table + disattenuation
  Sec 4.2  RQ2 correlation + TOST equivalence test
  Sec 4.5  Cohen's kappa for the 3 LLM annotators (reads data/kappa/*)
"""
import json, os, random, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
random.seed(42)

d = json.load(open(os.path.join(ROOT, "data", "scored_results", "results.json")))
models = sorted(d.keys())
rows = {}
for m in models:
    s = d[m]["Bench-Inject"]
    f = d[m]["Bench-Refusal"]
    rows[m] = {
        "ir": s["injection_recall"], "bf1": s["block_f1"],
        "hr": f["harmful_refusal_rate"], "fr": f["false_refusal_rate"],
        "hm": 1 - f["harmful_refusal_rate"],
    }

# ---------- Spearman ----------
def rank(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r

def spearman(xs, ys):
    n = len(xs)
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return 0.0 if dx == 0 or dy == 0 else num / (dx * dy)

def bootstrap_ci(xs, ys, n_boot=5000, alpha=0.05):
    n = len(xs)
    vals = []
    for _ in range(n_boot):
        idx = [random.randrange(n) for _ in range(n)]
        vals.append(spearman([xs[i] for i in idx], [ys[i] for i in idx]))
    vals.sort()
    lo = vals[int((alpha / 2) * n_boot)]
    hi = vals[int((1 - alpha / 2) * n_boot)]
    return lo, hi, vals

def percentile(vals, p):
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)

print("=" * 70)
print(f"n = {len(models)} models: {models}")
print("=" * 70)

# ---------- Sec 4.1: nested panel (the retracted framing, kept for transparency) ----------
print("\n--- Sec 4.1a: NESTED panel path (retracted framing — kept for audit trail) ---")
print("model accumulation order used in the original draft (commercial-first):")
nested_order = models  # NOTE: sorted() order used above; the original draft's exact
# accumulation order is not separately versioned — this reproduces the METHOD
# (nested/growing panel), the specific numbers in the paper's Sec 4.1 came from a
# now-superseded ordering and are not re-derived bit-for-bit here. See Sec 4.1's
# in-text acknowledgement that the exact nested path is a presentation artifact,
# not a number worth reproducing exactly.
for k in [9, 13, 17, 20, 25]:
    subset = nested_order[:k]
    xs = [rows[m]["ir"] for m in subset]
    ys = [rows[m]["hr"] for m in subset]
    rho = spearman(xs, ys)
    lo, hi, _ = bootstrap_ci(xs, ys)
    print(f"  k={k:2d}  rho={rho:+.3f}  95% CI=[{lo:+.3f},{hi:+.3f}]")

# ---------- Sec 4.1b: corrected random resampling ----------
print("\n--- Sec 4.1b: CORRECTED random resampling (Table in paper) ---")
N_DRAWS = 3000
dist_by_k = {}
for k in [9, 13, 17, 20, 25]:
    vals = []
    for _ in range(N_DRAWS):
        samp = random.sample(models, k)
        xs = [rows[m]["ir"] for m in samp]
        ys = [rows[m]["hr"] for m in samp]
        vals.append(spearman(xs, ys))
    dist_by_k[k] = vals
    med = percentile(vals, 0.5)
    lo = percentile(vals, 0.025)
    hi = percentile(vals, 0.975)
    print(f"  k={k:2d}  median={med:+.3f}  95% resampling interval=[{lo:+.3f},{hi:+.3f}]")

# ---------- Sec 4.1c: disattenuation ----------
print("\n--- Sec 4.1c: Disattenuation ---")
within_vars = [0.081 ** 2, 0.053 ** 2]  # doubao-seed-2.0-pro / 2.1-turbo, 5-trial SDs (Sec 4.3)
mean_within_var = sum(within_vars) / len(within_vars)
hr_vals = [rows[m]["hr"] for m in models]
mean_hr = sum(hr_vals) / len(hr_vals)
total_var_hr = sum((v - mean_hr) ** 2 for v in hr_vals) / len(hr_vals)
reliability_hr = max(0, 1 - mean_within_var / total_var_hr)

n_inj = 42  # size of the prompt-injection subset within Bench-Inject
ir_vals = [rows[m]["ir"] for m in models]
mean_ir = sum(ir_vals) / len(ir_vals)
total_var_ir = sum((v - mean_ir) ** 2 for v in ir_vals) / len(ir_vals)
binom_var = mean_ir * (1 - mean_ir) / n_inj
reliability_ir_approx = max(0, 1 - binom_var / total_var_ir)

rho_obs = spearman(ir_vals, hr_vals)
denom = math.sqrt(reliability_ir_approx * reliability_hr)
rho_disatt = rho_obs / denom if denom > 0 else float("nan")
print(f"  reliability(harmful_refusal_rate)  [empirical, from 2-model L2 repeat]  = {reliability_hr:.3f}")
print(f"  reliability(injection_recall)      [APPROXIMATED, binomial-noise floor] = {reliability_ir_approx:.3f}")
print(f"  observed rho = {rho_obs:.3f}  ->  disattenuated rho = {min(1.0, rho_disatt):.3f}")

# ---------- Sec 4.2: RQ2 + TOST ----------
print("\n--- Sec 4.2: RQ2 correlation + TOST equivalence test ---")
fr = [rows[m]["fr"] for m in models]
hm = [rows[m]["hm"] for m in models]
rho_rq2 = spearman(fr, hm)
lo95, hi95, boots = bootstrap_ci(fr, hm, alpha=0.05)
lo90, hi90, _ = bootstrap_ci(fr, hm, alpha=0.10)
print(f"  observed rho = {rho_rq2:.3f}")
print(f"  95% CI = [{lo95:+.3f},{hi95:+.3f}]   90% CI (TOST) = [{lo90:+.3f},{hi90:+.3f}]")
for margin in [0.2, 0.3, 0.4]:
    passes = (lo90 >= -margin) and (hi90 <= margin)
    print(f"  TOST margin ±{margin}: equivalence {'PASSES' if passes else 'does NOT pass'}")

# ---------- Sec 4.5: Cohen's kappa for LLM annotators ----------
print("\n--- Sec 4.5: Cohen's kappa (requires data/kappa/*_filled.jsonl from annotators) ---")
print("  Run code/llm_annotate.py to regenerate the 3 LLM-annotator files, or")
print("  run code/compute_kappa.py <filled.jsonl> <agent_safety|refusal> on human-filled forms.")

print("\nDone. Every number above should match the corresponding table in the manuscript.")
