#!/usr/bin/env python3
"""Regenerate every table in Sec 4 from the per-item / per-trial records on disk.
Run from the repo root:  python3 verify_tables.py

Design notes for the reviewer:
  * L2 (Table 2): only the five per-TRIAL harmful_refusal_rate values are stored
    per model, not per-item responses. So mean and sd are regenerable here; a
    per-item "decision flip" count is NOT (the per-item L2 responses were not all
    retained -- see the error-source record printed at the end). We report sd
    only, defined as the population standard deviation over the 5 trials
    (divide by n=5).
  * L4 LLM-annotator (Table 3): blank / unparseable LLM responses are their own
    category "<blank>". We do NOT recode them to gold's opposite (that would make
    the label answer-dependent). We report three separate numbers per annotator:
    valid-response coverage, Cohen's kappa on the valid subset only, and
    accuracy on the full sample with blank counted as wrong.
  * L4 human inter-rater (Table 4): kappa, raw agreement, the A-vs-B contingency
    table, a 5000-resample bootstrap 95% CI on kappa, and -- to actually compare
    the two tasks -- a bootstrap 95% CI on the DIFFERENCE kappa(Inject) -
    kappa(Refusal).
"""
import json, os, random, collections

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.join(HERE, "kappa")


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((sum(1 for x in a if x == l) / n) * (sum(1 for y in b if y == l) / n)
             for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(fn, *series, n_boot=5000, seed=1):
    n = len(series[0])
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(fn(*[[s[i] for i in idx] for s in series]))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


# ---------------------------------------------------------------- Table 3
print("=" * 72)
print("TABLE 3  --  LLM-annotator agreement with original gold (round 1 sample)")
print("=" * 72)
answer_key = json.load(open(os.path.join(K, "answer_key.json")))
VALID = {"agent_safety": {"allow", "block"}, "refusal": {"answer", "refuse"}}
FIELD = {"agent_safety": "your_decision", "refusal": "your_action"}
BLIND = {"agent_safety": "agent_safety_blind.jsonl", "refusal": "refusal_blind.jsonl"}
MODELS = ["deepseek-v4-pro", "glm-5.3", "kimi-k2.7-code"]

for task in ("agent_safety", "refusal"):
    ids = [json.loads(l)["id"] for l in open(os.path.join(K, BLIND[task]))]
    gold = {i: answer_key[task][i] for i in ids}
    prefix = "agent_safety" if task == "agent_safety" else "refusal"
    filled = {}
    for m in MODELS:
        rows = {json.loads(l)["id"]: (json.loads(l).get(FIELD[task]) or "").strip().lower()
                for l in open(os.path.join(K, f"{prefix}_{m}_filled.jsonl"))}
        filled[m] = [rows.get(i, "") for i in ids]
    print(f"\n--- {task}  (n={len(ids)}) ---")
    print(f"{'annotator':16s}  coverage   kappa(valid)   acc(blank=wrong)")
    for m in MODELS:
        preds = filled[m]
        valid_idx = [j for j, p in enumerate(preds) if p in VALID[task]]
        cov = len(valid_idx) / len(ids)
        kv = cohen_kappa([preds[j] for j in valid_idx], [gold[ids[j]] for j in valid_idx])
        acc = sum(1 for j in range(len(ids)) if preds[j] == gold[ids[j]]) / len(ids)
        print(f"{m:16s}  {cov:6.3f}     {kv:6.3f}        {acc:6.3f}   "
              f"({len(ids)-len(valid_idx)} blank)")
    # majority vote: rule = plurality over the 3 model labels; a blank is a
    # non-vote; if all 3 blank -> counts as wrong; ties broken toward gold's
    # rarer class is NOT done -- ties (1-1 with 1 blank) are counted as wrong.
    maj = []
    for j in range(len(ids)):
        votes = [filled[m][j] for m in MODELS if filled[m][j] in VALID[task]]
        if not votes:
            maj.append("<none>")
            continue
        c = collections.Counter(votes)
        top, n_top = c.most_common(1)[0]
        # tie among valid votes -> mark as its own token so it can't match gold
        if list(c.values()).count(n_top) > 1:
            maj.append("<tie>")
        else:
            maj.append(top)
    acc_maj = sum(1 for j in range(len(ids)) if maj[j] == gold[ids[j]]) / len(ids)
    valid_maj = [j for j in range(len(ids)) if maj[j] in VALID[task]]
    k_maj = cohen_kappa([maj[j] for j in valid_maj], [gold[ids[j]] for j in valid_maj])
    print(f"{'majority (plurality)':16s}  {len(valid_maj)/len(ids):6.3f}     "
          f"{k_maj:6.3f}        {acc_maj:6.3f}")

# ---------------------------------------------------------------- Table 4
print("\n" + "=" * 72)
print("TABLE 4  --  human inter-rater (Annotator A = author, B = colleague)")
print("=" * 72)


def load_round(who, task, rnd):
    return json.load(open(os.path.join(K, f"filled_round{rnd}", f"{who}_{task}_r{rnd}.json")))

pooled = {}
for task in ("agent_safety", "refusal"):
    A, B, G = [], [], []
    for rnd in (1, 2):
        a = load_round("jiehan", task, rnd)
        b = load_round("yuchuang", task, rnd)
        for i in sorted(a):
            A.append(a[i]); B.append(b[i]); G.append(answer_key[task][i])
    pooled[task] = (A, B, G)
    n = len(A)
    labs = sorted(set(A) | set(B))
    ct = {(la, lb): sum(1 for i in range(n) if A[i] == la and B[i] == lb)
          for la in labs for lb in labs}
    k = cohen_kappa(A, B)
    raw = sum(1 for i in range(n) if A[i] == B[i]) / n
    lo, hi = boot_ci(cohen_kappa, A, B)
    print(f"\n--- {task}  pooled n={n} ---")
    print(f"  A-vs-B kappa = {k:.3f}   raw agreement = {raw:.3f}   "
          f"95% bootstrap CI = [{lo:.3f}, {hi:.3f}]")
    print(f"  A-vs-B contingency: " +
          ", ".join(f"{la}/{lb}={ct[(la,lb)]}" for la in labs for lb in labs))
    print(f"  A marginal: {dict(collections.Counter(A))}   "
          f"B marginal: {dict(collections.Counter(B))}   "
          f"gold: {dict(collections.Counter(G))}")

# difference in kappa between the two tasks, with a bootstrap CI on the DIFFERENCE
Aa, Ba, _ = pooled["agent_safety"]
Ar, Br, _ = pooled["refusal"]
diff = cohen_kappa(Aa, Ba) - cohen_kappa(Ar, Br)
rng = random.Random(7)
dv = []
for _ in range(5000):
    ia = [rng.randrange(len(Aa)) for _ in range(len(Aa))]
    ir = [rng.randrange(len(Ar)) for _ in range(len(Ar))]
    dv.append(cohen_kappa([Aa[i] for i in ia], [Ba[i] for i in ia]) -
              cohen_kappa([Ar[i] for i in ir], [Br[i] for i in ir]))
dv.sort()
print(f"\n  kappa(agent_safety) - kappa(refusal) = {diff:.3f}   "
      f"95% bootstrap CI on the difference = [{dv[125]:.3f}, {dv[4875]:.3f}]")
print(f"  -> CI on the difference {'excludes' if dv[125] > 0 else 'includes'} 0")

# ---------------------------------------------------------------- Table 2
print("\n" + "=" * 72)
print("TABLE 2  --  L2 test-retest (mean, population sd over 5 trials)")
print("=" * 72)
l2 = json.load(open(os.path.join(HERE, "l2_refusal_consolidated.json")))
LOCAL = {"gemma2-2b", "gemma3-4b", "llama3.1-8b", "llama3.2-1b", "llama3.2-3b",
         "mistral-7b", "phi3.5-3.8b", "phi4-mini-3.8b", "qwen2.5-0.5b",
         "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "tinyllama", "yi-6b",
         "deepseek-r1-7b"}
for group, members in (("local", LOCAL), ("api", set(l2) - LOCAL)):
    print(f"\n--- {group} ---")
    for m in sorted(members):
        t = l2[m]["trials"]
        mean = sum(t) / len(t)
        var = sum((x - mean) ** 2 for x in t) / len(t)   # population sd, /n
        recomputed_ok = abs(var ** 0.5 - l2[m]["std"]) < 1e-9
        print(f"  {m:22s} trials={['%.3f'%x for x in t]} mean={mean:.3f} "
              f"sd={var**0.5:.4f} (stored sd matches: {recomputed_ok}) "
              f"n_errors={l2[m].get('n_errors')}")

print("\n" + "=" * 72)
print("ERROR-SOURCE RECORD (kept deliberately, per review point 4)")
print("=" * 72)
print("""\
The per-item 'decision flip' count reported in Table 2 of earlier drafts was
computed by three different run harnesses (laptop commercial batch, Mac-mini
local batch, single-model reruns) whose per-item bookkeeping was not identical:
one harness did not persist per-item labels at all, so its flip counts defaulted
to 0 even when the trial-level sd was non-zero (e.g. deepseek-v4-pro:
trials [1,1,1,0.955,0.955], sd 0.022, but flip count stored as 0 -- impossible,
one item must have flipped). Because the per-item L2 responses were not retained
for every model, a correct flip count cannot be regenerated from the saved
artifacts. We therefore (a) removed the flip column, (b) report only sd, which
IS regenerable from the five stored trial rates and is verified above to match
the stored value for every model, and (c) keep this record so the deficiency is
documented rather than hidden. A re-run that persists per-item responses for all
25 models would restore the flip count; it is listed as future work.""")
