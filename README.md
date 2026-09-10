# Reproducibility Package

This is the anonymized reproducibility bundle referenced in the manuscript's
Sec. 7. It is self-contained: no API calls are required to reproduce any number
reported in the paper — everything is derived from the frozen prediction files
in `data/predictions/`.

Benchmark names are anonymized to match the manuscript: **Bench-Inject**
(agent tool-call injection defense, the paper's primary L1/RQ1 benchmark) and
**Bench-Refusal** (defensive-refusal on dual-use security questions, the
paper's RQ2 benchmark). `Bench-AttackAttrib` and `Bench-FuncCall` are two
auxiliary benchmarks used only to compute the leaderboard's non-headline
columns, not discussed in the paper's main analysis.

## Structure

```
data/
  scored_results/results.json      # single source of truth: every model x
                                    # benchmark x metric number in the paper
  predictions/<Bench-*>/preds_*.jsonl   # per-model, per-item raw model output
                                    # (raw_text field) + parsed label — the
                                    # audit trail that caught the Sec 4.4
                                    # scorer bug
  predictions/<Bench-*>/gold_bench.jsonl  # gold labels for each item
  kappa/                           # Sec 4.5 blind-annotation materials:
                                    # blind samples (no gold), the withheld
                                    # answer key, round1_raw_labels/ and
                                    # round2_raw_labels/ (both rounds' actual
                                    # per-item human labels), and (once run)
                                    # LLM-annotator filled files
  l2_retest/                       # Sec 4.3/4.1: raw 5-trial repeat-measure
                                    # results for 11/25 models (refusal_side_
                                    # 11models.json + _doubao15pro.json) --
                                    # feeds compute_all_statistics.py's
                                    # empirical disattenuation input
code/
  backends.py                      # model-calling client (not needed to
                                    # reproduce statistics from frozen data;
                                    # only needed to re-run models live)
  run_eval.py                      # full evaluation harness (live re-run)
  rerun_refusal_only.py            # re-score Bench-Refusal only (used for the
                                    # Sec 4.4 scorer-bug fix and Sec 4.3 L2 reruns)
  repeat_measure.py                # Sec 4.3 test-retest (5x reruns)
  llm_annotate.py                  # Sec 4.5 LLM-as-annotator blind labeling
  compute_kappa.py                 # Cohen's kappa from a filled blind-annotation form
  compute_all_statistics.py        # *** START HERE ***: reproduces every
                                    # headline number (Sec 4.1, 4.2) from the
                                    # frozen data/, no API calls needed --
                                    # including the disattenuation calc from
                                    # data/l2_retest/. Sec 4.5's pooled n=54
                                    # kappa is not yet wired into this script
                                    # (compute manually from kappa/round{1,2}_
                                    # raw_labels/ + answer_key.json, or see
                                    # kappa/interrater_results_summary.json
                                    # for the precomputed result)
```

## Quickest path to verifying the paper's numbers

```bash
cd code
python3 compute_all_statistics.py
```

This reads `data/scored_results/results.json` and `data/l2_retest/` and prints:
- Sec. 4.1's random-resampling table (the corrected L1 finding)
- Sec. 4.1's disattenuation calculation
- Sec. 4.2's RQ2 correlation and TOST equivalence test

**A note on exact-match precision.** Numbers involving bootstrap resampling
(the CIs, the random-resampling medians) will differ from the manuscript at
roughly the third decimal place on a re-run, even with the same seed — the
exact value consumed from Python's random stream depends on the order and
count of prior random calls, which is itself a small illustration of the
paper's own point: report intervals, not point estimates, at this level of
precision. The nested-panel numbers in Sec. 4.1's *retracted* framing are not
independently reproducible byte-for-byte, because the exact model-accumulation
order used in that superseded analysis was not separately version-pinned —
this is disclosed in the script's comments and is consistent with Sec. 4.1's
finding that the nested ordering itself was the flaw, not a number worth
preserving exactly.

## Re-running the underlying evaluation (not required to check the paper's numbers)

`code/run_eval.py`, `code/rerun_refusal_only.py`, and `code/repeat_measure.py`
call live model APIs and are provided for transparency and extension (e.g. to
audit additional models), not because they are needed to verify the numbers
already reported. Model access credentials are not included; see each
script's `models_*.json`-style config argument for the expected shape.

## Sec. 4.5 human inter-rater kappa

`data/kappa/Bench-Inject_blind27.jsonl` / `Bench-Refusal_blind27.jsonl` (round 1,
n=27 each) and `*_round2.jsonl` (round 2, a second, non-overlapping n=27 each,
drawn to tighten round 1's wide confidence interval) are the blind samples given to
the three LLM annotators and two independent human annotators (identified in the
manuscript only as Annotator A / Annotator B). `answer_key.json` is withheld from
annotators during labeling; `code/compute_kappa.py <filled.jsonl>
<agent_safety|refusal>` reproduces one annotator's kappa against this key from a
filled blind form. `data/kappa/interrater_results_summary.json` has the final
computed values. Both rounds use the **same two annotators** (Annotator A / the
author, Annotator B / an LLM-focused postdoc colleague); round 1's raw per-item
labels were initially believed lost (only the aggregate κ had been recorded at
the time) but were later recovered from the annotators' own records and verified
to exactly reproduce the previously-published round-1 aggregate numbers before
being used for anything further. With both rounds' raw labels available, the
statistically correct **pooled n=54 statistic** is the headline number: κ=0.771
(agent-safety, substantial agreement) and κ=0.469 (defensive-refusal, moderate,
below the conventional 0.6 threshold for substantial agreement) — noting that
neither individual n=27 round was a reliable estimate of this pooled value on
its own (round 1: κ=1.000 / κ=0.242; round 2: κ=0.550 / κ=0.867). See the
manuscript's Sec. 4.5 for the full discussion of why the individual-round
numbers are unreliable and the pooled one is preferred.
`data/kappa/round1_raw_labels/` and `data/kappa/round2_raw_labels/` contain
each round's actual per-item labels (`annotator_A_*.json` /
`annotator_B_*.json`, id → label only, no annotator-identifying content), so
every number above is independently reproducible from raw data for both
rounds — round 1's raw labels are no longer an exception to this.
