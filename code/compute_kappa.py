#!/usr/bin/env python3
"""计算第二标注者 vs 原始 gold 的 Cohen's κ。
用法: python3 compute_kappa.py <filled_blind.jsonl> <bench: agent_safety|refusal>
标注者填好 your_decision / your_action 后运行。"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
key = json.load(open(os.path.join(HERE, "answer_key.json"), encoding="utf-8"))

def cohen_kappa(a, b):
    """a, b: 等长的标签列表。返回 (kappa, raw_agreement, n)。"""
    n = len(a)
    labels = sorted(set(a) | set(b))
    # 观测一致率
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # 期望一致率（按各自边缘分布）
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for y in b if y == lab) / n
        pe += pa * pb
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return kappa, po, n

def main():
    if len(sys.argv) < 3:
        print("用法: compute_kappa.py <filled.jsonl> <agent_safety|refusal>")
        return 2
    filled = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    bench = sys.argv[2]
    field = "your_decision" if bench == "agent_safety" else "your_action"
    gold_map = key[bench]
    annot, gold, skipped = [], [], 0
    for r in filled:
        v = (r.get(field) or "").strip().lower()
        if v not in ("allow", "block", "answer", "refuse"):
            skipped += 1
            continue
        annot.append(v)
        gold.append(gold_map[r["id"]])
    if not annot:
        print("没有有效标注，请先填 %s 字段" % field)
        return 1
    kappa, po, n = cohen_kappa(annot, gold)
    print(f"bench={bench}  n={n}  skipped={skipped}")
    print(f"raw_agreement (原始一致率) = {po:.3f}")
    print(f"Cohen's kappa            = {kappa:.3f}")
    interp = ("差<0.20" if kappa < 0.2 else "一般0.20-0.40" if kappa < 0.4 else
              "中等0.40-0.60" if kappa < 0.6 else "较好0.60-0.80" if kappa < 0.8 else "很好>0.80")
    print(f"解读: {interp}  (κ<0.6 视为 gold 标签不够可靠，需先修基准)")

if __name__ == "__main__":
    sys.exit(main())
