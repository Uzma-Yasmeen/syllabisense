"""
How much do two people agree about what "properly taught" means?

    python score_agreement.py

Raw agreement on its own flatters: with three classes, two people answering at random agree a
third of the time. Cohen's kappa subtracts that, so 0 means chance and 1 means perfect.

The number that matters most is the last one. The project's headline figure - how often the
application's verdict is right - was measured against one person's labels. If it moves when
scored against a second person's, then it was partly measuring that person. If it holds, the
measurement is about the system.

Writes results_verdicts/agreement.json
"""
import csv
import json
import math
import os
import sys

import numpy as np

from depth import judge

sys.stdout.reconfigure(encoding="utf-8")
OUT = "results_verdicts"
FIRST = "dataset/depth_labels.csv"
SECOND = "dataset/depth_labels_second.csv"
CANDIDATES = "dataset/depth_candidates.json"
CLASSES = ["NOT", "BARELY", "PROPERLY"]
SHORT = {"NOT TAUGHT": "NOT", "BARELY TAUGHT": "BARELY", "PROPERLY TAUGHT": "PROPERLY"}


def read(path):
    if not os.path.exists(path):
        return {}
    return {(r["syllabus"], r["topic"]): r["label"]
            for r in csv.DictReader(open(path, encoding="utf8")) if r["label"] in CLASSES}


def kappa(a, b):
    """Cohen's kappa: agreement after removing what chance would give you anyway."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in CLASSES)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0, observed, expected


def reading(k):
    for limit, words in ((0.0, "no better than chance"), (0.20, "slight"), (0.40, "fair"),
                         (0.60, "moderate"), (0.80, "substantial"), (1.01, "almost perfect")):
        if k < limit:
            return words
    return "almost perfect"


def ci(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def main():
    first, second = read(FIRST), read(SECOND)
    if not second:
        print(f"No second opinion yet. Run:\n\n    streamlit run agreement_review.py\n")
        return
    shared = sorted(set(first) & set(second))
    if not shared:
        print("The two label files have no topics in common.")
        return

    a = [first[k] for k in shared]
    b = [second[k] for k in shared]
    k, observed, expected = kappa(a, b)
    n = len(shared)

    print(f"topics labelled by both readers: {n}\n")
    print(f"  they agree on          : {observed * 100:.1f}%  (±{ci(observed, n) * 100:.1f})")
    print(f"  chance alone would give: {expected * 100:.1f}%")
    print(f"  Cohen's kappa          : {k:.3f}  — {reading(k)}")

    print("\n  where they differ (rows = first reader, columns = second)")
    print(f"  {'':<10}" + "".join(f"{c:>10}" for c in CLASSES))
    table = {}
    for i in CLASSES:
        row = [sum(1 for x, y in zip(a, b) if x == i and y == j) for j in CLASSES]
        table[i] = dict(zip(CLASSES, row))
        print(f"  {i:<10}" + "".join(f"{v:>10}" for v in row))

    # the question this exercise exists to answer
    cands = {(c["syllabus"], c["topic"]): c
             for c in json.load(open(CANDIDATES, encoding="utf8"))}
    sys_verdict = {}
    for key in shared:
        c = cands.get(key)
        if c is None:
            continue
        ev = [(e["file"], e["loc"], e["prob"], e["text"]) for e in c["evidence"]]
        sys_verdict[key] = SHORT.get(judge(ev)[0], "NOT")
    scored = [k2 for k2 in shared if k2 in sys_verdict]
    acc_a = float(np.mean([sys_verdict[k2] == first[k2] for k2 in scored]))
    acc_b = float(np.mean([sys_verdict[k2] == second[k2] for k2 in scored]))
    both = [k2 for k2 in scored if first[k2] == second[k2]]
    acc_both = float(np.mean([sys_verdict[k2] == first[k2] for k2 in both])) if both else 0.0

    print(f"\n  THE POINT OF THE EXERCISE — the system scored against each reader ({len(scored)} topics)")
    print(f"    against the first reader : {acc_a * 100:.1f}%  (±{ci(acc_a, len(scored)) * 100:.1f})")
    print(f"    against the second       : {acc_b * 100:.1f}%  (±{ci(acc_b, len(scored)) * 100:.1f})")
    print(f"    difference               : {abs(acc_a - acc_b) * 100:.1f} points")
    print(f"    on the {len(both)} topics they agreed about: {acc_both * 100:.1f}%")
    print("\n  A small difference means the headline number describes the system, not the person\n"
          "  who labelled it. A large one means it was partly measuring the labeller.")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"n_topics": n, "raw_agreement": observed, "chance_agreement": expected,
               "cohens_kappa": k, "interpretation": reading(k),
               "confusion": table,
               "system_accuracy": {"against_first": acc_a, "against_second": acc_b,
                                   "difference": abs(acc_a - acc_b),
                                   "where_readers_agreed": acc_both,
                                   "n_scored": len(scored), "n_agreed": len(both)}},
              open(f"{OUT}/agreement.json", "w"), indent=1)
    print(f"wrote {OUT}/agreement.json")


if __name__ == "__main__":
    main()
