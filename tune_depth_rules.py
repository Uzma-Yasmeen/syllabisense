"""
Tune the thresholds in depth.py against the hand-labelled verdicts.

The rules that decide properly / barely / not taught were written by hand with no data to check
them against. There are 150 labelled topics now, so the thresholds can be fitted instead of
guessed - and, more importantly, the rules can be given a way to say NOT TAUGHT when evidence
exists but is too weak to be teaching. Today they only say that when the evidence list is empty,
which is why they almost never agree with a human who says a topic is not taught.

Subjects are split in two: thresholds are chosen on one half and reported on the other, so the
improvement is measured on topics the tuning never saw.

    python tune_depth_rules.py

Writes results_verdicts/tuned_rules.json
"""
import csv
import itertools
import json
import os
import random
import sys

import numpy as np

from depth import evidence_kinds, STRONG, SUPPORT

sys.stdout.reconfigure(encoding="utf-8")
OUT = "results_verdicts"
CLASSES = ["NOT", "BARELY", "PROPERLY"]


def judge_p(ev, p):
    """The same decision as depth.judge, with every threshold exposed as a parameter."""
    if not ev:
        return "NOT"
    kinds = set()
    for _, _, _, text in ev:
        kinds |= evidence_kinds(text)
    pages = len(ev)
    best = ev[0][2]
    has_strong = bool(kinds & set(STRONG))
    has_support = bool(kinds & set(SUPPORT))

    # NEW: evidence can exist and still not be teaching
    if best < p["not_conf"] and not has_strong:
        return "NOT"
    if pages <= p["not_pages"] and not has_strong and not has_support:
        return "NOT"

    if pages >= p["proper_pages_support"] and has_strong and has_support:
        level = "PROPERLY"
    elif pages >= p["proper_pages_strong"] and has_strong:
        level = "PROPERLY"
    else:
        level = "BARELY"
    if level == "PROPERLY" and best < p["strong_conf"]:
        level = "BARELY"
    return level


def accuracy(rows, p):
    return float(np.mean([judge_p(r["ev"], p) == r["y"] for r in rows])) if rows else 0.0


def macro_f1(rows, p):
    pred = [judge_p(r["ev"], p) for r in rows]
    true = [r["y"] for r in rows]
    fs = []
    for c in CLASSES:
        tp = sum(1 for a, b in zip(true, pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(true, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(true, pred) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs))


def main():
    os.makedirs(OUT, exist_ok=True)
    cands = {(c["syllabus"], c["topic"]): c
             for c in json.load(open("dataset/depth_candidates.json", encoding="utf8"))}
    rows = []
    for r in csv.DictReader(open("dataset/depth_labels.csv", encoding="utf8")):
        c = cands.get((r["syllabus"], r["topic"]))
        if c is None or r["label"] not in CLASSES:
            continue
        rows.append({"subject": r["subject"], "y": r["label"],
                     "ev": [(e["file"], e["loc"], e["prob"], e["text"]) for e in c["evidence"]]})
    print(f"labelled topics usable: {len(rows)}")

    subjects = sorted({r["subject"] for r in rows})
    random.Random(7).shuffle(subjects)
    half = set(subjects[:len(subjects) // 2])
    tune = [r for r in rows if r["subject"] in half]
    test = [r for r in rows if r["subject"] not in half]
    print(f"  tuning on {len(tune)} topics from {len(half)} subjects")
    print(f"  reporting on {len(test)} topics from {len(subjects) - len(half)} subjects\n")

    current = {"not_conf": 0.0, "not_pages": 0, "proper_pages_support": 2,
               "proper_pages_strong": 3, "strong_conf": 0.90}
    print(f"rules as they stand : tune {accuracy(tune, current) * 100:.1f}%  "
          f"test {accuracy(test, current) * 100:.1f}%  (macro F1 {macro_f1(test, current):.3f})")

    grid = {"not_conf": [0.0, 0.80, 0.90, 0.95, 0.97, 0.99],
            "not_pages": [0, 1, 2],
            "proper_pages_support": [2, 3, 4],
            "proper_pages_strong": [3, 4, 5, 99],
            "strong_conf": [0.0, 0.90, 0.95, 0.99]}
    keys = list(grid)
    best, best_score = None, -1.0
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, combo))
        s = macro_f1(tune, p) + accuracy(tune, p)      # balance the classes, not just the total
        if s > best_score:
            best, best_score = p, s
    print(f"\nsearched {np.prod([len(v) for v in grid.values()])} combinations")
    print("best thresholds found on the tuning half:")
    for k, v in best.items():
        print(f"  {k:<22}{current[k]}  ->  {v}")

    a_t, f_t = accuracy(test, best), macro_f1(test, best)
    a_c, f_c = accuracy(test, current), macro_f1(test, current)
    print(f"\nON THE HELD-OUT SUBJECTS ({len(test)} topics)")
    print(f"  current rules : {a_c * 100:.1f}%  macro F1 {f_c:.3f}")
    print(f"  tuned rules   : {a_t * 100:.1f}%  macro F1 {f_t:.3f}")
    print(f"  change        : {(a_t - a_c) * 100:+.1f} points,  F1 {f_t - f_c:+.3f}")

    print("\n  rows = human, columns = tuned rules")
    print(f"  {'':<10}" + "".join(f"{c:>10}" for c in CLASSES))
    for c in CLASSES:
        line = [sum(1 for r in test if r["y"] == c and judge_p(r["ev"], best) == d)
                for d in CLASSES]
        print(f"  {c:<10}" + "".join(f"{v:>10}" for v in line))

    json.dump({"tuned": best, "previous": current,
               "held_out_topics": len(test),
               "accuracy_before": a_c, "accuracy_after": a_t,
               "macro_f1_before": f_c, "macro_f1_after": f_t,
               "protocol": "thresholds chosen on one half of the subjects, reported on the other"},
              open(f"{OUT}/tuned_rules.json", "w"), indent=1)
    print(f"\nwrote {OUT}/tuned_rules.json")


if __name__ == "__main__":
    main()
