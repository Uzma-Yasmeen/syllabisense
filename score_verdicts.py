"""
How often is the verdict the student actually reads correct?

Every other number in this project scores one (topic, page) decision. This one scores the thing
the application puts on screen: properly taught / barely taught / not taught, per syllabus topic.

It compares the rules in depth.py - which is what the application uses - against the hand-labelled
verdicts in dataset/depth_labels.csv, and does the same for the trained depth model when one
exists, so the two can be compared on equal terms.

    python score_verdicts.py

Writes results_verdicts/metrics.json and results_verdicts/confusion_matrix.png
"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from depth import judge

sys.stdout.reconfigure(encoding="utf-8")
OUT = "results_verdicts"
CANDIDATES = "dataset/depth_candidates.json"
LABELS = "dataset/depth_labels.csv"
CLASSES = ["NOT", "BARELY", "PROPERLY"]
FULL = {"NOT": "NOT TAUGHT", "BARELY": "BARELY TAUGHT", "PROPERLY": "PROPERLY TAUGHT"}
SHORT = {v: k for k, v in FULL.items()}


def ci(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    cands = {(c["syllabus"], c["topic"]): c
             for c in json.load(open(CANDIDATES, encoding="utf8"))}
    labels = [r for r in csv.DictReader(open(LABELS, encoding="utf8"))
              if r["label"] in CLASSES]
    print(f"hand-labelled topics: {len(labels)}")

    y_true, y_rule, missing = [], [], 0
    for r in labels:
        c = cands.get((r["syllabus"], r["topic"]))
        if c is None:
            missing += 1
            continue
        ev = [(e["file"], e["loc"], e["prob"], e["text"]) for e in c["evidence"]]
        level, _, _ = judge(ev)
        y_true.append(r["label"])
        y_rule.append(SHORT.get(level, "NOT"))
    if missing:
        print(f"  ({missing} labelled topics are not in the candidate file, skipped)")
    n = len(y_true)
    if not n:
        print("nothing to score")
        return

    acc = float(np.mean([a == b for a, b in zip(y_true, y_rule)]))
    macro = float(f1_score(y_true, y_rule, average="macro", labels=CLASSES, zero_division=0))
    cm = confusion_matrix(y_true, y_rule, labels=CLASSES)

    print(f"\nTHE NUMBER THAT WAS MISSING — how often the on-screen verdict is right")
    print(f"  topics scored : {n}")
    print(f"  accuracy      : {acc * 100:.1f}%  (95% CI ±{ci(acc, n) * 100:.1f} points)")
    print(f"  macro F1      : {macro:.3f}")
    print(f"\n  for reference, always guessing the most common class would score "
          f"{max(y_true.count(c) for c in CLASSES) / n * 100:.1f}%")

    print("\n  rows = what you said, columns = what the app says")
    print(f"  {'':<10}" + "".join(f"{c:>10}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:<10}" + "".join(f"{cm[i, j]:>10}" for j in range(len(CLASSES))))

    rep = classification_report(y_true, y_rule, labels=CLASSES, zero_division=0)
    print("\n" + rep)
    open(f"{OUT}/classification_report.txt", "w", encoding="utf8").write(rep)

    out = {"what": "accuracy of the per-topic verdict shown in the application",
           "judged_by": "the rules in depth.py, which is what the application uses",
           "n_topics": n, "accuracy": acc, "accuracy_ci95": ci(acc, n), "macro_f1": macro,
           "classes": CLASSES,
           "confusion_matrix": {f"true_{CLASSES[i]}": {f"pred_{CLASSES[j]}": int(cm[i, j])
                                                       for j in range(len(CLASSES))}
                                for i in range(len(CLASSES))},
           "per_class": {c: {"recall": float(cm[i].sum() and cm[i, i] / cm[i].sum()),
                             "n": int(cm[i].sum())} for i, c in enumerate(CLASSES)}}

    model_path = "results_depth/depth_ann.keras"
    if os.path.exists(model_path):
        fn = json.load(open("results_depth/feature_names.json", encoding="utf8"))
        feat_names = fn["features"] if isinstance(fn, dict) else fn
        import tensorflow as tf
        m = tf.keras.models.load_model(model_path)
        X = np.array([[cands[(r["syllabus"], r["topic"])]["features"][f] for f in feat_names]
                      for r in labels if (r["syllabus"], r["topic"]) in cands], np.float32)
        pred = m.predict(X, verbose=0).argmax(1)
        y_model = [CLASSES[i] for i in pred]
        macc = float(np.mean([a == b for a, b in zip(y_true, y_model)]))
        out["trained_depth_model"] = {
            "accuracy": macc,
            "macro_f1": float(f1_score(y_true, y_model, average="macro", labels=CLASSES,
                                       zero_division=0)),
            "note": "scored on ALL labelled topics including the ones it trained on, so this is "
                    "optimistic; results_depth/metrics.json holds the held-out figure"}
        print(f"the trained depth model on the same topics: {macc * 100:.1f}% "
              f"(optimistic — it trained on most of them)")

    json.dump(out, open(f"{OUT}/metrics.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASSES)), [FULL[c].title() for c in CLASSES], fontsize=8)
    ax.set_yticks(range(len(CLASSES)), [FULL[c].title() for c in CLASSES], fontsize=8)
    ax.set_xlabel("what the application says")
    ax.set_ylabel("what the human said")
    ax.set_title(f"Per-topic verdict — {acc * 100:.1f}% on {n} topics", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=140)
    print(f"\nwrote {OUT}/metrics.json and {OUT}/confusion_matrix.png")


if __name__ == "__main__":
    main()
