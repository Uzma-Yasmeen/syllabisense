"""
Evaluate the trained model against the HUMAN-verified labels.

Honest protocol:
  - gold test set  : reviewed items from the CS split (CS is never trained on)
  - threshold      : tuned on the reviewed items from the train/val splits, never on the gold test
  - weak-label test: the same model scored against the automatic labels, for comparison

    python evaluate_gold.py --model results_embed
"""
import argparse
import csv
import json
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="results_tfidf")
ap.add_argument("--drop-sim", action="store_true")
ap.add_argument("--features", default="tfidf", choices=["tfidf", "embed"],
                help="must match the feature set the model was trained on")
a = ap.parse_args()

key = lambda r: (r["syllabus"], r["topic"], r["file"], r["page"])
human = {key(r): r["human_label"] for r in csv.DictReader(open("dataset/review_done.csv", encoding="utf8"))
         if r["human_label"] in ("0", "1")}
print(f"human-labelled items: {len(human)}")

model = tf.keras.models.load_model(f"{a.model}/evidence_ann.keras")
scored = {}
for split in ("train", "val", "test"):
    rows = list(csv.DictReader(open(f"dataset/pairs_{split}.csv", encoding="utf8")))
    X = np.load(f"dataset/{a.features}_{split}.npz")["X"]
    if a.drop_sim:
        X = X[:, :-3]
    idx = [i for i, r in enumerate(rows) if key(r) in human]
    if not idx:
        continue
    p = model.predict(X[idx], verbose=0).ravel()
    for i, prob in zip(idx, p):
        r = rows[i]
        scored[key(r)] = {"prob": float(prob), "human": int(human[key(r)]),
                          "weak": int(r["label"]), "split": split, "domain": r["domain"],
                          "kind": r["kind"], "topic": r["topic"], "file": r["file"], "page": r["page"]}
print(f"scored: {len(scored)} of them")

tune = [v for v in scored.values() if v["split"] != "test"]
gold = [v for v in scored.values() if v["split"] == "test"]
print(f"threshold tuned on {len(tune)} non-test reviewed items; gold test = {len(gold)} CS items")

ty = np.array([v["human"] for v in tune]); tp = np.array([v["prob"] for v in tune])
grid = np.arange(0.05, 0.96, 0.01)
f1s = [precision_recall_fscore_support(ty, tp >= t, average="binary", zero_division=0)[2] for t in grid]
thr = float(grid[int(np.argmax(f1s))])
print(f"threshold from human-labelled non-test items: {thr:.2f}")

gy = np.array([v["human"] for v in gold]); gp = np.array([v["prob"] for v in gold])
wy = np.array([v["weak"] for v in gold])
out = {"model": a.model, "threshold_human_tuned": thr, "n_gold": len(gold)}
for name, yy in (("human_labels", gy), ("weak_labels", wy)):
    pred = (gp >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yy, pred, average="binary", zero_division=0)
    cm = confusion_matrix(yy, pred)
    out[name] = {"accuracy": float((pred == yy).mean()), "precision": float(pr), "recall": float(rc),
                 "f1": float(f1), "roc_auc": float(roc_auc_score(yy, gp)),
                 "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                      "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}}
out["weak_vs_human_agreement"] = float((wy == gy).mean())
json.dump(out, open(f"{a.model}/metrics_human.json", "w"), indent=1)
print(json.dumps(out, indent=1))

wrong = sorted([v for v in gold if (v["prob"] >= thr) != bool(v["human"])], key=lambda v: -abs(v["prob"] - thr))
print(f"\nmodel disagrees with you on {len(wrong)} of {len(gold)}; worst 8:")
for v in wrong[:8]:
    print(f"  p={v['prob']:.2f} you={'relevant' if v['human'] else 'not relevant'} "
          f"| {v['topic'][:46]} | {v['file'].split('/')[-1][:28]} p.{v['page']}")
