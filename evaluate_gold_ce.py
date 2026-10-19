"""
Evaluate the fine-tuned cross-encoder against the HUMAN-verified labels, using exactly the same
protocol as evaluate_gold.py does for the ANN - otherwise the two are not comparable.

  gold test set : reviewed items from the CS split (CS is never trained on)
  threshold     : tuned on the reviewed items from the OTHER domains, never on the gold test

    python evaluate_gold_ce.py

Rewrites results_crossencoder/metrics_human.json
"""
import csv
import json
import sys

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
OUT = "results_crossencoder"
MAXLEN = 192
key = lambda r: (r["syllabus"], r["topic"], r["file"], r["page"])


def predict(model, tok, rows, batch=32):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            enc = tok([r["topic"] for r in chunk], [r["text"] for r in chunk], padding=True,
                      truncation=True, max_length=MAXLEN, return_tensors="pt")
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    reviewed = {key(r): r for r in csv.DictReader(open("dataset/review_done.csv", encoding="utf8"))
                if r["human_label"] in ("0", "1")}
    print(f"human-labelled items: {len(reviewed)}")

    rows = []
    for split in ("train", "val", "test"):
        for r in csv.DictReader(open(f"dataset/pairs_{split}.csv", encoding="utf8")):
            if key(r) in reviewed:
                r["_human"] = int(reviewed[key(r)]["human_label"])
                r["_domain"] = reviewed[key(r)]["domain"]
                rows.append(r)
    gold = [r for r in rows if r["_domain"] == "CS"]
    tune = [r for r in rows if r["_domain"] != "CS"]
    print(f"gold (CS, unseen): {len(gold)}      threshold-tuning (other domains): {len(tune)}")

    tok = AutoTokenizer.from_pretrained(f"{OUT}/model")
    model = AutoModelForSequenceClassification.from_pretrained(f"{OUT}/model")

    print("scoring the threshold-tuning items ...")
    pt = predict(model, tok, tune)
    yt = np.array([r["_human"] for r in tune])
    grid = np.arange(0.05, 0.96, 0.01)
    f1s = [precision_recall_fscore_support(yt, pt >= t, average="binary", zero_division=0)[2]
           for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    print(f"threshold tuned on human labels (non-CS): {thr:.2f}")

    print("scoring the gold items ...")
    pg = predict(model, tok, gold)
    yg = np.array([r["_human"] for r in gold])
    pred = (pg >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yg, pred, average="binary", zero_division=0)
    cm = confusion_matrix(yg, pred)

    out = {"n_gold": len(gold), "n_threshold_tuning": len(tune),
           "threshold_human_tuned": thr,
           "protocol": "threshold tuned on human-labelled items from other domains; "
                       "reported on human-labelled CS items, which no model trained on",
           "human_labels": {"accuracy": float((pred == yg).mean()), "precision": float(pr),
                            "recall": float(rc), "f1": float(f1),
                            "roc_auc": float(roc_auc_score(yg, pg)),
                            "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}}}
    json.dump(out, open(f"{OUT}/metrics_human.json", "w"), indent=1)
    print(json.dumps(out["human_labels"], indent=1))
    rep = classification_report(yg, pred, target_names=["not evidence", "evidence"],
                                zero_division=0)
    open(f"{OUT}/classification_report_human.txt", "w", encoding="utf8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
