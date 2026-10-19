"""
Which model - or which combination - should the application actually use?

Scores the human-labelled items with BOTH the evidence ANN and the fine-tuned cross-encoder, then
compares five decision rules under one honest protocol:

    threshold / weight tuned on the 99 human-labelled items from the NON-CS domains
    reported on the 201 human-labelled CS items, which no model ever trained on

    python compare_fusion.py

Writes results_fusion/metrics.json and results_fusion/comparison.png
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze import TfidfFeaturiser

sys.stdout.reconfigure(encoding="utf-8")
OUT = "results_fusion"
CE_DIR = "results_crossencoder"
MAXLEN = 192
GRID = np.arange(0.05, 0.96, 0.01)
key = lambda r: (r["syllabus"], r["topic"], r["file"], r["page"])


def ce_scores(rows, batch=32):
    tok = AutoTokenizer.from_pretrained(f"{CE_DIR}/model")
    model = AutoModelForSequenceClassification.from_pretrained(f"{CE_DIR}/model").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            c = rows[i:i + batch]
            enc = tok([r["topic"] for r in c], [r["text"] for r in c], padding=True,
                      truncation=True, max_length=MAXLEN, return_tensors="pt")
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).numpy())
            print(f"  cross-encoder {min(i + batch, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(out)


def ann_scores(rows):
    model = tf.keras.models.load_model("results_tfidf/evidence_ann.keras")
    feat = TfidfFeaturiser()
    out = []
    for r in rows:                       # one topic at a time, as the app does
        X = feat.topic_page(r["topic"], [(r["file"], r["page"], r["text"])])
        out.append(float(model.predict(X, verbose=0).ravel()[0]))
    return np.array(out)


def best_threshold(y, p):
    f1s = [precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)[2]
           for t in GRID]
    return float(GRID[int(np.argmax(f1s))])


def report(y, p, thr):
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    return {"threshold": thr, "accuracy": float((pred == y).mean()), "precision": float(pr),
            "recall": float(rc), "f1": float(f1), "roc_auc": float(roc_auc_score(y, p)),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def main():
    os.makedirs(OUT, exist_ok=True)
    reviewed = {key(r): r for r in csv.DictReader(open("dataset/review_done.csv", encoding="utf8"))
                if r["human_label"] in ("0", "1")}
    rows = []
    for split in ("train", "val", "test"):
        for r in csv.DictReader(open(f"dataset/pairs_{split}.csv", encoding="utf8")):
            if key(r) in reviewed:
                r["_y"] = int(reviewed[key(r)]["human_label"])
                r["_cs"] = reviewed[key(r)]["domain"] == "CS"
                rows.append(r)
    print(f"human-labelled items: {len(rows)}  "
          f"(gold CS {sum(r['_cs'] for r in rows)}, tuning {sum(not r['_cs'] for r in rows)})")

    print("scoring with the evidence ANN ...")
    pa = ann_scores(rows)
    print("scoring with the cross-encoder ...")
    pc = ce_scores(rows)
    y = np.array([r["_y"] for r in rows])
    cs = np.array([r["_cs"] for r in rows])
    tune, gold = ~cs, cs

    # the blend weight is chosen on the tuning items only, like every other hyperparameter
    weights = np.arange(0, 1.01, 0.05)
    scored = []
    for w in weights:
        p = (1 - w) * pa + w * pc
        t = best_threshold(y[tune], p[tune])
        f1 = precision_recall_fscore_support(y[tune], p[tune] >= t, average="binary",
                                             zero_division=0)[2]
        scored.append((f1, w, t))
    _, w_best, t_blend = max(scored)
    print(f"blend weight chosen on the tuning items: {w_best:.2f} "
          f"(0 = ANN only, 1 = cross-encoder only), threshold {t_blend:.2f}")

    rules = {}
    rules["ann_only"] = report(y[gold], pa[gold], best_threshold(y[tune], pa[tune]))
    rules["crossencoder_only"] = report(y[gold], pc[gold], best_threshold(y[tune], pc[tune]))
    mean = 0.5 * (pa + pc)
    rules["mean_of_both"] = report(y[gold], mean[gold], best_threshold(y[tune], mean[tune]))
    blend = (1 - w_best) * pa + w_best * pc
    rules[f"weighted_blend_w{w_best:.2f}"] = report(y[gold], blend[gold], t_blend)

    # retrieve-then-rerank: the ANN must pass a recall-oriented gate, the cross-encoder decides
    gate = float(np.percentile(pa[tune][y[tune] == 1], 10))      # keeps 90% of true evidence
    rerank = np.where(pa >= gate, pc, 0.0)
    rules["rerank_ann_then_ce"] = report(y[gold], rerank[gold], best_threshold(y[tune],
                                                                              rerank[tune]))
    rules["rerank_ann_then_ce"]["ann_gate"] = gate

    best = max(rules, key=lambda k: rules[k]["f1"])
    out = {"n_gold": int(gold.sum()), "n_tuning": int(tune.sum()),
           "protocol": "every threshold and weight tuned on the non-CS human-labelled items; "
                       "reported on the 201 human-labelled CS items",
           "blend_weight": float(w_best), "best_rule": best, "rules": rules}
    json.dump(out, open(f"{OUT}/metrics.json", "w"), indent=1)

    print(f"\n{'rule':<28}{'acc':>7}{'prec':>7}{'rec':>7}{'F1':>7}{'AUC':>7}")
    for k, v in rules.items():
        print(f"{k:<28}{v['accuracy']*100:6.1f}%{v['precision']:7.3f}{v['recall']:7.3f}"
              f"{v['f1']:7.3f}{v['roc_auc']:7.3f}")
    print(f"\nbest by F1 on the gold set: {best}")

    names = list(rules)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    for i, (m, c) in enumerate([("accuracy", "#1F3864"), ("precision", "#2E74B5"),
                                ("recall", "#7FA9D4"), ("f1", "#BDD3EA")]):
        ax.bar(x + (i - 1.5) * 0.2, [rules[n][m] for n in names], 0.2, label=m.title(), color=c)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Decision rules on the 201 human-verified Computer Science items")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/comparison.png", dpi=150)
    print(f"\nwrote {OUT}/metrics.json and {OUT}/comparison.png")


if __name__ == "__main__":
    main()
