"""
Fine-tune a transformer cross-encoder end to end: does this page teach this topic?

This is the deep-learning model of the project. Unlike the feature-based ANN, the transformer is
TRAINED here - gradients flow through all encoder layers, not just a classifier on frozen vectors.

    topic [SEP] page text  ->  MiniLM encoder (6 layers, self-attention)  ->  linear head  ->  sigmoid

Base model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters), pretrained for relevance.
Training: AdamW, lr 2e-5, binary cross-entropy, batch 16, 2 epochs, linear warmup, CPU.
Test split is Computer Science, a subject never seen in training.

    python train_crossencoder.py --epochs 2 --max-train 9000

Writes results_crossencoder/: metrics.json, metrics_human.json, history.png,
confusion_matrix.png, classification_report.txt, model/
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.stdout.reconfigure(encoding="utf-8")
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUT = "results_crossencoder"
MAXLEN = 192


class Pairs(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r["topic"], r["text"], float(r["label"])


def collate(batch, tok):
    a, b, y = zip(*batch)
    enc = tok(list(a), list(b), padding=True, truncation=True, max_length=MAXLEN,
              return_tensors="pt")
    enc["labels"] = torch.tensor(y, dtype=torch.float)
    return enc


def load(split):
    return list(csv.DictReader(open(f"dataset/pairs_{split}.csv", encoding="utf8")))


@torch.no_grad()
def predict(model, rows, tok, bs=64):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        enc = tok([r["topic"] for r in batch], [r["text"] for r in batch], padding=True,
                  truncation=True, max_length=MAXLEN, return_tensors="pt")
        logits = model(**enc).logits.squeeze(-1)
        out += torch.sigmoid(logits).tolist()
        if i and i % (bs * 20) == 0:
            print(f"      scored {i}/{len(rows)}", flush=True)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=9000, help="cap training rows to fit the time budget")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(7)
    random.seed(7)
    torch.set_num_threads(os.cpu_count() or 8)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {BASE}: {n_params/1e6:.1f}M parameters, all trainable")

    train, val, test = load("train"), load("val"), load("test")
    random.shuffle(train)
    train = train[:a.max_train]
    print(f"train {len(train)}  val {len(val)}  test {len(test)} (CS, unseen)")

    dl = DataLoader(Pairs(train, tok), batch_size=a.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(dl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossfn = torch.nn.BCEWithLogitsLoss()

    history = {"loss": [], "val_loss": [], "val_accuracy": []}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        running = 0.0
        for step, enc in enumerate(dl, 1):
            labels = enc.pop("labels")
            loss = lossfn(model(**enc).logits.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                done = ep * len(dl) + step
                rate = (time.time() - t0) / done
                print(f"  epoch {ep+1} step {step}/{len(dl)}  loss {running/step:.4f}  "
                      f"eta {rate*(steps-done)/60:.0f} min", flush=True)
        pv = predict(model, val, tok)
        yv = np.array([int(r["label"]) for r in val])
        vl = float(-np.mean(yv * np.log(pv + 1e-9) + (1 - yv) * np.log(1 - pv + 1e-9)))
        history["loss"].append(running / len(dl))
        history["val_loss"].append(vl)
        history["val_accuracy"].append(float(((pv >= 0.5) == yv).mean()))
        print(f"  epoch {ep+1}: train loss {running/len(dl):.4f}  val loss {vl:.4f}  "
              f"val acc {history['val_accuracy'][-1]:.3f}", flush=True)

    # threshold on validation, never on test
    pv = predict(model, val, tok)
    yv = np.array([int(r["label"]) for r in val])
    grid = np.arange(0.05, 0.96, 0.01)
    f1s = [precision_recall_fscore_support(yv, pv >= t, average="binary", zero_division=0)[2] for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    print(f"threshold from validation: {thr:.2f}")

    print("scoring the test split (CS, unseen) ...")
    pt = predict(model, test, tok)
    yt = np.array([int(r["label"]) for r in test])
    pred = (pt >= thr).astype(int)
    cm = confusion_matrix(yt, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
    metrics = {"base_model": BASE, "parameters_millions": round(n_params / 1e6, 1),
               "threshold": thr, "epochs": a.epochs, "batch_size": a.batch, "learning_rate": a.lr,
               "max_length": MAXLEN, "n_train": len(train),
               "test": {"accuracy": float((pred == yt).mean()), "precision": float(pr),
                        "recall": float(rc), "f1": float(f1), "roc_auc": float(roc_auc_score(yt, pt))},
               "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                    "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
               "history": history, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    print(json.dumps(metrics["test"], indent=1))
    open(f"{OUT}/classification_report.txt", "w", encoding="utf8").write(
        classification_report(yt, pred, target_names=["not relevant", "relevant"], zero_division=0))

    # human-verified gold: the reviewed CS items
    human = {(r["syllabus"], r["topic"], r["file"], r["page"]): int(r["human_label"])
             for r in csv.DictReader(open("dataset/review_done.csv", encoding="utf8"))
             if r["human_label"] in ("0", "1")}
    gold = [(r, p) for r, p in zip(test, pt)
            if (r["syllabus"], r["topic"], r["file"], r["page"]) in human]
    if gold:
        gy = np.array([human[(r["syllabus"], r["topic"], r["file"], r["page"])] for r, _ in gold])
        gp = np.array([p for _, p in gold])
        gpred = (gp >= thr).astype(int)
        gpr, grc, gf1, _ = precision_recall_fscore_support(gy, gpred, average="binary", zero_division=0)
        gcm = confusion_matrix(gy, gpred)
        json.dump({"n_gold": len(gold), "threshold": thr,
                   "human_labels": {"accuracy": float((gpred == gy).mean()), "precision": float(gpr),
                                    "recall": float(grc), "f1": float(gf1),
                                    "roc_auc": float(roc_auc_score(gy, gp)),
                                    "confusion_matrix": {"tn": int(gcm[0, 0]), "fp": int(gcm[0, 1]),
                                                         "fn": int(gcm[1, 0]), "tp": int(gcm[1, 1])}}},
                  open(f"{OUT}/metrics_human.json", "w"), indent=1)
        print(f"human-verified ({len(gold)} items): acc {(gpred == gy).mean():.3f}  f1 {gf1:.3f}")

    ep = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["val_accuracy"], marker="o", label="validation")
    ax[0].set_title("Cross-encoder — Accuracy vs Epochs"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, history["loss"], marker="o", label="train")
    ax[1].plot(ep, history["val_loss"], marker="o", label="validation")
    ax[1].set_title("Cross-encoder — Loss vs Epochs"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/history.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["not relevant", "relevant"]); ax.set_yticks([0, 1], ["not relevant", "relevant"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Cross-encoder — confusion matrix (unseen subject)")
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=130); plt.close(fig)

    model.save_pretrained(f"{OUT}/model")
    tok.save_pretrained(f"{OUT}/model")
    print(f"\nsaved to {OUT}/  ({metrics['minutes']} minutes)")


if __name__ == "__main__":
    main()
