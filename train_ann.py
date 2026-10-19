"""
Train the evidence ANN:  (syllabus topic, page of material) -> relevant / not relevant

Architecture (frozen embeddings in, dense network trained):
    Input 1154 -> Dense 256 ReLU -> Dropout 0.3 -> Dense 64 ReLU -> Dropout 0.2 -> Dense 1 sigmoid
    Adam (lr 1e-3), binary cross-entropy, batch 64, up to 40 epochs, early stopping on val loss

Test and validation sets are entirely unseen SUBJECTS, not random rows, so the reported numbers
say whether the model generalises to a new subject.

Writes to results/: history.png, confusion_matrix.png, roc_curve.png, metrics.json,
classification_report.txt, predictions_test.csv, evidence_ann.keras
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

sys.stdout.reconfigure(encoding="utf-8")
tf.keras.utils.set_random_seed(7)
EPOCHS, BATCH, LR, DROPOUT = 40, 64, 1e-3, (0.3, 0.2)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
ap.add_argument("--drop-sim", action="store_true",
                help="ablation: remove the 3 hand-crafted similarity features (cosine, word "
                     "overlap, length) so only the learned text representation remains")
ap.add_argument("--features", default="tfidf", choices=["tfidf", "embed"],
                help="which feature set to train on: tfidf_<split>.npz from features_tfidf.py, "
                     "or embed_<split>.npz from embed_features.py")
ARGS = ap.parse_args()
OUT = ARGS.out
os.makedirs(OUT, exist_ok=True)


def load(split):
    d = np.load(f"dataset/{ARGS.features}_{split}.npz", allow_pickle=True)
    X = d["X"][:, :-3] if ARGS.drop_sim else d["X"]
    return X, d["y"].astype(np.float32), d["domain"], d["kind"]


def build(dim):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(dim,), name="features"),
        tf.keras.layers.Dense(256, activation="relu", name="hidden_1"),
        tf.keras.layers.Dropout(DROPOUT[0]),
        tf.keras.layers.Dense(64, activation="relu", name="hidden_2"),
        tf.keras.layers.Dropout(DROPOUT[1]),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return m


def main():
    Xtr, ytr, _, _ = load("train")
    Xva, yva, _, _ = load("val")
    Xte, yte, dte, kte = load("test")
    print(f"train {Xtr.shape} val {Xva.shape} test {Xte.shape}")

    model = build(Xtr.shape[1])
    model.summary()
    with open(f"{OUT}/model_summary.txt", "w", encoding="utf8") as fh:
        model.summary(print_fn=lambda s: fh.write(s + "\n"))

    # positives are 1/3 of the data; weight the classes so the model does not favour "not relevant"
    w = {0: 1.0, 1: float((ytr == 0).sum() / max(1, (ytr == 1).sum()))}
    print("class weights:", w)

    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
                     class_weight=w, verbose=2,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                                                 restore_best_weights=True)])

    # threshold chosen on VALIDATION (never on test)
    pv = model.predict(Xva, verbose=0).ravel()
    grid = np.arange(0.05, 0.96, 0.01)
    f1s = [precision_recall_fscore_support(yva, pv >= t, average="binary", zero_division=0)[2] for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    print(f"threshold from validation: {thr:.2f} (val F1 {max(f1s):.3f})")

    pt = model.predict(Xte, verbose=0).ravel()
    pred = (pt >= thr).astype(int)
    cm = confusion_matrix(yte, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    acc = float((pred == yte).mean())
    auc = float(roc_auc_score(yte, pt))

    metrics = {
        "threshold": thr, "test": {"accuracy": acc, "precision": float(pr), "recall": float(rc),
                                   "f1": float(f1), "roc_auc": auc},
        "train_final": {"accuracy": float(hist.history["accuracy"][-1]),
                        "loss": float(hist.history["loss"][-1])},
        "val_final": {"accuracy": float(hist.history["val_accuracy"][-1]),
                      "loss": float(hist.history["val_loss"][-1])},
        "epochs_run": len(hist.history["loss"]),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "hyperparameters": {"optimizer": "Adam", "learning_rate": LR, "loss": "binary_crossentropy",
                            "batch_size": BATCH, "max_epochs": EPOCHS, "dropout": DROPOUT,
                            "hidden_units": [256, 64], "input_dim": int(Xtr.shape[1]), "drop_sim_ablation": bool(ARGS.drop_sim)},
        "per_domain": {}, "per_kind": {},
    }
    for d in sorted(set(dte)):
        m = dte == d
        metrics["per_domain"][str(d)] = {
            "n": int(m.sum()), "accuracy": float((pred[m] == yte[m]).mean()),
            "f1": float(precision_recall_fscore_support(yte[m], pred[m], average="binary",
                                                        zero_division=0)[2])}
    for k in sorted(set(kte)):
        m = kte == k
        metrics["per_kind"][str(k)] = {"n": int(m.sum()), "accuracy": float((pred[m] == yte[m]).mean())}
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    print(json.dumps(metrics["test"], indent=1))
    print("per domain:", json.dumps(metrics["per_domain"], indent=1))
    print("per kind:", json.dumps(metrics["per_kind"], indent=1))

    with open(f"{OUT}/classification_report.txt", "w", encoding="utf8") as fh:
        fh.write(classification_report(yte, pred, target_names=["not relevant", "relevant"],
                                       zero_division=0))

    # graphs the rubric asks for
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy vs Epochs"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="validation")
    ax[1].set_title("Loss vs Epochs"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/history.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["not relevant", "relevant"]); ax.set_yticks([0, 1], ["not relevant", "relevant"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion Matrix (unseen subjects)")
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=130); plt.close(fig)

    fpr, tpr, _ = roc_curve(yte, pt)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/roc_curve.png", dpi=130); plt.close(fig)

    import csv
    rows = list(csv.DictReader(open("dataset/pairs_test.csv", encoding="utf8")))
    with open(f"{OUT}/predictions_test.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "topic", "file", "page", "kind", "actual", "probability", "predicted"])
        for r, p, q in zip(rows, pt, pred):
            w.writerow([r["domain"], r["topic"], r["file"].split("/")[-1], r["page"], r["kind"],
                        r["label"], f"{p:.4f}", q])
    model.save(f"{OUT}/evidence_ann.keras")
    print(f"\nsaved model + graphs + metrics to {OUT}/")


if __name__ == "__main__":
    main()
