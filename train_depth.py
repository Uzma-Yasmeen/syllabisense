"""
Train the DEPTH model: how well is a topic taught — properly / barely / not taught.

Input features describe the evidence the first model found (how many pages, how confident, how
often the topic's own words appear, which kinds of evidence are present). Labels come from
dataset/depth_labels.csv, which a human filled in by reading the evidence.

Architecture:
    Input (18) -> Dense 64 ReLU -> Dropout 0.3 -> Dense 32 ReLU -> Dropout 0.2 -> Dense 3 Softmax
    Adam (lr 1e-3), sparse categorical cross-entropy, batch 16, up to 120 epochs, early stopping

Held-out SUBJECTS form the test set, so the score says whether it generalises to a new subject.
Writes results_depth/: metrics.json, confusion_matrix.png, history.png, classification_report.txt,
depth_ann.keras, feature_names.json
"""
import collections
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.stdout.reconfigure(encoding="utf-8")
tf.keras.utils.set_random_seed(7)
OUT = "results_depth"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["NOT", "BARELY", "PROPERLY"]          # ordered by increasing depth
MIN_TEST_SUBJECTS = 2


def main():
    cands = {(c["syllabus"], c["topic"]): c
             for c in json.load(open("dataset/depth_candidates.json", encoding="utf8"))}
    rows = [r for r in csv.DictReader(open("dataset/depth_labels.csv", encoding="utf8"))
            if r["label"] in CLASSES]
    print(f"labelled: {len(rows)}   distribution: {dict(collections.Counter(r['label'] for r in rows))}")
    if len(rows) < 60:
        sys.exit("need at least ~60 labels before training is meaningful")

    names = sorted(next(iter(cands.values()))["features"])
    X, y, subj = [], [], []
    for r in rows:
        c = cands.get((r["syllabus"], r["topic"]))
        if not c:
            continue
        X.append([c["features"][n] for n in names])
        y.append(CLASSES.index(r["label"]))
        subj.append(r["subject"])
    X, y, subj = np.array(X, np.float32), np.array(y), np.array(subj)
    print(f"usable: {len(X)} examples, {X.shape[1]} features, {len(set(subj))} subjects")

    # hold out whole subjects for testing
    counts = collections.Counter(subj)
    order = [s for s, _ in counts.most_common()][::-1]
    test_subjects, n = [], 0
    for s in order:
        if n >= max(len(X) * 0.25, 25) and len(test_subjects) >= MIN_TEST_SUBJECTS:
            break
        test_subjects.append(s)
        n += counts[s]
    te = np.isin(subj, test_subjects)
    Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
    print(f"train {len(Xtr)} / test {len(Xte)} (test subjects: {', '.join(test_subjects)})")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],), name="evidence_features"),
        tf.keras.layers.Dense(64, activation="relu", name="hidden_1"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu", name="hidden_2"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="depth"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    with open(f"{OUT}/model_summary.txt", "w", encoding="utf8") as fh:
        model.summary(print_fn=lambda s: fh.write(s + "\n"))

    freq = collections.Counter(ytr.tolist())
    weights = {c: len(ytr) / (len(CLASSES) * max(1, freq[c])) for c in range(len(CLASSES))}
    hist = model.fit(Xtr, ytr, validation_split=0.2, epochs=120, batch_size=16, verbose=2,
                     class_weight=weights,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                                 restore_best_weights=True)])

    pred = model.predict(Xte, verbose=0).argmax(1)
    acc = float((pred == yte).mean())
    cm = confusion_matrix(yte, pred, labels=range(len(CLASSES)))
    metrics = {"test_accuracy": acc,
               "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
               "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
               "test_subjects": test_subjects, "classes": CLASSES,
               "confusion_matrix": cm.tolist(), "features": names,
               "epochs_run": len(hist.history["loss"]),
               "hyperparameters": {"optimizer": "Adam", "learning_rate": 1e-3, "batch_size": 16,
                                   "loss": "sparse_categorical_crossentropy",
                                   "hidden_units": [64, 32], "dropout": [0.3, 0.2]}}
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    json.dump({"features": names, "mean": mu.tolist(), "std": sd.tolist(), "classes": CLASSES},
              open(f"{OUT}/feature_names.json", "w"), indent=1)
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "macro_f1", "n_train", "n_test")}, indent=1))
    rep = classification_report(yte, pred, labels=range(len(CLASSES)), target_names=CLASSES,
                                zero_division=0)
    open(f"{OUT}/classification_report.txt", "w", encoding="utf8").write(rep)
    print(rep)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="validation")
    ax[0].set_title("Depth model — Accuracy vs Epochs"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="validation")
    ax[1].set_title("Depth model — Loss vs Epochs"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/history.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual (human)")
    ax.set_title("Depth model — unseen subjects")
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=130); plt.close(fig)
    model.save(f"{OUT}/depth_ann.keras")
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
