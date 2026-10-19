"""
Train the diagram CNN: does this page contain a diagram?

A CNN trained from scratch on 450 images failed (recall 0.11 - it learned to answer "no diagram"
almost always; kept as results_cnn_scratch for comparison). With a small dataset the standard
remedy is transfer learning, so the convolutional base is MobileNetV2 pretrained on ImageNet and
only a small head is trained.

    Input 128x128x3 page image
    MobileNetV2 convolutional base (pretrained, frozen) - convolution + pooling + ReLU throughout
    GlobalAveragePooling2D -> Dense(64) ReLU -> Dropout(0.4) -> Dense(1) Sigmoid
    Adam (lr 1e-3), binary cross-entropy, batch 32, up to 40 epochs, early stopping

Pages from one course look alike, so the split is BY COURSE: test pages come from courses the
network never saw.

Writes results_cnn/: metrics.json, history.png, confusion_matrix.png, examples.png,
classification_report.txt, diagram_cnn.keras
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
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

sys.stdout.reconfigure(encoding="utf-8")
tf.keras.utils.set_random_seed(7)
IMG_DIR = "dataset/page_images"
OUT = "results_cnn"
SIZE = 128


def main():
    os.makedirs(OUT, exist_ok=True)
    index = {r["image"]: r for r in csv.DictReader(open(f"{IMG_DIR}/index.csv", encoding="utf8"))}
    labels = [r for r in csv.DictReader(open("dataset/diagram_labels.csv", encoding="utf8"))
              if r["image"] in index]
    print(f"labelled images: {len(labels)}  {dict(collections.Counter(r['label'] for r in labels))}")

    X, y, course = [], [], []
    for r in labels:
        img = Image.open(os.path.join(IMG_DIR, r["image"])).convert("RGB").resize((SIZE, SIZE))
        X.append(np.asarray(img, np.float32))
        y.append(int(r["label"]))
        course.append(index[r["image"]]["unit"])
    X = np.array(X)
    y = np.array(y, np.float32)
    course = np.array(course)
    print(f"images {X.shape} from {len(set(course))} courses")

    counts = collections.Counter(course)
    test_courses, n = [], 0
    for c, k in sorted(counts.items(), key=lambda x: -x[1]):
        if n >= 0.25 * len(X):
            break
        if y[course == c].sum() == 0 and n > 0.1 * len(X):
            continue
        test_courses.append(c)
        n += k
    te = np.isin(course, test_courses)
    Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
    print(f"train {len(Xtr)} (pos {int(ytr.sum())})   test {len(Xte)} (pos {int(yte.sum())}) "
          f"from {len(test_courses)} held-out courses")

    base = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False,
                                             weights="imagenet")
    base.trainable = False
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SIZE, SIZE, 3), name="page_image"),
        tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input,
                               name="preprocess"),
        base,
        tf.keras.layers.GlobalAveragePooling2D(name="global_pool"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"trainable parameters: {trainable:,} (MobileNetV2 base frozen)")
    with open(f"{OUT}/model_summary.txt", "w", encoding="utf8") as fh:
        model.summary(print_fn=lambda s: fh.write(s + "\n"))

    w = {0: 1.0, 1: float((ytr == 0).sum() / max(1, (ytr == 1).sum()))}
    hist = model.fit(Xtr, ytr, validation_split=0.2, epochs=40, batch_size=32, verbose=2,
                     class_weight=w,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                                 patience=10,
                                                                 restore_best_weights=True)])

    p = model.predict(Xte, verbose=0).ravel()
    # threshold chosen on the training split, never on the test split
    ptr = model.predict(Xtr, verbose=0).ravel()
    grid = np.arange(0.1, 0.91, 0.02)
    thr = float(grid[int(np.argmax([precision_recall_fscore_support(ytr, ptr >= t, average="binary",
                                                                    zero_division=0)[2]
                                    for t in grid]))])
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    cm = confusion_matrix(yte, pred)
    metrics = {"approach": "MobileNetV2 (ImageNet) frozen base + trained head",
               "threshold": thr,
               "test": {"accuracy": float((pred == yte).mean()), "precision": float(pr),
                        "recall": float(rc), "f1": float(f1),
                        "roc_auc": float(roc_auc_score(yte, p)) if len(set(yte)) > 1 else None},
               "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
               "trainable_parameters": int(trainable),
               "test_courses": list(test_courses), "epochs_run": len(hist.history["loss"]),
               "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                    "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
               "hyperparameters": {"image_size": SIZE, "base": "MobileNetV2", "dense": 64,
                                   "dropout": 0.4, "optimizer": "Adam", "learning_rate": 1e-3,
                                   "loss": "binary_crossentropy", "batch_size": 32,
                                   "max_epochs": 40}}
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    print(json.dumps(metrics["test"], indent=1))
    rep = classification_report(yte, pred, target_names=["no diagram", "has diagram"],
                                zero_division=0)
    open(f"{OUT}/classification_report.txt", "w", encoding="utf8").write(rep)
    print(rep)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="validation")
    ax[0].set_title("Diagram CNN — Accuracy vs Epochs"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="validation")
    ax[1].set_title("Diagram CNN — Loss vs Epochs"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/history.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["no diagram", "has diagram"])
    ax.set_yticks([0, 1], ["no diagram", "has diagram"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Diagram CNN — held-out courses")
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=130); plt.close(fig)

    order = np.argsort(-p)[:4].tolist() + np.argsort(p)[:4].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for axx, i in zip(axes.ravel(), order):
        axx.imshow(Xte[i].astype(np.uint8))
        axx.set_title(f"p={p[i]:.2f} · actual {'diagram' if yte[i] else 'none'}", fontsize=9)
        axx.axis("off")
    fig.suptitle("Diagram CNN — most and least confident predictions")
    fig.tight_layout(); fig.savefig(f"{OUT}/examples.png", dpi=130); plt.close(fig)

    model.save(f"{OUT}/diagram_cnn.keras")
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
