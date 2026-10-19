"""
Train the diagram CNN: does this page contain a diagram?

A page image goes in, and the network says whether it contains a figure, chart, circuit,
flowchart or labelled drawing. This lets the report say "explained in text, but no diagram
anywhere in your material" - something the text models cannot know.

    Input 128x128x1 greyscale page image
    Conv2D(32,3x3) ReLU -> MaxPool(2x2)
    Conv2D(64,3x3) ReLU -> MaxPool(2x2)
    Conv2D(128,3x3) ReLU -> MaxPool(2x2)
    Flatten -> Dense(128) ReLU -> Dropout(0.5) -> Dense(1) Sigmoid
    Adam (lr 1e-3), binary cross-entropy, batch 32, up to 40 epochs, early stopping

Pages from the same course look alike, so the split is BY COURSE: the test pages come from
courses the network never trained on.

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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

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
    print(f"labelled images: {len(labels)}  "
          f"{dict(collections.Counter(r['label'] for r in labels))}")

    X, y, course = [], [], []
    for r in labels:
        img = Image.open(os.path.join(IMG_DIR, r["image"])).convert("L").resize((SIZE, SIZE))
        X.append(np.asarray(img, np.float32) / 255.0)
        y.append(int(r["label"]))
        course.append(index[r["image"]]["unit"])
    X = np.array(X)[..., None]
    y = np.array(y, np.float32)
    course = np.array(course)
    print(f"images {X.shape} from {len(set(course))} courses")

    # hold out whole courses, so the test pages come from material never seen in training
    counts = collections.Counter(course)
    test_courses, n = [], 0
    for c, k in sorted(counts.items(), key=lambda x: -x[1]):
        if n >= 0.25 * len(X):
            break
        pos = int(y[course == c].sum())
        if pos == 0 and n > 0.1 * len(X):      # keep some positives in the test set
            continue
        test_courses.append(c)
        n += k
    te = np.isin(course, test_courses)
    Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
    print(f"train {len(Xtr)} (pos {int(ytr.sum())})   test {len(Xte)} (pos {int(yte.sum())}) "
          f"from {len(test_courses)} held-out courses")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SIZE, SIZE, 1), name="page_image"),
        tf.keras.layers.RandomTranslation(0.05, 0.05),      # pages are scanned/cropped differently
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv_1"),
        tf.keras.layers.MaxPooling2D(2, name="pool_1"),
        tf.keras.layers.Conv2D(64, 3, activation="relu", name="conv_2"),
        tf.keras.layers.MaxPooling2D(2, name="pool_2"),
        tf.keras.layers.Conv2D(128, 3, activation="relu", name="conv_3"),
        tf.keras.layers.MaxPooling2D(2, name="pool_3"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    model.summary()
    with open(f"{OUT}/model_summary.txt", "w", encoding="utf8") as fh:
        model.summary(print_fn=lambda s: fh.write(s + "\n"))

    w = {0: 1.0, 1: float((ytr == 0).sum() / max(1, (ytr == 1).sum()))}
    print("class weights:", w)
    hist = model.fit(Xtr, ytr, validation_split=0.2, epochs=40, batch_size=32, verbose=2,
                     class_weight=w,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                                 restore_best_weights=True)])

    p = model.predict(Xte, verbose=0).ravel()
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    cm = confusion_matrix(yte, pred)
    metrics = {"test": {"accuracy": float((pred == yte).mean()), "precision": float(pr),
                        "recall": float(rc), "f1": float(f1),
                        "roc_auc": float(roc_auc_score(yte, p)) if len(set(yte)) > 1 else None},
               "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
               "test_courses": list(test_courses), "epochs_run": len(hist.history["loss"]),
               "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                    "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
               "hyperparameters": {"image_size": SIZE, "filters": [32, 64, 128], "kernel": "3x3",
                                   "pool": "2x2", "dense": 128, "dropout": 0.5, "optimizer": "Adam",
                                   "learning_rate": 1e-3, "loss": "binary_crossentropy",
                                   "batch_size": 32, "max_epochs": 40}}
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

    # a page of predictions, so the result can be seen rather than only read
    order = np.argsort(-p)[:4].tolist() + np.argsort(p)[:4].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for axx, i in zip(axes.ravel(), order):
        axx.imshow(Xte[i, :, :, 0], cmap="gray")
        axx.set_title(f"p={p[i]:.2f} · actual {'diagram' if yte[i] else 'none'}", fontsize=9)
        axx.axis("off")
    fig.suptitle("Diagram CNN — most and least confident predictions")
    fig.tight_layout(); fig.savefig(f"{OUT}/examples.png", dpi=130); plt.close(fig)

    model.save(f"{OUT}/diagram_cnn.keras")
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
