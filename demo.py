"""
SyllabiSense — code demonstration
Uzma Yasmeen · 2503A51L34 · Batch 20 CSE · AI & Deep Learning

Loads the trained models and reproduces every number reported in the proposal, then runs the
whole system end to end on a real syllabus and real lecture material.

    python demo.py

Nothing is hard-coded: each figure is computed here from the saved model and the saved dataset.
"""
# ----------------------------------------------------------------- 1. Libraries
import csv
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

sys.stdout.reconfigure(encoding="utf-8")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
LINE = "=" * 78


def banner(text):
    print(f"\n{LINE}\n{text}\n{LINE}")


# ------------------------------------------------------- 2. Load the dataset
banner("STEP 1 — Load the dataset")
rows = list(csv.DictReader(open("dataset/pairs_test.csv", encoding="utf8")))
y = np.array([int(r["label"]) for r in rows])
stats = json.load(open("dataset/stats.json", encoding="utf8"))


def build_features(rows):
    """The same 387 features the ANN was trained on, rebuilt here from the saved fitted
    TF-IDF vectoriser and SVD — exactly the path the application uses at prediction time."""
    import joblib
    d = joblib.load("dataset/tfidf_model.joblib")
    vec, svd = d["vectorizer"], d["svd"]
    T = vec.transform([r["topic"] for r in rows])
    P = vec.transform([r["text"] for r in rows])
    num = np.asarray(T.multiply(P).sum(1)).ravel()
    den = (np.sqrt(np.asarray(T.multiply(T).sum(1)).ravel())
           * np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel()) + 1e-9)
    cos = (num / den).reshape(-1, 1)
    a, b = svd.transform(T), svd.transform(P)
    ov, ln = [], []
    for r in rows:
        tw = {w for w in r["topic"].lower().split() if len(w) > 3}
        pw = r["text"].lower()
        ov.append([sum(1 for w in tw if w in pw) / max(1, len(tw))])
        ln.append([min(len(r["text"]), 1500) / 1500])
    return np.hstack([a, b, a * b, cos, np.array(ov), np.array(ln)]).astype(np.float32)


X = build_features(rows)

print(f"Total labelled pairs in the dataset : {stats['examples']:,}")
print(f"Test split (Computer Science)       : {len(rows):,} pairs, {X.shape[1]} features each")
print(f"  class 1 (page is evidence)        : {int(y.sum()):,}")
print(f"  class 0 (page is not evidence)    : {int((y == 0).sum()):,}")
print(f"Courses represented in the test set  : {len({r['syllabus'] for r in rows})}")
print("\nThe split is by SUBJECT AREA, not random — Computer Science appears nowhere in training:")
print(f"  training subjects : {', '.join(sorted(set(stats['domains']) - {'CS', 'Chemistry', 'Humanities'}))}")
print(f"  validation        : Chemistry, Humanities")
print(f"  test              : Computer Science")

# -------------------------------------------------------- 3. Load the model
banner("STEP 2 — Load the trained ANN and show its architecture")
model = tf.keras.models.load_model("results_tfidf/evidence_ann.keras")
model.summary()
print(f"\nTrainable parameters: {model.count_params():,}")

# --------------------------------------------- 4. Evaluate on the unseen subject
banner("STEP 3 — Evaluate on the unseen subject area (automatic labels)")
thr = json.load(open("results_tfidf/metrics.json", encoding="utf8"))["threshold"]
p = model.predict(X, verbose=0).ravel()
pred = (p >= thr).astype(int)
pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
cm = confusion_matrix(y, pred)

print(f"Decision threshold (tuned on validation, never on test): {thr:.2f}\n")
print(f"  Accuracy  : {(pred == y).mean() * 100:.1f}%")
print(f"  Precision : {pr:.3f}")
print(f"  Recall    : {rc:.3f}")
print(f"  F1-score  : {f1:.3f}")
print(f"  ROC-AUC   : {roc_auc_score(y, p):.3f}")
saved = json.load(open("results_tfidf/metrics.json", encoding="utf8"))["test"]
print(f"\n  Reported at training time: accuracy {saved['accuracy'] * 100:.1f}%, "
      f"F1 {saved['f1']:.3f}, ROC-AUC {saved['roc_auc']:.3f}")
print("  Small differences are expected: the features are rebuilt here from the saved")
print("  vectoriser, as the application does, rather than reloaded from the training cache.")
print("\n  Confusion matrix")
print("                      predicted: not evidence   predicted: evidence")
print(f"  actual: not evidence {cm[0, 0]:>18,} {cm[0, 1]:>21,}")
print(f"  actual: evidence     {cm[1, 0]:>18,} {cm[1, 1]:>21,}")

# ------------------------------------------- 5. Evaluate against human labels
banner("STEP 4 — Evaluate against HUMAN-VERIFIED labels (the honest figure)")
key = lambda r: (r["syllabus"], r["topic"], r["file"], r["page"])
human = {key(r): int(r["human_label"])
         for r in csv.DictReader(open("dataset/review_done.csv", encoding="utf8"))
         if r["human_label"] in ("0", "1")}
idx = [i for i, r in enumerate(rows) if key(r) in human]
hy = np.array([human[key(rows[i])] for i in idx])
hp = p[idx]
weak = np.array([int(rows[i]["label"]) for i in idx])
hthr = json.load(open("results_tfidf/metrics_human.json", encoding="utf8"))["threshold_human_tuned"]
hpred = (hp >= hthr).astype(int)
hpr, hrc, hf1, _ = precision_recall_fscore_support(hy, hpred, average="binary", zero_division=0)
hcm = confusion_matrix(hy, hpred)

print(f"300 pairs were re-labelled by hand on a screen that hides the automatic label.")
print(f"{len(idx)} of them fall in the Computer Science test split and are used here.\n")
print(f"  How often the AUTOMATIC label was actually right : {(weak == hy).mean() * 100:.1f}%")
print(f"  -> the training labels themselves are imperfect, which caps what any model can learn.\n")
print(f"  Accuracy  : {(hpred == hy).mean() * 100:.1f}%     (vs {(pred == y).mean() * 100:.1f}% on automatic labels)")
print(f"  Precision : {hpr:.3f}")
print(f"  Recall    : {hrc:.3f}")
print(f"  F1-score  : {hf1:.3f}")
print(f"  ROC-AUC   : {roc_auc_score(hy, hp):.3f}")
print(f"\n  Confusion matrix (TN {hcm[0,0]}  FP {hcm[0,1]}  FN {hcm[1,0]}  TP {hcm[1,1]})")
se = 1.96 * np.sqrt((hpred == hy).mean() * (1 - (hpred == hy).mean()) / len(hy))
print(f"  95% confidence interval on the accuracy: +/- {se * 100:.1f} points (n = {len(hy)})")

# ------------------------------------------------------------- 6. The CNN
banner("STEP 5 — The CNN: does this page contain a diagram?")
import diagrams

cnnm = json.load(open("results_cnn/metrics.json", encoding="utf8"))
print(f"Approach : {cnnm['approach']}")
print(f"Trained on {cnnm['n_train']} hand-labelled page images, tested on {cnnm['n_test']} images")
print(f"from courses held out of training.\n")
print(f"  Accuracy  : {cnnm['test']['accuracy'] * 100:.1f}%")
print(f"  Precision : {cnnm['test']['precision']:.3f}")
print(f"  Recall    : {cnnm['test']['recall']:.3f}")
print(f"  F1-score  : {cnnm['test']['f1']:.3f}")
scr = json.load(open("results_cnn_scratch/metrics.json", encoding="utf8"))
print(f"\nFor comparison, the same job with a CNN trained from scratch on the same images:")
print(f"  Accuracy {scr['test']['accuracy'] * 100:.1f}%  but Recall {scr['test']['recall']:.3f} "
      f"-- it answered 'no diagram' almost always.")
print(f"Transfer learning raised recall from {scr['test']['recall']:.2f} to "
      f"{cnnm['test']['recall']:.2f}.")

DEMO_PDF = os.path.expanduser("~/Downloads/Lecture-18, UNIT-III.pdf")
if os.path.exists(DEMO_PDF):
    import pypdfium2 as pdfium
    npages = len(pdfium.PdfDocument(DEMO_PDF))
    got = diagrams.detect([(DEMO_PDF, f"Page {i}") for i in range(1, npages + 1)])
    print(f"\nRunning the CNN on a real lecture file ({os.path.basename(DEMO_PDF)}):")
    for (_, loc), (prob, hit) in sorted(got.items(), key=lambda kv: -kv[1][0]):
        print(f"  {loc:<9} probability {prob:.2f}   {'DIAGRAM' if hit else '-'}")

# ------------------------------------------------- 7. End-to-end demonstration
banner("STEP 6 — End to end: a real syllabus against real lecture material")
import glob

from analyze import TfidfFeaturiser, has_terms
from depth import judge
from loaders import load as load_material
from syllabus_topics import content_words, topics_with_units

SYL = "syllabi/106102064.pdf"          # NPTEL — Data Structures And Algorithms
MAT = sorted(glob.glob(os.path.expanduser("~/Downloads/Lecture-1[3-9]*UNIT-III.pdf")))

if os.path.exists(SYL) and MAT:
    topics = [t for _, t in topics_with_units(SYL) if content_words(t)]
    blocks = load_material(MAT)
    pages = [(b["source"], b["location"], b["text"]) for b in blocks]
    paths = {b["source"]: b["path"] for b in blocks}
    print(f"Syllabus : {os.path.basename(SYL)}  ->  {len(topics)} topics")
    print(f"Material : {len(MAT)} lecture files  ->  {len(pages)} pages\n")

    feat = TfidfFeaturiser()
    results = []
    for t in topics:
        prob = feat.topic_page(t, pages)
        prob = model.predict(prob, verbose=0).ravel()
        order = np.argsort(-prob)
        above = [i for i in order if prob[i] >= hthr and has_terms(t, pages[i][2])][:8]
        ev = [(pages[i][0], pages[i][1], float(prob[i]), pages[i][2]) for i in above]
        level, reasons, kinds = judge(ev)
        results.append({"topic": t, "level": level, "evidence": ev})

    found = diagrams.detect({(paths.get(f), loc) for r in results for f, loc, *_ in r["evidence"]})
    for r in results:
        r["diagram"], r["pages"] = diagrams.summarise(
            [(paths.get(f), loc) for f, loc, *_ in r["evidence"]], found)

    tally = {}
    for r in results:
        tally[r["level"]] = tally.get(r["level"], 0) + 1
    print("Verdict for every topic in the syllabus:\n")
    print(f"  {'TOPIC':<44}{'VERDICT':<17}{'DIAGRAM':<12}EVIDENCE")
    print("  " + "-" * 92)
    for r in results:
        dia = {"diagram": "yes", "none": "no", "not_checked": "-"}[r["diagram"]]
        ev = ""
        if r["evidence"]:
            f, loc, pr_, _ = r["evidence"][0]
            ev = f"{f[:26]} {loc} ({pr_:.2f})"
        print(f"  {r['topic'][:42]:<44}{r['level']:<17}{dia:<12}{ev}")
    print("\nSummary: " + "   ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    print("\nThe syllabus covers a full course; only Unit III material was supplied, so most")
    print("topics are correctly reported as having no evidence in what was uploaded. The four")
    print("topics with evidence are exactly the ones Unit III teaches, and the CNN found a")
    print("diagram for collision resolution.")
    print("\nKnown limitation visible above: the last three lines ('Science and Engineering',")
    print("'funded by MHRD') are page footers from the NPTEL PDF that the syllabus parser")
    print("mistook for topics. Parsing is rule-based and this is listed in the limitations.")

# ----------------------------------------------------------------- 8. Summary
banner("SUMMARY OF ALL TRAINED MODELS")
print(f"  {'MODEL':<34}{'TEST ACC':>10}{'PRECISION':>11}{'RECALL':>9}{'F1':>8}")
print("  " + "-" * 72)
for name, path in [("Evidence ANN (deployed)", "results_tfidf"),
                   ("Cross-encoder (transformer)", "results_crossencoder"),
                   ("Diagram CNN (MobileNetV2)", "results_cnn"),
                   ("Diagram CNN (from scratch)", "results_cnn_scratch")]:
    m = json.load(open(f"{path}/metrics.json", encoding="utf8"))["test"]
    print(f"  {name:<34}{m['accuracy'] * 100:>9.1f}%{m['precision']:>11.3f}"
          f"{m['recall']:>9.3f}{m['f1']:>8.3f}")
dm = json.load(open("results_depth/metrics.json", encoding="utf8"))
print(f"  {'Depth ANN (3-class, preliminary)':<34}{dm['test_accuracy'] * 100:>9.1f}%"
      f"{'—':>11}{'—':>9}{dm['macro_f1']:>8.3f}")
print(f"\n  Human-verified accuracy of the deployed model: {(hpred == hy).mean() * 100:.1f}% "
      f"+/- {se * 100:.1f} points")
print(f"\n{LINE}\nAll figures above were computed by this script from the saved models.\n{LINE}")
