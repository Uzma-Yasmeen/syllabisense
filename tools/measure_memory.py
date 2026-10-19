"""
How much memory does the application actually need?

Streamlit Community Cloud gives a free app 1 GB. This walks the same path the app walks - import,
load both models, run one analysis over the bundled example - and reports resident memory after
each step, so the answer is measured rather than guessed.

    python measure_memory.py
"""
import os
import sys

import psutil

sys.stdout.reconfigure(encoding="utf-8")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
proc = psutil.Process()
LIMIT_MB = 1024
steps = []


def mark(label):
    mb = proc.memory_info().rss / 1048576
    delta = mb - steps[-1][1] if steps else mb
    steps.append((label, mb, delta))
    print(f"  {label:<44}{mb:>8.0f} MB   ({delta:+.0f})")


print(f"measuring against a {LIMIT_MB} MB limit\n")
mark("bare python")

import numpy as np                                                    # noqa: E402
mark("numpy")

import tensorflow as tf                                               # noqa: E402
mark("tensorflow imported")

sys.path.insert(0, ".")
from analyze import TfidfFeaturiser, has_terms                        # noqa: E402
from depth import judge                                               # noqa: E402
from loaders import load                                              # noqa: E402
from syllabus_topics import content_words, topics_with_units          # noqa: E402
import diagrams, report                                               # noqa: E402,E401
mark("app modules")

feat = TfidfFeaturiser()
mark("TF-IDF vectoriser + SVD (21 MB on disk)")

model = tf.keras.models.load_model("results_tfidf/evidence_ann.keras")
mark("evidence ANN")

files = sorted(os.path.join("demo_files", f) for f in os.listdir("demo_files"))
syl = next(f for f in files if "syllabus" in os.path.basename(f).lower())
mats = [f for f in files if f != syl]
topics = [t for _, t in topics_with_units(syl) if content_words(t)]
blocks = load(mats)
pages = [(b["source"], b["location"], b["text"]) for b in blocks]
paths = {b["source"]: b["path"] for b in blocks}
mark(f"material read ({len(pages)} pages, {len(topics)} topics)")

import json                                                           # noqa: E402
thr = json.load(open("results_tfidf/metrics_human.json"))["threshold_human_tuned"]
results = []
for t in topics:
    p = model.predict(feat.topic_page(t, pages), verbose=0).ravel()
    order = np.argsort(-p)
    above = [i for i in order if p[i] >= thr and has_terms(t, pages[i][2])][:8]
    ev = [(pages[i][0], pages[i][1], float(p[i]), pages[i][2]) for i in above]
    lvl, reasons, kinds = judge(ev)
    results.append({"topic": t, "unit": "Course", "order": len(results), "level": lvl,
                    "reasons": reasons, "kinds": kinds, "evidence": ev,
                    "closest": (pages[order[0]][0], pages[order[0]][1], float(p[order[0]]))})
mark("scored every topic against every page")

found = diagrams.detect({(paths.get(f), l) for r in results for f, l, *_ in r["evidence"]})
for r in results:
    r["diagram"], r["diagram_pages"] = diagrams.summarise(
        [(paths.get(f), l) for f, l, *_ in r["evidence"]], found)
mark("diagram CNN loaded and run")

pdf = report.to_pdf(results, {"syllabus": os.path.basename(syl),
                              "material": [os.path.basename(m) for m in mats]})
mark(f"report PDF built ({len(pdf) // 1024} KB)")

peak = max(s[1] for s in steps)
print(f"\n  PEAK: {peak:.0f} MB of a {LIMIT_MB} MB limit  "
      f"({peak / LIMIT_MB * 100:.0f}% used)")
print(f"  headroom: {LIMIT_MB - peak:.0f} MB")
print("\n  note: Streamlit itself adds roughly 80-120 MB on top of this,")
print("  and a second simultaneous visitor does not double it (the models are cached).")
if peak + 120 > LIMIT_MB:
    print("\n  -> TOO TIGHT. Dropping TensorFlow for the ANN would remove most of it.")
else:
    print(f"\n  -> fits, with about {LIMIT_MB - peak - 120:.0f} MB spare after Streamlit.")
