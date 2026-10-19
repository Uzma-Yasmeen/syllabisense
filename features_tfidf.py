"""
Offline feature builder (no model download needed).

TF-IDF over words, reduced to 128 dimensions with LSA (truncated SVD), which gives a compact
semantic-ish representation of each text. Used if the pretrained embedding model cannot be
downloaded; embed_features.py is the preferred path.

Features per example (387 numbers):
  topic LSA vector            128
  page LSA vector             128
  topic * page (elementwise)  128
  tf-idf cosine similarity      1
  word overlap fraction         1
  page length (scaled)          1
Fitted on the TRAINING split only, then applied to val/test, so no information leaks from the
held-out subjects.
Output: dataset/tfidf_<split>.npz  (same format as embed_features.py writes to embed_*)
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding="utf-8")
DIMS = 128
MODEL_PATH = "dataset/tfidf_model.joblib"
# --refit forces a fresh fit; by default an existing fitted model is reused so the feature files
# stay in step with the ANN that was trained on them
REUSE = "--refit" not in sys.argv


def main():
    splits = {s: list(csv.DictReader(open(f"dataset/pairs_{s}.csv", encoding="utf8")))
              for s in ("train", "val", "test")}

    if REUSE and os.path.exists(MODEL_PATH):
        # Rebuild the feature files WITHOUT refitting. The trained ANN and the app's threshold
        # belong to one particular fitted vectoriser; refitting on a dataset that has since been
        # regenerated produces a slightly different projection and silently invalidates them.
        import joblib
        d = joblib.load(MODEL_PATH)
        vec, svd = d["vectorizer"], d["svd"]
        print(f"reusing the fitted model in {MODEL_PATH}: vocabulary {len(vec.vocabulary_)}, "
              f"LSA dims {svd.n_components}")
    else:
        train_text = [r["topic"] for r in splits["train"]] + [r["text"] for r in splits["train"]]
        vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=3, max_df=0.5,
                              sublinear_tf=True, max_features=60000)
        vec.fit(train_text)
        svd = TruncatedSVD(n_components=DIMS, random_state=7)
        svd.fit(vec.transform(train_text))
        print(f"vocabulary {len(vec.vocabulary_)}  LSA dims {DIMS} "
              f"(explains {svd.explained_variance_ratio_.sum():.1%} of variance)")

    for name, rows in splits.items():
        T = vec.transform([r["topic"] for r in rows])
        P = vec.transform([r["text"] for r in rows])
        cos = np.asarray(T.multiply(P).sum(1)) / (
            np.sqrt(np.asarray(T.multiply(T).sum(1))) * np.sqrt(np.asarray(P.multiply(P).sum(1))) + 1e-9)
        a, b = svd.transform(T), svd.transform(P)
        ov, ln = [], []
        for r in rows:
            tw = {w for w in r["topic"].lower().split() if len(w) > 3}
            pw = r["text"].lower()
            ov.append([sum(1 for w in tw if w in pw) / max(1, len(tw))])
            ln.append([min(len(r["text"]), 1500) / 1500])
        X = np.hstack([a, b, a * b, cos, np.array(ov), np.array(ln)]).astype(np.float32)
        y = np.array([int(r["label"]) for r in rows], np.int8)
        np.savez_compressed(f"dataset/tfidf_{name}.npz", X=X, y=y,
                            domain=np.array([r["domain"] for r in rows]),
                            kind=np.array([r["kind"] for r in rows]))
        print(f"  {name}: X={X.shape} positives={int(y.sum())}/{len(y)}")
    json.dump({"model": "tfidf+LSA", "dims": int(X.shape[1]), "lsa_components": DIMS},
              open("dataset/tfidf_meta.json", "w"), indent=1)
    # save the fitted vectoriser so analyze.py / the app can build the same features for new files
    import joblib
    joblib.dump({"vectorizer": vec, "svd": svd}, "dataset/tfidf_model.joblib")
    print("  saved dataset/tfidf_model.joblib")


if __name__ == "__main__":
    main()
