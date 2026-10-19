"""
Build candidates for depth labelling: topic + the evidence the model found + numeric features.

For every (syllabus, content) pair in the frozen benchmark, each topic is scored against every page
of its matched material with the trained evidence ANN (same guards as analyze.py). The result is
one candidate per topic, carrying:
  - the evidence pages (so a human can judge depth by reading, not searching)
  - numeric features describing that evidence, which the depth model will learn from

Output: dataset/depth_candidates.json
"""
import collections
import json
import os
import sys

import numpy as np
import tensorflow as tf

from analyze import TfidfFeaturiser, distinctive, has_terms
from depth import CUES
from syllabus_topics import content_words

sys.stdout.reconfigure(encoding="utf-8")
MODEL_DIR = "results/tfidf"
MAX_PAGES = 1200          # cap per pair so one huge textbook does not dominate


def page_features(topic, pages, prob, thr):
    """Describe the evidence for one topic as numbers the depth model can learn from."""
    keep = [i for i in np.argsort(-prob) if prob[i] >= thr and has_terms(topic, pages[i][2])][:8]
    ev = [(pages[i][0], pages[i][1], float(prob[i]), pages[i][2]) for i in keep]
    d = distinctive(topic)
    best_text = ev[0][3] if ev else ""
    low = best_text.lower()
    mentions = sum(low.count(w[:6].lower()) for w in d) if d else 0
    kinds = {k: 0 for k in CUES}
    for _, _, _, text in ev:
        for k, pat in CUES.items():
            if pat.search(text):
                kinds[k] = 1
    feats = {
        "n_evidence": len(ev),
        "n_files": len({e[0] for e in ev}),
        "max_prob": float(prob.max()),
        "mean_top3": float(np.mean(sorted(prob)[-3:])),
        "prob_spread": float(prob.max() - np.median(prob)),
        "frac_pages_above": float((prob >= thr).mean()),
        "mentions_best_page": float(min(mentions, 30)),
        "topic_words": float(len(content_words(topic))),
        "distinctive_words": float(len(d)),
        "best_page_chars": float(len(best_text)) / 1500.0,
        **{f"has_{k}": float(v) for k, v in kinds.items()},
    }
    return ev, feats


def main():
    model = tf.keras.models.load_model("models/evidence_ann.keras")
    thr = json.load(open(f"{MODEL_DIR}/metrics.json", encoding="utf8"))["threshold"]
    hm = f"{MODEL_DIR}/metrics_human.json"
    if os.path.exists(hm):
        thr = json.load(open(hm, encoding="utf8"))["threshold_human_tuned"]
    feat = TfidfFeaturiser()
    M = json.load(open("match_matrix.json", encoding="utf8"))

    pairs = {}
    for src, key in (("FINAL_PAIRS.json", "pairs"), ("cs_pairs.json", None)):
        if not os.path.exists(src):
            continue
        d = json.load(open(src, encoding="utf8"))
        d = d[key] if key else d
        for name, r in d.items():
            cid = r.get("syllabus_id") or r.get("syllabus")
            units = r.get("content_units") or r.get("units")
            pairs[f"{name}|{cid}"] = (cid, units)
    print(f"pairs: {len(pairs)}")

    cache = {}

    def unit_pages(uid):
        if uid not in cache:
            d = json.load(open(f"cache/units/{uid}.json", encoding="utf8"))
            rows = [(os.path.basename(p["file"]), f"Page {p['page']}", p["text"][:1500])
                    for p in d["pages"] if len(p["text"].strip()) > 200]
            cache[uid] = rows[:MAX_PAGES]
        return cache[uid]

    out = []
    for n, (name, (cid, units)) in enumerate(sorted(pairs.items()), 1):
        if cid not in M["syllabi"]:
            continue
        pages = [p for u in units for p in unit_pages(u)]
        topics = M["syllabi"][cid]["topics"]
        if not pages or not topics:
            continue
        for t in topics:
            prob = model.predict(feat.topic_page(t, pages), verbose=0, batch_size=512).ravel()
            ev, f = page_features(t, pages, prob, thr)
            out.append({"subject": name.split("|")[0], "syllabus": cid, "topic": t,
                        "features": f,
                        "evidence": [{"file": a, "loc": b, "prob": round(c, 3), "text": d[:700]}
                                     for a, b, c, d in ev[:5]]})
        print(f"  [{n}/{len(pairs)}] {name.split('|')[0]:<22} {len(topics):3d} topics, "
              f"{len(pages):5d} pages", flush=True)

    json.dump(out, open("dataset/depth_candidates.json", "w", encoding="utf8"), indent=1)
    dist = collections.Counter(min(c["features"]["n_evidence"], 3) for c in out)
    print(f"\ncandidates: {len(out)}  (evidence pages 0/1/2/3+: "
          f"{dist[0]}/{dist[1]}/{dist[2]}/{dist[3]})")


if __name__ == "__main__":
    main()
