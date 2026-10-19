"""
Build the labelled dataset for the evidence model:
    (syllabus topic, page of course material) -> relevant / not relevant

Labels (weak supervision, stated plainly in the proposal):
  POSITIVE      page matched the topic by the verified matcher (idf-weighted terms within a
                30-word window, navigation/reference/scanned pages excluded)
  HARD NEGATIVE another page from the SAME course that does not match the topic
  EASY NEGATIVE a page from a course in an unrelated subject group
A page is only used as a negative if fewer than half the topic's words appear in it, so true
evidence is not mislabelled as negative.

Splits are by DOMAIN so the test set contains subjects never seen in training.
Outputs (dataset/):
  pairs_train.csv pairs_val.csv pairs_test.csv   weakly-labelled
  review_sample.csv                              stratified sample for human verification
  stats.json
"""
import collections
import glob
import json
import os
import random
import re

import numpy as np
import sys

from match import is_garbage, is_navigation, strip_references
from syllabus_topics import content_words

sys.stdout.reconfigure(encoding="utf-8")
random.seed(7)
OUT = "dataset"
os.makedirs(OUT, exist_ok=True)
PAGE_CHARS = 1500          # how much of a page the model sees
# CS is held out as the TEST subject: the model never trains on it, and the student (a CS
# undergraduate) can verify CS labels confidently, which non-CS subjects would not allow.
TEST_DOMAINS = {"CS"}
VAL_DOMAINS = {"Chemistry", "Humanities"}

M = json.load(open("match_matrix.json", encoding="utf8"))
H = json.load(open("match_hits.json", encoding="utf8"))
UNITS = M["units"]
SINGLE_DF = json.load(open("cache/single_word_df.json", encoding="utf8"))


def generic(t):
    w = set(content_words(t))
    return len(w) == 1 and SINGLE_DF.get(next(iter(w)), 0) > 0.01


# best valid pair per syllabus
best = {}
for r in json.load(open("valid_pairs.json", encoding="utf8")):
    k = r["syllabus"]
    if k not in best or r["score"] > best[k]["score"]:
        best[k] = r
print(f"syllabi with a valid pair: {len(best)}")

# page text per unit, excluding pages that are never evidence
print("loading pages ...", flush=True)
pages = {}
for f in glob.glob("cache/units/*.json"):
    d = json.load(open(f, encoding="utf8"))
    keep = []
    for p in d["pages"]:
        t = strip_references(p["text"])
        if len(t.strip()) < 200 or is_navigation(p["text"]) or is_garbage(p["text"]):
            continue
        keep.append((p["file"], p["page"], re.sub(r"\s+", " ", t)[:PAGE_CHARS]))
    pages[d["unit"]] = keep
print(f"usable pages: {sum(len(v) for v in pages.values())} in {len(pages)} units")

by_group = collections.defaultdict(list)
for uid, meta in UNITS.items():
    if pages.get(uid):
        by_group[meta["group"]].append(uid)


def ok_negative(topic_words, text):
    low = text.lower()
    hit = sum(1 for w in topic_words if w[:6] in low)
    return hit * 2 < len(topic_words)


# The human review found that 23% of "hard negatives" picked by word overlap alone were actually
# relevant pages, which taught the model to call same-course pages irrelevant. Training negatives
# now also have to be far from the topic in TF-IDF space. The TEST split is left untouched so the
# human-labelled gold items stay valid and the before/after comparison is fair.
STRICT_NEG_COS = 0.06
_vec = None


def neg_filter(topic, texts):
    """Return the indexes of texts that are confidently unrelated to the topic."""
    global _vec
    if _vec is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        sample = [t for u in list(pages)[:60] for _, _, t in pages[u][:400]]
        _vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, sublinear_tf=True,
                               max_features=50000).fit(sample)
    T = _vec.transform([topic])
    P = _vec.transform(texts)
    num = np.asarray(P.multiply(T).sum(1)).ravel()
    den = np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel()) * np.sqrt(T.multiply(T).sum()) + 1e-9
    return set(np.where(num / den < STRICT_NEG_COS)[0].tolist())


rows, skipped = [], collections.Counter()
for cid, r in best.items():
    s = M["syllabi"][cid]
    dom = r["domain"]
    group = s["group"]
    topics = [t for t in s["topics"] if not generic(t)]
    unit_pages = {u: pages.get(u, []) for u in r["units"]}
    if not any(unit_pages.values()):
        skipped["no pages"] += 1
        continue
    other_groups = [g for g in by_group if g != group]
    for t in topics:
        tw = [w for w in content_words(t) if len(w) > 3]
        if not tw:
            continue
        # positive
        hit = None
        for u in r["units"]:
            h = H.get(f"{cid}|{u}", {}).get(t)
            if h:
                hit = (u, h)
                break
        if not hit:
            continue
        u, (file, page) = hit
        txt = next((x[2] for x in pages.get(u, []) if x[0] == file and x[1] == page), None)
        if not txt:
            skipped["positive page filtered out"] += 1
            continue
        rows.append(dict(domain=dom, syllabus=cid, topic=t, unit=u, file=file, page=page,
                         text=txt, label=1, kind="positive"))
        # hard negative: same course, different page
        pool = [x for x in pages.get(u, []) if not (x[0] == file and x[1] == page)]
        random.shuffle(pool)
        pool = pool[:25]
        strict = dom not in TEST_DOMAINS          # test split keeps the original rule
        far = neg_filter(t, [x[2] for x in pool]) if (strict and pool) else set(range(len(pool)))
        for j, (f2, p2, t2) in enumerate(pool):
            if j in far and ok_negative(tw, t2):
                rows.append(dict(domain=dom, syllabus=cid, topic=t, unit=u, file=f2, page=p2,
                                 text=t2, label=0, kind="hard_negative"))
                break
        # easy negative: unrelated subject group
        if other_groups:
            gu = random.choice(by_group[random.choice(other_groups)])
            f3, p3, t3 = random.choice(pages[gu])
            if ok_negative(tw, t3):
                rows.append(dict(domain=dom, syllabus=cid, topic=t, unit=gu, file=f3, page=p3,
                                 text=t3, label=0, kind="easy_negative"))

print(f"examples: {len(rows)}  skipped: {dict(skipped)}")


def split_of(dom):
    return "test" if dom in TEST_DOMAINS else "val" if dom in VAL_DOMAINS else "train"


import csv

counts = collections.Counter()
files = {}
for name in ("train", "val", "test"):
    fh = open(f"{OUT}/pairs_{name}.csv", "w", newline="", encoding="utf8")
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    files[name] = (fh, w)
for row in rows:
    name = split_of(row["domain"])
    files[name][1].writerow(row)
    counts[(name, row["kind"])] += 1
for fh, _ in files.values():
    fh.close()

# Stratified sample for human verification -> the gold test set.
# Weighted towards the CS test split, which the student can judge reliably; the rest give
# per-subject numbers and may be marked "unsure" without hurting the main result.
sample = []
for name in ("test", "val", "train"):
    pool = [r for r in rows if split_of(r["domain"]) == name]
    for kind in ("positive", "hard_negative", "easy_negative"):
        k = [r for r in pool if r["kind"] == kind]
        random.shuffle(k)
        take = {"test": 67, "val": 17, "train": 16}[name]
        sample += [dict(r, split=name) for r in k[:take]]
random.shuffle(sample)
with open(f"{OUT}/review_sample.csv", "w", newline="", encoding="utf8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(sample[0].keys()) + ["human_label", "note"])
    w.writeheader()
    for r in sample:
        w.writerow(dict(r, human_label="", note=""))

stats = {"examples": len(rows), "by_split_kind": {f"{a}/{b}": c for (a, b), c in sorted(counts.items())},
         "domains": dict(collections.Counter(r["domain"] for r in rows)),
         "test_domains": sorted(TEST_DOMAINS), "val_domains": sorted(VAL_DOMAINS),
         "review_sample": len(sample), "page_chars": PAGE_CHARS}
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1)[:1200])
