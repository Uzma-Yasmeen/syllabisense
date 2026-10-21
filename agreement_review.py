"""
Second-annotator screen, for measuring inter-annotator agreement.

    streamlit run agreement_review.py

Every depth label in this project came from one person, which means nobody - including that
person - can tell "the model is wrong" apart from "I would have said something else on a
different day". This screen shows a second reader the same topics and the same evidence pages,
with the first reader's answer hidden, and writes to a separate file.

    dataset/depth_labels_second.csv

Then `python score_agreement.py` reports how often the two agree, Cohen's kappa, and - the part
that matters - whether the system's measured accuracy changes depending on whose labels it is
scored against.

A hundred items is enough. They are drawn from the ones already labelled, in a fixed order, so
the two sets line up.
"""
import csv
import json
import os
import random

import streamlit as st

CAND = "dataset/depth_candidates.json"
FIRST = "dataset/depth_labels.csv"
OUT = "dataset/depth_labels_second.csv"
FIELDS = ["subject", "syllabus", "topic", "label", "n_evidence", "note"]
TARGET = 100
CHOICES = [("PROPERLY", "✅ Properly taught", "enough here to actually learn the topic"),
           ("BARELY", "⚠️ Barely taught", "mentioned or listed, but not explained"),
           ("NOT", "❌ Not taught", "these pages are not about this topic")]

st.set_page_config(page_title="Second opinion", layout="wide")


@st.cache_data
def shared_items():
    """The already-labelled topics, in a fixed order, with the first answer withheld."""
    if not os.path.exists(FIRST):
        return []
    first = {(r["syllabus"], r["topic"]) for r in csv.DictReader(open(FIRST, encoding="utf8"))
             if r["label"] in ("PROPERLY", "BARELY", "NOT")}
    cands = [c for c in json.load(open(CAND, encoding="utf8"))
             if (c["syllabus"], c["topic"]) in first]
    random.Random(23).shuffle(cands)      # a fixed seed, so the sample is reproducible
    return cands[:TARGET]


def done_keys():
    if not os.path.exists(OUT):
        return set()
    return {(r["syllabus"], r["topic"]) for r in csv.DictReader(open(OUT, encoding="utf8"))}


def save(c, label):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"subject": c["subject"], "syllabus": c["syllabus"], "topic": c["topic"],
                    "label": label, "n_evidence": c["features"]["n_evidence"], "note": ""})


items = shared_items()
if not items:
    st.error(f"No labelled topics found in {FIRST}. The first pass has to happen before a "
             f"second opinion can be taken.")
    st.stop()

done = done_keys()
todo = [c for c in items if (c["syllabus"], c["topic"]) not in done]

st.title("How well does this material teach this topic?")
st.caption("You are the second reader. Judge only what these pages show — someone else has "
           "already answered these, and their answer is deliberately hidden so it cannot pull "
           "yours. Disagreeing is useful; that is the point of the exercise.")
st.progress(min(len(done) / TARGET, 1.0), text=f"{len(done)} of {TARGET}")

if not todo:
    st.success(f"All {len(done)} done. Thank you — now run `python score_agreement.py`.")
    st.stop()

c = todo[0]
st.markdown(f"### {c['topic']}")
st.caption(f"subject: {c['subject']}  ·  {c['features']['n_evidence']} evidence page(s) found")

if not c["evidence"]:
    st.warning("The model found no evidence pages for this topic in the material.")
else:
    for i, e in enumerate(c["evidence"]):
        with st.expander(f"{e['file']} — {e['loc']}  ·  confidence {e['prob']:.2f}",
                         expanded=i < 2):
            st.write(e["text"])

st.divider()
for col, (key, label, hint) in zip(st.columns(3), CHOICES):
    with col:
        if st.button(label, use_container_width=True, help=hint):
            save(c, key)
            st.rerun()
        st.caption(hint)

st.divider()
if done and st.button("↩ Undo my last answer"):
    with open(OUT, encoding="utf8") as fh:
        lines = fh.read().splitlines()
    if len(lines) > 1:
        with open(OUT, "w", encoding="utf8") as fh:
            fh.write("\n".join(lines[:-1]) + "\n")
        st.rerun()

st.caption("If the pages only list the topic as a bullet, or name it inside the explanation of "
           "something else, that is **barely taught** rather than properly taught.")
