"""
Depth labelling screen — the training data for the depth model.

    streamlit run depth_review.py

You see a syllabus topic and the pages the model found for it, and answer ONE question:
how well does this material teach the topic?

    PROPERLY TAUGHT  explained enough to learn from
    BARELY TAUGHT    only mentioned, listed, or touched in passing
    NOT TAUGHT       the pages are not really about this topic at all

Answers go to dataset/depth_labels.csv after every click, so stopping is safe.
Aim for 150-200; a balanced spread of answers matters more than the total.
"""
import csv
import json
import os
import random

import streamlit as st

CAND = "dataset/depth_candidates.json"
OUT = "dataset/depth_labels.csv"
FIELDS = ["subject", "syllabus", "topic", "label", "n_evidence", "note"]
CHOICES = [("PROPERLY", "✅ Properly taught", "enough here to actually learn the topic"),
           ("BARELY", "⚠️ Barely taught", "mentioned or listed, but not explained"),
           ("NOT", "❌ Not taught", "these pages are not about this topic")]

st.set_page_config(page_title="Depth labelling", layout="wide")


@st.cache_data
def load_candidates():
    c = json.load(open(CAND, encoding="utf8"))
    random.Random(11).shuffle(c)          # mix subjects so early stopping stays representative
    return c


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


cands = load_candidates()
done = done_keys()
todo = [c for c in cands if (c["syllabus"], c["topic"]) not in done]

st.title("How well does this material teach this topic?")
st.progress(min(len(done) / 200, 1.0), text=f"{len(done)} labelled — aiming for 150–200")

if not todo:
    st.success("No candidates left.")
    st.stop()

c = todo[0]
st.markdown(f"### {c['topic']}")
st.caption(f"subject: {c['subject']}  ·  {c['features']['n_evidence']} evidence page(s) found")

if not c["evidence"]:
    st.warning("The model found no evidence pages for this topic in the material.")
else:
    for e in c["evidence"]:
        with st.expander(f"{e['file']} — {e['loc']}  ·  confidence {e['prob']:.2f}",
                         expanded=c["evidence"].index(e) < 2):
            st.write(e["text"])

st.divider()
cols = st.columns(3)
for col, (key, label, hint) in zip(cols, CHOICES):
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
        open(OUT, "w", encoding="utf8").write("\n".join(lines[:-1]) + "\n")
        st.rerun()

st.caption("Judge only what these pages show. If the pages merely list the topic as a bullet or "
           "name it inside another topic, that is **barely taught**, not properly taught.")
