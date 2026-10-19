"""
Human verification screen for the evidence dataset.

    streamlit run review_tool.py

Shows a syllabus topic and a page of course material, and asks one question:
is this page useful evidence for that topic?

The automatic (weak) label is deliberately hidden while judging - seeing it would nudge the
answer and inflate the agreement we report. It is revealed only after the answer is saved.
Answers are appended to dataset/review_done.csv after every click, so stopping is safe.
"""
import csv
import os

import streamlit as st

SAMPLE = "dataset/review_sample.csv"
DONE = "dataset/review_done.csv"
FIELDS = ["domain", "syllabus", "topic", "unit", "file", "page", "kind", "label", "human_label", "note"]

st.set_page_config(page_title="SyllabiSense — evidence review", layout="wide")


@st.cache_data
def load_sample():
    return list(csv.DictReader(open(SAMPLE, encoding="utf8")))


def load_done():
    if not os.path.exists(DONE):
        return {}
    return {(r["syllabus"], r["topic"], r["file"], r["page"]): r
            for r in csv.DictReader(open(DONE, encoding="utf8"))}


def save(row, human, note=""):
    new = not os.path.exists(DONE)
    with open(DONE, "a", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS} | {"human_label": human, "note": note})


rows = load_sample()
done = load_done()
todo = [r for r in rows if (r["syllabus"], r["topic"], r["file"], r["page"]) not in done]

st.title("Is this page useful evidence for this topic?")
st.progress(len(done) / len(rows), text=f"{len(done)} of {len(rows)} reviewed")

if not todo:
    st.success(f"All {len(rows)} reviewed. Agreement with the automatic labels:")
    agree = sum(1 for r in done.values() if r["human_label"] == r["label"])
    st.metric("agreement", f"{100*agree/max(1,len(done)):.0f}%")
    st.stop()

r = todo[0]
left, right = st.columns([1, 2])
with left:
    st.subheader("Syllabus topic")
    st.markdown(f"### {r['topic']}")
    st.caption(f"subject: {r['domain']}  ·  syllabus {r['syllabus']}")
with right:
    st.subheader("Page from the course material")
    st.caption(f"{r['file'].split('/')[-1]} — page {r['page']}")
    st.write(r["text"] if "text" in r else "")

st.divider()
c1, c2, c3 = st.columns(3)
if c1.button("✅ Relevant — this page is evidence for the topic", use_container_width=True):
    save(r, "1"); st.rerun()
if c2.button("❌ Not relevant", use_container_width=True):
    save(r, "0"); st.rerun()
if c3.button("🤔 Unsure / can't tell", use_container_width=True):
    save(r, "", "unsure"); st.rerun()

st.caption("Judge only whether the page is about this topic. Partly useful counts as relevant; "
           "same subject but a different topic counts as not relevant.")

st.divider()
if done and st.button("↩ Undo my last answer", help="Removes the most recent answer so you can redo it"):
    with open(DONE, encoding="utf8") as fh:
        lines = fh.read().splitlines()
    if len(lines) > 1:                       # keep the header
        last = lines[-1].split(",")[2][:60]
        with open(DONE, "w", encoding="utf8") as fh:
            fh.write("\n".join(lines[:-1]) + "\n")
        st.warning(f"Removed the last answer — it will come back for review ({last}...)")
        st.rerun()
