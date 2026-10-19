"""
Diagram labelling screen for the CNN.

    streamlit run image_review.py

Nine page images at a time. Tick the ones that contain a diagram, chart, circuit, flowchart or
labelled figure; leave plain text, equations-only and tables unticked. Press Save and next.

A grid is much faster than one page at a time - about 10 seconds per screen, so ~350 labels in
roughly 10 minutes.

Output: dataset/diagram_labels.csv  (image, label)  label 1 = has a diagram
"""
import csv
import os

import streamlit as st

IMG = "dataset/page_images"
OUT = "dataset/diagram_labels.csv"
GRID = 9
COLS = 3

st.set_page_config(page_title="Diagram labelling", layout="wide")


def all_images():
    """Read the index if rendering has finished, otherwise just list what exists so labelling
    can start while the renderer is still working."""
    if os.path.exists(f"{IMG}/index.csv"):
        return [r["image"] for r in csv.DictReader(open(f"{IMG}/index.csv", encoding="utf8"))]
    return sorted(f for f in os.listdir(IMG) if f.endswith(".jpg")) if os.path.isdir(IMG) else []


def labelled():
    if not os.path.exists(OUT):
        return set()
    return {r["image"] for r in csv.DictReader(open(OUT, encoding="utf8"))}


def save(batch, ticked):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "label"])
        if new:
            w.writeheader()
        for name in batch:
            w.writerow({"image": name, "label": 1 if name in ticked else 0})


images = all_images()
done = labelled()
todo = [i for i in images if i not in done]

st.title("Tick the pages that contain a diagram")
st.progress(min(len(done) / 350, 1.0), text=f"{len(done)} labelled — aiming for ~350")
st.caption("**Diagram** = figure, chart, graph, circuit, flowchart, labelled drawing, screenshot. "
           "**Not a diagram** = plain text, equations only, plain tables, title pages.")

if not todo:
    st.success(f"All {len(images)} images labelled.")
    st.stop()

batch = todo[:GRID]
ticked = set()
for row_start in range(0, len(batch), COLS):
    cols = st.columns(COLS)
    for col, name in zip(cols, batch[row_start:row_start + COLS]):
        with col:
            st.image(os.path.join(IMG, name), use_column_width=True)
            if st.checkbox("has a diagram", key=f"cb_{name}"):
                ticked.add(name)

st.divider()
c1, c2 = st.columns([1, 3])
if c1.button("Save and next ▶", type="primary", use_container_width=True):
    save(batch, ticked)
    for name in batch:
        st.session_state.pop(f"cb_{name}", None)
    st.rerun()
if c2.button("↩ Undo last screen"):
    if os.path.exists(OUT):
        lines = open(OUT, encoding="utf8").read().splitlines()
        if len(lines) > 1:
            open(OUT, "w", encoding="utf8").write("\n".join(lines[:-GRID]) + "\n")
            st.rerun()
