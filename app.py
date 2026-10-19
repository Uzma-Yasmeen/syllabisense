"""
SyllabiSense — web interface.

    streamlit run app.py

Upload a syllabus and your course material (PDF / PPTX / DOCX / TXT) and get a report organised
UNIT BY UNIT: for each unit, which topics are properly taught, barely taught, not taught, or had
no material uploaded at all — with the file and page as evidence.

The decision for each (topic, page) pair comes from the trained ANN; the depth level on top is the
transparent rule in depth.py.
"""
import json
import os
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf

import diagrams
import report
from analyze import TfidfFeaturiser, has_terms
from depth import advice, judge, study_order
from loaders import SUPPORTED, load
from syllabus_topics import content_words, topics_with_units
from units import applicable, file_units, unit_of_label

MODEL_DIR = "results/tfidf"
DEMO_DIR = "demo_files"
LEVELS = ["PROPERLY TAUGHT", "BARELY TAUGHT", "NOT TAUGHT", "NO MATERIAL PROVIDED"]
BADGE = {"PROPERLY TAUGHT": "✅", "BARELY TAUGHT": "⚠️", "NOT TAUGHT": "❌",
         "NO MATERIAL PROVIDED": "➖"}
SHORT = {"PROPERLY TAUGHT": "properly taught", "BARELY TAUGHT": "barely taught",
         "NOT TAUGHT": "not taught", "NO MATERIAL PROVIDED": "no material given"}

st.set_page_config(page_title="SyllabiSense", layout="wide")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/evidence_ann.keras")
    thr = json.load(open(f"{MODEL_DIR}/metrics.json", encoding="utf8"))["threshold"]
    human = f"{MODEL_DIR}/metrics_human.json"
    if os.path.exists(human):        # prefer the threshold tuned on human-checked items
        thr = json.load(open(human, encoding="utf8"))["threshold_human_tuned"]
    return model, thr, TfidfFeaturiser()


def demo_files():
    """The bundled example, so the app can be tried without uploading anything.
    Returns (syllabus path, [material paths]) or None when the folder is absent."""
    if not os.path.isdir(DEMO_DIR):
        return None
    files = sorted(os.path.join(DEMO_DIR, f) for f in os.listdir(DEMO_DIR)
                   if os.path.splitext(f)[1].lower() in SUPPORTED)
    syl = next((f for f in files if "syllabus" in os.path.basename(f).lower()), None)
    mats = [f for f in files if f != syl]
    return (syl, mats) if syl and mats else None


def save_uploads(files):
    tmp = tempfile.mkdtemp()
    paths = []
    for f in files:
        p = os.path.join(tmp, f.name)
        with open(p, "wb") as fh:
            fh.write(f.getbuffer())
        paths.append(p)
    return paths


st.title("SyllabiSense")
st.caption("Does your course material actually cover your syllabus?")

c1, c2 = st.columns(2)
with c1:
    syl_file = st.file_uploader("1. Your syllabus", type=["pdf", "docx", "txt"])
with c2:
    mat_files = st.file_uploader(f"2. Your course material ({', '.join(s[1:] for s in SUPPORTED)})",
                                 type=[s[1:] for s in SUPPORTED], accept_multiple_files=True)
st.caption("Tip: name files with their unit — `Unit-2 notes.pdf`, `U3_slides.pptx` — and each "
           "unit is judged only against its own material.")

use_demo = False
if demo_files():
    use_demo = st.checkbox("Nothing to hand? Use the built-in example instead — an NPTEL Data "
                           "Structures syllabus against seven MIT OpenCourseWare lecture PDFs.")

if st.button("Analyse", type="primary", disabled=not (use_demo or (syl_file and mat_files))):
    if use_demo:
        syl_path, mat_paths = demo_files()
    else:
        syl_path, mat_paths = save_uploads([syl_file])[0], save_uploads(mat_files)
    model, thr, feat = load_model()
    with st.status("Working...", expanded=True) as status:
        st.write("Reading the syllabus ...")
        tu = [(u, t) for u, t in topics_with_units(syl_path) if content_words(t)]
        topics, units = [t for _, t in tu], [u for u, _ in tu]
        nunits = len({u for u in units if u})
        st.write(f"found {len(topics)} topics" + (f" in {nunits} units" if nunits else ""))

        st.write("Reading the course material ...")
        blocks = load(mat_paths)
        pages = [(b["source"], b["location"], b["text"]) for b in blocks]
        paths = {b["source"]: b["path"] for b in blocks}   # for rendering pages for the CNN
        st.write(f"{len(pages)} pages/slides/sections from {len(set(p[0] for p in pages))} file(s)")
        funits = file_units(pages)
        have = sorted({u for u in funits.values() if u})
        if have:
            st.write("material covers unit(s): " + ", ".join(str(u) for u in have)
                     + " — a unit is judged only against its own material")
        if not topics or not pages:
            status.update(label="Nothing to compare — check the files", state="error")
            st.stop()

        st.write("Scoring every topic against every page ...")
        bar = st.progress(0.0)
        results = []
        for n, t in enumerate(topics):
            prob = model.predict(feat.topic_page(t, pages), verbose=0).ravel()
            order = np.argsort(-prob)
            tunit = unit_of_label(units[n])
            above = [i for i in order
                     if prob[i] >= thr and has_terms(t, pages[i][2])
                     and applicable(tunit, funits.get(pages[i][0]))][:8]
            evidence = [(pages[i][0], pages[i][1], float(prob[i]), pages[i][2]) for i in above]
            level, reasons, kinds = judge(evidence)
            if not evidence and tunit is not None and have and tunit not in have \
                    and not any(u is None for u in funits.values()):
                level, reasons = "NO MATERIAL PROVIDED", [f"no file for unit {tunit} was uploaded"]
            results.append({"topic": t, "unit": units[n] or "Course", "order": n, "level": level,
                            "reasons": reasons, "kinds": kinds, "evidence": evidence,
                            "closest": (pages[order[0]][0], pages[order[0]][1],
                                        float(prob[order[0]]))})
            bar.progress((n + 1) / len(topics))

        # the CNN runs only on the pages the evidence model already chose, not the whole document
        if diagrams.available():
            st.write("Checking the evidence pages for diagrams ...")
            want = {(paths.get(f), loc) for r in results for f, loc, *_ in r["evidence"]}
            found = diagrams.detect(want)
            for r in results:
                ev = [(paths.get(f), loc) for f, loc, *_ in r["evidence"]]
                r["diagram"], r["diagram_pages"] = diagrams.summarise(ev, found)
            withfig = sum(1 for r in results if r["diagram"] == "diagram")
            st.write(f"{len(found)} page(s) checked — {withfig} topic(s) have a diagram")
        status.update(label="Done", state="complete")
    st.session_state["results"] = results
    st.session_state["meta"] = {"syllabus": os.path.basename(syl_path),
                                "material": [os.path.basename(m) for m in mat_paths]}

if "results" in st.session_state:
    results = st.session_state["results"]
    count = lambda rs, lv: sum(1 for r in rs if r["level"] == lv)

    a, b, c, d, e = st.columns(5)
    a.metric("Topics", len(results))
    b.metric("✅ Properly taught", count(results, "PROPERLY TAUGHT"))
    c.metric("⚠️ Barely taught", count(results, "BARELY TAUGHT"))
    d.metric("❌ Not taught", count(results, "NOT TAUGHT"))
    e.metric("➖ No material", count(results, "NO MATERIAL PROVIDED"))

    meta = st.session_state.get("meta", {})
    d1, d2 = st.columns(2)
    d1.download_button("⬇  Download the full report (PDF)", report.to_pdf(results, meta),
                       file_name="syllabus_coverage_report.pdf", mime="application/pdf",
                       use_container_width=True)
    d2.download_button("⬇  Download the topic table (CSV)", report.to_csv(results),
                       file_name="syllabus_coverage.csv", mime="text/csv",
                       use_container_width=True)

    todo = study_order(results)
    if todo:
        st.subheader("What to study first")
        for i, r in enumerate(todo[:8], 1):
            why = "nothing in the material" if r["level"] == "NOT TAUGHT" else "material is too thin"
            st.write(f"**{i}. {r['topic']}** — {why}  ·  _{r['unit']}_")

    st.subheader("Unit by unit")
    unit_names = list(dict.fromkeys(r["unit"] for r in results))
    tabs = st.tabs([u.split("—")[0].strip() for u in unit_names])
    for tab, u in zip(tabs, unit_names):
        rs = [r for r in results if r["unit"] == u]
        with tab:
            st.markdown(f"#### {u}")
            cols = st.columns(4)
            for col, lv in zip(cols, LEVELS):
                col.metric(f"{BADGE[lv]} {SHORT[lv]}", count(rs, lv))

            if count(rs, "NO MATERIAL PROVIDED") == len(rs):
                st.info("You did not upload any material for this unit, so it could not be judged.")
                continue

            for lv in LEVELS:
                group = [r for r in rs if r["level"] == lv]
                if not group:
                    continue
                st.markdown(f"**{BADGE[lv]} {SHORT[lv].title()} — {len(group)}**")
                for r in group:
                    with st.expander(r["topic"]):
                        st.caption(" · ".join(r["reasons"]))
                        st.write(f"**What to do:** {advice(r['level'], r['kinds'])}")
                        if r.get("diagram") == "diagram":
                            df, dloc, dp = r["diagram_pages"][0]
                            st.caption(f"🖼 Diagram found on {os.path.basename(df)} — {dloc} "
                                       f"(CNN confidence {dp:.2f})")
                        elif r.get("diagram") == "none":
                            st.caption("🖼 No diagram on any of these pages — this topic is "
                                       "explained in words only.")
                        for f, loc, pr, text in r["evidence"][:3]:
                            st.markdown(f"**{f} — {loc}**  ·  confidence {pr:.2f}")
                            st.caption(text[:400] + " ...")
                        if not r["evidence"] and r["level"] == "NOT TAUGHT":
                            f, loc, pr = r["closest"]
                            st.caption(f"closest page: {f} — {loc} "
                                       f"(scored {pr:.2f}, below the threshold)")

    st.divider()
    st.caption("Evidence model: ANN trained on 11,615 (topic, page) pairs from 12 non-computing "
               "subject areas, tested on 109 Computer Science courses. "
               "Diagram detection: MobileNetV2 CNN trained on 576 hand-labelled page images "
               "(84% accurate, PDF pages only — slides and Word files are not checked). "
               "Depth levels are rules over the evidence found, not learned labels.")
