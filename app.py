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

import diagrams
import report
from evidence import load_evidence_model
from analyze import TfidfFeaturiser, has_terms
from depth import advice, judge, study_order
from loaders import SUPPORTED, load, unreadable
from syllabus_topics import content_words, topics_with_units
from units import applicable, file_units, unit_of_label, units_from_scores

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
    model = load_evidence_model()
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



def build_results(topics, units, pages, prob, funits, thr):
    """Turn the score matrix into a verdict per topic. Separate from the scoring so that
    changing a file's unit re-decides everything instantly, with no model calls."""
    have = sorted({u for u in funits.values() if u})
    all_placed = not any(u is None for u in funits.values())
    out = []
    for n, t in enumerate(topics):
        row = prob[n]
        order = np.argsort(-row)
        tunit = unit_of_label(units[n])
        above = [i for i in order
                 if row[i] >= thr and has_terms(t, pages[i][2])
                 and applicable(tunit, funits.get(pages[i][0]))][:8]
        evidence = [(pages[i][0], pages[i][1], float(row[i]), pages[i][2]) for i in above]
        level, reasons, kinds = judge(evidence)
        if not evidence and tunit is not None and have and tunit not in have and all_placed:
            level, reasons = "NO MATERIAL PROVIDED", [f"no file for unit {tunit} was uploaded"]
        out.append({"topic": t, "unit": units[n] or "Course", "order": n, "level": level,
                    "reasons": reasons, "kinds": kinds, "evidence": evidence,
                    "closest": (pages[order[0]][0], pages[order[0]][1], float(row[order[0]]))})
    return out


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
st.caption("Each unit is judged only against its own material. SyllabiSense works out which "
           "unit a file belongs to from its name, from a unit stated inside it, or from which "
           "unit's topics it answers — and you can correct it afterwards.")

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

        # a scanned file reads as zero pages, which would otherwise look like "not taught"
        scanned = unreadable(mat_paths)
        if scanned:
            st.session_state["scanned"] = [(os.path.basename(p), b, n) for p, b, n in scanned]
            st.write(f"⚠️ {len(scanned)} file(s) appear to be scanned images and could not be read")
        else:
            st.session_state.pop("scanned", None)

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
        prob = np.zeros((len(topics), len(pages)), np.float32)
        for n, t in enumerate(topics):
            prob[n] = model.predict(feat.topic_page(t, pages), verbose=0).ravel()
            bar.progress((n + 1) / len(topics))

        # Files whose name and text say nothing about a unit get placed by what they contain:
        # the scores above already say which unit's topics each file answers. Only files that
        # are still unplaced are considered, so a name or a title page always wins.
        topic_units = [unit_of_label(u) for u in units]
        page_sources = [pg[0] for pg in pages]
        unplaced = [s for s in dict.fromkeys(page_sources) if funits.get(s) is None]
        if unplaced and len({u for u in topic_units if u is not None}) >= 2:
            guessed = {s: u for s, u in
                       units_from_scores(prob, topic_units, page_sources).items()
                       if s in unplaced}
            if guessed:
                funits.update(guessed)
                st.write("placed by content: "
                         + ", ".join(f"{s} → unit {u}" for s, u in guessed.items()))

        results = build_results(topics, units, pages, prob, funits, thr)

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
    st.session_state["state"] = {"topics": topics, "units": units, "pages": pages,
                                 "prob": prob, "funits": funits, "thr": thr, "paths": paths}
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

    scanned = st.session_state.get("scanned")
    if scanned:
        names = ", ".join(f"{n} ({b} of {t} pages)" for n, b, t in scanned)
        st.warning(
            f"**{len(scanned)} file(s) could not be read: {names}.** These look like scanned "
            f"images or photographs with no text layer. Their pages were not searched, so any "
            f"topic they cover is reported below as if the material were missing. Re-export them "
            f"as text-based PDFs, or treat those topics as unchecked rather than untaught.")

    state = st.session_state.get("state")
    if state and len({u for u in (unit_of_label(u) for u in state["units"]) if u}) >= 2:
        srcs = list(dict.fromkeys(pg[0] for pg in state["pages"]))
        unit_nums = sorted({u for u in (unit_of_label(u) for u in state["units"]) if u})
        with st.expander("Which unit does each file belong to?"):
            st.caption("Detected from the file name, from a unit stated inside it, or from which "
                       "unit's topics it answers. Change anything that is wrong — the report "
                       "updates without re-running the models.")
            chosen, cols = {}, st.columns(min(3, len(srcs)) or 1)
            for i, src in enumerate(srcs):
                cur = state["funits"].get(src)
                options = ["whole course"] + [f"Unit {u}" for u in unit_nums]
                idx = options.index(f"Unit {cur}") if cur in unit_nums else 0
                with cols[i % len(cols)]:
                    pick = st.selectbox(src, options, index=idx, key=f"unit_{src}")
                chosen[src] = None if pick == "whole course" else int(pick.split()[-1])
            if chosen != state["funits"]:
                if st.button("Apply and re-judge", type="primary"):
                    state["funits"] = chosen
                    st.session_state["results"] = build_results(
                        state["topics"], state["units"], state["pages"],
                        state["prob"], chosen, state["thr"])
                    if diagrams.available():
                        res = st.session_state["results"]
                        want = {(state["paths"].get(f), loc)
                                for r in res for f, loc, *_ in r["evidence"]}
                        found = diagrams.detect(want)
                        for r in res:
                            ev = [(state["paths"].get(f), loc) for f, loc, *_ in r["evidence"]]
                            r["diagram"], r["diagram_pages"] = diagrams.summarise(ev, found)
                    st.rerun()

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
