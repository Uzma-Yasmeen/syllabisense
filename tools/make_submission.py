"""
Assemble the submission folder: Word proposal + runnable code + results.

    python make_submission.py

Produces submission/
    PROPOSAL.docx          the proposal document (converted from PROPOSAL_DRAFT.md)
    README.md              how to run everything
    requirements.txt       exact packages used
    run_app.bat            double-click to start the web interface
    code/                  the pipeline that trains and runs the model
    code/tools/            dataset-collection scripts (provenance, not needed to run the model)
    results/               graphs, confusion matrix, metrics, trained model
    sample_report.md       a real report produced by the system
"""
import os
import re
import shutil
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

sys.stdout.reconfigure(encoding="utf-8")
OUT = "submission"

CORE = ["analyze.py", "app.py", "loaders.py", "syllabus_topics.py", "units.py", "depth.py",
        "build_dataset.py", "features_tfidf.py", "embed_features.py", "train_ann.py",
        "evaluate_gold.py", "review_tool.py", "match.py",
        "build_depth_candidates.py", "depth_review.py", "train_depth.py"]
TOOLS = ["collect_nptel.py", "collect_ocw.py", "extract_text.py", "inventory.py",
         "select_pairs.py", "select_cs.py", "finalize_pairs.py", "write_final.py",
         "spot_check.py", "generic_topics.py", "test_page_types.py"]
RESULT_DIRS = ["results_tfidf", "results_embed", "results_tfidf_nosim", "results_embed_nosim",
               "results_depth"]

REQUIREMENTS = """tensorflow==2.15.0
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
streamlit==1.32.0
pypdfium2
python-pptx==1.0.2
python-docx==1.1.2
joblib
torch          # only needed for the embedding variant (embed_features.py)
transformers   # only needed for the embedding variant
"""


# ---------- markdown -> docx ----------
TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
SEP_ROW = re.compile(r"^\|[\s:|-]+\|\s*$")


def add_runs(par, text):
    """Handle **bold** and `code` inside a line."""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        else:
            par.add_run(part)


def md_to_docx(md_path, out_path):
    lines = open(md_path, encoding="utf8").read().splitlines()
    d = docx.Document()
    for s in d.styles["Normal"], :
        s.font.name = "Calibri"
        s.font.size = Pt(10.5)

    i, in_code = 0, False
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = d.add_paragraph()
            r = p.add_run(ln)
            r.font.name = "Consolas"
            r.font.size = Pt(9)
            p.paragraph_format.space_after = Pt(0)
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("#"):
            level = len(ln) - len(ln.lstrip("#"))
            text = ln.lstrip("#").strip()
            if level == 1:
                h = d.add_heading(text, 0)
                h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                d.add_heading(text, min(level - 1, 4))
            i += 1
            continue
        if ln.startswith(">"):
            p = d.add_paragraph(style="Intense Quote")
            add_runs(p, ln.lstrip("> ").strip())
            i += 1
            continue
        if TABLE_ROW.match(ln) and i + 1 < len(lines) and SEP_ROW.match(lines[i + 1]):
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            body = []
            j = i + 2
            while j < len(lines) and TABLE_ROW.match(lines[j]):
                body.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            t = d.add_table(rows=1, cols=len(header))
            t.style = "Light Grid Accent 1"
            for c, text in zip(t.rows[0].cells, header):
                c.text = ""
                run = c.paragraphs[0].add_run(re.sub(r"\*\*", "", text))
                run.bold = True
            for row in body:
                cells = t.add_row().cells
                for c, text in zip(cells, row):
                    c.text = ""
                    add_runs(c.paragraphs[0], text)
            d.add_paragraph()
            i = j
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            p = d.add_paragraph(style="List Bullet")
            add_runs(p, re.sub(r"^\s*[-*]\s+", "", ln))
            i += 1
            continue
        if re.match(r"^\s*\d+\.\s+", ln):
            p = d.add_paragraph(style="List Number")
            add_runs(p, re.sub(r"^\s*\d+\.\s+", "", ln))
            i += 1
            continue
        if set(ln.strip()) <= {"-", "="} and len(ln.strip()) >= 3:
            i += 1
            continue
        p = d.add_paragraph()
        add_runs(p, ln)
        i += 1

    # append the result images so the document carries its own evidence
    d.add_page_break()
    d.add_heading("Result graphs", 1)
    for title, path in [("Accuracy and loss vs epochs — evidence model", "results_tfidf/history.png"),
                        ("Confusion matrix — evidence model (unseen subject)",
                         "results_tfidf/confusion_matrix.png"),
                        ("ROC curve — evidence model", "results_tfidf/roc_curve.png"),
                        ("Accuracy and loss vs epochs — depth model", "results_depth/history.png"),
                        ("Confusion matrix — depth model", "results_depth/confusion_matrix.png")]:
        if os.path.exists(path):
            d.add_heading(title, 3)
            d.add_picture(path, width=docx.shared.Inches(5.8))
    d.save(out_path)


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/code/tools", exist_ok=True)
    os.makedirs(f"{OUT}/results", exist_ok=True)

    md_to_docx("PROPOSAL_DRAFT.md", f"{OUT}/PROPOSAL.docx")
    shutil.copy("PROPOSAL_DRAFT.md", f"{OUT}/PROPOSAL.md")

    for f in CORE:
        if os.path.exists(f):
            shutil.copy(f, f"{OUT}/code/{f}")
    for f in TOOLS:
        if os.path.exists(f):
            shutil.copy(f, f"{OUT}/code/tools/{f}")
    for d in RESULT_DIRS:
        if os.path.isdir(d):
            shutil.copytree(d, f"{OUT}/results/{d}")
    for f in ("report_SE.md", "FINAL_PAIRS.md", "dataset/stats.json"):
        if os.path.exists(f):
            shutil.copy(f, f"{OUT}/{os.path.basename(f)}")
    if os.path.exists("dataset/tfidf_model.joblib"):
        os.makedirs(f"{OUT}/dataset", exist_ok=True)
        shutil.copy("dataset/tfidf_model.joblib", f"{OUT}/dataset/tfidf_model.joblib")

    open(f"{OUT}/requirements.txt", "w", encoding="utf8").write(REQUIREMENTS)
    open(f"{OUT}/run_app.bat", "w", encoding="utf8").write(
        "@echo off\r\nREM Start the SyllabiSense web interface\r\n"
        "cd /d \"%~dp0code\"\r\nstreamlit run app.py\r\npause\r\n")
    open(f"{OUT}/README.md", "w", encoding="utf8").write(README)
    n = sum(len(fs) for _, _, fs in os.walk(OUT))
    print(f"submission/ built: {n} files")
    for root, _, fs in os.walk(OUT):
        if fs:
            print(f"  {root.replace(os.sep, '/'):<34} {len(fs)} file(s)")


README = """# SyllabiSense — AI & Deep Learning project

**Uzma Yasmeen · Roll 2503A51L34 · Batch 20 · Individual project**

Checks whether course material actually covers a syllabus: for every syllabus topic it reports
**properly taught / barely taught / not taught**, with the file and page as evidence.

## Run the web interface

Double-click `run_app.bat`, or:

    pip install -r requirements.txt
    cd code
    streamlit run app.py

Upload a syllabus (PDF/DOCX) and course material (PDF, PPTX, DOCX, TXT), then press **Analyse**.
Naming files by unit (`Unit-2 notes.pdf`, `U3_slides.pptx`) makes each unit be judged only against
its own material.

## Run from the command line

    cd code
    python analyze.py --syllabus my_syllabus.pdf --material notes_folder --out report.md

## The two models

| Model | Task | Architecture |
|---|---|---|
| Evidence ANN | is this page evidence for this topic? | 387 -> 256 -> 64 -> 1 (sigmoid) |
| Depth ANN | properly / barely / not taught | 18 -> 64 -> 32 -> 3 (softmax) |

## Reproducing the training

    python build_dataset.py        # build labelled (topic, page) pairs
    python features_tfidf.py       # TF-IDF + LSA features
    python train_ann.py --out ../results/results_tfidf
    python evaluate_gold.py --model ../results/results_tfidf   # score against human labels

    python build_depth_candidates.py   # topics + evidence + features
    streamlit run depth_review.py      # human depth labelling
    python train_depth.py              # train the depth model

## Folders

- `code/` — the pipeline
- `code/tools/` — scripts used to collect and verify the dataset (not needed to run the model)
- `results/` — graphs, confusion matrices, metrics, trained models
- `report_SE.md` — a real report the system produced
- `FINAL_PAIRS.md` — the frozen benchmark pairs the dataset was built from
"""

if __name__ == "__main__":
    main()
