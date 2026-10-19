"""
Assemble the clean, shareable repository at Documents/GitHub/syllabisense.

The working folder has ~30 scripts because a dataset was built from scratch. The repo keeps only
what someone needs to run, train or understand the project; the collection scripts move to tools/
as provenance. The 5.5 GB of PDFs stays in the data folder and is git-ignored.

    python build_repo.py [--dest PATH]
"""
import argparse
import os
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

RUN = ["app.py", "analyze.py", "demo.py"]
LIB = ["loaders.py", "syllabus_topics.py", "units.py", "depth.py", "match.py", "diagrams.py",
       "report.py", "evidence.py"]
TRAIN = ["build_dataset.py", "features_tfidf.py", "embed_features.py", "train_ann.py",
         "train_crossencoder.py", "evaluate_gold.py", "evaluate_gold_ce.py", "score_verdicts.py", "export_ann.py", "tune_depth_rules.py",
         "render_pages.py", "train_cnn.py", "train_cnn_scratch.py", "compare_fusion.py"]
LABEL = ["review_tool.py", "build_depth_candidates.py", "depth_review.py", "train_depth.py",
         "image_review.py"]
TOOLS = ["collect_nptel.py", "collect_ocw.py", "extract_text.py", "inventory.py",
         "select_pairs.py", "select_cs.py", "finalize_pairs.py", "write_final.py",
         "spot_check.py", "generic_topics.py", "test_page_types.py", "build_repo.py",
         "make_submission.py", "make_figures.py", "make_proposal.py",
         "make_code_submission.py", "make_proposal_html.py", "make_space.py", "measure_memory.py"]
RESULTS = ["results_tfidf", "results_embed", "results_tfidf_nosim", "results_embed_nosim",
           "results_depth", "results_crossencoder", "results_cnn", "results_cnn_scratch",
           "results_fusion", "results_verdicts"]
DOCS = [("PROPOSAL_DRAFT.md", "PROPOSAL.md"), ("report_SE.md", "example_report.md"),
        ("FINAL_PAIRS.md", "dataset_pairs.md"), ("MANIFEST.md", "dataset_sources.md")]

CROSSENCODER_NOTE = """# Cross-encoder weights are not in this repository

`model.safetensors` is 87 MB. The application never loads it — the transformer was trained as a
measured comparison against the evidence ANN, not as a component of the system.

To reproduce it (about an hour on a CPU):

    python train_crossencoder.py --epochs 2 --max-train 9000

Its measured results, graphs and confusion matrices are in `results/crossencoder/`.
"""

GITIGNORE = """# data lives outside the repo (5.5 GB of PDFs)
models/crossencoder/
syllabi/
material/
material_quarantine/
cache/
dataset/*.csv
dataset/*.npz
dataset/*.json
!dataset/stats.json
*.log
models/crossencoder/
__pycache__/
.venv/
"""

README = """# SyllabiSense

Checks whether course material actually covers a syllabus. For every syllabus topic it reports
**properly taught / barely taught / not taught**, with the file and page as evidence.

Built for the AI & Deep Learning course at SR University.
**Uzma Yasmeen · Roll 2503A51L34 · Batch 20**

## What it does

Upload a syllabus and your course material (PDF, PowerPoint, Word). The system extracts the
syllabus topics, reads every page of the material, and decides which pages are evidence for which
topic. Topics with no evidence are reported as gaps — the thing a document chatbot cannot do,
because it has no idea what the syllabus promised.

Files named by unit (`Unit-2 notes.pdf`, `U3_slides.pptx`) are matched to that unit only, and a
unit with no uploaded material is reported as *no material provided* rather than *not taught*.

## Run it

    pip install -r requirements.txt
    streamlit run app.py          # or double-click run_app.bat

Command line:

    python analyze.py --syllabus syllabus.pdf --material notes_folder --out report.md

## Models

| Model | Question | Architecture |
|---|---|---|
| Cross-encoder (deep learning) | is this page evidence for this topic? | MiniLM transformer, 6 layers, fine-tuned end to end |
| Evidence ANN | same question, from frozen features | 387 → 256 → 64 → 1, sigmoid |
| Depth ANN | properly / barely / not taught | 18 → 64 → 32 → 3, softmax |
| Diagram CNN | does this page contain a diagram? | MobileNetV2 base + trained head |

Trained on 17,887 (topic, page) pairs built from 1,295 NPTEL syllabi and ~110 MIT OpenCourseWare
courses plus 7 OpenStax textbooks (100,414 pages). Splits are by **subject**, so the test subject
is never seen during training. Accuracy is reported against **human-verified labels**, not the
automatic ones — see `docs/PROPOSAL.md` for the numbers and the limitations.

## Training pipeline

    python build_dataset.py            # labelled (topic, page) pairs, split by subject
    python features_tfidf.py           # TF-IDF + LSA features
    python train_ann.py                # the feature-based ANN
    python train_crossencoder.py       # the fine-tuned transformer
    python evaluate_gold.py            # score against human labels

    python evaluate_gold_ce.py         # same, for the transformer (identical protocol)

    python build_depth_candidates.py   # topics + evidence + features
    streamlit run depth_review.py      # human depth labelling
    python train_depth.py              # the 3-class depth model

    python render_pages.py             # page images for the CNN
    streamlit run image_review.py      # human diagram labelling
    python train_cnn_scratch.py        # CNN from scratch (fails - kept for comparison)
    python train_cnn.py                # MobileNetV2 transfer learning

`review_tool.py` is the screen used to collect the human-verified relevance labels.

## Layout

    app.py analyze.py          run the system
    loaders.py                 PDF / PPTX / DOCX / TXT readers, one common shape
    syllabus_topics.py         syllabus parser (units and topics)
    units.py                   match material files to syllabus units
    depth.py                   evidence -> teaching depth
    match.py                   page-type filters and the matcher used to build the dataset
    models/                    trained models
    results/                   metrics, confusion matrices, graphs
    docs/                      proposal, example report, dataset details
    tools/                     scripts used to collect the dataset (not needed to run anything)

The dataset itself (5.5 GB) is not in the repo. `docs/dataset_sources.md` lists every source.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/Documents/GitHub/syllabisense"))
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    # Clear the contents but never .git. Wiping the whole folder loses the remote and the
    # history, and on Windows it fails half way through on git's read-only objects - which
    # leaves the repository empty and unrecoverable.
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(dest):
        if name == ".git":
            continue
        path = os.path.join(dest, name)
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
    for sub in ("tools", "results", "docs", "models"):
        os.makedirs(os.path.join(dest, sub), exist_ok=True)

    def copy(src, dst):
        if os.path.exists(src):
            shutil.copy(src, dst)
            return True
        return False

    n = 0
    for f in RUN + LIB + TRAIN + LABEL:
        if copy(f, os.path.join(dest, f)):
            n += 1
    for f in TOOLS:
        copy(f, os.path.join(dest, "tools", f))

    # models in one place
    for src, dst in [("results_tfidf/evidence_ann.npz", "evidence_ann.npz"),
                     ("results_tfidf/evidence_ann.keras", "evidence_ann.keras"),
                     ("results_depth/depth_ann.keras", "depth_ann.keras"),
                     ("results_cnn/diagram_cnn.keras", "diagram_cnn.keras"),
                     ("dataset/tfidf_model.joblib", "tfidf_model.joblib"),
                     ("results_depth/feature_names.json", "depth_features.json")]:
        copy(src, os.path.join(dest, "models", dst))
    # the fine-tuned transformer's weights are 87 MB and the application never loads them -
    # train_crossencoder.py reproduces them in about an hour, so they stay out of the repository
    if os.path.isdir("results_crossencoder/model"):
        note = os.path.join(dest, "models", "crossencoder_NOT_INCLUDED.md")
        open(note, "w", encoding="utf8").write(CROSSENCODER_NOTE)

    # results: metrics and graphs only, never the weights (they live in models/)
    for d in RESULTS:
        if not os.path.isdir(d):
            continue
        out = os.path.join(dest, "results", d.replace("results_", ""))
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith((".png", ".json", ".txt", ".csv")):
                shutil.copy(os.path.join(d, f), os.path.join(out, f))

    for src, dst in DOCS:
        copy(src, os.path.join(dest, "docs", dst))
    copy("submission/PROPOSAL.docx", os.path.join(dest, "docs", "PROPOSAL.docx"))
    copy("submission/PROPOSAL.pdf", os.path.join(dest, "docs", "PROPOSAL.pdf"))
    if os.path.isdir("figures"):
        os.makedirs(os.path.join(dest, "docs", "figures"), exist_ok=True)
        for f in os.listdir("figures"):
            copy(os.path.join("figures", f), os.path.join(dest, "docs", "figures", f))
    os.makedirs(os.path.join(dest, "dataset"), exist_ok=True)
    copy("dataset/stats.json", os.path.join(dest, "dataset", "stats.json"))

    # the copies look for models in models/ and results/, not in the working folder's layout
    for f in ("app.py", "analyze.py", "build_depth_candidates.py", "diagrams.py", "evidence.py"):
        p = os.path.join(dest, f)
        if not os.path.exists(p):
            continue
        s = open(p, encoding="utf8").read()
        s = s.replace('MODEL_DIR = "results_tfidf"', 'MODEL_DIR = "results/tfidf"')
        s = s.replace('default="results_tfidf/evidence_ann.keras"', 'default="models/evidence_ann.keras"')
        s = s.replace('default="results_tfidf/metrics.json"', 'default="results/tfidf/metrics.json"')
        s = s.replace('f"{MODEL_DIR}/evidence_ann.keras"', '"models/evidence_ann.keras"')
        s = s.replace('def __init__(self, path="dataset/tfidf_model.joblib")',
                      'def __init__(self, path="models/tfidf_model.joblib")')
        # the CNN: weights live in models/, its metrics in results/cnn/. Without this the repo
        # copy finds no model, diagrams.available() returns False and the diagram line silently
        # disappears from the report.
        s = s.replace('MODEL = "results_cnn/diagram_cnn.keras"', 'MODEL = "models/diagram_cnn.keras"')
        s = s.replace('EXPORT = "results_tfidf/evidence_ann.npz"', 'EXPORT = "models/evidence_ann.npz"')
        s = s.replace('KERAS = "results_tfidf/evidence_ann.keras"', 'KERAS = "models/evidence_ann.keras"')
        s = s.replace('METRICS = "results_cnn/metrics.json"', 'METRICS = "results/cnn/metrics.json"')
        open(p, "w", encoding="utf8").write(s)

    open(os.path.join(dest, "README.md"), "w", encoding="utf8").write(README)
    open(os.path.join(dest, ".gitignore"), "w", encoding="utf8").write(GITIGNORE)
    copy("submission/requirements.txt", os.path.join(dest, "requirements.txt"))
    open(os.path.join(dest, "run_app.bat"), "w", encoding="utf8").write(
        "@echo off\r\ncd /d \"%~dp0\"\r\nstreamlit run app.py\r\npause\r\n")

    total = sum(len(fs) for _, _, fs in os.walk(dest))
    print(f"repo built at {dest}\n{total} files total\n")
    for root, dirs, fs in sorted(os.walk(dest)):
        rel = os.path.relpath(root, dest)
        if fs:
            print(f"  {('.' if rel == '.' else rel):<16} {len(fs):3d} file(s)  "
                  f"{', '.join(sorted(fs)[:6])}{' ...' if len(fs) > 6 else ''}")


if __name__ == "__main__":
    main()
