"""
Assemble a Hugging Face Space from the working folder.

The Space runs the application only, so it carries the two models the app loads and none of the
training code. torch and transformers are deliberately absent: nothing on the app's import path
needs them, which keeps the build small and fast.

    python make_space.py [--dest PATH]

Then, in the destination folder:

    git init && git add -A && git commit -m "SyllabiSense"
    git remote add origin https://huggingface.co/spaces/<user>/syllabisense
    git push -u origin main
"""
import argparse
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

APP = ["app.py", "analyze.py", "diagrams.py", "evidence.py", "quiz.py", "report.py", "depth.py",
       "loaders.py", "syllabus_topics.py", "units.py", "match.py"]
MODELS = [("results_tfidf/evidence_ann.npz", "results_tfidf/evidence_ann.npz"),
          ("results_tfidf/metrics.json", "results_tfidf/metrics.json"),
          ("results_tfidf/metrics_human.json", "results_tfidf/metrics_human.json"),
          ("results_cnn/diagram_cnn.keras", "results_cnn/diagram_cnn.keras"),
          ("results_cnn/metrics.json", "results_cnn/metrics.json"),
          ("dataset/tfidf_model.joblib", "dataset/tfidf_model.joblib")]

REQUIREMENTS = """streamlit==1.32.0
scikit-learn==1.5.2
numpy==1.26.4
joblib==1.4.2
matplotlib==3.9.2
reportlab==4.2.5
pypdfium2==4.30.0
Pillow==10.4.0
python-pptx==1.0.2
python-docx==1.1.2

# TensorFlow is optional: the evidence ANN runs in numpy (see evidence.py) and only the
# diagram CNN needs a framework. Without it the app runs and simply omits the diagram line.
# Uncomment on a host with the memory and a matching Python version:
# tensorflow-cpu==2.15.0
"""

README = """---
title: SyllabiSense
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# SyllabiSense

Checks whether course material actually covers a syllabus. For every syllabus topic it reports
**properly taught / barely taught / not taught**, citing the file and page as evidence.

Upload a syllabus and your course material (PDF, PowerPoint or Word), or tick the box to run the
built-in example. Name files with their unit — `Unit-2 notes.pdf` — and each unit is judged only
against its own material; a unit with no material is reported as *no material provided* rather
than as a gap in teaching.

## What is running

| Model | Job | Measured |
|---|---|---|
| Evidence ANN, 387 → 256 → 64 → 1 | is this page evidence for this topic? | 93.0% on an unseen subject area; **83.1% against hand-checked labels** |
| Diagram CNN, MobileNetV2 + trained head | does this page contain a diagram? | 84.2% on pages from held-out courses |

Trained on 17,887 (topic, page) pairs built from NPTEL, MIT OpenCourseWare and OpenStax. The split
is by **subject area**, not random, so the test subject is never seen during training.

## What it cannot do yet

The per-topic verdict — the line a student actually reads — is right about **56%** of the time on
101 hand-labelled topics. The errors run one way: the system never wrongly says *not taught*, but
it does call a topic taught when the page only mentions it in passing. Treat the report as a study
aid, not a verdict on the teaching.

Diagram detection reads PDF pages only; slides and Word files are reported as *not checked*, never
as *no diagram*. Scanned pages with no text layer are skipped.

Built for the AI & Deep Learning course at SR University by Uzma Yasmeen (2503A51L34).
"""

IGNORE = """__pycache__/
*.pyc
.streamlit/secrets.toml
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/Documents/GitHub/syllabisense-space"))
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    # clear the contents but keep .git: once the Space has been pushed, the folder is a
    # repository and wiping it would throw away the remote and the history
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(dest):
        if name == ".git":
            continue
        path = os.path.join(dest, name)
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)

    missing = []
    for f in APP:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(dest, f))
        else:
            missing.append(f)
    for src, rel in MODELS:
        if os.path.exists(src):
            os.makedirs(os.path.join(dest, os.path.dirname(rel)), exist_ok=True)
            shutil.copy(src, os.path.join(dest, rel))
        else:
            missing.append(src)
    if os.path.isdir("demo_files"):
        shutil.copytree("demo_files", os.path.join(dest, "demo_files"))

    open(os.path.join(dest, "requirements.txt"), "w", encoding="utf8").write(REQUIREMENTS)
    open(os.path.join(dest, "README.md"), "w", encoding="utf8").write(README)
    open(os.path.join(dest, ".gitignore"), "w", encoding="utf8").write(IGNORE)

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(dest) for f in fs)
    n = sum(len(fs) for _, _, fs in os.walk(dest))
    print(f"Space assembled at {dest}")
    print(f"  {n} files, {total / 1048576:.1f} MB")
    if missing:
        print("  MISSING (the Space will not run without these):")
        for m in missing:
            print(f"    {m}")
    big = [(os.path.join(r, f), os.path.getsize(os.path.join(r, f)))
           for r, _, fs in os.walk(dest) for f in fs
           if os.path.getsize(os.path.join(r, f)) > 10 * 1024 * 1024]
    for path, size in big:
        print(f"  note: {os.path.relpath(path, dest)} is {size / 1048576:.1f} MB "
              f"(fine for Spaces; over 10 MB git may warn)")


if __name__ == "__main__":
    main()
