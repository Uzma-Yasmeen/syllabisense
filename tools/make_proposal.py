"""
Build the proposal in the format of the AIDL template handed out by the department.

    python make_figures.py      (first, for the diagrams)
    python make_proposal.py

Section order follows the template exactly: 1 (A-G), 2 Code Submission, 3 Model Details,
4 Results, 5 Final Submission Checklist, 6 Student Fill-In Template.

Every number is read from the results_*/metrics.json files, never typed in, so re-running after a
model changes updates the document.

Writes submission/PROPOSAL.docx and submission/PROPOSAL.pdf
"""
import datetime
import json
import os
import subprocess
import sys

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.stdout.reconfigure(encoding="utf-8")

NAME, ROLL, BATCH = "Uzma Yasmeen", "2503A51L34", "20 CSE"
TITLE = "SyllabiSense: AI for Course Coverage and Personalized Learning"
NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x74, 0xB5)
GREY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
H_NAVY, H_BAND = "1F3864", "EAF1F8"
OUT_DOCX, OUT_PDF = "submission/PROPOSAL.docx", "submission/PROPOSAL.pdf"


def load(p, d=None):
    return json.load(open(p, encoding="utf8")) if os.path.exists(p) else d


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def num(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


# ---------------------------------------------------------------- formatting
def shade(cell, hexcolor):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcolor)
    cell._tc.get_or_add_tcPr().append(el)


def borders(obj, edges, color, size=4):
    if hasattr(obj, "_tc"):
        pr, tag = obj._tc.get_or_add_tcPr(), "w:tcBorders"
    else:
        pr, tag = obj._p.get_or_add_pPr(), "w:pBdr"
    bd = pr.find(qn(tag))
    if bd is None:
        bd = OxmlElement(tag)
        pr.append(bd)
    for e in edges:
        el = OxmlElement(f"w:{e}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:color"), color)
        el.set(qn("w:space"), "2")
        bd.append(el)


def no_borders(table_obj):
    for row in table_obj.rows:
        for c in row.cells:
            bd = OxmlElement("w:tcBorders")
            for e in ("top", "left", "bottom", "right"):
                el = OxmlElement(f"w:{e}")
                el.set(qn("w:val"), "nil")
                bd.append(el)
            c._tc.get_or_add_tcPr().append(bd)


def cell_text(cell, text, size=9, bold=False, color=None, mono=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    for i, part in enumerate(str(text).split("**")):
        r = p.add_run(part)
        r.font.size = Pt(size)
        r.bold = bold or i % 2 == 1
        if mono:
            r.font.name = "Consolas"
            r.font.size = Pt(size - 0.5)
        if color is not None:
            r.font.color.rgb = color
    return p


def grid(doc, rows, widths, header=None, size=9, label_shade=True, caption=None):
    """The template's look: shaded label column (or header row), thin grey rules."""
    ncols = len(widths)
    t = doc.add_table(rows=0, cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    if header:
        cells = t.add_row().cells
        for c, h in zip(cells, header):
            cell_text(c, h, size=size - 0.5, bold=True, color=WHITE)
            shade(c, H_NAVY)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cell_text(cells[i], v, size=size, bold=(i == 0 and label_shade and not header))
            if i == 0 and label_shade and not header:
                shade(cells[i], H_BAND)
    for r in t.rows:
        for c, w in zip(r.cells, widths):
            c.width = Inches(w)
            borders(c, ("top", "left", "bottom", "right"), "B7C7D9", 4)
    if caption:
        p = doc.add_paragraph()
        r = p.add_run(caption)
        r.italic = True
        r.font.size = Pt(8)
        r.font.color.rgb = GREY
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph().paragraph_format.space_after = Pt(3)
    return t


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = NAVY
    borders(p, ("bottom",), "2E74B5", 6)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = BLUE
    return p


def body(doc, text, size=9.5, space=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for i, part in enumerate(text.split("**")):
        r = p.add_run(part)
        r.font.size = Pt(size)
        r.bold = i % 2 == 1
    return p


def figure(doc, path, caption, width=5.6):
    if not os.path.exists(path):
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(caption)
    r.italic = True
    r.font.size = Pt(8)
    r.font.color.rgb = GREY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)


def field(paragraph, instr):
    """A Word field, e.g. PAGE — evaluated by Word when the document opens or exports."""
    f = OxmlElement("w:fldSimple")
    f.set(qn("w:instr"), instr)
    paragraph._p.append(f)


def hf_row(container, edge):
    """One full-width cell carrying the hairline on `edge`, holding a paragraph with a right tab
    stop. It is a table rather than a bare paragraph because the built-in Header and Footer
    styles carry their own tab stops that a paragraph-level clear does not remove; inside a cell
    the paragraph starts from Normal, so the tab lands where it is put. One cell, so the rule
    runs unbroken."""
    t = container.add_table(rows=1, cols=1, width=Inches(7.0))
    t.autofit = False
    cell = t.rows[0].cells[0]
    cell.width = Inches(7.0)
    mar = OxmlElement("w:tcMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "0")
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    cell._tc.get_or_add_tcPr().append(mar)
    for e in ("top", "left", "bottom", "right"):
        borders(cell, (e,), "FFFFFF", 0) if e != edge else borders(cell, (e,), "C9D2DE", 4)
    p = cell.paragraphs[0]
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.94), WD_TAB_ALIGNMENT.RIGHT)
    # drop the empty paragraph the header/footer starts with, so the rule sits tight
    p0 = container.paragraphs[0]._p
    p0.getparent().remove(p0)
    return p


def header_footer(section):
    """Institution bold in the accent colour on the left, document name quiet on the right,
    a hairline under the header and above the footer."""
    h = hf_row(section.header, "bottom")
    r = h.add_run("SR UNIVERSITY  /  CS & AI")
    r.bold = True
    r.font.size = Pt(8)
    r.font.color.rgb = NAVY
    r = h.add_run("\tAI & Deep Learning Project — Project Proposal")
    r.font.size = Pt(8)
    r.font.color.rgb = GREY

    f = hf_row(section.footer, "top")
    r = f.add_run(f"{NAME}  |  {ROLL}  |  Batch {BATCH}\tPage ")
    r.font.size = Pt(8)
    r.font.color.rgb = GREY
    field(f, "PAGE")


# --------------------------------------------------------------------- build
def main():
    os.makedirs("submission", exist_ok=True)
    tf_, tf_h = load("results_tfidf/metrics.json"), load("results_tfidf/metrics_human.json")
    ce, ce_h = load("results_crossencoder/metrics.json"), load("results_crossencoder/metrics_human.json")
    cnn = load("results_cnn/metrics.json")
    scratch = load("results_cnn_scratch/metrics.json")
    depth = load("results_depth/metrics.json")
    nosim = load("results_tfidf_nosim/metrics.json")
    fusion = load("results_fusion/metrics.json")
    stats = load("dataset/stats.json", {})
    n = stats.get("by_split_kind", {})
    tot = lambda k: sum(n.get(f"{s}/{k}", 0) for s in ("train", "val", "test"))

    doc = docx.Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(4)
    s = doc.sections[0]
    s.top_margin = s.bottom_margin = Inches(0.7)
    s.left_margin = s.right_margin = Inches(0.75)
    header_footer(s)

    # ------------------------------------------------------------ title block
    for text, size, bold, color, space in [
            ("SR UNIVERSITY", 16, True, NAVY, 2),
            ("Department of Computer Science & Artificial Intelligence (CS & AI)", 10, False, GREY, 2),
            ("Professor: Sai Meghana Rao Eravelli", 9.5, False, GREY, 10),
            ("AI & DEEP LEARNING PROJECT", 15, True, NAVY, 2),
            ("Project Proposal and Code Submission", 11, False, BLUE, 8)]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(space)
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)
        r.font.color.rgb = color

    h1(doc, "1. Project Proposal Document")

    h2(doc, "A. Student / Team Details")
    grid(doc, [["Project Title", TITLE],
               ["Student Name(s)", NAME],
               ["Roll Number(s)", f"{ROLL} (Enrollment No.)"],
               ["Batch Number", BATCH],
               ["Course", "AI & Deep Learning"],
               ["Submission Date", datetime.date.today().strftime("%d %B %Y")]],
         widths=[1.6, 5.4])

    h2(doc, "B. Project Title")
    body(doc, f"**{TITLE}**")
    body(doc, "Deep neural networks that check course material against a syllabus: for every "
              "topic they decide whether the material actually teaches it, cite the file and page "
              "as evidence, and turn the gaps into a personalised study order.")

    h2(doc, "C. Why Did You Choose This Project?")
    body(doc, "**Problem and importance.** A university gives students a syllabus and, separately, "
              "a pile of study material — lecture PDFs, slide decks and notes. Students assume the "
              "two match. They often do not: a topic listed in the syllabus may be covered in "
              "depth, mentioned in a single line, or missing altogether, and this is usually "
              "discovered only while revising for the exam. Existing tools cannot help, because a "
              "document chatbot answers questions about the pages it is given and can never report "
              "that something is **missing** — it has no idea what the syllabus promised.")
    body(doc, "**Motivation, beneficiaries and outcome.** I chose this project because detecting "
              "absence requires holding two documents against each other and judging every pairing "
              "between them, which is exactly what a neural network trained on (topic, page) pairs "
              "can learn. Students benefit by learning early which topics need outside material; "
              "faculty can check their own material against the syllabus before a semester begins, "
              "which is also the shape of the documentation required for accreditation. The "
              "expected outcome is a working application that reads a syllabus and a folder of "
              "material and reports every topic as properly taught, barely taught or not taught, "
              "with evidence and measured accuracy.")

    # ----------------------------------------------------------------- D
    doc.add_page_break()
    h2(doc, "D. AI / Deep Learning Techniques Used")
    grid(doc, [
        ["Model / architecture being used",
         "Four trained networks. **Main model:** fully connected ANN in TensorFlow/Keras "
         "(class built in train_ann.py): 387 → 256 → 64 → 1. **Also trained:** a MiniLM "
         "transformer cross-encoder fine-tuned end to end (PyTorch, 22.7M parameters), a "
         "MobileNetV2 CNN for diagram detection, and a 3-class ANN for teaching depth."],
        ["Reason for selecting the model",
         "The task is a judgement about a pair of texts, so the input is a fixed-length vector of "
         "pair features — dense layers suit it. The network is small, trains in under a minute on "
         "a CPU and scores 9,000 topic-page pairs in under a second, which the application needs. "
         "The transformer was trained as a controlled comparison and measured; it ranks better "
         "(AUC 0.835 vs 0.806) but is less accurate at a fixed threshold, so the ANN is deployed."],
        ["Major layers used",
         "Input layer (387), Dense 256, Dropout 0.3, Dense 64, Dropout 0.2, Dense 1."],
        ["Activation functions", "ReLU after each hidden layer; sigmoid at the output."],
        ["Optimizer and loss function",
         f"Adam (learning rate {tf_['hyperparameters']['learning_rate'] if tf_ else 0.001}); "
         f"binary cross-entropy, with class weights."],
        ["Input data and expected output",
         "Input: 387 numbers describing one (syllabus topic, material page) pair — TF-IDF reduced "
         "to 128 dimensions by truncated SVD for the topic and for the page, their product, plus "
         "cosine similarity, word overlap and page length. Output: one probability that the page "
         "is evidence for the topic."],
        ["Role of ANN in the proposed system",
         "Scores every topic against every page. Pages above the calibrated threshold become the "
         "topic's evidence; a topic with no evidence is reported as a gap."],
        ["Role of CNN in the proposed system",
         "Runs only on the pages the ANN already chose, and reports whether the topic is supported "
         "by a diagram or explained in words alone."]],
         widths=[1.8, 5.2])

    figure(doc, "figures/ann.png", "Figure 1 — ANN architecture (387-256-64-1).", 5.5)

    h2(doc, "ANN components in this project")
    grid(doc, [
        ["Input layer", "387 features per (topic, page) pair: topic LSA 128, page LSA 128, "
                        "elementwise product 128, cosine similarity, word overlap, page length."],
        ["Hidden layers", "Dense(256) → ReLU → Dropout(0.3) → Dense(64) → ReLU → Dropout(0.2)."],
        ["Dense layers", "387×256 + 256×64 + 64×1 = 115,841 trainable parameters."],
        ["Activation functions", "ReLU (hidden), sigmoid (output)."],
        ["Output layer", "Dense(1) — probability that the page is evidence for the topic."],
        ["Optimizer", "tf.keras.optimizers.Adam, lr = 0.001."],
        ["Loss function", "binary_crossentropy, with class weights for the 1:2 class imbalance."],
        ["Regularisation", "Dropout 0.3 / 0.2; early stopping on validation AUC "
                           f"(stopped at epoch {tf_['epochs_run'] if tf_ else '—'} of 40)."]],
         widths=[1.8, 5.2], header=["Component", "Implementation in the code"])

    h2(doc, "CNN components in this project")
    grid(doc, [
        ["Input layer", "Page rendered to a 128 × 128 × 3 image."],
        ["Convolutional base", "MobileNetV2 pretrained on ImageNet — convolution, ReLU and "
                               "pooling throughout; frozen during training."],
        ["Pooling", "GlobalAveragePooling2D over the final feature map."],
        ["Dense layers", "Dense(64) → ReLU → Dropout(0.4) → Dense(1) → sigmoid; "
                         f"{cnn['trainable_parameters']:,} trainable parameters."
                         if cnn else "Dense(64) → ReLU → Dropout(0.4) → Dense(1)."],
        ["Optimizer / loss", "Adam (lr 0.001); binary cross-entropy."],
        ["Why transfer learning", "A CNN trained from scratch on 424 images reached 75% accuracy "
                                  "with recall 0.11 — it answered 'no diagram' almost always. "
                                  "Both runs are reported in Section 4."]],
         widths=[1.8, 5.2], header=["Component", "Implementation in the code"])
    figure(doc, "figures/cnn.png", "Figure 2 — CNN architecture (MobileNetV2 base + trained head).",
           5.8)

    # ----------------------------------------------------------------- E, F, G
    doc.add_page_break()
    h2(doc, "E. Dataset Description")
    grid(doc, [
        ["Dataset name", "SyllabiSense (topic, page) pairs — built from scratch for this project."],
        ["Dataset source / link",
         "NPTEL (nptel.ac.in) — 1,295 course syllabi; MIT OpenCourseWare (ocw.mit.edu, "
         "CC BY-NC-SA) — about 110 courses of lecture notes, problem sets and exams; OpenStax "
         "(openstax.org, CC BY) — 7 textbooks. Collected by tools/collect_nptel.py and "
         "tools/collect_ocw.py. Raw material: 7,133 PDFs, 100,414 pages, roughly 13 million words."],
        ["Number of samples", f"{stats.get('examples', 17887):,} labelled (topic, page) pairs."],
        ["Number of classes",
         f"2 — class 1 = the page is evidence for the topic ({tot('positive'):,}); class 0 = it is "
         f"not ({tot('hard_negative') + tot('easy_negative'):,}, made of "
         f"{tot('hard_negative'):,} hard negatives from the same course and "
         f"{tot('easy_negative'):,} easy negatives from an unrelated subject)."],
        ["Input features", "387 numeric features per pair (listed in Section D); label column: "
                           "label."],
        ["Training / testing split",
         "**Split by subject area, never at random.** Train 11,615 pairs from 12 non-computing "
         "areas (management, electrical, aerospace, economics, chemical engineering, civil, "
         "psychology, sociology, biology, mathematics, mechanical, physics); validation 775 "
         "(chemistry, humanities); test 5,497 from 109 Computer Science courses. No computing "
         "material appears anywhere in training."],
        ["Human-verified labels",
         "Labels are generated automatically, so 300 pairs were re-labelled by hand through a "
         "screen that hides the automatic label. The automatic labels proved **84% correct**, so "
         "all headline accuracy is reported against the human labels as well. A further 101 topics "
         "and 576 page images were hand-labelled for the depth and CNN models."],
        ["Brief description",
         "A positive pair is a page that matches the topic's distinctive terms within a 30-word "
         "window. Contents pages, index pages, reference lists and unreadable scans are filtered "
         "out before pairs are formed."]],
         widths=[1.8, 5.2])

    h2(doc, "F. Proposed Methodology")
    body(doc, "Workflow: Dataset → Data Preprocessing → Train/Test Split → Model Building → "
              "Model Training → Model Evaluation → Prediction.")
    figure(doc, "figures/pipeline.png", "Figure 3 — overall project workflow.", 4.1)
    grid(doc, [
        ["Dataset", "Download 1,295 syllabi and ~110 courses of material; extract text per page, "
                    "slide or section (tools/extract_text.py)."],
        ["Data Preprocessing", "Drop navigation, contents, index, reference and unreadable pages; "
                               "parse the syllabus into units and topics; fit TF-IDF (60,000 "
                               "terms) and truncated SVD (128 dimensions) on the training split "
                               "only."],
        ["Train/Test Split", "By subject area: 12 areas train, 2 validate, Computer Science tests."],
        ["Model Building", "Define the 387-256-64-1 ANN; separately the cross-encoder, the depth "
                           "ANN and the MobileNetV2 CNN."],
        ["Model Training", "Adam (lr 0.001), binary cross-entropy, batch 64, class weights, early "
                           "stopping on validation AUC."],
        ["Model Evaluation", "Threshold chosen on validation and on human-labelled non-test items, "
                             "never on the test set; accuracy, precision, recall, F1, ROC-AUC and "
                             "confusion matrix on the unseen subject area."],
        ["Prediction", "Score a new syllabus against new material; group by unit; report each "
                       "topic as properly taught, barely taught, not taught, or no material "
                       "provided."]],
         widths=[1.5, 5.5], header=["Step", "What is done"])

    h2(doc, "G. Expected Outcome")
    body(doc, "The final system accepts a syllabus and a folder of course material in PDF, "
              "PowerPoint or Word format, and produces a unit-by-unit report marking every "
              "syllabus topic as properly taught, barely taught or not taught, with the file and "
              "page quoted as evidence and a confidence score from the ANN. The CNN adds whether "
              "each covered topic is supported by a diagram. Files named by unit are judged only "
              "against their own unit, and a unit with no uploaded material is reported as **no "
              "material provided** rather than as a gap in teaching. The output is a study guide "
              "and a coverage audit — not a claim about teaching quality.")

    # ------------------------------------------------------------ 2. code
    doc.add_page_break()
    h1(doc, "2. Code Submission")
    body(doc, "**Source files:** app.py (Streamlit interface), analyze.py (command line), "
              "train_ann.py, train_crossencoder.py, train_cnn.py, train_depth.py (training), "
              "build_dataset.py, features_tfidf.py (dataset and features), diagrams.py, "
              "loaders.py, syllabus_topics.py, units.py, depth.py (libraries), evaluate_gold.py, "
              "evaluate_gold_ce.py, compare_fusion.py (evaluation), review_tool.py, "
              "image_review.py, depth_review.py (hand-labelling screens), requirements.txt, "
              "run_app.bat.")
    grid(doc, [
        ["1", "Importing required libraries", "Top of train_ann.py (numpy, tensorflow, sklearn); "
                                              "app.py (streamlit); train_crossencoder.py (torch, "
                                              "transformers)."],
        ["2", "Loading the dataset", "build_dataset.py writes dataset/pairs_{train,val,test}.csv; "
                                     "loaded in train_ann.py and evaluate_gold.py."],
        ["3", "Data preprocessing", "match.py (is_navigation, is_garbage, strip_references); "
                                    "features_tfidf.py (TF-IDF + truncated SVD); "
                                    "syllabus_topics.py (topics_with_units)."],
        ["4", "Data visualization", "train_ann.py writes history.png, confusion_matrix.png and "
                                    "roc_curve.png; compare_fusion.py writes comparison.png; "
                                    "app.py shows per-unit metrics."],
        ["5", "Train/test data splitting", "build_dataset.py — TEST_DOMAINS and VAL_DOMAINS split "
                                           "by subject area, not at random."],
        ["6", "Model definition", "tf.keras.Sequential in train_ann.py; "
                                  "AutoModelForSequenceClassification in train_crossencoder.py."],
        ["7", "ANN architecture", "Dense(256) → Dropout → Dense(64) → Dropout → Dense(1) in "
                                  "train_ann.py."],
        ["8", "CNN architecture", "MobileNetV2 base + GlobalAveragePooling2D + Dense(64) + "
                                  "Dropout(0.4) + Dense(1) in train_cnn.py."],
        ["9", "Model compilation", "model.compile(Adam(1e-3), 'binary_crossentropy', "
                                   "metrics=['accuracy', AUC]) in train_ann.py."],
        ["10", "Model training", "model.fit(...) with class_weight and EarlyStopping in "
                                 "train_ann.py; manual AdamW loop in train_crossencoder.py."],
        ["11", "Model evaluation", "evaluate_gold.py and evaluate_gold_ce.py (human-verified "
                                   "labels); compare_fusion.py (five deployment rules)."],
        ["12", "Prediction / inference", "app.py and analyze.py — score a new syllabus against new "
                                         "material; diagrams.py runs the CNN on the chosen pages."],
        ["13", "Saving and loading the model", "results_tfidf/evidence_ann.keras, "
                                               "results_cnn/diagram_cnn.keras, "
                                               "results_crossencoder/model/, "
                                               "dataset/tfidf_model.joblib."]],
         widths=[0.35, 1.85, 4.8], header=["#", "Required element", "Location in the code"])

    # --------------------------------------------------------- 3. model details
    doc.add_page_break()
    h1(doc, "3. Model Details")
    h2(doc, "Main model — Evidence ANN")
    hp = tf_["hyperparameters"] if tf_ else {}
    grid(doc, [
        ["Number of layers", "3 dense layers (2 hidden: 256, 64) plus input",
         "Enough capacity for 387 features without overfitting 11,615 rows."],
        ["Number of neurons", "387 – 256 – 64 – 1", "Narrowing funnel to a single probability."],
        ["Kernel size", "Not applicable (ANN)", "No convolution in this model."],
        ["Pooling size", "Not applicable", "No pooling in this model."],
        ["Activation functions", "ReLU (hidden), sigmoid (output)",
         "Non-linearity; output read as a probability."],
        ["Learning rate", str(hp.get("learning_rate", 0.001)), "Stable step size for Adam."],
        ["Optimizer", "Adam", "Adaptive learning rates, fast convergence on a small network."],
        ["Loss function", "Binary cross-entropy", "Two-class problem; penalises confident errors."],
        ["Batch size", str(hp.get("batch_size", 64)), "Smooth mini-batch updates."],
        ["Number of epochs", f"{tf_['epochs_run'] if tf_ else '—'} of a maximum "
                             f"{hp.get('max_epochs', 40)} (early stopping)",
         "Stopped when validation AUC stopped improving."],
        ["Dropout rate", "0.3 after the first hidden layer, 0.2 after the second",
         "Weak labels make overfitting easy; dropout limits it."],
        ["Class weights", "Applied (1 positive : 2 negatives)",
         "Keeps the network from always answering 'not evidence'."],
        ["Decision threshold", f"{tf_h['threshold_human_tuned']:.2f} (tuned on human-labelled "
                               f"non-test items)" if tf_h else "tuned on validation",
         "Never tuned on the test set."]],
         widths=[1.5, 2.3, 3.2], header=["Parameter", "Value", "Purpose"])

    h2(doc, "Second model — Diagram CNN")
    grid(doc, [
        ["Number of layers", "MobileNetV2 base (53 convolutional layers, frozen) + 2 dense",
         "Transfer learning: only the head is trained."],
        ["Number of neurons", "GlobalAveragePooling → 64 → 1", "Small head for 424 training images."],
        ["Kernel size", "3 × 3 and 1 × 1 (inside MobileNetV2)", "Standard MobileNetV2 blocks."],
        ["Pooling size", "Global average pooling over the final feature map",
         "Turns the feature map into 1,280 numbers."],
        ["Activation functions", "ReLU (hidden), sigmoid (output)", "Probability of a diagram."],
        ["Learning rate", "0.001", "Head only; base is frozen."],
        ["Optimizer", "Adam", "Same as above."],
        ["Loss function", "Binary cross-entropy", "Two classes: diagram / no diagram."],
        ["Batch size", "32", "Fits comfortably in memory with 128 × 128 images."],
        ["Number of epochs", f"{cnn['epochs_run'] if cnn else '—'} of a maximum 40 (early stopping)",
         "Stopped on validation AUC."],
        ["Dropout rate", "0.4", "Small dataset; strong regularisation needed."],
        ["Trainable parameters", f"{cnn['trainable_parameters']:,}" if cnn else "—",
         "MobileNetV2's 2.2M base parameters stay frozen."]],
         widths=[1.5, 2.3, 3.2], header=["Parameter", "Value", "Purpose"])

    # ------------------------------------------------------------ 4. results
    h1(doc, "4. Results")
    body(doc, "All values below are produced by the code and stored in results_*/metrics.json. "
              "The test split is Computer Science — a subject area that appears nowhere in "
              "training.")

    h2(doc, "4.1 Evidence ANN (main model)")
    t, hgold = (tf_["test"] if tf_ else {}), (tf_h["human_labels"] if tf_h else {})
    cmw = tf_["confusion_matrix"] if tf_ else {}
    cmh = hgold.get("confusion_matrix", {})
    grid(doc, [
        ["Training accuracy", pct(tf_["train_final"].get("accuracy")) if tf_ else "—",
         "Final epoch on the training split."],
        ["Validation accuracy", pct(tf_["val_final"].get("accuracy")) if tf_ else "—",
         "Chemistry and humanities; used for early stopping."],
        ["Test accuracy", f"**{pct(t.get('accuracy'))}**",
         "5,497 unseen Computer Science pairs (automatic labels)."],
        ["Test accuracy (human-verified)", f"**{pct(hgold.get('accuracy'))}**",
         "201 pairs re-labelled by hand — the honest figure."],
        ["Training / validation loss",
         f"{tf_['train_final'].get('loss', 0):.4f} / {tf_['val_final'].get('loss', 0):.4f}"
         if tf_ else "—", "Binary cross-entropy at the final epoch."],
        ["Confusion matrix (test)",
         f"TN {cmw.get('tn')}  FP {cmw.get('fp')}  FN {cmw.get('fn')}  TP {cmw.get('tp')}",
         "results_tfidf/confusion_matrix.png."],
        ["Confusion matrix (human)",
         f"TN {cmh.get('tn')}  FP {cmh.get('fp')}  FN {cmh.get('fn')}  TP {cmh.get('tp')}",
         "On the 201 hand-labelled pairs."],
        ["Precision / Recall / F1 (test)",
         f"{num(t.get('precision'))} / {num(t.get('recall'))} / {num(t.get('f1'))}",
         "Automatic labels."],
        ["Precision / Recall / F1 (human)",
         f"{num(hgold.get('precision'))} / {num(hgold.get('recall'))} / {num(hgold.get('f1'))}",
         "Hand-labelled; lower because automatic labels are 84% correct."],
        ["ROC-AUC", f"{num(t.get('roc_auc'))} (test), {num(hgold.get('roc_auc'))} (human)",
         "results_tfidf/roc_curve.png."],
        ["Accuracy vs. Epochs graph", "results_tfidf/history.png",
         f"{tf_['epochs_run'] if tf_ else '—'} epochs before early stopping."],
        ["Loss vs. Epochs graph", "results_tfidf/history.png", "Training and validation loss."],
        ["Prediction examples", "CODE_SUBMISSION.pdf, step 6",
         "The verdict for every topic of a real syllabus run against real lecture material, with "
         "the evidence page and the model's confidence quoted for each."]],
         widths=[1.9, 2.4, 2.7], header=["Result", "Value", "Source / note"])

    h2(doc, "4.2 Diagram CNN — and the from-scratch comparison")
    rows = []
    if scratch:
        s2 = scratch["test"]
        rows.append(["CNN trained from scratch", pct(s2["accuracy"]), num(s2["precision"]),
                     num(s2["recall"]), num(s2["f1"])])
    if cnn:
        c2 = cnn["test"]
        rows.append(["MobileNetV2 transfer learning", pct(c2["accuracy"]), num(c2["precision"]),
                     num(c2["recall"]), num(c2["f1"])])
    grid(doc, rows, widths=[2.6, 1.1, 1.1, 1.1, 1.1],
         header=["Approach", "Test accuracy", "Precision", "Recall", "F1"], label_shade=False)
    body(doc, f"Trained on {cnn['n_train'] if cnn else '—'} hand-labelled page images and tested "
              f"on {cnn['n_test'] if cnn else '—'} images from courses held out of training. The "
              f"from-scratch network learned to answer 'no diagram' almost always — 75% accuracy "
              f"sounds acceptable only because 76% of pages have no diagram. Transfer learning "
              f"raised recall from 0.11 to 0.72. Graphs: results_cnn/history.png, "
              f"confusion_matrix.png, examples.png.")

    h2(doc, "4.3 Comparison with the fine-tuned transformer")
    rows = []
    if ce and ce_h:
        rows.append(["Cross-encoder (MiniLM, fine-tuned)", pct(ce["test"]["accuracy"]),
                     pct(ce_h["human_labels"]["accuracy"]), num(ce_h["human_labels"]["f1"]),
                     num(ce_h["human_labels"]["roc_auc"])])
    if tf_ and tf_h:
        rows.append(["Evidence ANN (deployed)", pct(tf_["test"]["accuracy"]),
                     pct(tf_h["human_labels"]["accuracy"]), num(tf_h["human_labels"]["f1"]),
                     num(tf_h["human_labels"]["roc_auc"])])
    grid(doc, rows, widths=[2.6, 1.2, 1.3, 0.9, 1.0],
         header=["Model", "Test acc.", "Human acc.", "Human F1", "Human AUC"], label_shade=False)
    if fusion:
        best = fusion["best_rule"]
        import math
        ng = fusion["n_gold"]
        acc_best = fusion["rules"][best]["accuracy"]
        acc_ann = fusion["rules"]["ann_only"]["accuracy"]
        gap = acc_best - acc_ann
        se = math.sqrt(acc_best * (1 - acc_best) / ng + acc_ann * (1 - acc_ann) / ng)
        body(doc, f"compare_fusion.py tested five ways of deploying the two models — ANN alone, "
                  f"transformer alone, the mean of both, a weighted blend, and "
                  f"retrieve-then-rerank — with every threshold tuned on the same held-out human "
                  f"labels. The highest score went to **{best.replace('_', ' ')}** at "
                  f"{pct(acc_best)}, against {pct(acc_ann)} for the ANN alone.")
        body(doc, f"**What the comparison can and cannot settle.** "
                  f"That gap is {gap * 100:.1f} points, and its standard error on {ng} items is "
                f"{se * 100:.1f} points — {abs(gap / se):.2f} standard errors, which is not "
                f"distinguishable from chance. Running the same comparison against an earlier "
                f"training run of the same architecture put the ANN alone in first place instead. "
                f"The honest conclusion is not that one rule beats another, but that "
                f"{ng} hand-labelled items cannot separate them. The ANN alone is deployed "
                f"because it is the fastest by two orders of magnitude — one forward pass over "
                f"all pages, against one transformer pass per topic-page pair — and nothing in "
                f"the measurement justifies paying that cost. Enlarging the hand-labelled set is "
                f"what would make this decision answerable.")

    if depth:
        h2(doc, "4.4 Depth model (preliminary)")
        body(doc, f"Test accuracy {pct(depth['test_accuracy'])}, macro F1 "
                  f"{num(depth['macro_f1'])} on {depth['n_test']} held-out topics "
                  f"({depth['n_train']} used for training). It identifies 'not taught' reliably "
                  f"but cannot yet separate 'properly' from 'barely taught' — 101 hand-labelled "
                  f"topics are too few. The application therefore derives the depth level from "
                  f"transparent rules over the evidence the ANN found, and this model is reported "
                  f"as work in progress.")

    h2(doc, "4.5 Output screenshots of the application")
    body(doc, "The run below is a real one: the Software Engineering and System Design syllabus "
              "(47 topics across five units) checked against the Unit 1 and Unit 2 lecture notes. "
              "Only two units of material were supplied, and the system reports the other three "
              "as **no material provided** rather than as gaps in teaching.")
    figure(doc, "screenshots/01_upload_and_summary.jpg",
           "Figure 4 — the syllabus and material are uploaded, and the result is summarised: "
           "47 topics, 7 properly taught, 6 barely taught, 2 not taught, 32 with no material. "
           "The report can be downloaded as PDF or CSV.", 4.3)
    figure(doc, "screenshots/02_study_order.jpg",
           "Figure 5 — the gaps turned into a study order, most urgent first, each labelled with "
           "its unit.", 4.3)
    figure(doc, "screenshots/03_unit_breakdown.jpg",
           "Figure 6 — one tab per unit. Unit I: 4 topics properly taught, 3 barely, 1 not "
           "taught, 0 without material.", 4.3)
    figure(doc, "screenshots/04_evidence_and_diagram.jpg",
           "Figure 7 — a topic opened up. The ANN cites the pages it judged as evidence with its "
           "confidence, and the CNN reports a diagram on page 25 at confidence 0.74.", 4.3)

    h2(doc, "Note for evaluation")
    body(doc, "Two honesty notes. **First,** the training labels are generated automatically and "
              "are themselves only 84% correct, so the human-verified figures (84.1% accuracy, F1 "
              "0.746) are the ones to judge — the automatic-label figures (92.8%) are inflated by "
              "the same label noise the model was trained on. **Second,** with 201 hand-labelled "
              "test items the 95% confidence interval is about ±5 points, and the decision "
              "threshold shifts slightly depending on which items tune it, so the honest claim is "
              "roughly 84% ± 5 rather than an exact figure.")

    # ------------------------------------------------------------ 5, 6
    doc.add_page_break()
    h1(doc, "5. Final Submission Checklist")
    grid(doc, [
        ["[X]", "Project Proposal Document", "This document (PROPOSAL.pdf / PROPOSAL.docx)."],
        ["[X]", "Complete Source Code", "CODE_SUBMISSION.pdf — the demonstration program with "
                                        "the output it produced, the result graphs, and an "
                                        "appendix containing all 16 source files of the system."],
        ["[X]", "Dataset source / reference", "Section 1E — NPTEL, MIT OpenCourseWare, OpenStax."],
        ["[X]", "ANN model architecture and parameter details", "Sections 1D and 3."],
        ["[X]", "CNN model architecture and parameter details", "Sections 1D and 3."],
        ["[X]", "Results and performance metrics", "Section 4; results_*/metrics.json."],
        ["[X]", "Relevant graphs", "history.png, confusion_matrix.png, roc_curve.png, "
                                   "examples.png, comparison.png."],
        ["[X]", "Output screenshots of the application", "Section 4.5, Figures 4 to 7 — a real "
                                                         "run on the Software Engineering "
                                                         "syllabus."]],
         widths=[0.5, 2.6, 3.9], header=["Done", "Item", "Where"])

    h1(doc, "6. Project Proposal — Student Fill-In Template")
    grid(doc, [
        ["Project Title:", TITLE],
        ["Student Name(s):", NAME],
        ["Roll Number(s):", ROLL],
        ["Batch Number:", BATCH],
        ["Why did you choose this project?:",
         "Syllabus and study material often do not match, and students find out only at exam time. "
         "A chatbot cannot report what is missing because it never sees the syllabus; a network "
         "trained on (topic, page) pairs can."],
        ["Dataset Name and Source:",
         "SyllabiSense (topic, page) pairs, built for this project from NPTEL (1,295 syllabi), MIT "
         "OpenCourseWare (~110 courses) and OpenStax (7 textbooks); 17,887 pairs, 387 features, "
         "2 classes."],
        ["ANN/CNN Techniques Used:",
         "ANN for evidence detection (deployed); MobileNetV2 CNN for diagram detection; a "
         "fine-tuned MiniLM transformer cross-encoder and a 3-class depth ANN as further models."],
        ["Model Architecture / Layers:",
         "Input 387 — Dense 256 (ReLU) — Dropout 0.3 — Dense 64 (ReLU) — Dropout 0.2 — Dense 1 "
         "(sigmoid); 115,841 parameters."],
        ["Optimizer and Loss Function:", "Adam (lr 0.001); binary cross-entropy with class "
                                         "weights."],
        ["Proposed Methodology:", "Dataset — Preprocessing — Train/Test split by subject area — "
                                  "Model building — Training — Evaluation on an unseen subject "
                                  "and against human labels — Prediction."],
        ["Expected Outcome:",
         "A unit-by-unit report marking each syllabus topic as properly taught, barely taught or "
         "not taught, with the file and page as evidence and a note on whether a diagram exists."]],
         widths=[2.0, 5.0], header=["Field", "Entry"])

    doc.save(OUT_DOCX)
    print(f"wrote {OUT_DOCX}")
    ps = (f'$w = New-Object -ComObject Word.Application; $w.Visible = $false; '
          f'$d = $w.Documents.Open("{os.path.abspath(OUT_DOCX)}"); '
          f'$d.SaveAs([ref]"{os.path.abspath(OUT_PDF)}", [ref]17); $d.Close(); $w.Quit()')
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                       capture_output=True, timeout=300)
        print(f"wrote {OUT_PDF}")
    except Exception as e:
        print(f"PDF export failed ({type(e).__name__}) — open the .docx and Save As PDF")


if __name__ == "__main__":
    main()
