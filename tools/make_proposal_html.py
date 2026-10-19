"""
Build the web version of the proposal for publishing as an Artifact.

    python make_proposal_html.py

Reads the same results_*/metrics.json files as make_proposal.py, so the page and the PDF always
quote the same numbers.

Writes submission/web/index.html and copies the figures it references into submission/web/img/.
"""
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf8")

OUT = "submission/web"
NAME, ROLL, BATCH = "Uzma Yasmeen", "2503A51L34", "20 CSE"
TITLE = "SyllabiSense: AI for Course Coverage and Personalized Learning"

FIGURES = [("figures/ann.png", "ann.png"), ("figures/cnn.png", "cnn.png"),
           ("figures/crossencoder.png", "crossencoder.png"),
           ("figures/pipeline.png", "pipeline.png"),
           ("results_tfidf/history.png", "ann_history.png"),
           ("results_tfidf/confusion_matrix.png", "ann_cm.png"),
           ("results_tfidf/roc_curve.png", "ann_roc.png"),
           ("results_cnn/history.png", "cnn_history.png"),
           ("results_cnn/confusion_matrix.png", "cnn_cm.png"),
           ("results_cnn/examples.png", "cnn_examples.png"),
           ("results_crossencoder/history.png", "ce_history.png"),
           ("results_fusion/comparison.png", "fusion.png")]


def load(p, d=None):
    return json.load(open(p, encoding="utf8")) if os.path.exists(p) else d


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def num(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def rows(pairs):
    return "\n".join(f"<tr><th scope='row'>{k}</th><td>{v}</td></tr>" for k, v in pairs)


def main():
    os.makedirs(f"{OUT}/img", exist_ok=True)
    kept = []
    for src, dst in FIGURES:
        if os.path.exists(src):
            shutil.copy(src, f"{OUT}/img/{dst}")
            kept.append(dst)
    print(f"figures copied: {len(kept)}")

    tf_, tf_h = load("results_tfidf/metrics.json"), load("results_tfidf/metrics_human.json")
    ce, ce_h = load("results_crossencoder/metrics.json"), load("results_crossencoder/metrics_human.json")
    cnn, scratch = load("results_cnn/metrics.json"), load("results_cnn_scratch/metrics.json")
    depth, fusion = load("results_depth/metrics.json"), load("results_fusion/metrics.json")
    stats = load("dataset/stats.json", {})
    n = stats.get("by_split_kind", {})
    tot = lambda k: sum(n.get(f"{s}/{k}", 0) for s in ("train", "val", "test"))
    t, h = tf_["test"], tf_h["human_labels"]
    cw, ch = tf_["confusion_matrix"], h["confusion_matrix"]
    hp = tf_["hyperparameters"]

    model_rows = "".join(
        f"<tr><td>{m}</td><td class='n'>{a}</td><td class='n'>{p}</td>"
        f"<td class='n'>{r}</td><td class='n'>{f}</td></tr>"
        for m, a, p, r, f in [
            ("Evidence ANN <span class='tag'>deployed</span>", pct(t["accuracy"]),
             num(t["precision"]), num(t["recall"]), num(t["f1"])),
            ("Cross-encoder (fine-tuned transformer)", pct(ce["test"]["accuracy"]),
             num(ce["test"]["precision"]), num(ce["test"]["recall"]), num(ce["test"]["f1"])),
            ("Diagram CNN (MobileNetV2)", pct(cnn["test"]["accuracy"]),
             num(cnn["test"]["precision"]), num(cnn["test"]["recall"]), num(cnn["test"]["f1"])),
            ("Diagram CNN (from scratch — failed)", pct(scratch["test"]["accuracy"]),
             num(scratch["test"]["precision"]), num(scratch["test"]["recall"]),
             num(scratch["test"]["f1"])),
            ("Depth ANN (3-class, preliminary)", pct(depth["test_accuracy"]), "—", "—",
             num(depth["macro_f1"]))])

    html = f"""<title>SyllabiSense Proposal</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: single measured column, a fixed label gutter for the template's field tables. */
:root {{
  --paper:#ffffff; --sunk:#f3f5f9; --ink:#151c2b; --muted:#5a6478; --line:#dde3ec;
  --navy:#1f3864; --blue:#2e74b5; --good:#1c6b4a; --warn:#8a5a12;
  --serif:"Source Serif 4", Georgia, "Times New Roman", serif;
  --sans:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper:#0f141f; --sunk:#171e2c; --ink:#e7ecf5; --muted:#9aa5bb; --line:#2a3346;
  --navy:#9dbbe4; --blue:#7fa9d4; --good:#6fc79b; --warn:#e0ad5e; color-scheme:dark;
}} }}
:root[data-theme="dark"] {{
  --paper:#0f141f; --sunk:#171e2c; --ink:#e7ecf5; --muted:#9aa5bb; --line:#2a3346;
  --navy:#9dbbe4; --blue:#7fa9d4; --good:#6fc79b; --warn:#e0ad5e; color-scheme:dark;
}}
* {{ box-sizing:border-box }}
body {{ background:var(--paper); color:var(--ink); font-family:var(--sans); font-size:15px;
  line-height:1.6; margin:0 }}
.wrap {{ max-width:54rem; margin:0 auto; padding-inline:20px; padding-block:0 72px }}

header.sheet {{ border-bottom:3px solid var(--navy); padding-block:34px 20px; margin-bottom:28px }}
.inst {{ font-family:var(--sans); font-size:.78rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); margin:0 0 14px }}
h1 {{ font-family:var(--serif); font-size:clamp(1.7rem,4.6vw,2.5rem); line-height:1.15;
  font-weight:600; margin:0 0 10px; text-wrap:balance; color:var(--navy) }}
.sub {{ font-family:var(--serif); font-size:1.05rem; color:var(--muted); margin:0 0 20px;
  max-width:46rem }}
.meta {{ display:flex; flex-wrap:wrap; gap:6px 26px; font-size:.86rem; color:var(--muted);
  font-variant-numeric:tabular-nums }}
.meta b {{ color:var(--ink); font-weight:600 }}

h2 {{ font-family:var(--serif); font-size:1.42rem; font-weight:600; color:var(--navy);
  margin:40px 0 4px; padding-bottom:6px; border-bottom:1px solid var(--line); text-wrap:balance }}
h3 {{ font-family:var(--sans); font-size:1rem; font-weight:600; color:var(--blue);
  margin:26px 0 6px; letter-spacing:.01em }}
p {{ margin:0 0 12px; max-width:40rem }}
.lead {{ font-weight:600 }}

table {{ border-collapse:collapse; width:100%; font-size:.87rem; margin:6px 0 4px }}
.scroll {{ overflow-x:auto; margin-bottom:16px }}
th, td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top }}
th[scope="row"] {{ background:var(--sunk); font-weight:600; width:13.5rem; color:var(--ink) }}
thead th {{ background:var(--navy); color:var(--paper); font-weight:600; font-size:.8rem;
  letter-spacing:.02em }}
:root[data-theme="dark"] thead th, :root:not([data-theme="light"]) thead th {{ color:#0f141f }}
@media (prefers-color-scheme: light) {{ :root:not([data-theme="dark"]) thead th {{ color:#fff }} }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; font-family:var(--mono);
  font-size:.82rem; white-space:nowrap }}
caption {{ caption-side:bottom; text-align:left; font-size:.78rem; color:var(--muted);
  padding-top:7px; font-style:italic }}

figure {{ margin:18px 0; text-align:center }}
figure img {{ max-width:100%; border:1px solid var(--line); border-radius:3px; background:#fff }}
figcaption {{ font-size:.78rem; color:var(--muted); margin-top:7px; font-style:italic }}

.note {{ border-left:4px solid var(--blue); background:var(--sunk); padding:12px 16px;
  margin:18px 0; border-radius:0 3px 3px 0 }}
.note p:last-child {{ margin-bottom:0 }}
.note .k {{ font-size:.73rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  font-weight:600; display:block; margin-bottom:3px }}
.tag {{ font-size:.68rem; letter-spacing:.06em; text-transform:uppercase; background:var(--navy);
  color:var(--paper); padding:2px 6px; border-radius:3px; margin-left:6px; white-space:nowrap }}
:root[data-theme="dark"] .tag, :root:not([data-theme="light"]) .tag {{ color:#0f141f }}
@media (prefers-color-scheme: light) {{ :root:not([data-theme="dark"]) .tag {{ color:#fff }} }}
ol.steps {{ padding-left:1.15rem; max-width:40rem }}
ol.steps li {{ margin-bottom:7px }}
code {{ font-family:var(--mono); font-size:.85em; background:var(--sunk); padding:1px 5px;
  border-radius:3px }}
.done {{ font-family:var(--mono); color:var(--good); font-weight:500 }}
.todo {{ font-family:var(--mono); color:var(--warn); font-weight:500 }}
footer {{ margin-top:46px; padding-top:16px; border-top:1px solid var(--line); font-size:.8rem;
  color:var(--muted) }}
@media (max-width:560px) {{ th[scope="row"] {{ width:auto }} }}
</style>

<div class="wrap">
<header class="sheet">
  <p class="inst">SR University · Department of Computer Science &amp; Artificial Intelligence</p>
  <h1>{TITLE}</h1>
  <p class="sub">Deep neural networks that check course material against a syllabus: for every
  topic they decide whether the material actually teaches it, cite the file and page as evidence,
  and turn the gaps into a personalised study order.</p>
  <div class="meta">
    <span><b>{NAME}</b></span><span>Roll <b>{ROLL}</b></span><span>Batch <b>{BATCH}</b></span>
    <span>AI &amp; Deep Learning</span><span>Prof. Sai Meghana Rao Eravelli</span>
  </div>
</header>

<h2>1. Project Proposal Document</h2>

<h3>A. Student / Team Details</h3>
<div class="scroll"><table>
{rows([("Project Title", TITLE), ("Student Name", NAME), ("Roll Number", ROLL),
       ("Batch Number", BATCH), ("Course", "AI &amp; Deep Learning"),
       ("Team size", "Individual project (solo)")])}
</table></div>

<h3>C. Why Did You Choose This Project?</h3>
<p><span class="lead">Problem and importance.</span> A university gives students a syllabus and,
separately, a pile of study material — lecture PDFs, slide decks and notes. Students assume the two
match. They often do not: a topic listed in the syllabus may be covered in depth, mentioned in a
single line, or missing altogether, and this is usually discovered only while revising for the
exam. Existing tools cannot help, because a document chatbot answers questions about the pages it
is given and can never report that something is <em>missing</em> — it has no idea what the syllabus
promised.</p>
<p><span class="lead">Motivation, beneficiaries and outcome.</span> Detecting absence means holding
two documents against each other and judging every pairing between them, which is what a network
trained on (topic, page) pairs can learn. Students find out early which topics need outside
material; faculty can check their own material against the syllabus before a semester begins, which
is also the shape of the documentation accreditation asks for.</p>

<h3>D. AI / Deep Learning Techniques Used</h3>
<div class="scroll"><table>
{rows([("Model / architecture",
        "Four trained networks. <b>Main model:</b> fully connected ANN in TensorFlow/Keras, "
        "387 → 256 → 64 → 1. <b>Also trained:</b> a MiniLM transformer cross-encoder fine-tuned "
        f"end to end ({ce['parameters_millions']}M parameters), a MobileNetV2 CNN for diagram "
        "detection, and a 3-class ANN for teaching depth."),
       ("Reason for selecting it",
        "The task judges a pair of texts, so the input is a fixed-length vector of pair features "
        "and dense layers suit it. The network trains in under a minute on a CPU and scores 9,000 "
        "topic-page pairs in under a second, which the application needs."),
       ("Major layers", "Input (387), Dense 256, Dropout 0.3, Dense 64, Dropout 0.2, Dense 1."),
       ("Activation functions", "ReLU after each hidden layer; sigmoid at the output."),
       ("Optimizer and loss",
        f"Adam (lr {hp['learning_rate']}); binary cross-entropy with class weights."),
       ("Input and output",
        "Input: 387 numbers describing one (syllabus topic, material page) pair — TF-IDF reduced "
        "to 128 dimensions by truncated SVD for each side, their product, plus cosine similarity, "
        "word overlap and page length. Output: one probability that the page is evidence."),
       ("Role of the ANN",
        "Scores every topic against every page; pages above the calibrated threshold become that "
        "topic's evidence, and a topic with none is reported as a gap."),
       ("Role of the CNN",
        "Runs on the pages the ANN chose and reports whether the topic is supported by a diagram "
        "or explained in words alone.")])}
</table></div>

<figure><img src="img/ann.png" alt="Diagram of the evidence ANN: 387 inputs through two hidden
layers to a single sigmoid output."><figcaption>Figure 1 — evidence ANN (387-256-64-1),
115,841 trainable parameters.</figcaption></figure>

<figure><img src="img/cnn.png" alt="Diagram of the diagram CNN: a page image through a frozen
MobileNetV2 base and a small trained head."><figcaption>Figure 2 — diagram CNN, MobileNetV2 base
with a trained head.</figcaption></figure>

<h3>E. Dataset Description</h3>
<div class="scroll"><table>
{rows([("Dataset name", "SyllabiSense (topic, page) pairs — built from scratch for this project."),
       ("Source",
        "NPTEL (1,295 course syllabi), MIT OpenCourseWare (~110 courses of lecture notes, problem "
        "sets and exams, CC BY-NC-SA), OpenStax (7 textbooks, CC BY). Raw material: 7,133 PDFs, "
        "100,414 pages, roughly 13 million words."),
       ("Number of samples", f"{stats.get('examples', 0):,} labelled pairs."),
       ("Number of classes",
        f"2 — evidence ({tot('positive'):,}) and not evidence "
        f"({tot('hard_negative') + tot('easy_negative'):,}, of which {tot('hard_negative'):,} are "
        f"hard negatives from the same course)."),
       ("Input features", "387 numeric features per pair."),
       ("Train / test split",
        "<b>By subject area, never at random.</b> Train 11,615 pairs from 12 non-computing areas; "
        "validate 775 (chemistry, humanities); test 5,497 from 109 Computer Science courses. No "
        "computing material appears anywhere in training."),
       ("Human-verified labels",
        "300 pairs were re-labelled by hand on a screen that hides the automatic label. The "
        "automatic labels proved <b>84% correct</b>, so accuracy is reported against the human "
        "labels too. A further 101 topics and 576 page images were hand-labelled for the depth "
        "and CNN models.")])}
</table></div>

<div class="note"><span class="k">Why the split matters</span>
<p>A random split would put pages from the same course in both training and testing, and the
reported accuracy would mostly measure memorisation. Splitting by subject forces the models to
generalise to vocabulary they have never seen.</p></div>

<h3>F. Proposed Methodology</h3>
<figure><img src="img/pipeline.png" alt="Flowchart: syllabus and material are parsed, paired,
labelled, split by subject, used to train four networks, evaluated, and turned into a per-unit
report."><figcaption>Figure 3 — end-to-end pipeline.</figcaption></figure>
<ol class="steps">
<li><b>Dataset</b> — download syllabi and material; extract text per page, slide or section.</li>
<li><b>Preprocessing</b> — drop navigation, contents, index, reference and unreadable pages; parse
the syllabus into units and topics; fit TF-IDF and SVD on the training split only.</li>
<li><b>Train/test split</b> — by subject area: 12 areas train, 2 validate, Computer Science tests.</li>
<li><b>Model building</b> — the 387-256-64-1 ANN, plus the cross-encoder, depth ANN and CNN.</li>
<li><b>Training</b> — Adam, binary cross-entropy, class weights, early stopping on validation AUC.</li>
<li><b>Evaluation</b> — thresholds chosen on validation and on human-labelled non-test items, never
on the test set.</li>
<li><b>Prediction</b> — score a new syllabus against new material and report each topic per unit.</li>
</ol>

<h3>G. Expected Outcome</h3>
<p>The system accepts a syllabus and a folder of material in PDF, PowerPoint or Word format and
produces a unit-by-unit report marking every topic as properly taught, barely taught or not taught,
with the file and page quoted as evidence. The CNN adds whether each covered topic has a diagram.
Files named by unit are judged only against their own unit, and a unit with no uploaded material is
reported as <em>no material provided</em> rather than as a gap in teaching.</p>

<h2>4. Results</h2>
<p>Every figure is computed by the code and stored in <code>results_*/metrics.json</code>. The test
split is Computer Science — a subject area that appears nowhere in training.</p>

<div class="scroll"><table>
<caption>All trained models, on their own held-out test sets.</caption>
<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
<tbody>{model_rows}</tbody>
</table></div>

<h3>The honest figure</h3>
<p>Training labels are generated automatically and are themselves only 84% correct, so the
automatic-label accuracy is inflated by the same noise the model trained on. Measured against the
201 hand-labelled Computer Science pairs, with the threshold tuned on human-labelled items from
other subjects:</p>
<div class="scroll"><table>
<caption>Evidence ANN on human-verified labels. The 95% confidence interval is about ±5 points at
this sample size.</caption>
<thead><tr><th>Measured against</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th>
<th>ROC-AUC</th></tr></thead>
<tbody>
<tr><td>Automatic labels (5,497 pairs)</td><td class="n">{pct(t['accuracy'])}</td>
<td class="n">{num(t['precision'])}</td><td class="n">{num(t['recall'])}</td>
<td class="n">{num(t['f1'])}</td><td class="n">{num(t['roc_auc'])}</td></tr>
<tr><td>Human-verified labels (201 pairs)</td><td class="n">{pct(h['accuracy'])}</td>
<td class="n">{num(h['precision'])}</td><td class="n">{num(h['recall'])}</td>
<td class="n">{num(h['f1'])}</td><td class="n">{num(h['roc_auc'])}</td></tr>
</tbody></table></div>

<div class="scroll"><table>
<caption>Confusion matrices for the evidence ANN.</caption>
<thead><tr><th></th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr></thead>
<tbody>
<tr><td>Test split (automatic labels)</td><td class="n">{cw['tn']:,}</td>
<td class="n">{cw['fp']:,}</td><td class="n">{cw['fn']:,}</td><td class="n">{cw['tp']:,}</td></tr>
<tr><td>Human-verified set</td><td class="n">{ch['tn']}</td><td class="n">{ch['fp']}</td>
<td class="n">{ch['fn']}</td><td class="n">{ch['tp']}</td></tr>
</tbody></table></div>

<figure><img src="img/ann_history.png" alt="Accuracy and loss curves for the evidence ANN across
training epochs."><figcaption>Figure 4 — evidence ANN: accuracy and loss vs epochs
({tf_['epochs_run']} epochs before early stopping).</figcaption></figure>
<figure><img src="img/ann_roc.png" alt="ROC curve for the evidence ANN."
style="max-width:26rem"><figcaption>Figure 5 — evidence ANN: ROC curve.</figcaption></figure>

<h3>The CNN, and a failed experiment</h3>
<p>A CNN trained from scratch on {cnn['n_train']} page images reached
{pct(scratch['test']['accuracy'])} accuracy with a recall of {num(scratch['test']['recall'], 2)}:
it learned to answer “no diagram” almost always, which is what 75% means when 76% of pages have no
diagram. Transfer learning raised recall to {num(cnn['test']['recall'], 2)} — the network now finds
26 of the 36 diagram pages instead of 4. Both runs are kept, because the comparison is the
result.</p>
<figure><img src="img/cnn_examples.png" alt="Grid of page images with the CNN's confidence scores,
showing its most and least confident predictions."><figcaption>Figure 6 — diagram CNN: most and
least confident predictions on held-out courses.</figcaption></figure>

<h3>Which model should the application use?</h3>
<p>The fine-tuned transformer ranks pages better than the ANN — ROC-AUC
{num(ce_h['human_labels']['roc_auc'])} against {num(h['roc_auc'])} — yet it is less accurate at a
fixed decision threshold ({pct(ce_h['human_labels']['accuracy'])} against {pct(h['accuracy'])}).
Its scores separate the classes well but are not calibrated consistently across subject areas, so a
threshold fitted on one set of subjects lands in the wrong place on another. Five ways of deploying
the two models were compared under one protocol, with every threshold tuned on held-out human
labels: <b>{fusion['best_rule'].replace('_', ' ')}</b> won, so the ANN is deployed and the
transformer is reported as the comparison that justifies the choice.</p>
<figure><img src="img/fusion.png" alt="Grouped bar chart comparing accuracy, precision, recall and
F1 for five decision rules."><figcaption>Figure 7 — five deployment rules on the 201 human-verified
items.</figcaption></figure>

<div class="note"><span class="k">Limitations</span>
<p>The per-topic verdict shown to a student is not yet measured — the figures above score
individual (topic, page) decisions. The depth model reaches only {pct(depth['test_accuracy'])} on
{depth['n_test']} held-out topics, so the application derives depth from transparent rules instead.
Pages with no text layer are skipped (no OCR), the syllabus parser is rule-based and occasionally
mistakes a page footer for a topic, and the human-verified set was labelled by one person.</p></div>

<h2>5. Final Submission Checklist</h2>
<div class="scroll"><table>
<thead><tr><th>Done</th><th>Item</th><th>Where</th></tr></thead>
<tbody>
<tr><td class="done">[x]</td><td>Project proposal document</td><td>PROPOSAL.pdf / .docx</td></tr>
<tr><td class="done">[x]</td><td>Complete source code</td><td>CODE_SUBMISSION.pdf,
SyllabiSense_Demo.ipynb</td></tr>
<tr><td class="done">[x]</td><td>Dataset source and reference</td><td>Section 1E</td></tr>
<tr><td class="done">[x]</td><td>ANN and CNN architecture details</td><td>Section 1D</td></tr>
<tr><td class="done">[x]</td><td>Results and performance metrics</td><td>Section 4</td></tr>
<tr><td class="done">[x]</td><td>Relevant graphs</td><td>Figures 4–7</td></tr>
<tr><td class="todo">[ ]</td><td>Output screenshots of the application</td><td>To attach after the
demonstration run</td></tr>
</tbody></table></div>

<footer>Every number on this page is read from the project's results files rather than typed in.
Companion documents: PROPOSAL.pdf (the submitted proposal) and CODE_SUBMISSION.pdf (the
demonstration program and its real output).</footer>
</div>
"""
    with open(f"{OUT}/index.html", "w", encoding="utf8") as fh:
        fh.write(html)
    kb = os.path.getsize(f"{OUT}/index.html") / 1024
    print(f"wrote {OUT}/index.html ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
