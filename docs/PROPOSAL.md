# AI & Deep Learning — Project Proposal

**SR University — Department of Computer Science & Artificial Intelligence**
Faculty: Prof. Sai Meghana Rao Eravelli

> Numbers marked (provisional) are being re-measured after a labelling fix; everything else is final.

## A. Student / Team Details

| | |
|---|---|
| **Project Title** | SyllabiSense: Detecting Syllabus Coverage Gaps in Course Material Using a Deep Neural Network |
| **Student Name** | Uzma Yasmeen (individual project) |
| **Roll Number** | 2503A51L34 |
| **Batch Number** | 20 |
| **Course** | AI & Deep Learning |

## B. Project Title

**SyllabiSense: Detecting Syllabus Coverage Gaps in Course Material Using a Deep Neural Network**

## C. Why Did You Choose This Project?

Students receive a syllabus from the university and a separate pile of study material — lecture
PDFs, slide decks and notes — and are left to assume the two match. They usually do not. A topic
listed in the syllabus may be covered in depth, mentioned in one line, or missing from the
material entirely, and a student normally discovers this only while revising for the exam. Existing
tools do not solve this: a document chatbot can answer questions about the pages it is given, but
it can never tell a student that something is *absent*, because it has no notion of what the
syllabus promised.

This project addresses that gap. SyllabiSense reads a syllabus, reads the course material, and
decides for every syllabus topic whether the material actually teaches it — citing the exact file
and page as evidence. The deep learning component is a neural network that judges whether a given
page is genuine evidence for a given topic; this decision is what separates "covered" from
"missing". Students benefit by learning early which topics need outside material, and faculty
benefit by checking their own material against the syllabus before a semester begins. The expected
outcome is a working system that produces a per-topic coverage report with page-level evidence,
together with measured accuracy on subjects the model was never trained on.

## D. AI / Deep Learning Techniques Used

**Model:** a feed-forward Artificial Neural Network (ANN) that performs binary classification on a
(syllabus topic, page of material) pair.

**Why this model:** the task is a judgement over a *pair* of texts, so each text is first converted
into a fixed-length numeric representation and the ANN learns the relationship between the two.
A fully connected network is appropriate because the input is a single feature vector rather than
an image or a sequence, and it is small enough to train and run on a CPU.

**Input representation (1,155 features per example):**

| Part | Size | Meaning |
|---|---|---|
| Topic vector | 384 | pretrained sentence-embedding of the syllabus topic |
| Page vector | 384 | pretrained sentence-embedding of the page text |
| Elementwise product | 384 | where the two representations agree |
| Cosine similarity | 1 | overall closeness of topic and page |
| Word overlap | 1 | fraction of topic words occurring on the page |
| Page length | 1 | scaled length of the page text |

A second variant replaces the pretrained embeddings with TF-IDF reduced to 128 dimensions by
Truncated SVD (LSA), giving 387 features; both are reported.

**Architecture (layers):**

```
Input (1155)
Dense(256)  ReLU        - learns combinations of the topic/page features
Dropout(0.3)            - reduces overfitting
Dense(64)   ReLU        - compresses to a smaller decision space
Dropout(0.2)
Dense(1)    Sigmoid     - probability that the page is evidence for the topic
```

- **Activation functions:** ReLU in hidden layers (fast, avoids vanishing gradients), Sigmoid at the
  output because the task is binary.
- **Optimizer:** Adam, learning rate 0.001.
- **Loss function:** binary cross-entropy, the standard loss for two-class problems.
- **Class weighting:** positives are one third of the data, so the positive class is weighted to stop
  the network from simply predicting "not relevant".
- **Input:** topic text + page text. **Output:** a probability in [0, 1], converted to 0/1 using a
  threshold chosen on validation data, never on the test data.
- **Role in the system:** the ANN is the component that decides whether evidence exists for a topic.
  Topics with no page above the threshold are reported to the student as *not covered*.

## E. Dataset Description

**Name:** SyllabiBench (built for this project).

**Sources (all public, verified by download):**

| Source | Use | Licence |
|---|---|---|
| NPTEL (nptel.ac.in) | 1,295 course syllabi | free for educational use |
| MIT OpenCourseWare | ~110 courses of lecture notes, problem sets, exams | CC BY-NC-SA |
| OpenStax | 7 full textbooks | CC BY |

**Raw scale:** 7,133 content PDFs, 100,414 pages, about 13 million words (5.5 GB).

**Labelled examples: 17,887** (topic, page) pairs in three classes of example:

| Example type | How it is labelled | Count |
|---|---|---|
| Positive (relevant) | page matched the topic's specific terms within a 30-word window | 6,037 |
| Hard negative | a different page from the *same* course | 6,013 |
| Easy negative | a page from a course in an unrelated subject | 5,837 |

**Split — by subject, not randomly**, so the test set is a subject the model has never seen:

| Split | Subjects | Examples |
|---|---|---|
| Train | 12 subjects (Management, Electrical, Aero, Economics, Biology, …) | 11,615 |
| Validation | Chemistry, Humanities | 775 |
| **Test** | **Computer Science only** | **5,497** |

**Human-verified gold set:** 300 examples (201 CS) were reviewed manually, one by one, without
showing the automatic label. This measures the true accuracy and showed that the automatic labels
are 84% correct — the honest result reported below rather than the inflated one.

**Input features:** 1,155 numeric features per example (table above). **Classes:** 2 (relevant /
not relevant).

## F. Proposed Methodology

```
Syllabus PDF ─► topic extraction (rule-based parser)
                                                   ├─► pair generation ─► feature extraction
Course material (PDF/PPTX/DOCX) ─► page extraction ─┘        │
                                                             ▼
                      weak labelling (matcher) ─► train / validation / test split by subject
                                                             │
                                                             ▼
                                   ANN training (Adam, binary cross-entropy, early stopping)
                                                             │
                                                             ▼
                        evaluation on unseen subject + human-verified gold set
                                                             │
                                                             ▼
                      prediction → per-topic coverage report with file and page evidence
```

1. **Data collection** — syllabi and course material downloaded from NPTEL, MIT OCW and OpenStax.
2. **Preprocessing** — text extracted per page/slide/section; contents pages, index pages,
   reference lists and unreadable scanned pages removed; syllabus topics parsed from the document.
3. **Pair generation and labelling** — positives from term-proximity matching, hard negatives from
   the same course, easy negatives from unrelated subjects.
4. **Train/test split** — by subject, so the test subject is never seen in training.
5. **Model building** — embeddings/TF-IDF features, then the ANN described above.
6. **Model training** — Adam, binary cross-entropy, batch 64, up to 40 epochs, early stopping.
7. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, ROC-AUC, plus a
   human-verified gold set and an ablation study.
8. **Prediction** — `analyze.py` takes a new syllabus and new material and outputs the coverage report.

## G. Expected Outcome

A working system that accepts a syllabus and a folder of course material in PDF, PowerPoint or
Word format and produces a report listing every syllabus topic as **covered** (with the file and
page number as evidence) or **not covered**. Demonstrated on an NPTEL Deep Learning syllabus
against MIT's Deep Learning course: 54 of 60 topics covered, with the missing ones correctly
identified as VGGNet, GoogLeNet, ZF-Net, AdaGrad and RMSProp.

## H. Model Details

| Parameter | Value | Why |
|---|---|---|
| Layers | 3 dense (256, 64, 1) + 2 dropout | enough capacity without overfitting 11.6k examples |
| Neurons | 256, 64, 1 | narrowing funnel to a single decision |
| Activations | ReLU, ReLU, Sigmoid | ReLU trains fast; sigmoid gives a probability |
| Learning rate | 0.001 | Adam default; stable here |
| Optimizer | Adam | adaptive rates, works well without tuning |
| Loss | Binary cross-entropy | standard for binary classification |
| Batch size | 64 | balances speed and gradient stability on CPU |
| Epochs | up to 40, early stopping (patience 6) | stops when validation loss stops improving |
| Dropout | 0.3 and 0.2 | the main regulariser in a fully connected network |

## I. Results (provisional — being re-measured)

**Test subject = Computer Science, never seen during training.**

| Evaluation | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Against automatic labels (5,497 items) | 92.8% | — | — | 0.889 | 0.955 |
| **Against human labels (201 items)** | **84.1%** | — | — | **0.746** | 0.806 |

**Ablation — what the model actually uses:**

| Features | Accuracy | F1 |
|---|---|---|
| Full feature set | 92.8% | 0.889 |
| Without the 3 similarity features | 61.6% | 0.442 |

**Honest findings to state in the viva:**
1. Accuracy measured against automatic labels (92.8%) is roughly 9 points higher than against human
   labels (84.1%). Only the human-verified number is trustworthy.
2. The ablation shows much of the performance comes from the hand-crafted similarity features, not
   from the learned text representation alone.
3. The human review found that 23% of "hard negative" training labels were actually relevant pages,
   which taught the model to under-detect evidence; the labelling rule was then tightened.

## J. Submission Checklist

- [x] Proposal document (this file)
- [x] Source code (`build_dataset.py`, `embed_features.py`, `features_tfidf.py`, `train_ann.py`,
      `evaluate_gold.py`, `analyze.py`, `loaders.py`, `review_tool.py`)
- [x] Dataset source/reference (NPTEL, MIT OpenCourseWare, OpenStax — listed above)
- [x] ANN architecture and parameters (sections D and H)
- [x] Results and performance metrics (section I)
- [x] Graphs: accuracy vs epochs, loss vs epochs, confusion matrix, ROC curve (`results_*/`)
