# SyllabiSense

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
