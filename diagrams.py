"""
Diagram detection for the report: does the evidence page carry a diagram?

Runs the trained CNN (results_cnn/diagram_cnn.keras) over only the pages the evidence model
already chose, so a run costs a few dozen images rather than the whole document.

Honest about what it cannot do: only PDF pages can be rendered to an image here, so slides and
Word files come back as "not checked" rather than "no diagram".

    from diagrams import detect, summarise
    found = detect([(path, "Page 7"), ...])     # -> {(path, location): (probability, is_diagram)}
"""
import json
import os
import re
import sys

import numpy as np

MODEL = "models/diagram_cnn.keras"
METRICS = "results/cnn/metrics.json"
SIZE = 128
SCALE = 0.55          # the scale the training images were rendered at - keep these in step
_model, _thr = None, 0.5


def available():
    return os.path.exists(MODEL)


def _load():
    global _model, _thr
    if _model is None:
        import tensorflow as tf
        # safe_mode=False: the saved CNN wraps MobileNetV2's preprocess_input in a Lambda layer,
        # which Keras will not deserialise otherwise. The file is one we trained ourselves.
        _model = tf.keras.models.load_model(MODEL, safe_mode=False)
        if os.path.exists(METRICS):
            _thr = float(json.load(open(METRICS, encoding="utf8")).get("threshold", 0.5))
    return _model, _thr


def page_number(location):
    """'Page 12' -> 12;  'Slide 4' or 'Heading > sub' -> None."""
    if not location or not location.lower().startswith("page"):
        return None
    m = re.search(r"(\d+)", location)
    return int(m.group(1)) if m else None


def renderable(path, location):
    return bool(path) and path.lower().endswith(".pdf") and page_number(location) is not None


def detect(pairs):
    """pairs: iterable of (file path, location). Returns {(path, location): (prob, is_diagram)}
    for the pages that could be rendered; pages that could not are simply absent."""
    if not available():
        return {}
    todo = {(p, loc): page_number(loc) for p, loc in pairs if renderable(p, loc)}
    if not todo:
        return {}

    import pypdfium2 as pdfium
    from PIL import Image                                      # noqa: F401  (pypdfium2 -> PIL)
    model, thr = _load()
    keys, images, docs = [], [], {}
    for (path, loc), pg in todo.items():
        try:
            if path not in docs:
                docs[path] = pdfium.PdfDocument(path)
            doc = docs[path]
            if pg > len(doc):
                continue
            img = (doc[pg - 1].render(scale=SCALE, grayscale=True)
                   .to_pil().convert("RGB").resize((SIZE, SIZE)))
        except Exception:
            continue
        keys.append((path, loc))
        images.append(np.asarray(img, np.float32))
    for d in docs.values():
        try:
            d.close()
        except Exception:
            pass
    if not keys:
        return {}
    p = model.predict(np.array(images), verbose=0).ravel()
    return {k: (float(v), bool(v >= thr)) for k, v in zip(keys, p)}


def summarise(evidence, found):
    """evidence: [(path, location, ...), ...] for one topic.
    Returns (state, pages) where state is 'diagram', 'none' or 'not_checked'."""
    checked = [(pa, lo) for pa, lo, *_ in evidence if (pa, lo) in found]
    if not checked:
        return "not_checked", []
    hits = [(pa, lo, found[(pa, lo)][0]) for pa, lo in checked if found[(pa, lo)][1]]
    if hits:
        return "diagram", sorted(hits, key=lambda h: -h[2])
    return "none", []


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)
    import pypdfium2 as pdfium
    path = sys.argv[1]
    n = len(pdfium.PdfDocument(path))
    got = detect([(path, f"Page {i}") for i in range(1, n + 1)])
    yes = [k for k, v in got.items() if v[1]]
    print(f"{os.path.basename(path)}: {len(got)} pages checked, {len(yes)} with a diagram")
    for (_, loc), (p, hit) in sorted(got.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {loc:<10} {p:.2f}  {'diagram' if hit else '-'}")
