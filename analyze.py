"""
SyllabiSense end-to-end run: syllabus + course material -> what is covered, what is missing.

    python analyze.py --syllabus path/to/syllabus.pdf --material file.pdf another.pdf or_a_folder/
    python analyze.py --syllabus s.pdf --material notes/ --top 3 --out report.md

Steps: extract syllabus topics -> read every page of the material -> embed both ->
score each (topic, page) pair with the trained ANN -> a topic counts as COVERED if its best
page scores above the threshold chosen on validation data.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import tensorflow as tf

from depth import advice, judge, study_order
from loaders import SUPPORTED, load
from units import applicable, file_units, unit_of_label
from syllabus_topics import content_words, topics_with_units

sys.stdout.reconfigure(encoding="utf-8")
EMB = "BAAI/bge-small-en-v1.5"
PAGE_CHARS = 1500


def read_pages(paths):
    """Any supported format -> (source, location, text). PDF pages, PowerPoint slides
    (including speaker notes) and Word sections all arrive in the same shape."""
    return [(b["source"], b["location"], b["text"]) for b in load(paths)]


class TfidfFeaturiser:
    """Same 387 features the TF-IDF model was trained on, using the saved fitted vectoriser."""

    def __init__(self, path="models/tfidf_model.joblib"):
        import joblib
        d = joblib.load(path)
        self.vec, self.svd = d["vectorizer"], d["svd"]

    def topic_page(self, topic, pages):
        import numpy as np
        T = self.vec.transform([topic])
        P = self.vec.transform([p[2] for p in pages])
        num = np.asarray(P.multiply(T).sum(1)).ravel()
        den = (np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel())
               * np.sqrt(T.multiply(T).sum()) + 1e-9)
        cos = (num / den).reshape(-1, 1)
        a = np.repeat(self.svd.transform(T), len(pages), axis=0)
        b = self.svd.transform(P)
        tw = {w for w in topic.lower().split() if len(w) > 3}
        ov = np.array([[sum(1 for w in tw if w in p[2].lower()) / max(1, len(tw))] for p in pages])
        ln = np.array([[min(len(p[2]), PAGE_CHARS) / PAGE_CHARS] for p in pages])
        return np.hstack([a, b, a * b, cos, ov, ln]).astype(np.float32)


class Embedder:
    """Only used by the embedding variant. torch and transformers are imported here rather than
    at the top of the file so the application - which never builds an Embedder - does not drag
    ~2.5 GB of dependencies into a deployment that will not call them."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(EMB)
        self.mdl = AutoModel.from_pretrained(EMB).eval()
        torch.set_num_threads(8)

    def __call__(self, texts, bs=32):
        torch = self.torch
        out = np.zeros((len(texts), 384), np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=256,
                               return_tensors="pt")
                h = self.mdl(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out[i:i + enc["input_ids"].shape[0]] = torch.nn.functional.normalize(
                    v, dim=-1).numpy()
        return out


def features(tvec, pvecs, topic, pages):
    a = np.repeat(tvec[None, :], len(pages), axis=0)
    cos = (a * pvecs).sum(1, keepdims=True)
    tw = {w for w in topic.lower().split() if len(w) > 3}
    ov = np.array([[sum(1 for w in tw if w in p[2].lower()) / max(1, len(tw))] for p in pages], np.float32)
    ln = np.array([[min(len(p[2]), PAGE_CHARS) / PAGE_CHARS] for p in pages], np.float32)
    return np.hstack([a, pvecs, a * pvecs, cos, ov, ln]).astype(np.float32)



def distinctive(topic):
    """Topic words specific enough to expect on a page that teaches it."""
    stop = {"introduction", "overview", "basics", "concepts", "principles", "types", "models",
            "review", "study", "using", "based", "various", "general"}
    return [w for w in content_words(topic) if len(w) > 4 and w not in stop]


def has_terms(topic, text, need=0.5, min_mentions=3):
    """Evidence requires two things:
    - at least `need` of the topic's distinctive words appear on the page, and
    - those words appear `min_mentions` times in total.
    The second rule separates teaching from a passing mention: an SDLC overview names "testing"
    once as a phase, while a page teaching testing repeats the term throughout."""
    d = distinctive(topic)
    if not d:
        return True
    low = text.lower()
    counts = [low.count(w[:6].lower()) for w in d]
    present = sum(1 for c in counts if c)
    return present >= max(1, round(need * len(d))) and sum(counts) >= min_mentions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--material", nargs="+", required=True,
                    help="files or folders; " + ", ".join(SUPPORTED))
    ap.add_argument("--model", default="models/evidence_ann.keras")
    ap.add_argument("--metrics", default="results/tfidf/metrics.json")
    ap.add_argument("--features", choices=["tfidf", "embed"], default="tfidf",
                    help="must match the model: tfidf -> 387 features, embed -> 1155")
    ap.add_argument("--top", type=int, default=2, help="evidence pages to show per covered topic")
    ap.add_argument("--out", default="coverage_report.md")
    a = ap.parse_args()

    thr = json.load(open(a.metrics, encoding="utf8"))["threshold"]
    human = a.metrics.replace("metrics.json", "metrics_human.json")
    if os.path.exists(human):          # prefer the threshold tuned on human-checked items
        thr = json.load(open(human, encoding="utf8"))["threshold_human_tuned"]
    model = tf.keras.models.load_model(a.model)

    tu = [(u, t) for u, t in topics_with_units(a.syllabus) if content_words(t)]
    topics = [t for _, t in tu]
    units = [u for u, _ in tu]
    print(f"syllabus topics found: {len(topics)}"
          + (f" in {len(set(units))} units" if units and units[0] else ""))
    pages = read_pages(a.material)
    print(f"usable pages of material: {len(pages)}")
    funits = file_units(pages)
    have = sorted({u for u in funits.values() if u})
    if have:
        print("material covers unit(s): " + ", ".join(str(u) for u in have)
              + "  (pages from one unit are not used as evidence for another)")
    if not topics or not pages:
        sys.exit("nothing to compare - check the files")

    if a.features == "tfidf":
        feat = TfidfFeaturiser()
        make = lambda t, i: feat.topic_page(t, pages)
    else:
        emb = Embedder()
        print("embedding ...", flush=True)
        tvecs = emb(list(topics))
        pvecs = emb([p[2] for p in pages])
        make = lambda t, i: features(tvecs[i], pvecs, t, pages)

    results = []
    for n, t in enumerate(topics):
        prob = model.predict(make(t, n), verbose=0).ravel()
        order = np.argsort(-prob)
        tunit = unit_of_label(units[n])
        above = [i for i in order
                 if prob[i] >= thr and has_terms(t, pages[i][2])
                 and applicable(tunit, funits.get(pages[i][0]))][:8]
        evidence = [(pages[i][0], pages[i][1], float(prob[i]), pages[i][2]) for i in above]
        level, reasons, kinds = judge(evidence)
        if not evidence and tunit is not None and have and tunit not in have \
                and not any(u is None for u in funits.values()):
            level = "NO MATERIAL PROVIDED"
            reasons = [f"no file for unit {tunit} was uploaded"]
        closest = (pages[order[0]][0], pages[order[0]][1], float(prob[order[0]]))
        results.append({"topic": t, "unit": units[n], "order": n, "level": level,
                        "reasons": reasons, "kinds": kinds, "evidence": evidence,
                        "closest": closest})

    LEVELS = ("PROPERLY TAUGHT", "BARELY TAUGHT", "NOT TAUGHT", "NO MATERIAL PROVIDED")
    groups = {lv: [r for r in results if r["level"] == lv] for lv in LEVELS}
    lines = ["# Coverage report", "",
             f"- syllabus: `{os.path.basename(a.syllabus)}` — {len(topics)} topics",
             f"- material: {len(pages)} pages/slides/sections from "
             f"{len(set(p[0] for p in pages))} file(s)",
             f"- decision threshold: {thr:.2f} (chosen on validation data)",
             f"- model: `{a.model}`",
             ("- material covers unit(s): " + ", ".join(str(u) for u in have)
              + " — pages from one unit are not used as evidence for another"
              if have else "- material has no unit labels: every page may support every topic"), "",
             "| | topics |", "|---|---|",
             f"| Properly taught | {len(groups['PROPERLY TAUGHT'])} |",
             f"| Barely taught | {len(groups['BARELY TAUGHT'])} |",
             f"| Not taught | {len(groups['NOT TAUGHT'])} |",
             f"| No material provided | {len(groups['NO MATERIAL PROVIDED'])} |", ""]

    if any(r["unit"] for r in results):
        lines += ["## By unit", "", "| Unit | Topics | ✅ properly | ⚠️ barely | ❌ not taught | ➖ no material |",
                  "|---|---|---|---|---|---|"]
        for u in dict.fromkeys(r["unit"] for r in results):
            rs = [r for r in results if r["unit"] == u]
            cnt = {lv: sum(1 for r in rs if r["level"] == lv) for lv in groups}
            lines.append(f"| {u} | {len(rs)} | {cnt['PROPERLY TAUGHT']} | {cnt['BARELY TAUGHT']} | "
                         f"{cnt['NOT TAUGHT']} | {cnt['NO MATERIAL PROVIDED']} |")
        lines.append("")

    for lv in LEVELS:
        rs = groups[lv]
        if lv == "NO MATERIAL PROVIDED" and not rs:
            continue
        lines += [f"## {lv.title()} — {len(rs)}", ""]
        for r in rs:
            lines.append(f"**{r['topic']}**" + (f"  _({r['unit']})_" if r["unit"] else "") + "  ")
            lines.append(f"  {' · '.join(r['reasons'])}  ")
            lines.append(f"  → {advice(r['level'], r['kinds'])}  ")
            for f, loc, pr, _ in r["evidence"][:a.top]:
                lines.append(f"  📄 {f} — {loc} ({pr:.2f})  ")
            if not r["evidence"]:
                f, loc, pr = r["closest"]
                lines.append(f"  (closest page: {f} — {loc}, scored {pr:.2f})  ")
            lines.append("")

    todo = study_order(results)
    lines += ["## What to do, in order", ""]
    if not todo:
        lines.append("Everything in this syllabus is properly covered by the material.")
    for i, r in enumerate(todo, 1):
        why = "no material at all" if r["level"] == "NOT TAUGHT" else "material is too thin"
        lines.append(f"{i}. **{r['topic']}** — {why}; "
                     f"{'earlier topics usually come first' if r['order'] < len(topics) / 2 else 'later in the syllabus'}")
    lines += ["", "---", "*Levels come from the evidence the neural network found: how many pages, "
              "how confident, and which kinds of evidence (definition, explanation, example, "
              "formula, figure) appear on them.*"]
    open(a.out, "w", encoding="utf8").write("\n".join(lines))

    print(f"\nPROPERLY TAUGHT {len(groups['PROPERLY TAUGHT'])}   "
          f"BARELY {len(groups['BARELY TAUGHT'])}   NOT TAUGHT {len(groups['NOT TAUGHT'])}   "
          f"NO MATERIAL {len(groups['NO MATERIAL PROVIDED'])}")
    for r in todo[:8]:
        print(f"   {r['level']:<16} {r['topic'][:60]}")
    print(f"\nfull report -> {a.out}")


if __name__ == "__main__":
    main()
