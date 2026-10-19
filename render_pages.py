"""
Render page images for the diagram CNN.

Samples pages across many different courses (so the CNN does not learn one textbook's style) and
renders each to a small greyscale JPEG. Pages that are mostly blank are skipped.

    python render_pages.py --n 2500

Output: dataset/page_images/*.jpg  and  dataset/page_images/index.csv
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

import pypdfium2 as pdfium

sys.stdout.reconfigure(encoding="utf-8")
OUT = "dataset/page_images"
SCALE = 0.55          # ~450 px wide for a typical page, small enough to label quickly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--per-course", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    random.seed(7)

    units = sorted(glob.glob("cache/units/*.json"))
    random.shuffle(units)
    picks = []
    for u in units:
        d = json.load(open(u, encoding="utf8"))
        pages = [(p["file"], p["page"], len(p["text"].strip())) for p in d["pages"]]
        random.shuffle(pages)
        picks += [(d["unit"], f, pg, n) for f, pg, n in pages[:a.per_course]]
        if len(picks) >= a.n * 1.4:
            break
    random.shuffle(picks)

    rows, done, open_docs = [], 0, {}
    for unit, f, pg, nchars in picks:
        if done >= a.n:
            break
        if not os.path.exists(f):
            continue
        try:
            if f not in open_docs:
                open_docs[f] = pdfium.PdfDocument(f)
            doc = open_docs[f]
            if pg > len(doc):
                continue
            bitmap = doc[pg - 1].render(scale=SCALE, grayscale=True)
            img = bitmap.to_pil()
        except Exception:
            continue
        name = f"{done:05d}.jpg"
        img.save(os.path.join(OUT, name), quality=70)
        rows.append({"image": name, "unit": unit, "file": os.path.basename(f), "page": pg,
                     "chars": nchars})
        done += 1
        if done % 250 == 0:
            print(f"  rendered {done}/{a.n}", flush=True)
    for d in open_docs.values():
        try:
            d.close()
        except Exception:
            pass

    with open(f"{OUT}/index.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "unit", "file", "page", "chars"])
        w.writeheader()
        w.writerows(rows)
    print(f"rendered {len(rows)} page images from {len({r['unit'] for r in rows})} courses -> {OUT}")


if __name__ == "__main__":
    main()
