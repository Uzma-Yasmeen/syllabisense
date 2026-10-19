"""
Extract page text from every content PDF once and cache it.

Each content "unit" is one source a syllabus could be paired with:
  - one MIT OCW course folder
  - one OpenStax textbook
  - the NPTEL marketing lecture set
Output: cache/units/<unit_id>.json = {"unit", "source", "pages": [{"file", "page", "text"}]}
Every page keeps its file + page number, so any match can be traced back and checked by hand.
"""
import glob
import json
import os
import sys

import pypdfium2 as pdfium

sys.stdout.reconfigure(encoding="utf-8")
OUT = "cache/units"
os.makedirs(OUT, exist_ok=True)


def pages_of(path):
    out = []
    try:
        pdf = pdfium.PdfDocument(path)
    except Exception as e:
        print(f"    unreadable {path}: {type(e).__name__}")
        return out
    for i in range(len(pdf)):
        try:
            t = pdf[i].get_textpage().get_text_range()
        except Exception:
            t = ""
        out.append({"file": path.replace("\\", "/"), "page": i + 1, "text": t})
    pdf.close()
    return out


def units():
    for d in sorted(glob.glob("material/ocw/*/")):
        slug = os.path.basename(os.path.normpath(d))
        yield "ocw__" + slug, "MIT OCW " + slug, sorted(glob.glob(d + "*.pdf"))
    for f in sorted(glob.glob("material/openstax/*.pdf")):
        b = os.path.splitext(os.path.basename(f))[0]
        yield "openstax__" + b, "OpenStax " + b, [f]
    mk = sorted(glob.glob("material/122106032/*.pdf"))
    if mk:
        yield "nptel__122106032", "NPTEL 122106032 Intro to Marketing notes", mk


def main():
    for uid, source, files in units():
        dst = os.path.join(OUT, uid + ".json")
        if os.path.exists(dst):
            # re-extract if the folder changed since caching (e.g. cached mid-download)
            cached = json.load(open(dst, encoding="utf8"))
            if cached.get("n_files") == len(files):
                continue
        pages = []
        for f in files:
            pages.extend(pages_of(f))
        chars = sum(len(p["text"]) for p in pages)
        json.dump({"unit": uid, "source": source, "n_files": len(files), "pages": pages},
                  open(dst, "w", encoding="utf8"))
        print(f"  {uid[:60]:60s} {len(files):4d} files {len(pages):5d} pages {chars/1e6:6.2f}M chars", flush=True)


if __name__ == "__main__":
    main()
