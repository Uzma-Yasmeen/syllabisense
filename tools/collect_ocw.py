"""
Collect course material from MIT OpenCourseWare.

Per course: read the /download/ page (server-rendered HTML), find the course ZIP,
download it, keep only real lecture-note PDFs, drop video transcripts, delete the ZIP.

Usage:
    python collect_ocw.py                 # 2 courses per department
    python collect_ocw.py --per-dept 3    # more per department
    python collect_ocw.py --dept EECS/CS  # one department only
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

UA = {"User-Agent": "Mozilla/5.0"}
OUT = "material/ocw"
MIN_PDFS = 5          # a course must yield at least this many good PDFs to count
MAX_ZIP_MB = 220      # skip anything larger, it is mostly video (override: --max-zip-mb)
ZIP_TIMEOUT = 300     # seconds for the course ZIP download (override: --timeout)


def fetch(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def find_zip(slug):
    """The /download/ page is plain HTML; the course ZIP link is in it."""
    try:
        html = fetch(f"https://ocw.mit.edu/courses/{slug}/download/", 60).decode("utf8", "ignore")
    except Exception:
        return None
    m = re.findall(r'https://ocw\.mit\.edu/courses/[^"\' <>]+\.zip', html)
    return m[0] if m else None


YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def is_material(name):
    """Keep lecture notes, chapters, problem sets. Drop only video transcripts and video
    thumbnails, which are named by an 11-char YouTube id (e.g. 'e98MPnMHLxE.pdf').
    Short real names like 'ch8.pdf' or 'hw3.pdf' must be kept - an earlier version
    required long names and silently dropped whole textbooks."""
    base = name.split("/")[-1]
    if not base.lower().endswith(".pdf"):
        return False
    if "transcript" in base.lower():
        return False
    stem = re.sub(r"^[a-f0-9]{32}_", "", base)[:-4]        # strip content-hash prefix and .pdf
    if YOUTUBE_ID.match(stem) and re.search(r"[A-Z]", stem) and re.search(r"[a-z]", stem) \
            and not re.match(r"^(MIT|mit)", stem):
        return False                                       # video id: mixed case, exactly 11 chars
    return True


def collect(slug):
    zurl = find_zip(slug)
    if not zurl:
        return 0, 0, "no zip link"
    try:
        blob = fetch(zurl, ZIP_TIMEOUT)
    except Exception as e:
        return 0, 0, f"download failed: {type(e).__name__}"
    if len(blob) > MAX_ZIP_MB * 1_000_000:
        return 0, 0, f"zip too large ({len(blob)//10**6} MB)"
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except Exception:
        return 0, 0, "not a valid zip"

    keep = [i for i in z.infolist() if is_material(i.filename)]
    if len(keep) < MIN_PDFS:
        return 0, len(blob), f"only {len(keep)} usable PDFs"

    d = os.path.join(OUT, slug)
    os.makedirs(d, exist_ok=True)
    total = 0
    for i in keep:
        base = re.sub(r"^[a-f0-9]{32}_", "", i.filename.split("/")[-1])
        with open(os.path.join(d, base), "wb") as f:
            data = z.read(i)
            f.write(data)
            total += len(data)
    return len(keep), total, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dept", type=int, default=2)
    ap.add_argument("--dept", default=None)
    ap.add_argument("--slugs", default=None, help="text file, one course slug per line")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download courses already on disk and restore files an older filter dropped")
    ap.add_argument("--max-zip-mb", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    a = ap.parse_args()
    global MAX_ZIP_MB, ZIP_TIMEOUT
    MAX_ZIP_MB = a.max_zip_mb or MAX_ZIP_MB
    ZIP_TIMEOUT = a.timeout or ZIP_TIMEOUT

    if a.slugs:
        wanted = [s.strip() for s in open(a.slugs, encoding="utf8") if s.strip() and not s.startswith("#")]
        manifest = json.load(open("ocw_manifest.json", encoding="utf8")) if os.path.exists("ocw_manifest.json") else {}
        for slug in wanted:
            d = os.path.join(OUT, slug)
            before = len(os.listdir(d)) if os.path.isdir(d) else 0
            if before and not a.refresh:
                print(f"  have {slug}", flush=True)
                continue
            n, size, msg = collect(slug)
            if before and msg == "ok":
                print(f"  REFR {slug[:58]:58s} {before:3d} -> {len(os.listdir(d)):3d} files", flush=True)
                continue
            flag = "OK  " if msg == "ok" else "--  "
            print(f"  {flag} {slug[:58]:58s} " + (f"{n:3d} PDFs  {size/1e6:5.1f} MB" if msg == "ok" else msg), flush=True)
            if msg == "ok":
                manifest.setdefault("targeted", []).append({"slug": slug, "pdfs": n, "bytes": size})
        json.dump(manifest, open("ocw_manifest.json", "w"), indent=1)
        return

    courses = json.load(open("ocw_courses.json", encoding="utf8"))
    depts = [a.dept] if a.dept else sorted(courses)
    manifest = {}
    os.makedirs(OUT, exist_ok=True)

    for dept in depts:
        got = 0
        print(f"\n=== {dept} ===", flush=True)
        for slug in courses.get(dept, []):
            if got >= a.per_dept:
                break
            n, size, msg = collect(slug)
            if msg == "ok":
                got += 1
                print(f"  OK   {slug[:58]:58s} {n:3d} PDFs  {size/1e6:5.1f} MB", flush=True)
                manifest.setdefault(dept, []).append({"slug": slug, "pdfs": n, "bytes": size})
            else:
                print(f"  --   {slug[:58]:58s} {msg}", flush=True)
        if got == 0:
            print(f"  !! nothing collected for {dept}", flush=True)

    json.dump(manifest, open("ocw_manifest.json", "w"), indent=1)
    tot = sum(c["pdfs"] for v in manifest.values() for c in v)
    mb = sum(c["bytes"] for v in manifest.values() for c in v) / 1e6
    print(f"\nTOTAL: {tot} PDFs across {len(manifest)} departments, {mb:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
