"""
Find NPTEL syllabus PDFs by probing the course-id space.

NPTEL ids are 9 digits: <discipline><institute><serial>.
The syllabus endpoint is a static file, so a HEAD-ish GET tells us if a course exists.

Usage:
    python collect_nptel.py 101 103 104 109     # discipline codes to scan
"""
import concurrent.futures as cf
import os
import re
import sys
import time
import urllib.error
import urllib.request

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8")   # some course titles contain non-cp1252 characters
UA = {"User-Agent": "Mozilla/5.0"}
INSTITUTES = ["101", "102", "103", "104", "105", "106", "107", "108"]
SERIALS = range(1, 60)
OUT = "syllabi"


FAILED = []   # ids that errored on every attempt - logged, never silently treated as "absent"


def try_id(cid):
    """Only an HTTP 404 means the course has no syllabus. Timeouts and other errors are retried;
    an earlier version treated them as 'absent' and silently skipped real syllabi (e.g. DBMS)."""
    url = f"https://archive.nptel.ac.in/content/syllabus_pdf/{cid}.pdf"
    for attempt in range(4):
        try:
            d = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
            return d if d.startswith(b"%PDF") else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    FAILED.append(cid)
    return None


def title_of(path):
    try:
        with pdfplumber.open(path) as p:
            txt = p.pages[0].extract_text() or ""
    except Exception:
        return "?"
    lines = [x.strip() for x in txt.split("\n") if x.strip()]
    for ln in lines[:5]:
        if "NPTEL" in ln.upper() or len(ln) < 4:
            continue
        return re.sub(r"\s+", " ", ln)[:58]
    return lines[0][:58] if lines else "?"


def scan(disc):
    ids = [f"{disc}{inst}{n:03d}" for inst in INSTITUTES for n in SERIALS
           if not os.path.exists(os.path.join(OUT, f"{disc}{inst}{n:03d}.pdf"))]
    found = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for cid, blob in zip(ids, ex.map(try_id, ids)):
            if not blob:
                continue
            path = os.path.join(OUT, f"{cid}.pdf")
            with open(path, "wb") as f:
                f.write(blob)
            t = title_of(path)
            # skip placeholder/empty syllabi
            if t in ("?", "") or len(t) < 5:
                os.remove(path)
                continue
            found.append((cid, t))
            print(f"  {cid}  {t}", flush=True)
    return found


def main():
    # usage: python collect_nptel.py [--serials 60-260] 108 109 110
    global SERIALS
    args = sys.argv[1:]
    if args and args[0] == "--serials":
        lo, hi = map(int, args[1].split("-"))
        SERIALS = range(lo, hi + 1)
        args = args[2:]
    discs = args or ["101", "103", "104", "109"]
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for d in discs:
        print(f"\n=== discipline {d} (serials {SERIALS.start}-{SERIALS.stop - 1}) ===", flush=True)
        total += len(scan(d))
    print(f"\nTOTAL new syllabi: {total}")
    if FAILED:
        with open("nptel_failed_ids.txt", "a", encoding="utf8") as fh:
            fh.write("\n".join(FAILED) + "\n")
        print(f"WARNING: {len(FAILED)} ids failed after retries -> nptel_failed_ids.txt (re-run to retry)")


if __name__ == "__main__":
    main()
