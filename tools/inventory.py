"""Build a per-domain inventory of every syllabus and content PDF collected."""
import glob
import json
import os
import re
import sys

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8")

# NPTEL: first 3 digits of the course id = discipline
NPTEL_DISC = {
    "101": "Aero", "102": "Biology", "103": "ChemEng", "104": "Chemistry",
    "105": "Civil", "106": "CS", "108": "Electrical", "109": "Humanities",
    "110": "Management", "111": "Mathematics", "112": "Mechanical",
    "113": "Materials", "115": "Physics", "122": "Management", "127": "Mechanical",
}

# discipline 109 lumps together several very different subjects; split on the title
H_SPLIT = [
    ("Economics", r"econom|game theory"),
    ("Social science", r"psycholog|behavio|human resource"),
    ("Humanities", r"sociolog|society|philosoph|literature|linguistic|writing|ethic|cultur|histor|language|translat|communicat"),
]

# MIT OCW: leading course number = department
OCW_DEPT = {
    "1": "Civil", "2": "Mechanical", "3": "Materials", "5": "Chemistry", "6": "CS",
    "7": "Biology", "8": "Physics", "9": "Social science", "10": "ChemEng",
    "14": "Economics", "15": "Management", "16": "Aero", "18": "Mathematics",
    "21": "Humanities", "24": "Humanities",
}

OPENSTAX = {
    "introductory-statistics.pdf": ("Mathematics", "Introductory Statistics"),
    "introduction-to-sociology-3e.pdf": ("Humanities", "Introduction to Sociology 3e"),
    "psychology-2e.pdf": ("Social science", "Psychology 2e"),
    "principles-of-microeconomics-2e-draft.pdf": ("Economics", "Principles of Microeconomics 2e"),
    "biology-2e.pdf": ("Biology", "Biology 2e"),
}

CACHE = "titles.json"
titles = json.load(open(CACHE, encoding="utf8")) if os.path.exists(CACHE) else {}


def title_of(path):
    key = os.path.basename(path)
    if key in titles:
        return titles[key]
    try:
        with pdfplumber.open(path) as p:
            txt = p.pages[0].extract_text() or ""
    except Exception:
        txt = ""
    lines = [x.strip() for x in txt.split("\n") if x.strip()]
    t = "?"
    for ln in lines[:5]:
        if "NPTEL" in ln.upper() or len(ln) < 4:
            continue
        t = re.sub(r"\s+", " ", ln)[:52]
        break
    titles[key] = t
    return t


def classify(cid, title):
    d = NPTEL_DISC.get(cid[:3], "Other")
    if d == "Humanities":
        low = title.lower()
        for name, pat in H_SPLIT:
            if re.search(pat, low):
                return name
    return d


dom = {}


def add(d, kind, name, n, b):
    e = dom.setdefault(d, {"syllabi": [], "content": []})
    e[kind].append((name, n, b))


for f in sorted(glob.glob("syllabi/*.pdf")):
    cid = os.path.splitext(os.path.basename(f))[0]
    t = title_of(f)
    add(classify(cid, t), "syllabi", f"{cid} {t}", 1, os.path.getsize(f))

for f in glob.glob("material/openstax/*.pdf"):
    b = os.path.basename(f)
    if b in OPENSTAX:
        d, name = OPENSTAX[b]
        add(d, "content", name + " (OpenStax)", 1, os.path.getsize(f))

mk = glob.glob("material/122106032/*.pdf")
if mk:
    add("Management", "content", "Intro to Marketing lectures (NPTEL)", len(mk),
        sum(os.path.getsize(x) for x in mk))

for cdir in sorted(glob.glob("material/ocw/*/")):
    slug = os.path.basename(os.path.normpath(cdir))
    m = re.match(r"^(\d+)[-.]", slug)
    d = OCW_DEPT.get(m.group(1), "Other") if m else "Other"
    fs = glob.glob(cdir + "*.pdf")
    if fs:
        add(d, "content", slug[:44] + " (OCW)", len(fs), sum(os.path.getsize(x) for x in fs))

json.dump(titles, open(CACHE, "w"), indent=1)

print("%-15s %8s %8s   %s" % ("DOMAIN", "SYLLABI", "CONTENT", "STATUS"))
print("-" * 66)
complete = []
for d in sorted(dom):
    e = dom[d]
    s = sum(x[1] for x in e["syllabi"])
    c = sum(x[1] for x in e["content"])
    st = "COMPLETE" if s and c else ("syllabus only" if s else "content only")
    if st == "COMPLETE":
        complete.append(d)
    print("%-15s %8s %8s   %s" % (d, s or "-", c or "-", st))
print("-" * 66)
print("COMPLETE: %d domains -> %s" % (len(complete), ", ".join(complete)))

tot = sum(x[1] for d in dom for k in ("syllabi", "content") for x in dom[d][k])
mb = sum(x[2] for d in dom for k in ("syllabi", "content") for x in dom[d][k]) / 1e6
print("TOTAL: %d PDFs, %.0f MB" % (tot, mb))

with open("INVENTORY.txt", "w", encoding="utf8") as fh:
    for d in sorted(dom):
        e = dom[d]
        s = sum(x[1] for x in e["syllabi"])
        c = sum(x[1] for x in e["content"])
        fh.write("\n### %s  (%s syllabi, %s content PDFs)\n" % (d, s or 0, c or 0))
        for name, n, b in e["content"]:
            fh.write("   CONTENT  %-48s %4d file(s)  %7.1f MB\n" % (name[:48], n, b / 1e6))
        for name, n, b in e["syllabi"]:
            fh.write("   syllabus %-48s\n" % name[:48])
print("\nwrote INVENTORY.txt")
