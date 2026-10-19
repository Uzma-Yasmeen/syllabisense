"""
Score every syllabus against every content unit, topic by topic.

A topic counts as FOUND in a unit only if some page has the topic's key terms
close together (within WINDOW words) - not merely scattered across the page.
Every hit records file + page so it can be checked by hand.

Control check: each syllabus is also scored against units from OTHER domains.
A real pairing must clearly beat the best unrelated unit, otherwise the score
is just generic vocabulary and the pairing is rejected.

Outputs:
  cache/syllabus_topics.json   topics per syllabus
  match_matrix.json            coverage for every (syllabus, unit)
  match_hits.json              evidence pages for the best pairs
"""
import collections
import glob
import json
import math
import os
import re
import sys

from syllabus_topics import content_words, topics

sys.stdout.reconfigure(encoding="utf-8")
WINDOW = 30          # topic terms must sit within ~one sentence of each other
NEED = 0.75          # share of the topic's specificity weight that must be present
MIN_TOPICS = 8

NUMBERED_HEADING = re.compile(r"^\s*\d{1,2}(\.\d{1,2})*\.?\s+[A-Z][^.]{0,70}$")
CHAPTER_LINE = re.compile(r"^\s*(chapter|unit|part)\s+\d+\s*[:.]", re.I)
NUMBER_ONLY = re.compile(r"^\s*[\d.\-–,]+\s*$")
# TOC leaders: "Topic ....... 23", spaced ". . . . .", or long dot runs between entries
DOT_LEADER = re.compile(r"\.{4,}\s*\d|(\.\s){4,}|\.{8,}")
# bibliography entries: "Abela, J. R., & Hankin, B. L. (2011)." / "Coe, W. C. (2009)."
CITATION = re.compile(r"\((19|20)\d\d[a-z]?\)|\bet al\.|\b(19|20)\d\d\.\s+[A-Z]|doi\.org|https?://", re.I)
REVIEW_OUTLINE = re.compile(r"(\s[–•]\s[^–•]{2,40}){8,}")   # "– Sample space – Discrete – Bayes ..."
SENTENCE = re.compile(r"[a-z][^.?!\n]{40,}[.?!]")
REAL_WORD = re.compile(r"^[A-Za-z][a-z]{2,}[.,;:)]?$")
# broken OCR produces tokens like "EF1Cf5", "CL/.1r7T", "oCL" - letters and digits/case jumbled
OCR_JUNK = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d/.,'~|!]{3,}$|^[a-z]+[A-Z]{2,}|[~|]{1,}")


def is_navigation(text):
    """Contents, index, reading lists and schedules MENTION topics without teaching them.
    They must not count as coverage evidence. Data tables in worked examples, exam covers
    and exercises contain real sentences, and that protects them from being flagged."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 6:
        return False
    n = len(lines)
    if len(DOT_LEADER.findall(text)) >= 5:                             # "Topic ........ 23" (not code "...")
        return True
    flat = re.sub(r"\s+", " ", text)
    if len(CITATION.findall(flat)) >= 8 and len(CITATION.findall(flat)) / n > 0.2:
        return True                                                    # reference list / bibliography
    if REVIEW_OUTLINE.search(flat) and len(SENTENCE.findall(flat)) < 3:
        return True                                                    # "– Sample space – Bayes' theorem" outlines
    if sum(1 for l in lines if CHAPTER_LINE.match(l)) >= 5:            # "Chapter 8: Memory" lists
        return True
    numbered = sum(1 for l in lines if NUMBERED_HEADING.match(l)) / n
    words_per_line = sum(len(l.split()) for l in lines) / n
    numeric = sum(1 for l in lines if NUMBER_ONLY.match(l)) / n
    # these two layouts are unambiguous; checked before the sentence test because flattening
    # a list of headings can glue it into fake "sentences"
    if numbered > 0.7:                                                 # "2. The Science of Biology"
        return True
    long_lines = sum(1 for l in lines if len(l.split()) >= 6) / n
    # fragment tables / schedules: tiny lines, numbers, and NO full lines at all
    # (plot-heavy lecture slides also have tiny numeric lines, but keep a title/explanation)
    if words_per_line < 2.5 and numeric > 0.12 and long_lines < 0.04:
        return True
    # PDF text wraps sentences across lines, so look for sentences in whitespace-flattened text
    sentences = len(SENTENCE.findall(re.sub(r"\s+", " ", text)))
    if sentences >= 3:
        return False
    if numbered > 0.5:
        return True
    if numeric > 0.12 and words_per_line < 4 and \
            re.search(r"\b(readings?|lectures?|sessions?|chapters?|weeks?)\b", text, re.I):
        return True                                                    # schedule tables: "32", "10.7"
    return False


REF_HEADING = re.compile(r"\n\s*(References|Bibliography|Works Cited|Further Reading)\s*\n")


def strip_references(text):
    """Pages often end a chapter and then start its reference list. Keep the prose above the
    heading; the citations below it mention topics without teaching them."""
    m = REF_HEADING.search(text)
    return text[: m.start()] if m else text


def is_garbage(text):
    """Scanned pages with a broken text layer ('--L P LLY s t -p 1 I'). They need OCR in
    preprocessing; until then they are neither evidence for nor against a topic."""
    toks = text.split()
    if len(toks) < 30:
        return False
    real = sum(1 for t in toks if REAL_WORD.match(t)) / len(toks)
    junk = sum(1 for t in toks if OCR_JUNK.search(t)) / len(toks)
    # maths pages have few real words too, but not jumbled letter/digit tokens
    return real < 0.25 and junk > 0.12

# ---------- domains ----------
NPTEL_DISC = {"101": "Aero", "102": "Biology", "103": "ChemEng", "104": "Chemistry", "105": "Civil",
              "106": "CS", "108": "Electrical", "110": "Management", "111": "Mathematics",
              "112": "Mechanical", "113": "Materials", "115": "Physics", "127": "Mechanical"}
OCW_DEPT = {"1": "Civil", "2": "Mechanical", "3": "Materials", "5": "Chemistry", "6": "CS",
            "7": "Biology", "8": "Physics", "9": "Psychology", "10": "ChemEng", "14": "Economics",
            "15": "Management", "16": "Aero", "18": "Mathematics", "21": "Humanities", "24": "Humanities"}
OCW_OVERRIDE = {"6-002": "Electrical", "6-003": "Electrical", "1-151": "Mathematics",
                "1-00": "CS",   # MIT 1.00 is a Java / data-structures course, not civil engineering
                # MIT course 6 is all of EECS; these are electrical-engineering courses
                "6-334": "Electrical", "6-061": "Electrical", "6-013": "Electrical",
                "6-012": "Electrical", "6-071j": "Electrical", "6-622": "Electrical",
                "18-404j": "CS",   # Theory of Computation is filed under maths at MIT, core CS in India
                "15-773": "CS",    # Hands-On Deep Learning is filed under the business school
                # CS subjects taught in other MIT departments
                "1-264j": "CS", "16-355j": "CS", "1-124j": "CS", "15-071": "CS", "15-062": "CS"}
# Controls must come from a genuinely unrelated field: thermodynamics legitimately appears
# in Mechanical, ChemEng, Chemistry and Materials, so those are not controls for each other.
GROUP = {"Mechanical": "eng", "Aero": "eng", "Civil": "eng", "Materials": "eng", "Physics": "eng",
         "ChemEng": "eng", "Chemistry": "eng", "Electrical": "eng", "Mathematics": "eng",
         "CS": "comp", "Biology": "life",
         "Economics": "social", "Psychology": "social", "Sociology": "social",
         "Humanities": "social", "Management": "social"}
GENERIC_DF = 0.05   # a word on >5% of all pages carries no topic-specific evidence
OPENSTAX = {"introductory-statistics": "Mathematics", "introduction-to-sociology-3e": "Sociology",
            "psychology-2e": "Psychology", "principles-of-microeconomics-2e-draft": "Economics",
            "biology-2e": "Biology", "introduction-to-business": "Management",
            "principles-of-macroeconomics-2e-draft": "Economics"}
H_SPLIT = [("Economics", r"econom|game theory"),
           ("Psychology", r"psycholog|behavio|cognition"),
           ("Sociology", r"sociolog|society|population")]


def syllabus_domain(cid, title):
    if cid.startswith("122103039"):
        return "Biology"                    # Basic Biology, misfiled under 122 prefix
    if cid.startswith("122"):
        return "Management"
    if cid.startswith("109"):
        low = title.lower()
        for name, pat in H_SPLIT:
            if re.search(pat, low):
                return name
        return "Humanities"
    return NPTEL_DISC.get(cid[:3], "Other")


def unit_domain(uid):
    kind, name = uid.split("__", 1)
    if kind == "openstax":
        return OPENSTAX.get(name, "Other")
    if kind == "nptel":
        return "Management"
    for k, v in OCW_OVERRIDE.items():
        if name.startswith(k + "-"):
            return v
    m = re.match(r"^(\d+)[-.]", name)
    return OCW_DEPT.get(m.group(1), "Other") if m else "Other"


# ---------- content index ----------
class Unit:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf8"))
        self.uid, self.source = d["unit"], d["source"]
        self.domain = unit_domain(self.uid)
        self.pages = [(p["file"], p["page"]) for p in d["pages"]]
        self.toks = [content_words(strip_references(p["text"])) for p in d["pages"]]
        self.nav = {i for i, p in enumerate(d["pages"]) if is_navigation(p["text"])}
        self.garbage = {i for i, p in enumerate(d["pages"])
                        if i not in self.nav and is_garbage(p["text"])}
        self.index = collections.defaultdict(set)
        for i, ts in enumerate(self.toks):
            if i in self.nav or i in self.garbage:
                continue          # contents/index pages and unreadable scans are never evidence
            for t in set(ts):
                self.index[t].add(i)

    def find(self, weights):
        """weights: {term: idf} for the topic's specific terms.
        FOUND = some passage of <= WINDOW words contains the topic's most specific term
        and >= NEED of the topic's total idf weight. Returns (file, page) or None."""
        if not weights:
            return None
        key = max(weights, key=weights.get)
        total = sum(weights.values())
        cands = self.index.get(key, ())
        for i in cands:
            toks = self.toks[i]
            pos = [(j, t) for j, t in enumerate(toks) if t in weights]
            lo = 0
            for hi in range(len(pos)):
                while pos[hi][0] - pos[lo][0] > WINDOW:
                    lo += 1
                seen = {t for _, t in pos[lo:hi + 1]}
                if key in seen and sum(weights[t] for t in seen) >= NEED * total:
                    return self.pages[i]
        return None


def main():
    # topics for every syllabus (cached)
    tcache = "cache/syllabus_topics.json"
    titles = json.load(open("titles.json", encoding="utf8")) if os.path.exists("titles.json") else {}
    stop = json.load(open(tcache, encoding="utf8")) if os.path.exists(tcache) else {}
    for f in sorted(glob.glob("syllabi/*.pdf")):
        cid = os.path.splitext(os.path.basename(f))[0]
        if cid not in stop:
            try:
                stop[cid] = topics(f)
            except Exception:
                stop[cid] = []
    json.dump(stop, open(tcache, "w", encoding="utf8"), indent=1)

    units = [Unit(p) for p in sorted(glob.glob("cache/units/*.json"))]
    print(f"content units: {len(units)}")

    # document frequency of every term across ALL pages -> how specific a word is
    df = collections.Counter()
    for u in units:
        for toks in u.toks:
            df.update(set(toks))
    N = sum(len(u.toks) for u in units)
    print(f"pages indexed: {N}")

    def topic_weights(t):
        w = {}
        for term in set(content_words(t)):
            f = df.get(term, 0)
            if f == 0 or f / N > GENERIC_DF:
                continue          # unseen anywhere, or too common to be evidence
            w[term] = math.log(N / f)
        return w

    syl, generic = {}, {}
    for cid, ts in stop.items():
        # a "topic" that is just the course title matches every title page - not a topic
        title_words = set(content_words(titles.get(cid + ".pdf", "")))
        ts = [t for t in ts if not (set(content_words(t)) and set(content_words(t)) <= title_words)]
        tw = [(t, topic_weights(t)) for t in ts]
        keep = [(t, w) for t, w in tw if w]
        generic[cid] = [t for t, w in tw if not w]
        if len(keep) >= MIN_TOPICS:
            syl[cid] = keep
    print(f"syllabi with >= {MIN_TOPICS} specific topics: {len(syl)} of {len(stop)}")

    matrix, hits = {}, {}
    for n, (cid, tw) in enumerate(sorted(syl.items()), 1):
        title = titles.get(cid + ".pdf", "?")
        dom = syllabus_domain(cid, title)
        row = {}
        for u in units:
            found = {}
            for t, w in tw:
                h = u.find(w)
                if h:
                    found[t] = h
            row[u.uid] = round(len(found) / len(tw), 3)
            hits[f"{cid}|{u.uid}"] = found
        matrix[cid] = {"title": title, "domain": dom, "group": GROUP.get(dom, "other"),
                       "n_topics": len(tw), "topics": [t for t, _ in tw],
                       "generic_topics_skipped": generic[cid], "coverage": row}
        if n % 25 == 0:
            print(f"  scored {n}/{len(syl)}", flush=True)

    json.dump({"units": {u.uid: {"domain": u.domain, "group": GROUP.get(u.domain, "other"),
                                 "source": u.source, "pages": len(u.pages)} for u in units},
               "syllabi": matrix}, open("match_matrix.json", "w", encoding="utf8"), indent=1)
    json.dump(hits, open("match_hits.json", "w", encoding="utf8"))
    print("wrote match_matrix.json, match_hits.json")


if __name__ == "__main__":
    main()
