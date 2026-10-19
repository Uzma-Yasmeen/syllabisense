"""
Turn an NPTEL syllabus PDF into a clean list of topics.

Handles the three layouts seen so far:
  - NOC:         "COURSE PLAN : Week 1: Basic concepts / Week 2: Work and heat ..."
  - Web course:  "Module 1: Review  Review of basic concepts - systems, surroundings, ..."
  - Table:       "Module* Topics and Contents No. of Lectures  1. CELL STRUCTURE ..."
Two-column PDFs interleave references and coordinator names into the topic text,
so fragments that look like citations, people, or boilerplate are dropped.
"""
import re
import sys

import pypdfium2 as pdfium

STOP = set("""the a an and or of to in for with on at by is are be as from that this it its its
we you your they their his her not no all any can may will shall must should would could
which who whom what when where how why into out up down over under other more most such
than then them these those there here also been being have has had do does did but if
via using use used based various different general basic introduction introductory
concept concepts review overview elementary fundamental fundamentals principle principles
course lecture lectures week module topic topics chapter unit part study students application applications
""".split())

START = re.compile(r"COURSE\s+PLAN|COURSE\s+OUTLINE|COURSE\s+DETAIL|COURSE\s+CONTENT|SYLLABUS", re.I)
CUT = re.compile(r"ABOUT\s+INSTRUCTOR|INTENDED\s+AUDIENCE|PRE-?\s*REQUISITES|INDUSTRY\s+SUPPORT|"
                 r"REFERENCES|SUGGESTED\s+READING|TEXT\s*BOOKS?|ADDITIONAL\s+READING", re.I)
JUNK = re.compile(
    r"\b(19|20)\d\d\b|edition|press\b|wiley|springer|mcgraw|pearson|prentice|elsevier|publish|"
    r"\bdr\.|\bprof\b|professor|coordinator|department|institute|\biit\b|iisc|nptel|http|www\.|"
    r"@|isbn|\bpp\.|vol\.|\bet al\b|tutorial|assignment|quiz|exam\b|lectures?\s*\d|"
    r"no\.\s*of|hours?\b|credit|target group|objective|this course|the course|students will|"
    r"undergraduate|postgraduate|\bug\b|\bpg\b|faculty|gate\b|"
    r"^course (outline|detail|plan|content)$|^(web|video) course$|"
    r"guwahati|kanpur|madras|delhi|bombay|kharagpur|roorkee|bangalore|chennai|mumbai|"
    # course-description prose, not topics
    r"\bcourse\b|\bstudents?\b|practitioners|teachers|\baims?\b|introduce[sd]?\b|will be|"
    r"\bdiscussed\b|\bexamined\b|\bcovered\b|\bexplained\b|\bemphasis\b|in addition|"
    r"hyperlink|^contents?$|^index$|^references?$", re.I)

# single words that are never a topic on their own
GENERIC_SINGLE = set("""definition definitions technique techniques science sciences serial production
engineering contents personally structure structures function functions mechanism mechanisms
method methods theory theories analysis design model models system systems process processes
property properties application applications example examples importance scope nature types
measurement measurements significance characterization calculation evaluation classification
introduction overview summary review conclusion discussion approach approaches issues issue
problems problem solution solutions tools framework frameworks action actions""".split())

WORD = re.compile(r"[a-z][a-z\-]{2,}")


def stem(w):
    for suf in ("ations", "ation", "ities", "ings", "ing", "ies", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_words(s):
    return [stem(w) for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 2]


def pdf_text(path):
    pdf = pdfium.PdfDocument(path)
    t = "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    pdf.close()
    return t


CONNECTOR = re.compile(r"\b(and|of|the|in|for|to|with|on|by|&|a|an|or|between|from|into)\s*$", re.I)


def join_wrapped(body):
    """Two-column PDFs wrap one phrase over several lines; stitch those back together.
    A line continues the previous one if the previous ends on a connector word or hyphen,
    or this one starts lowercase / with a bracket. '§' marks a hard break (week/module)."""
    out = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if out and not s.startswith("§"):
            prev = out[-1]
            if CONNECTOR.search(prev) or prev.endswith("-") or s[0].islower() or s[0] in "([":
                out[-1] = prev.rstrip("-") + ("" if prev.endswith("-") else " ") + s
                continue
        out.append(s)
    return "\n".join(out)


def topics_from_text(text):
    # NOC layout: objectives and instructor bio come BEFORE the week plan, so prefer the plan
    m = re.search(r"COURSE\s+PLAN", text, re.I) or START.search(text)
    body = text[m.end():] if m else text
    # drop a trailing references / instructor section, but only if it comes after real content
    c = CUT.search(body)
    if c and c.start() > 200:
        body = body[: c.start()]
    # week/module markers are hard breaks; numbered list items too
    body = re.sub(r"(Week|Module|Unit|Lecture)\s*[\*\d]+\s*[:.\-–]?", "\n§ ", body, flags=re.I)
    body = re.sub(r"^\s*(\d{1,2}|[ivxIVX]{1,4})[.)]\s+", "§ ", body, flags=re.M)
    body = join_wrapped(body).replace("§", "")
    frags = re.split(r"[\n,;•●▪◦]|\s[–—-]\s|:\s", body)

    seen, out = set(), []
    for f in frags:
        f = re.sub(r"\s+", " ", f).strip(" .-–—*()")
        if len(f) < 4 or JUNK.search(f):
            continue
        cw = content_words(f)
        # a topic is a short noun phrase; long prose sentences are objectives, not topics
        if not 1 <= len(cw) <= 7:
            continue
        # single generic word is not a topic ("Applications", "Definition", "Techniques")
        if len(cw) == 1 and (len(cw[0]) < 6 or f.strip().lower() in GENERIC_SINGLE
                             or cw[0] in GENERIC_SINGLE):
            continue
        # fragments that start mid-sentence are wrap debris, not topics
        if re.match(r"^(and|or|with|of|in|the|such as|etc)\b", f, re.I):
            continue
        key = " ".join(sorted(set(cw)))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def topics(path):
    return topics_from_text(pdf_text(path))


UNIT_HEAD = re.compile(r"^\s*(unit|module)\s+([ivxlc]+|\d+)\b[ .:\-]*(.*)$", re.I)
NUMBERED = re.compile(r"^\s*(\d{1,2})[.)]\s+(.+)$")
# everything after these headings is not syllabus content
END_OF_SYLLABUS = re.compile(r"^\s*(reference|text\s*books?|learning resources|course outcomes|"
                             r"co[- ]po|evaluation|assessment|books?\s*:|web\s*resources|"
                             r"suggested reading|e-?resources)", re.I)
# URLs and share-ids only. A bare "/" is normal syllabus text ("Functional/Non-Functional",
# "White-box -Statement/Branch Coverage", "CI/CD") and must not disqualify a topic.
LINKISH = re.compile(r"https?://|www\.|\.(com|org|in|edu)\b|/{2,}|[A-Za-z0-9_-]{18,}")


def topics_with_units(path):
    """University syllabi are usually 'Unit I <name>' followed by numbered topics.
    Returns [(unit_label, topic), ...]. The unit heading itself is NOT a topic - treating it as
    one made 'Unit V Large-Scale Software Design' match unrelated Unit-1 pages.
    Falls back to the flat parser when a document has no unit headings."""
    text = pdf_text(path)
    lines = [l.rstrip() for l in text.splitlines()]
    units, cur, buf = [], None, []

    def flush():
        """Each numbered syllabus item is ONE topic, kept whole.
        Splitting '23. Testing: Levels -Unit, Integration, System, Acceptance' on commas produced
        fragments such as 'Integration' that match almost any page, so items are no longer split."""
        if cur and buf:
            joined = join_wrapped("\n".join(buf))
            items, carry = [], ""
            for ln in joined.splitlines():
                m = NUMBERED.match(ln)
                if m:
                    if carry:
                        items.append(carry)
                    carry = m.group(2)
                elif carry and ln.strip():
                    carry += " " + ln.strip()           # wrapped continuation of the same item
            if carry:
                items.append(carry)
            if not items:                               # unit without numbering: use the lines
                items = [l for l in joined.splitlines() if l.strip()]
            for frag in items:
                frag = re.sub(r"\s+", " ", frag).strip(" .-\u2013\u2014*()")
                frag = re.sub(r"^(introduction|overview)\s*[:\-]\s*", "", frag, flags=re.I)
                if len(frag) < 6 or JUNK.search(frag) or LINKISH.search(frag):
                    continue
                if not 1 <= len(content_words(frag)) <= 25:
                    continue
                units.append((cur, frag))
        buf.clear()

    for ln in lines:
        if END_OF_SYLLABUS.match(ln):
            flush()
            cur = None
            continue
        m = UNIT_HEAD.match(ln)
        if m:
            flush()
            name = re.sub(r"\s+", " ", m.group(3)).strip()
            cur = f"Unit {m.group(2).upper()}" + (f" — {name}" if name else "")
        elif cur:
            buf.append(ln)
    flush()

    if len(units) < 5:                      # no usable unit structure: fall back
        return [(None, t) for t in topics_from_text(text)]
    seen, out = set(), []
    for u, t in units:
        k = (u, " ".join(sorted(set(content_words(t)))))
        if k not in seen:
            seen.add(k)
            out.append((u, t))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        ts = topics(p)
        print(f"\n=== {p}  ({len(ts)} topics)")
        for t in ts:
            print("   -", t)
