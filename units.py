"""
Work out which unit a syllabus topic and a material file belong to, so that Unit-1 notes are not
counted as evidence for Unit-4 topics.

A file's unit comes from its name ("Unit-1 Notes.pdf", "U3_slides.pptx", "Module 2 notes.docx")
or, failing that, from a unit heading inside its text. Files with no unit marking are treated as
applying to the whole course, which is the safe default for a single combined PDF.
"""
import re

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
# the trailing boundary must also accept "_" and "." ("U3_slides.pptx", "unit2.pdf")
FILE_UNIT = re.compile(r"\b(?:unit|module|chapter|u)[\s_\-]*([0-9]{1,2}|[ivx]{1,4})(?![0-9A-Za-z])", re.I)
TEXT_UNIT = re.compile(r"^\s*(?:unit|module)\s*[\-–]?\s*([0-9]{1,2}|[ivx]{1,4})\b", re.I | re.M)
# the same marker anywhere in a line, not only at its start
TEXT_UNIT_ANY = re.compile(r"\b(?:unit|module)\s*[\-–]?\s*([0-9]{1,2}|[ivx]{1,4})(?![0-9A-Za-z])",
                           re.I)
TITLE_CHARS = 900        # roughly the first page, where a deck states which unit it belongs to
TITLE_WEIGHT = 3         # a marker there outweighs a passing mention later


def to_number(token):
    t = str(token).strip().lower()
    if t.isdigit():
        return int(t)
    return ROMAN.get(t)


def unit_of_label(label):
    """'Unit III — Software Design' -> 3 ; None if there is no unit."""
    if not label:
        return None
    m = re.match(r"\s*(?:unit|module)\s+([0-9]{1,2}|[ivxIVX]{1,4})", str(label), re.I)
    return to_number(m.group(1)) if m else None


def unit_of_file(name, text=""):
    """Unit number for a material file, from its name first and its text second.

    The text search deliberately does not require the marker to start a line: lecture slides
    carry it inline, as in "Course Code ... Program Core Unit 3". When several unit numbers
    appear, the most frequent one wins rather than the file being given up as ambiguous - a
    Unit 3 deck that mentions Unit 2 once in a recap is still a Unit 3 deck. A tie, or a single
    stray mention, still returns None so the file applies to the whole course.
    """
    m = FILE_UNIT.search(name)
    if m:
        n = to_number(m.group(1))
        if n:
            return n
    text = text or ""
    counts = {}
    for m in TEXT_UNIT_ANY.finditer(text):
        n = to_number(m.group(1))
        if not n:
            continue
        # a marker on the title page is the strongest signal there is - lecture decks carry
        # "Program Core Unit 3" once, at the front, and never repeat it
        weight = TITLE_WEIGHT if m.start() < TITLE_CHARS else 1
        counts[n] = counts.get(n, 0) + weight
    if not counts:
        return None
    best, score = max(counts.items(), key=lambda kv: kv[1])
    runner_up = max((v for k, v in counts.items() if k != best), default=0)
    return best if score >= 2 and score > runner_up else None


def file_units(pages):
    """pages: (source, location, text). Returns {source: unit or None}."""
    by_source = {}
    for src, _, text in pages:
        by_source.setdefault(src, []).append(text)
    return {src: unit_of_file(src, " ".join(t[:1500] for t in texts[:5]))
            for src, texts in by_source.items()}


def units_from_scores(prob, topic_units, page_sources, top_k=5, margin=1.15):
    """Work out a file's unit from what it actually contains.

    The application has already scored every topic against every page, so a file that belongs to
    Unit 3 will score highly on Unit 3's topics and poorly on the rest. That is enough to place a
    file whose name and text say nothing - no extra model, no extra pass.

        prob          [n_topics x n_pages] probabilities from the evidence model
        topic_units   unit number per topic, None where the syllabus gave none
        page_sources  the file each page came from, in the same order as prob's columns

    Returns {source: unit}, holding only the files where one unit clearly leads. A file whose
    best unit is not `margin` times better than the runner-up is left out, because a confident
    wrong answer here is worse than none: the caller treats an unplaced file as applying to the
    whole course.
    """
    import numpy as np
    prob = np.asarray(prob)
    units = sorted({u for u in topic_units if u is not None})
    if len(units) < 2 or prob.size == 0:
        return {}

    out = {}
    for src in dict.fromkeys(page_sources):
        cols = [i for i, s in enumerate(page_sources) if s == src]
        if not cols:
            continue
        strength = {}
        for u in units:
            rows = [i for i, tu in enumerate(topic_units) if tu == u]
            if not rows:
                continue
            block = prob[np.ix_(rows, cols)]
            # how well this file's best pages answer this unit's topics
            best_per_topic = block.max(axis=1)
            k = min(top_k, len(best_per_topic))
            strength[u] = float(np.sort(best_per_topic)[-k:].mean())
        if len(strength) < 2:
            continue
        ranked = sorted(strength.items(), key=lambda kv: -kv[1])
        (best_u, best_s), (_, second_s) = ranked[0], ranked[1]
        if second_s <= 1e-9 or best_s / second_s >= margin:
            out[src] = best_u
    return out


def applicable(topic_unit, page_unit):
    """Can this page count as evidence for this topic?
    Yes when either side is unmarked, or when both point at the same unit."""
    if topic_unit is None or page_unit is None:
        return True
    return topic_unit == page_unit
