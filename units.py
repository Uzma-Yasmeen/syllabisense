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
    """Unit number for a material file, from its name first and its text second."""
    m = FILE_UNIT.search(name)
    if m:
        n = to_number(m.group(1))
        if n:
            return n
    found = {to_number(x) for x in TEXT_UNIT.findall(text or "")}
    found.discard(None)
    return found.pop() if len(found) == 1 else None      # ambiguous text -> treat as whole course


def file_units(pages):
    """pages: (source, location, text). Returns {source: unit or None}."""
    by_source = {}
    for src, _, text in pages:
        by_source.setdefault(src, []).append(text)
    return {src: unit_of_file(src, " ".join(t[:1500] for t in texts[:5]))
            for src, texts in by_source.items()}


def applicable(topic_unit, page_unit):
    """Can this page count as evidence for this topic?
    Yes when either side is unmarked, or when both point at the same unit."""
    if topic_unit is None or page_unit is None:
        return True
    return topic_unit == page_unit
