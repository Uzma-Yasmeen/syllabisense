"""
Turn the model's evidence into a teaching-depth judgement for each syllabus topic.

The neural network decides WHICH pages are evidence for a topic. This module asks HOW WELL the
topic is taught, from the evidence it found:

    NOT TAUGHT      no page scored above the threshold
    BARELY TAUGHT   evidence exists but is thin - one page, or only one kind of evidence
    PROPERLY TAUGHT several pages and more than one kind of evidence

The kinds of evidence are detected with explicit text cues, so every judgement can be explained
and checked by hand. This layer is deliberately rule-based, not learned: we have no human labels
for teaching depth, and inventing them would make the result unverifiable.
"""
import re

CUES = {
    "definition": re.compile(r"\b(is defined as|defined as|definition|refers to|is called|are called|"
                             r"we define|is known as|means that|denotes)\b", re.I),
    "explanation": re.compile(r"\b(because|therefore|which means|in other words|this implies|"
                              r"the reason|as a result|intuitively|note that)\b", re.I),
    "example": re.compile(r"\b(for example|for instance|e\.g\.|such as|consider the|worked example|"
                          r"case study|suppose that)\b", re.I),
    "formula": re.compile(r"[=≈≤≥±∑∏∫√∂]|\b\d+\s*[+\-*/^]\s*\d+|\\frac|\bequation\b", re.I),
    "exercise": re.compile(r"\b(problem \d|exercise|question \d|solve for|prove that|homework|"
                           r"quiz|exam)\b", re.I),
    "figure": re.compile(r"\b(figure \d|fig\. ?\d|diagram|shown below|illustrated|see figure|"
                         r"the plot|schematic)\b", re.I),
}
STRONG = ("definition", "explanation")        # a topic needs at least one of these to count as taught
SUPPORT = ("example", "formula", "figure")    # supporting evidence that deepens it


def evidence_kinds(text):
    return {k for k, pat in CUES.items() if pat.search(text)}


def judge(evidence, strong_conf=0.90):
    """evidence: list of (source, location, probability, text) above the threshold, best first.
    Returns (level, reasons, kinds)."""
    if not evidence:
        return "NOT TAUGHT", ["nothing in the material scored above the threshold"], set()

    kinds = set()
    for _, _, _, text in evidence:
        kinds |= evidence_kinds(text)
    pages = len(evidence)
    sources = len({e[0] for e in evidence})
    best = evidence[0][2]

    reasons = [f"{pages} page{'s' if pages > 1 else ''}"
               + (f" across {sources} files" if sources > 1 else "")]
    if kinds:
        reasons.append("has " + ", ".join(sorted(kinds)))
    else:
        reasons.append("no definition or explanation detected")

    has_strong = bool(kinds & set(STRONG))
    has_support = bool(kinds & set(SUPPORT))

    if pages >= 2 and has_strong and has_support:
        level = "PROPERLY TAUGHT"
    elif pages >= 3 and has_strong:
        level = "PROPERLY TAUGHT"
    else:
        level = "BARELY TAUGHT"
        if pages == 1:
            reasons.append("single page only")
        if not has_strong:
            reasons.append("looks like a passing mention")
        if not has_support:
            reasons.append("no example, formula or figure")
    if level == "PROPERLY TAUGHT" and best < strong_conf:
        level = "BARELY TAUGHT"
        reasons.append(f"best evidence only scored {best:.2f}")
    return level, reasons, kinds


def advice(level, kinds):
    if level == "NO MATERIAL PROVIDED":
        return "you did not upload material for this unit - cannot judge it"
    if level == "NOT TAUGHT":
        return "not in your material at all - you will need an outside source"
    if level == "BARELY TAUGHT":
        missing = [k for k in ("example", "figure", "formula") if k not in kinds]
        tail = f" (missing: {', '.join(missing)})" if missing else ""
        return f"too thin to learn from - find extra material{tail}"
    return "well covered - revise from the pages listed"


def study_order(results):
    """results: list of dicts with topic, level, order (position in the syllabus).
    Missing topics first, then thin ones; within a level keep syllabus order, because earlier
    topics are usually prerequisites for later ones."""
    rank = {"NOT TAUGHT": 0, "BARELY TAUGHT": 1, "PROPERLY TAUGHT": 2}
    # topics we could not judge are not study advice - they need material first
    todo = [r for r in results if r["level"] in rank and r["level"] != "PROPERLY TAUGHT"]
    return sorted(todo, key=lambda r: (rank[r["level"]], r["order"]))
