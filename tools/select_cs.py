"""
CS block: one syllabus <-> content pair per CS SUBJECT (not one for all of CS).

Same rules as select_pairs.py (read from valid_pairs.json, which already enforces them):
margin >= 20% over unrelated content, coverage 40-92%, >= 5 topics covered and missing.
A pair counts for a subject only if BOTH sides are about it: the syllabus title matches the
subject and every content course in the pair belongs to it. NPTEL "Web" and "Video" versions of
one course count as the same syllabus. Output: cs_pairs.json + table.
"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

CS_SUBJECT = [  # MIT slug prefix -> CS subject
    (("6-0001", "6-100l", "6-087", "6-096", "6-s096", "6-092", "6-088", "1-00-"), "Programming"),
    (("6-0002", "15-071", "15-062"), "Data Science"),
    (("6-042j",), "Discrete Mathematics"),
    (("6-006", "6-851"), "Data Structures"),
    (("6-046j", "6-006"), "Design & Analysis of Algorithms"),
    (("6-111", "6-004"), "Digital Logic"),
    (("6-004", "6-823"), "Computer Organisation & Architecture"),
    (("6-828",), "Operating Systems"),
    (("6-033",), "Computer Systems"),
    (("6-830", "6-5830", "1-264j"), "Databases"),
    (("6-829", "6-263j", "6-02-"), "Computer Networks"),
    (("18-404j", "6-045j"), "Theory of Computation"),
    (("6-035",), "Compilers"),
    (("6-005", "6-170", "16-355j", "1-124j"), "Software Engineering"),
    (("6-034",), "Artificial Intelligence"),
    (("6-036", "6-867"), "Machine Learning"),
    (("6-s191", "6-7960", "15-773", "6-s087"), "Deep Learning"),
    (("6-864", "6-863j"), "Natural Language Processing"),
    (("6-801", "6-8300"), "Computer Vision"),
    (("6-858", "6-857", "6-875"), "Security & Cryptography"),
    (("6-824", "6-852j", "6-172"), "Distributed & Parallel Systems"),
    (("6-837",), "Computer Graphics"),
    (("6-831",), "Human-Computer Interaction"),
]


def subjects_of(uid):
    """A course can serve more than one subject (6.006 teaches data structures AND algorithms)."""
    name = uid.split("__", 1)[1]
    return {subj for prefixes, subj in CS_SUBJECT
            if any(name.startswith(p if p.endswith("-") else p + "-") for p in prefixes)}


# the SYLLABUS must be about the subject too - an earlier version only checked the content side
# and produced pairs like "Computer Vision <- Optimisation for Machine Learning"
TITLE_SUBJECT = [
    (r"deep learning|neural network", "Deep Learning"),
    (r"machine learning|pattern recognition", "Machine Learning"),
    (r"artificial intelligence", "Artificial Intelligence"),
    (r"natural language|\bnlp\b", "Natural Language Processing"),
    (r"computer vision|image processing|machine vision", "Computer Vision"),
    (r"data science|data analytics|data mining", "Data Science"),
    (r"compiler", "Compilers"),
    (r"theory of computation|automata|formal language|computability", "Theory of Computation"),
    (r"operating system", "Operating Systems"),
    (r"data ?base|dbms", "Databases"),
    (r"computer network|data communication|networking|\bnetworks\b", "Computer Networks"),
    (r"security|cryptograph", "Security & Cryptography"),
    (r"distributed|parallel|cloud computing|high performance", "Distributed & Parallel Systems"),
    (r"graphics", "Computer Graphics"),
    (r"human.computer|user interface|\bhci\b", "Human-Computer Interaction"),
    (r"software (engineering|testing|design|construction)|software conceptual", "Software Engineering"),
    (r"computer architecture|computer organi[sz]ation", "Computer Organisation & Architecture"),
    (r"switching|logic design|digital (logic|circuits|design)", "Digital Logic"),
    (r"discrete math|graph theory|mathematics for computer", "Discrete Mathematics"),
    (r"data structure", "Data Structures"),
    (r"algorithm", "Design & Analysis of Algorithms"),
    (r"computer systems?\b|system design", "Computer Systems"),
    (r"programming", "Programming"),
]
# advanced electives lose to the core syllabus of the same subject whenever the core one is valid
ADVANCED = re.compile(r"advanced|selected topics|special topics|multi-core|randomi[sz]ed|parallel algorithm|"
                      r"graph theory|circuit complexity|computational complexity|static analysis|"
                      r"functional programming|game theory|optimi[sz]ation for", re.I)


def title_subject(title):
    for pat, subj in TITLE_SUBJECT:
        if re.search(pat, title, re.I):
            return subj
    return None


def course_key(title):
    """'Compiler Design - Video course' and '... - Web course' are the same course."""
    t = re.sub(r"[-–]\s*(web|video)\s*(course)?.*$", "", title, flags=re.I)
    return re.sub(r"\W+", " ", t).strip().lower()


M = json.load(open("match_matrix.json", encoding="utf8"))
V = [r for r in json.load(open("valid_pairs.json", encoding="utf8")) if r["domain"] == "CS"]
cov = {cid: s["coverage"] for cid, s in M["syllabi"].items()}

best = {}   # subject -> best pair
for r in V:
    subj = title_subject(r["title"])
    if not subj:
        continue
    # every content course in the pair must belong to the syllabus's subject
    if not all(subj in subjects_of(u) for u in r["units"]):
        continue
    r = dict(r, subject=subj, core=not ADVANCED.search(r["title"]))
    rank = (r["core"], r["score"])      # core syllabus first, then best score
    if subj not in best or rank > (best[subj]["core"], best[subj]["score"]):
        best[subj] = r

# one course may appear at most once (it can only be the best pair for one subject)
used, final = set(), {}
for subj, r in sorted(best.items(), key=lambda x: -x[1]["score"]):
    k = course_key(r["title"])
    if k in used:
        continue
    used.add(k)
    final[subj] = r

json.dump(final, open("cs_pairs.json", "w", encoding="utf8"), indent=1)
print("%-36s %-5s %-10s %-40s %5s %6s %7s  %s" % ("CS SUBJECT", "KIND", "SYLLABUS", "TITLE", "TOPIC", "COVER",
                                                  "MARGIN", "CONTENT"))
for subj, r in sorted(final.items()):
    print("%-36s %-5s %-10s %-40s %5d %5.0f%% %+6.0f%%  %s" % (
        subj, "core" if r["core"] else "elect", r["syllabus"], r["title"][:40], r["n_topics"],
        100 * r["coverage"], 100 * r["margin"],
        " + ".join(u.split("__")[1][:30] for u in r["units"])))
missing = [s for _, s in CS_SUBJECT if s not in final]
print(f"\n{len(final)} CS subjects paired. No valid pair yet: {', '.join(missing) or 'none'}")
