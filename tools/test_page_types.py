"""
Regression test for the page-type detectors in match.py.
Every case is a real page that was misclassified at some point, checked by hand.
Run: python test_page_types.py
"""
import json
import random
import re
import sys

from match import is_garbage, is_navigation

sys.stdout.reconfigure(encoding="utf-8")

CASES = [
    # unit, file, page, expected
    ("openstax__biology-2e", "biology-2e.pdf", 1, "nav"),                  # table of contents
    ("openstax__biology-2e", "biology-2e.pdf", 2, "nav"),                  # table of contents
    ("openstax__biology-2e", "biology-2e.pdf", 750, "ok"),                 # autocrine signalling text
    ("ocw__2-43-advanced-thermodynamics-spring-2024", "mit2_43_s24_index_slides.pdf", 9, "nav"),  # index
    ("ocw__10-40-chemical-engineering-thermodynamics-fall-2003", "readings_1040.pdf", 8, "nav"),  # schedule
    ("ocw__1-151-probability-and-statistics-in-engineering-spring-2005", "app19_fin_regs.pdf", 1, "ok"),  # worked example w/ table
    ("openstax__psychology-2e", "psychology-2e.pdf", 8, "nav"),            # chapter list
    ("ocw__6-828-operating-system-engineering-fall-2012", "MIT6_828F12_assignment5.pdf", 1, "ok"),  # exercise
    ("ocw__7-016-introductory-biology-fall-2018", "MIT7_016F18exam2.pdf", 1, "ok"),  # exam cover
    ("ocw__5-61-physical-chemistry-fall-2017", "MIT5_61F17_lec15.pdf", 7, "ok"),     # maths-heavy lecture
    ("ocw__6-003-signals-and-systems-fall-2011", "MIT6_003F11_notes.pdf", 5, "nav"),  # brief contents
    ("ocw__15-010-economic-analysis-for-business-decisions-fall-2004", "clr_prcng_pwr_ii.pdf", 17, "ok"),  # slide
    ("ocw__6-003-signals-and-systems-fall-2011", "MIT6_003F11_lec06.pdf", 13, "ok"),  # slide
    ("ocw__9-00sc-introduction-to-psychology-fall-2011", "MIT9_00SCF11_lec14_intel.pdf", 58, "ok"),  # slide
    ("ocw__18-440-probability-and-random-variables-spring-2014", "MIT18_440S14_Lecture4.pdf", 4, "ok"),  # axioms (maths)
    ("ocw__6-003-signals-and-systems-fall-2011", "MIT6_003F11_sol03.pdf", 3, "ok"),   # Z-transform solution
    # added after the 70-passage precision check
    ("openstax__psychology-2e", "psychology-2e.pdf", 1343, "ok"),        # prose + refs: refs cut by strip_references
    ("openstax__psychology-2e", "psychology-2e.pdf", 266, "nav"),        # reference list
    ("ocw__8-01sc-classical-mechanics-fall-2016", "MIT8_01F16_TableOfContents.pdf", 1, "nav"),  # TOC, no page nums
    ("ocw__2-43-advanced-thermodynamics-spring-2024", "mit2_43_s24_toc_slides.pdf", 5, "nav"),  # spaced dots
    ("ocw__18-05-introduction-to-probability-and-statistics-spring-2022", "mit18_05_s22_classfinal_pset.pdf", 1, "nav"),  # review outline
    ("ocw__8-01sc-classical-mechanics-fall-2016", "mit8_01scs22_chapter31.pdf", 2, "ok"),  # non-inertial frames text
    ("openstax__psychology-2e", "psychology-2e.pdf", 1575, "ok"),        # exercise with prose
]


def kind(text):
    if is_navigation(text):
        return "nav"
    if is_garbage(text):
        return "garbage"
    return "ok"


def main():
    cache, ok = {}, 0
    for u, fn, pg, exp in CASES:
        if u not in cache:
            cache[u] = json.load(open(f"cache/units/{u}.json", encoding="utf8"))["pages"]
        page = next(p for p in cache[u] if p["file"].endswith(fn) and p["page"] == pg)
        got = kind(page["text"])
        ok += got == exp
        print(f"  {'OK  ' if got == exp else 'FAIL'} {fn[:34]:34s} p.{pg:<5d} got={got:8s} expected={exp}")
    print(f"known cases: {ok}/{len(CASES)}")

    # garbage: the scanned soil-mechanics pages must be caught
    soil = json.load(open("cache/units/ocw__1-322-soil-behavior-spring-2005.json", encoding="utf8"))["pages"] \
        if any(True for _ in [0]) else []
    g = [p for p in soil if kind(p["text"]) == "garbage"]
    print(f"soil-behavior pages flagged garbage: {len(g)}/{len(soil)}")
    random.seed(5)
    for p in random.sample(g, min(3, len(g))):
        print("    ", p["file"].split("/")[-1][:28], f"p.{p['page']}", "|", re.sub(r"\s+", " ", p["text"])[:90])
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
