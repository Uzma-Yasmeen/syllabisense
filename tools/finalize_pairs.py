"""
Final pair choice = automatic selection + a few documented overrides.

Overrides are only allowed where valid_pairs.json already contains the alternative
(so it passed every rule) AND there is a stated reason. Everything is recorded.
"""
import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

auto = json.load(open("selected_pairs.json", encoding="utf8"))
valid = json.load(open("valid_pairs.json", encoding="utf8"))   # every pair that passed all rules
shutil.copy("selected_pairs.json", "selected_pairs_auto.json")

OVERRIDES = {
    # domain: (syllabus, units, reason)
    "Mathematics": ("111101004", ["ocw__18-05-introduction-to-probability-and-statistics-spring-2022"],
                    "same subject (probability); margin +47% vs +24% for Advanced Engineering Mathematics, "
                    "whose linear-algebra/complex-analysis topics were being matched to probability pages"),
    "Physics": ("115106059", ["ocw__8-01sc-classical-mechanics-fall-2016"],
                "Classical Mechanics is the actual subject of MIT 8.01; Special Relativity was chosen on a near-tie"),
    "ChemEng": ("103101005", ["ocw__10-37-chemical-and-biological-reaction-engineering-spring-2007"],
                "same subject (reaction engineering); audit 4/5. Auto pick 'Molecular Simulations' won a "
                "near-tie (1.39 vs 1.37) with an odd content mix"),
    "Civil": ("105103027", ["ocw__1-010-uncertainty-in-engineering-fall-2008"],
              "drop 'Soil behavior': its only contribution in the audit was a false match; still +37%"),
    "Management": ("110106164", ["openstax__introduction-to-business"],
                   "business development matches the textbook's scope; auto pick 'Financial Accounting' is "
                   "covered by only one chapter of that book"),
    "Psychology": ("109104025", ["ocw__9-00sc-introduction-to-psychology-fall-2011", "openstax__psychology-2e"],
                   "auto pick 'Selected Topics in Psychology' has junk topics (lecture numbers + author names, "
                   "'funded by MHRD'), audit 1/5; Abnormal Psychology audited 5/5"),
}

final = {}
for dom, r in auto.items():
    r = dict(r)
    r["choice"] = "automatic"
    final[dom] = r
for dom, (sid, units, why) in OVERRIDES.items():
    match = [c for c in valid if c["domain"] == dom and c["syllabus"] == sid and c["units"] == units]
    if not match:
        sys.exit(f"override for {dom} is not a valid candidate - refusing")
    r = dict(match[0])
    r["choice"] = "override: " + why
    final[dom] = r

json.dump(final, open("selected_pairs.json", "w", encoding="utf8"), indent=1)
print("%-12s %-10s %-40s %5s %5s %6s %6s %7s  %s" %
      ("DOMAIN", "SYLLABUS", "TITLE", "TOPIC", "FOUND", "COVER", "CTRL", "MARGIN", "CONTENT"))
for d, r in sorted(final.items()):
    print("%-12s %-10s %-40s %5d %5d %5.0f%% %5.0f%% %+6.0f%%  %s%s" %
          (d, r["syllabus"], r["title"][:40], r["n_topics"], r["found"], 100 * r["coverage"],
           100 * r["control"], 100 * r["margin"], " + ".join(u.split("__")[1][:34] for u in r["units"]),
           "   *" if r["choice"] != "automatic" else ""))
print("\n* = documented override (reasons stored in selected_pairs.json)")
