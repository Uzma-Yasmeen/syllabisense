"""
Write the frozen benchmark pairing: FINAL_PAIRS.json + FINAL_PAIRS.md

Includes the hand-audit results (evidence passages judged by reading each one; two
independent random samples per subject) so the quality of every pair is on record.
"""
import datetime
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

sel = json.load(open("selected_pairs.json", encoding="utf8"))
units = json.load(open("match_matrix.json", encoding="utf8"))["units"]

# hand audit: genuine evidence passages out of those read, combining two independent random
# samples of 5 per subject (final sample seed 13 + earlier seed 7 or seed 11 for the same pair)
AUDIT = {"Biology": "8/10", "Civil": "8/10", "Humanities": "8/10",
         "ChemEng": "7/10", "Electrical": "7/10", "Psychology": "7/10",
         "Chemistry": "6/10", "Economics": "6/10", "Mechanical": "6/10", "Physics": "6/10",
         "CS": "5/10", "Aero": "4/10", "Mathematics": "4/10",
         "Management": "3/10", "Sociology": "3/10"}
TIER = {
    "A": (["Biology", "Civil", "Humanities", "ChemEng", "Electrical", "Psychology",
           "Chemistry", "Economics", "Mechanical", "Physics"],
          "same subject, confirmed by reading; >= 6/10 evidence passages genuine"),
    "B": (["CS", "Aero", "Mathematics"],
          "same subject; evidence noisy (review outlines, word ambiguity e.g. 'Euler' equation vs Eulerian path)"),
    "C": (["Management", "Sociology"],
          "same broad subject but weak evidence (3/10): generic syllabus topics and preface/front-matter "
          "pages match easily. Usable, but the least reliable pairs"),
}
DROPPED = {}

tier_of = {d: t for t, (ds, _) in TIER.items() for d in ds}
final = {}
for d, r in sorted(sel.items()):
    if d in DROPPED:
        continue
    final[d] = {
        "tier": tier_of.get(d, "?"),
        "syllabus_id": r["syllabus"], "syllabus_title": r["title"],
        "syllabus_file": f"syllabi/{r['syllabus']}.pdf",
        "content_units": r["units"],
        "content_sources": [units[u]["source"] for u in r["units"]],
        "n_topics": r["n_topics"], "topics_found": r["found"], "topics_missing": r["missing"],
        "coverage": r["coverage"], "control": r["control"], "control_unit": r["control_unit"],
        "margin": r["margin"], "audit_genuine": AUDIT.get(d), "choice": r["choice"],
    }

meta = {"frozen": datetime.date.today().isoformat(),
        "method": "topic-level keyword matching with specificity (idf) weighting, 30-word window, "
                  "navigation/reference/garbage pages excluded; control = best unit from an unrelated "
                  "subject group; rules: margin >= 20%, coverage 40-92%, >= 5 topics each class",
        "audit": "final set: 75 passages, fresh random sample (seed 13), 40/75 genuine (53%). Per-subject scores combine two independent samples (10 passages each). Known false-positive sources: preface/front-matter pages, review outlines, word ambiguity - page-type filtering moves to preprocessing.",
        "important": "these pairs define WHICH content goes with WHICH syllabus. Topic-level labels are "
                     "NOT taken from this matcher; they come from the retrieval pipeline + human checking.",
        "dropped": DROPPED}
json.dump({"meta": meta, "pairs": final}, open("FINAL_PAIRS.json", "w", encoding="utf8"), indent=1)

L = ["# SyllabiSense - frozen benchmark pairs", "", f"Frozen: {meta['frozen']}", "",
     "| Tier | Subject | Syllabus (NPTEL) | Content | Topics | Covered | Coverage | Unrelated control | Margin | Audit (genuine) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for d, p in sorted(final.items(), key=lambda x: (x[1]["tier"], x[0])):
    L.append(f"| {p['tier']} | {d} | {p['syllabus_id']} {p['syllabus_title'].replace('|', '/')} | "
             f"{' + '.join(s.replace('MIT OCW ', 'MIT ') for s in p['content_sources'])} | {p['n_topics']} | "
             f"{p['topics_found']} | {100*p['coverage']:.0f}% | {100*p['control']:.0f}% | "
             f"+{100*p['margin']:.0f}% | {p['audit_genuine']} |")
L += ["", "**Tiers**", ""]
for t, (ds, why) in TIER.items():
    L.append(f"- **{t}**: {why}")
L += (["", "**Dropped**", ""] + [f"- **{d}**: {why}" for d, why in DROPPED.items()]) if DROPPED else []
L += ["", "**Overrides of the automatic choice**", ""]
L += [f"- **{d}**: {p['choice'][10:]}" for d, p in final.items() if p["choice"].startswith("override")]
L += ["", "**Method**", "", meta["method"], "", "**Audit**", "", meta["audit"], "",
      "**Important**", "", meta["important"]]
open("FINAL_PAIRS.md", "w", encoding="utf8").write("\n".join(L))
print("\n".join(L))
