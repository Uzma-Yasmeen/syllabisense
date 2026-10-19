"""
Pick the final syllabus <-> content pairs from match_matrix.json.

Rules (fixed before looking at results):
  - control margin : coverage on own-domain content must beat the best unit from an
                     UNRELATED subject group by >= MARGIN, else it is generic vocabulary
  - coverage band  : between LO and HI  -> real overlap, but real gaps left to learn from
  - examples       : at least 5 topics in each class (covered / missing)
  - ranking        : margin x sqrt(smaller class) - clearly related first, then more examples
                     (ranking purely on balance favoured loosely related syllabi near 50%)
Content may combine up to 2 units, like a student receiving several PDFs.
"""
import itertools
import json
import sys

from syllabus_topics import content_words

sys.stdout.reconfigure(encoding="utf-8")
MARGIN, LO, HI = 0.20, 0.40, 0.92

M = json.load(open("match_matrix.json", encoding="utf8"))
H = json.load(open("match_hits.json", encoding="utf8"))
UNITS = M["units"]
# Engineering courses that contain a real computing/programming course inside them, so they are
# not "unrelated" to CS syllabi. Verified from evidence: 16.01 matched "Linked lists" to a linked-list
# problem set (pset8_sol_04.pdf), "list and vector" to Ada generics, "software design" to its
# software-process notes.
HAS_COMPUTING_THREAD = {"ocw__16-01-unified-engineering-i-ii-iii-iv-fall-2005-spring-2006"}


# One-word topics whose word is on >1% of all pages ("services", "standard", "probability") match
# unrelated books as easily as the real subject. Measured by generic_topics.py; they are removed
# from BOTH the numerator and denominator, for the real content and the control alike.
SINGLE_DF = json.load(open("cache/single_word_df.json", encoding="utf8"))
GENERIC_SINGLE_CUTOFF = 0.01


def generic_topics(topics):
    out = set()
    for t in topics:
        w = set(content_words(t))
        if len(w) == 1 and SINGLE_DF.get(next(iter(w)), 0) > GENERIC_SINGLE_CUTOFF:
            out.add(t)
    return out


def evaluate(cid, s):
    dom, grp = s["domain"], s["group"]
    drop = generic_topics(s["topics"])
    n_kept = s["n_topics"] - len(drop)
    if n_kept < 8:
        return []
    # recompute coverage for every unit from the stored per-topic hits, without the generic topics
    cov = {u: len(set(H.get(f"{cid}|{u}", {})) - drop) / n_kept for u in s["coverage"]}
    s = dict(s, n_topics=n_kept)
    own = [u for u in cov if UNITS[u]["domain"] == dom]
    # control = best score on a GENUINELY unrelated field (different subject group)
    other = [u for u in cov if UNITS[u]["group"] != grp
             and not (grp == "comp" and u in HAS_COMPUTING_THREAD)]
    if not own or not other:
        return []
    ctrl_u = max(other, key=lambda u: cov[u])
    ctrl = cov[ctrl_u]
    n = s["n_topics"]
    # top own units, then single + pairs
    own = sorted(own, key=lambda u: -cov[u])[:6]
    opts = [(u,) for u in own] + list(itertools.combinations(own, 2))
    res = []
    for combo in opts:
        found = {}
        for u in combo:
            for t, h in H.get(f"{cid}|{u}", {}).items():
                if t not in drop:
                    found.setdefault(t, (u, h))
        c = len(found) / n
        # a second unit must add something real, not just be carried along
        if len(combo) == 2 and c - max(cov[combo[0]], cov[combo[1]]) < 0.05:
            continue
        res.append({"syllabus": cid, "title": s["title"], "domain": dom, "units": list(combo),
                    "n_topics": n, "found": len(found), "missing": n - len(found),
                    "coverage": round(c, 3), "control_unit": ctrl_u, "control": ctrl,
                    "margin": round(c - ctrl, 3), "evidence": found})
    return res


def main():
    allres = []
    for cid, s in M["syllabi"].items():
        allres.extend(evaluate(cid, s))
    ok = [r for r in allres if r["margin"] >= MARGIN and LO <= r["coverage"] <= HI
          and min(r["found"], r["missing"]) >= 5]
    for r in allres:
        # rank by how clearly the pair beats the unrelated control, scaled by usable examples
        r["score"] = round(r["margin"] * min(r["found"], r["missing"]) ** 0.5, 3)
    by_dom = {}
    for r in ok:
        by_dom.setdefault(r["domain"], []).append(r)

    domains = sorted({s["domain"] for s in M["syllabi"].values()})
    chosen = {}
    print("%-12s %-10s %-34s %5s %5s %6s %6s %6s  %s" %
          ("DOMAIN", "SYLLABUS", "TITLE", "TOPIC", "FOUND", "COVER", "CTRL", "MARGIN", "CONTENT"))
    print("-" * 150)
    for d in domains:
        rs = sorted(by_dom.get(d, []), key=lambda r: -r["score"])
        if not rs:
            # show why it failed: best own-domain option regardless of rules
            cands = sorted([r for r in allres if r["domain"] == d], key=lambda r: -r["margin"])
            if cands:
                r = cands[0]
                print("%-12s %-10s %-34s %5d %5d %5.0f%% %5.0f%% %+5.0f%%  NO VALID PAIR (best shown) %s" %
                      (d, r["syllabus"], r["title"][:34], r["n_topics"], r["found"], 100 * r["coverage"],
                       100 * r["control"], 100 * r["margin"], " + ".join(r["units"])[:60]))
            else:
                print("%-12s  no syllabus with enough topics / no content" % d)
            continue
        r = rs[0]
        chosen[d] = r
        print("%-12s %-10s %-34s %5d %5d %5.0f%% %5.0f%% %+5.0f%%  %s" %
              (d, r["syllabus"], r["title"][:34], r["n_topics"], r["found"], 100 * r["coverage"],
               100 * r["control"], 100 * r["margin"], " + ".join(r["units"])))
    json.dump(chosen, open("selected_pairs.json", "w", encoding="utf8"), indent=1)
    json.dump(ok, open("valid_pairs.json", "w", encoding="utf8"))
    json.dump({d: sorted(v, key=lambda r: -r["score"])[:5]
               for d, v in by_dom.items()}, open("candidate_pairs.json", "w", encoding="utf8"), indent=1)
    print(f"\nselected {len(chosen)} domains -> selected_pairs.json (top-5 alternatives in candidate_pairs.json)")


if __name__ == "__main__":
    main()
