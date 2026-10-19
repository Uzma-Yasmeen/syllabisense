"""
Print real evidence for selected pairs so a human can judge whether matches are genuine.

For each domain: N random FOUND topics with the passage from the page that was counted,
and N random MISSING topics. Seeded, so the same sample can be re-checked later.

Usage: python spot_check.py [N] [domain]
"""
import json
import random
import re
import sys

from syllabus_topics import content_words

sys.stdout.reconfigure(encoding="utf-8")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ONLY = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "all" else None
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42

import os
sel = json.load(open(os.environ.get("SEL", "selected_pairs.json"), encoding="utf8"))
_pages = {}


def page_text(uid, file, page):
    if uid not in _pages:
        d = json.load(open(f"cache/units/{uid}.json", encoding="utf8"))
        _pages[uid] = {(p["file"], p["page"]): p["text"] for p in d["pages"]}
    return _pages[uid].get((file, page), "")


def snippet(text, topic, width=260):
    words = set(content_words(topic))
    flat = re.sub(r"\s+", " ", text)
    best, best_i = 0, 0
    toks = list(re.finditer(r"[A-Za-z][A-Za-z\-]{2,}", flat))
    for k, m in enumerate(toks):
        hit = sum(1 for m2 in toks[k:k + 40] if content_words(m2.group()) and content_words(m2.group())[0] in words)
        if hit > best:
            best, best_i = hit, m.start()
    s = max(0, best_i - 40)
    return flat[s:s + width]


for dom, r in sorted(sel.items()):
    if ONLY and dom != ONLY:
        continue
    rng = random.Random(SEED)
    found = list(r["evidence"].items())
    all_topics = json.load(open("match_matrix.json", encoding="utf8"))["syllabi"][r["syllabus"]]["topics"]
    missing = [t for t in all_topics if t not in r["evidence"]]
    print("=" * 110)
    print(f"{dom}: {r['syllabus']} {r['title']}  <->  {' + '.join(r['units'])}")
    print(f"coverage {100*r['coverage']:.0f}%  control {100*r['control']:.0f}% ({r['control_unit']})")
    for t, (uid, (file, page)) in rng.sample(found, min(N, len(found))):
        print(f"\n  FOUND  '{t}'")
        print(f"         {file.split('/')[-1]} p.{page}")
        print(f"         \"...{snippet(page_text(uid, file, page), t)}...\"")
    print("\n  MISSING:", " | ".join(rng.sample(missing, min(N * 2, len(missing)))))
