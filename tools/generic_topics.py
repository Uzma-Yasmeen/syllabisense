"""
Measure how common each ONE-WORD syllabus topic's word is across all content pages.

One-word topics like "Services", "Standards", "coding" match sociology and business books as easily
as the real subject, which both inflates coverage and erases the margin over unrelated content.
This computes the page frequency of every one-word topic so a cut-off can be chosen by looking at
what it removes. Output: cache/single_word_df.json {word: fraction_of_pages}
"""
import collections
import glob
import json
import sys

from syllabus_topics import content_words

sys.stdout.reconfigure(encoding="utf-8")

M = json.load(open("match_matrix.json", encoding="utf8"))
single = collections.Counter()
for s in M["syllabi"].values():
    for t in s["topics"]:
        w = content_words(t)
        if len(set(w)) == 1:
            single[w[0]] += 1
print(f"one-word topic words: {len(single)} distinct, {sum(single.values())} topic uses")

df, N = collections.Counter(), 0
for f in glob.glob("cache/units/*.json"):
    for p in json.load(open(f, encoding="utf8"))["pages"]:
        N += 1
        toks = set(content_words(p["text"])) & single.keys()
        df.update(toks)
frac = {w: df[w] / N for w in single}
json.dump(frac, open("cache/single_word_df.json", "w", encoding="utf8"), indent=1)

for lo, hi in [(0.02, 1), (0.01, 0.02), (0.005, 0.01), (0.0025, 0.005)]:
    ws = sorted((w for w in single if lo <= frac[w] < hi), key=lambda w: -single[w])
    print(f"\n{100*lo:.2f}%-{100*hi:.0f}% of pages: {len(ws)} words")
    print("   ", ", ".join(f"{w}({100*frac[w]:.1f}%)" for w in ws[:40]))
