"""
Build a quiz from the material the student actually has.

Questions are made only from pages the evidence model accepted, so the quiz never asks about a
topic the material does not cover - the coverage report is what reports those. Every question
carries the file and page it came from, so a disputed answer can be checked rather than argued
about.

No language model is involved. The hard part of a multiple-choice question is the wrong answers:
random ones are guessable at a glance. Here the distractors are drawn from other topics in the
same course and ranked by the project's own TF-IDF and LSA vectors, so they are near misses -
plausible to someone who half-knows the material, wrong to someone who knows it.

    from quiz import build_quiz
    questions = build_quiz(results, feat, n=10)

Each question is a Question: kind, stem, options, answer, why, topic, source.
"""
import random
import re
from dataclasses import dataclass, field

import numpy as np

from depth import CUES
from syllabus_topics import STOP, content_words

TOPIC_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")


def surface_terms(topic):
    """The topic's own words as they are spelled.

    content_words() stems — "Prototyping" becomes "prototyp" — which is right for matching and
    wrong for showing someone. A distractor has to look like a word a lecturer would write.
    """
    seen, out = set(), []
    for w in TOPIC_WORD.findall(str(topic)):
        low = w.lower()
        if low in STOP or low in seen or len(low) < 5:
            continue
        seen.add(low)
        out.append(low)
    return out


def clean_topic(topic):
    """The topic as a reader should see it in a question stem."""
    t = " ".join(str(topic).split()).strip(" ,;:-–")
    t = re.sub(r"^[IVX0-9]+[.)]\s*", "", t)
    return t

# "X is a technique that ...", "X refers to ..."
DEFINES = re.compile(
    r"\b([A-Z][\w\- ]{2,48}?)\s+(?:is|are)\s+(?:a|an|the)?\s*"
    r"(?:defined as|known as|called)?\s*([^.;]{25,190})", re.I)
# "Functional Requirements: These describe what the system should do." Lecture notes are written
# in bullets far more often than in prose, and this is the shape they take - on the material
# tested, four sentences matched DEFINES and ninety matched this.
LABELLED = re.compile(r"(?:^|[●•·▪\-]\s*)([A-Z][\w\-/ ]{3,46}):\s+([^.;]{25,190})")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
WORDISH = re.compile(r"^[A-Za-z][A-Za-z\-]{3,}$")
BAD_START = re.compile(r"^(this|that|these|those|it|they|there|here|we|you|i|he|she)\b", re.I)


@dataclass
class Question:
    kind: str                       # definition | term | true_false
    stem: str
    options: list
    answer: int                     # index into options
    topic: str
    source: tuple                   # (file, location)
    why: str = ""
    distractor_note: str = field(default="", repr=False)

    @property
    def correct(self):
        return self.options[self.answer]


def sentences(text):
    flat = " ".join(str(text).split())
    return [s.strip() for s in SENTENCE.split(flat) if 40 <= len(s.strip()) <= 240]


def usable(s):
    """A sentence worth asking about: prose, not a heading, header or list of fragments."""
    if BAD_START.match(s):
        return False
    words = s.split()
    if len(words) < 7:
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    if caps / len(words) > 0.6:              # a title or a slide heading
        return False
    if sum(c.isdigit() for c in s) / len(s) > 0.25:
        return False
    return True


class Distractors:
    """Picks wrong answers that are close to right.

    Topics are placed in the LSA space the evidence model already uses, so "the topic most
    similar to this one, that is not this one" is a lookup rather than a guess. A distractor
    drawn from a far-away topic is obviously wrong; one drawn from the nearest neighbour makes
    the question worth answering.
    """

    def __init__(self, topics, feat):
        self.topics = list(topics)
        vec, svd = feat.vec, feat.svd
        V = svd.transform(vec.transform(self.topics))
        norm = np.linalg.norm(V, axis=1, keepdims=True)
        self.V = V / np.where(norm == 0, 1, norm)

    def nearest(self, topic, k=6):
        if topic not in self.topics:
            return []
        i = self.topics.index(topic)
        sim = self.V @ self.V[i]
        sim[i] = -2.0
        return [self.topics[j] for j in np.argsort(-sim)[:k]]


TRAILING_JUNK = re.compile(r"[\s,;:\-–(]+$")
ENDS_BADLY = re.compile(r"(including|such as|e\.g|like|and|or|the|a|an|with|for|to|of|in)$", re.I)


def tidy_clause(clause):
    """A clause fit to show as an answer, or None.

    Lecture notes are full of bullets and half-sentences; an option that trails off mid-thought
    makes a question feel broken even when the answer is right.
    """
    c = " ".join(str(clause).split())
    c = re.sub(r"^[\s●•·▪\-–*]+", "", c)
    # a list that was bulleted in the slide reads as one run-on line once flattened
    if len(re.findall(r"[●•·▪]", c)) >= 2:
        return None
    c = re.sub(r"\s*[●•·▪]\s*", ", ", c)
    c = TRAILING_JUNK.sub("", c)
    if not (32 <= len(c) <= 180) or len(c.split()) < 6:
        return None
    if ENDS_BADLY.search(c) or re.search(r"\w+\s+\d+$", c):
        return None
    if sum(ch.isdigit() for ch in c) / len(c) > 0.12:
        return None
    return c


def definition_in(sentence):
    """(subject, definition) from either shape a lecturer writes, or None."""
    for pattern in (LABELLED, DEFINES):
        m = pattern.search(sentence)
        if m:
            subject = m.group(1).strip()
            clause = tidy_clause(m.group(2))
            if clause:
                return subject, clause
    return None


def about(topic, subject):
    """Is this sentence defining this topic? Compared on stems, so Requirement matches
    Requirements and Prototyping matches Prototype."""
    a, b = set(content_words(topic)), set(content_words(subject))
    return bool(a & b) if a else True


def _options(correct, pool, rng, n=3, usage=None):
    """`n` distinct wrong answers, or None. Duplicates make a question unanswerable, and a
    distractor equal to the right answer makes it worse than unanswerable."""
    seen, wrong = {str(correct).strip().lower()}, []
    if usage is not None:                       # least-used first, so answers stop repeating
        pool = sorted(pool, key=lambda c: usage.get(str(c).strip().lower(), 0))
    for c in pool:
        key = str(c).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        wrong.append(c)
        if len(wrong) == n:
            break
    if len(wrong) < n:
        return None
    if usage is not None:
        for c in wrong:
            key = str(c).strip().lower()
            usage[key] = usage.get(key, 0) + 1
    options = [correct] + wrong
    rng.shuffle(options)
    return options


def _definition_question(topic, evidence, pool, rng, used, usage):
    """'Which of these describes X?' - the real clause against clauses from neighbouring topics."""
    terms = set(surface_terms(topic))
    for src, loc, _, text in evidence:
        for s in sentences(text):
            if s in used or not usable(s):
                continue
            found = definition_in(s)
            if not found:
                continue
            subject, clause = found
            # the sentence must actually be defining this topic, not something next to it
            if not about(topic, subject):
                continue
            options = _options(clause, [c for c in pool if abs(len(c) - len(clause)) < 140],
                               rng, usage=usage)
            if not options:
                continue
            used.add(s)
            return Question(
                kind="definition",
                stem=f"Which of these describes **{clean_topic(topic)}**?",
                options=options, answer=options.index(clause), topic=topic,
                source=(src, loc),
                why="Taken from the sentence that defines it in your material.",
                distractor_note="wrong answers are definitions of nearby topics")
    return None


def _term_question(topic, evidence, neighbours, rng, used):
    """A sentence from the material with its key term removed."""
    terms = set(surface_terms(topic))
    if not terms:
        return None
    for src, loc, _, text in evidence:
        for s in sentences(text):
            if s in used or not usable(s):
                continue
            hit = next((w for w in s.split()
                        if WORDISH.match(w.strip(",.;:()")) and
                        w.strip(",.;:()").lower() in terms), None)
            if not hit:
                continue
            key = hit.strip(",.;:()")
            blanked = s.replace(key, "________", 1)
            # if the word still appears after blanking, the answer is sitting in the question
            if re.search(rf"\b{re.escape(key)}\b", blanked, re.I):
                continue
            pool = [w for nb in neighbours for w in surface_terms(nb) if w not in terms]
            options = _options(key.lower(), pool, rng)
            if not options:
                continue
            used.add(s)
            return Question(
                kind="term",
                stem=f"Fill the blank:\n\n> {blanked}",
                options=options, answer=options.index(key.lower()), topic=topic,
                source=(src, loc),
                why=f"The sentence in your material reads “{key}”.",
                distractor_note="wrong answers are terms from the most similar topics")
    return None


def _true_false_question(topic, evidence, neighbours, rng, used):
    """A real statement, or the same statement with its subject swapped for a near neighbour."""
    terms = set(surface_terms(topic))
    for src, loc, _, text in evidence:
        for s in sentences(text):
            if s in used or not usable(s) or not CUES["explanation"].search(s):
                continue
            if rng.random() < 0.5 or not neighbours:
                used.add(s)
                return Question(
                    kind="true_false", stem=f"True or false?\n\n> {s}",
                    options=["True", "False"], answer=0, topic=topic, source=(src, loc),
                    why="This sentence appears in your material as written.")
            hit = next((w for w in s.split() if w.strip(",.;:()").lower() in terms), None)
            if not hit:
                continue
            swap = next((w for nb in neighbours for w in surface_terms(nb) if w not in terms),
                        None)
            if not swap:
                continue
            used.add(s)
            altered = s.replace(hit, swap.capitalize() if hit[:1].isupper() else swap, 1)
            return Question(
                kind="true_false", stem=f"True or false?\n\n> {altered}",
                options=["True", "False"], answer=1, topic=topic, source=(src, loc),
                why=f"Your material says “{hit.strip(',.;:()')}” here, not “{swap}”.",
                distractor_note="one term swapped for one from a neighbouring topic")
    return None


def build_quiz(results, feat, n=10, seed=None, levels=("PROPERLY TAUGHT", "BARELY TAUGHT")):
    """Questions drawn from the topics the material actually covers.

    results : the application's result list, each with topic and evidence
    feat    : the TfidfFeaturiser, reused for its vectoriser and LSA space
    """
    rng = random.Random(seed)
    covered = [r for r in results if r["level"] in levels and r.get("evidence")]
    if len(covered) < 2:
        return []
    topics = [r["topic"] for r in covered]
    near = Distractors(topics, feat)
    by_topic = {r["topic"]: r for r in covered}

    # one definition clause per topic, so a question about A can borrow wrong answers from B
    clauses = {}
    for r in covered:
        for _, _, _, text in r["evidence"]:
            for s in sentences(text):
                found = definition_in(s) if usable(s) else None
                if found:
                    clauses[r["topic"]] = found[1]
                    break
            if r["topic"] in clauses:
                break

    order = list(covered)
    rng.shuffle(order)
    out, used, usage = [], set(), {}   # used: no sentence twice. usage: spread the distractors.
    builders = [_definition_question, _term_question, _true_false_question]
    for i, r in enumerate(order):
        if len(out) >= n:
            break
        topic, ev = r["topic"], r["evidence"]
        neighbours = near.nearest(topic)
        pool = [clauses[t] for t in neighbours if t in clauses and t != topic]
        # rotate which kind is tried first, so one easy-to-build kind cannot crowd out the rest
        rotated = builders[i % len(builders):] + builders[:i % len(builders)]
        for build in rotated:
            q = (build(topic, ev, pool, rng, used, usage) if build is _definition_question
                 else build(topic, ev, neighbours, rng, used))
            if q:
                out.append(q)
                break
    return out[:n]
