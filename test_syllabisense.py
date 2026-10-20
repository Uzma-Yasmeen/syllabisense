"""
Tests for the parts that decide what the user is told.

Runs with pytest if it is installed, and on its own if it is not:

    python test_syllabisense.py
    pytest test_syllabisense.py -q

Nothing here loads a trained model, so the whole file runs in about a second. The behaviour
being pinned down is the behaviour that has broken before: units read from a file's own text,
evidence that is present but too thin, and generated questions that are quietly unanswerable.
"""
import sys

import numpy as np

from depth import evidence_kinds, judge
from quiz import (Distractors, _options, definition_in, sentences, surface_terms, tidy_clause,
                  usable)
from units import applicable, file_units, to_number, unit_of_file, unit_of_label, units_from_scores


# --------------------------------------------------------------- units: labels
def test_unit_of_label_reads_roman_and_arabic():
    assert unit_of_label("Unit III — Software Design") == 3
    assert unit_of_label("Unit 4: Verification") == 4
    assert unit_of_label("Module 2") == 2


def test_unit_of_label_none_when_absent():
    assert unit_of_label("Introduction to Hashing") is None
    assert unit_of_label("") is None
    assert unit_of_label(None) is None


def test_to_number_handles_both_numerals():
    assert to_number("iv") == 4
    assert to_number("7") == 7
    assert to_number("zzz") is None


# ------------------------------------------------------------ units: filenames
def test_unit_from_filename_in_several_shapes():
    assert unit_of_file("Unit-2 notes.pdf") == 2
    assert unit_of_file("U3_slides.pptx") == 3
    assert unit_of_file("Module 5 handout.docx") == 5
    assert unit_of_file("Lecture-13,UNIT-III.pdf") == 3


def test_unit_from_filename_ignores_unrelated_digits():
    assert unit_of_file("notes2026.pdf") is None
    assert unit_of_file("chapter.pdf") is None


# ----------------------------------------------------------------- units: text
def test_unit_read_from_the_title_page():
    """A deck states its unit once, at the front - which is exactly the case that used to fail."""
    text = ("Course Code Course Title L R P Cr 24CS312PC216 Data Structures 3 1 4 6 "
            "Program Core Unit 3 - Hashing and Dictionaries")
    assert unit_of_file("notes.pdf", text) == 3


def test_single_mention_deep_in_the_text_is_not_enough():
    text = "x " * 600 + "see unit 4 for the proof"
    assert unit_of_file("notes.pdf", text) is None


def test_majority_wins_over_a_passing_mention():
    text = "Unit 3 overview. Recall from Unit 2 the hash table. Unit 3 continues."
    assert unit_of_file("notes.pdf", text) == 3


def test_file_units_maps_every_source():
    pages = [("Unit-1 Notes.pdf", "Page 1", "anything"),
             ("misc.pdf", "Page 1", "no marker here")]
    got = file_units(pages)
    assert got["Unit-1 Notes.pdf"] == 1
    assert got["misc.pdf"] is None


# ------------------------------------------------------------ units: from scores
def test_a_file_is_placed_by_the_topics_it_answers():
    """Two files, two units. File A answers unit 1's topics, file B answers unit 2's."""
    topic_units = [1, 1, 2, 2]
    sources = ["a.pdf", "a.pdf", "b.pdf", "b.pdf"]
    prob = np.array([[0.9, 0.9, 0.1, 0.1],
                     [0.9, 0.8, 0.1, 0.1],
                     [0.1, 0.1, 0.9, 0.9],
                     [0.1, 0.1, 0.8, 0.9]], np.float32)
    assert units_from_scores(prob, topic_units, sources) == {"a.pdf": 1, "b.pdf": 2}


def test_no_guess_when_no_unit_leads():
    topic_units = [1, 2]
    sources = ["a.pdf", "a.pdf"]
    prob = np.full((2, 2), 0.5, np.float32)
    assert units_from_scores(prob, topic_units, sources) == {}


def test_applicable_lets_unmarked_material_count_anywhere():
    assert applicable(2, 2) is True
    assert applicable(2, 3) is False
    assert applicable(None, 3) is True
    assert applicable(2, None) is True


# ---------------------------------------------------------------------- depth
def test_no_evidence_is_not_taught():
    level, reasons, kinds = judge([])
    assert level == "NOT TAUGHT"
    assert reasons and kinds == set()


def test_one_thin_page_is_barely_taught():
    ev = [("f.pdf", "Page 1", 0.99, "Hashing is mentioned here among other things.")]
    assert judge(ev)[0] == "BARELY TAUGHT"


def test_several_pages_with_definition_and_example_are_properly_taught():
    text = ("Separate chaining is defined as a collision resolution technique. "
            "For example, each slot points to a linked list.")
    ev = [("f.pdf", f"Page {i}", 0.99, text) for i in range(1, 4)]
    assert judge(ev)[0] == "PROPERLY TAUGHT"


def test_evidence_kinds_detects_what_it_claims_to():
    assert "definition" in evidence_kinds("A heap is defined as a complete binary tree.")
    assert "example" in evidence_kinds("For example, consider the array [3, 1, 2].")
    assert "figure" in evidence_kinds("As shown in Figure 4, the list grows.")
    assert evidence_kinds("Nothing of interest on this page.") == set()


# ----------------------------------------------------------------------- quiz
def test_sentences_splits_and_drops_fragments():
    text = "Short. A hash table stores key-value pairs for fast lookup by key. Also short."
    out = sentences(text)
    assert len(out) == 1 and out[0].startswith("A hash table")


def test_usable_rejects_headings_and_pronoun_openers():
    assert usable("A linked list stores each element in a separate node object.")
    assert not usable("This is why it matters to the reader of the page.")      # pronoun opener
    assert not usable("Unit III Hashing And Dictionaries Lecture Notes Slides")  # title case


def test_tidy_clause_rejects_clauses_that_trail_off():
    assert tidy_clause("a collision resolution technique where each slot holds a list") is not None
    assert tidy_clause("commonly specified in this step, including") is None      # ends badly
    assert tidy_clause("too short") is None
    assert tidy_clause("a list: 1 2 3 4 5 6 7 8 9 10 11 12 13 14") is None        # mostly digits


def test_definition_in_reads_the_bullet_shape():
    """Lecture notes are written as 'Term: description' far more often than 'X is a Y'."""
    got = definition_in("Functional Requirements: These describe what the software system "
                        "should do in response to input.")
    assert got is not None
    subject, clause = got
    assert subject.startswith("Functional Requirements")
    assert clause.startswith("These describe")


def test_definition_in_reads_the_prose_shape():
    got = definition_in("Hashing is a technique that maps keys to positions in a fixed table.")
    assert got is not None and "maps keys" in got[1]


def test_surface_terms_returns_words_not_stems():
    terms = surface_terms("Prototyping, Spiral Process Models")
    assert "prototyping" in terms          # not "prototyp"
    assert all(len(t) >= 5 for t in terms)


def test_options_refuses_to_build_a_question_with_duplicates():
    import random
    rng = random.Random(0)
    assert _options("right", ["wrong a", "wrong b", "wrong c"], rng) is not None
    assert _options("right", ["same", "same", "same"], rng) is None     # not 3 distinct
    assert _options("right", ["right", "RIGHT "], rng) is None          # a copy of the answer


def test_options_spreads_distractors_instead_of_repeating_one():
    import random
    rng = random.Random(0)
    usage = {"a": 5, "b": 0, "c": 0, "d": 0}
    opts = _options("answer", ["a", "b", "c", "d"], rng, usage=usage)
    assert "a" not in [o.strip() for o in opts]      # the overused one is passed over


def test_distractors_prefers_the_nearest_topic():
    class Fake:
        """Stands in for the fitted TF-IDF and SVD so the test needs no model files."""
        class vec:
            @staticmethod
            def transform(xs):
                return xs
        class svd:
            @staticmethod
            def transform(xs):
                table = {"hash tables": [1.0, 0.0], "hashing collisions": [0.95, 0.05],
                         "graph traversal": [0.0, 1.0]}
                return np.array([table[x] for x in xs])
    near = Distractors(["hash tables", "hashing collisions", "graph traversal"], Fake)
    assert near.nearest("hash tables")[0] == "hashing collisions"


# ------------------------------------------------------------------- the runner
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  .  {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"  F  {name}  {e}")
        except Exception as e:
            failed.append(name)
            print(f"  E  {name}  {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
