# SyllabiSense - frozen benchmark pairs

Frozen: 2026-09-19

| Tier | Subject | Syllabus (NPTEL) | Content | Topics | Covered | Coverage | Unrelated control | Margin | Audit (genuine) |
|---|---|---|---|---|---|---|---|---|---|
| A | Biology | 102103012 Cell Biology - Web course | OpenStax biology-2e | 77 | 54 | 70% | 21% | +49% | 8/10 |
| A | ChemEng | 103101005 Chemical Reaction Engineering I - Web | MIT 10-37-chemical-and-biological-reaction-engineering-spring-2007 | 26 | 19 | 73% | 27% | +46% | 7/10 |
| A | Chemistry | 104106047 Molecular spectroscopy I (Microwave, IR, | MIT 5-61-physical-chemistry-fall-2017 | 29 | 19 | 66% | 21% | +45% | 6/10 |
| A | Civil | 105103027 Probability Methods in Civil Engineering - | MIT 1-010-uncertainty-in-engineering-fall-2008 | 43 | 30 | 70% | 33% | +37% | 8/10 |
| A | Economics | 109104041 Micro Economics - Video course | MIT 14-01sc-principles-of-microeconomics-fall-2011 | 65 | 48 | 74% | 29% | +45% | 6/10 |
| A | Electrical | 108104099 APPLIED ELECTROMAGNETICS FOR | MIT 6-013-electromagnetics-and-applications-spring-2009 | 69 | 45 | 65% | 12% | +54% | 7/10 |
| A | Humanities | 109104032 Ethics - Video course | MIT 24-01-classics-of-western-philosophy-spring-2016 + MIT 24-00-problems-of-philosophy-fall-2019 | 18 | 11 | 61% | 11% | +50% | 8/10 |
| A | Mechanical | 112103016 Advanced Engineering Thermodynamics - | MIT 2-43-advanced-thermodynamics-spring-2024 | 32 | 20 | 62% | 38% | +25% | 6/10 |
| A | Physics | 115106059 Classical Mechanics - Video course | MIT 8-01sc-classical-mechanics-fall-2016 | 17 | 7 | 41% | 12% | +29% | 6/10 |
| A | Psychology | 109104025 Abnormal Psychology - Video | MIT 9-00sc-introduction-to-psychology-fall-2011 + OpenStax psychology-2e | 15 | 9 | 60% | 13% | +47% | 7/10 |
| B | Aero | 101103004 Principles of Fluid Dynamics - Web course | MIT 16-01-unified-engineering-i-ii-iii-iv-fall-2005-spring-2006 | 73 | 52 | 71% | 20% | +51% | 4/10 |
| B | CS | 106107050 Operating System - Video course | MIT 6-828-operating-system-engineering-fall-2012 | 43 | 32 | 74% | 37% | +37% | 5/10 |
| B | Mathematics | 111101004 Introduction to Probability Theory - Web | MIT 18-05-introduction-to-probability-and-statistics-spring-2022 | 34 | 25 | 74% | 24% | +50% | 4/10 |
| C | Management | 110106164 BUSINESS DEVELOPMENT: FROM | OpenStax introduction-to-business | 69 | 49 | 71% | 20% | +51% | 3/10 |
| C | Sociology | 109104074 Introduction to Sociology - Video | OpenStax introduction-to-sociology-3e | 49 | 41 | 84% | 43% | +41% | 3/10 |

**Tiers**

- **A**: same subject, confirmed by reading; >= 6/10 evidence passages genuine
- **B**: same subject; evidence noisy (review outlines, word ambiguity e.g. 'Euler' equation vs Eulerian path)
- **C**: same broad subject but weak evidence (3/10): generic syllabus topics and preface/front-matter pages match easily. Usable, but the least reliable pairs

**Overrides of the automatic choice**

- **ChemEng**: same subject (reaction engineering); audit 4/5. Auto pick 'Molecular Simulations' won a near-tie (1.39 vs 1.37) with an odd content mix
- **Civil**: drop 'Soil behavior': its only contribution in the audit was a false match; still +37%
- **Management**: business development matches the textbook's scope; auto pick 'Financial Accounting' is covered by only one chapter of that book
- **Mathematics**: same subject (probability); margin +47% vs +24% for Advanced Engineering Mathematics, whose linear-algebra/complex-analysis topics were being matched to probability pages
- **Physics**: Classical Mechanics is the actual subject of MIT 8.01; Special Relativity was chosen on a near-tie
- **Psychology**: auto pick 'Selected Topics in Psychology' has junk topics (lecture numbers + author names, 'funded by MHRD'), audit 1/5; Abnormal Psychology audited 5/5

**Method**

topic-level keyword matching with specificity (idf) weighting, 30-word window, navigation/reference/garbage pages excluded; control = best unit from an unrelated subject group; rules: margin >= 20%, coverage 40-92%, >= 5 topics each class

**Audit**

final set: 75 passages, fresh random sample (seed 13), 40/75 genuine (53%). Per-subject scores combine two independent samples (10 passages each). Known false-positive sources: preface/front-matter pages, review outlines, word ambiguity - page-type filtering moves to preprocessing.

**Important**

these pairs define WHICH content goes with WHICH syllabus. Topic-level labels are NOT taken from this matcher; they come from the retrieval pipeline + human checking.