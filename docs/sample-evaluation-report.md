# Crossword evaluation - nyt:14

14 puzzles x 2 system(s) = 28 runs, 0 error(s). Finished 2026-09-02T07:14:16+00:00.

## 1. Headline

| System | Solve rate (95% CI) | Cell acc | Word acc | $/puzzle | s/puzzle | n | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 50.0% [21.4, 78.6] | 96.8% | 94.1% | $0.9683 | 216.4 | 14 | 0 |
| `greedy-llm` | 0.0% [0.0, 0.0] | 71.2% | 49.3% | $0.0401 | 8.9 | 14 | 0 |

CIs are 2000 bootstrap resamples over puzzles (seed 0); a puzzle counts as solved only when every gradable entry is right.

## 2. Difficulty breakdown

Solve rate by day of week / difficulty label, with n per cell.

| Difficulty | `full` | `greedy-llm` |
| --- | --- | --- |
| Mon | 100.0% (2) | 0.0% (2) |
| Tue | 50.0% (2) | 0.0% (2) |
| Wed | 100.0% (2) | 0.0% (2) |
| Thu | 50.0% (2) | 0.0% (2) |
| Fri | 0.0% (2) | 0.0% (2) |
| Sat | 50.0% (2) | 0.0% (2) |
| Sun | 0.0% (2) | 0.0% (2) |

## 3. Ablation deltas vs `full`

| System | d solve rate | d cell acc | d word acc | d $/puzzle | `full` only | System only | McNemar p | Holm p | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `greedy-llm` | -50.0pp | -25.6pp | -44.8pp | -$0.9282 | 7 | 0 | 0.016 | 0.016 | 14 |

`full` only / System only are the discordant puzzles McNemar's exact test is computed on; everything else is uninformative about the difference.

## 4. Calibration

| System | ECE | Mean confidence | Entry accuracy | Entries |
| --- | --- | --- | --- | --- |
| `full` | 3.81% | 97.5% | 94.1% | 1178 |
| `greedy-llm` | 3.90% | 48.0% | 49.3% | 1178 |

Reliability, system `full` (10 equal-width bins):

| Confidence bin | n | Mean confidence | Accuracy | Gap |
| --- | --- | --- | --- | --- |
| 0.0-0.1 | 1 | 5.4% | 0.0% | -5.4pp |
| 0.1-0.2 | 4 | 14.0% | 0.0% | -14.0pp |
| 0.2-0.3 | 6 | 25.1% | 0.0% | -25.1pp |
| 0.3-0.4 | 5 | 33.4% | 40.0% | +6.6pp |
| 0.4-0.5 | 8 | 43.7% | 62.5% | +18.8pp |
| 0.5-0.6 | 11 | 55.7% | 27.3% | -28.4pp |
| 0.6-0.7 | 7 | 64.1% | 71.4% | +7.3pp |
| 0.7-0.8 | 7 | 73.7% | 57.1% | -16.6pp |
| 0.8-0.9 | 11 | 86.7% | 45.5% | -41.3pp |
| 0.9-1.0 | 1118 | 99.9% | 97.0% | -2.8pp |

Selective accuracy, system `full` (answer only above threshold):

| Confidence >= | Coverage | Accuracy |
| --- | --- | --- |
| 0.50 | 98.0% | 95.5% |
| 0.70 | 96.4% | 96.3% |
| 0.90 | 94.9% | 97.0% |
| 0.95 | 93.8% | 97.6% |
| 0.99 | 92.4% | 97.7% |

## 5. Failure taxonomy

| Category | `full` | `greedy-llm` |
| --- | --- | --- |
| proper-noun | 7 | 87 |
| wordplay | 4 | 24 |
| abbreviation | 1 | 13 |
| fill-in-blank | 4 | 45 |
| multi-word | 0 | 17 |
| crosswordese | 0 | 8 |
| theme | 0 | 3 |
| plural-tense | 0 | 1 |
| other | 53 | 399 |
| **total** | 69 | 597 |

## 6. Hardest 10 entries missed

| Puzzle | Entry | Clue | Predicted | Gold | Missed by | Max conf |
| --- | --- | --- | --- | --- | --- | --- |
| nyt-1989-01-01 | 107D | Fun and games for 82 Down | WAB | WAR | 2/2 | 100.0% |
| nyt-2000-09-05 | 45D | Eminem, e.g. | RAPSTER | RAPSTAR | 2/2 | 100.0% |
| nyt-1989-01-01 | 114A | Colorful spectral type | BEASTAR | REDSTAR | 2/2 | 100.0% |
| nyt-1994-12-31 | 16A | Personal choice | ATMOSTEA | CUPOFTEA | 2/2 | 100.0% |
| nyt-2000-09-05 | 67A | Notion | IDEE | IDEA | 2/2 | 100.0% |
| nyt-1994-12-31 | 2D | Bury | INTER | INURN | 2/2 | 100.0% |
| nyt-1979-11-23 | 26D | Cave men, of sorts | UNCOUNTESS | SPELUNKERS | 2/2 | 100.0% |
| nyt-1994-12-31 | 9D | Bakery gizmo | BEATER | GLAZER | 2/2 | 100.0% |
| nyt-1989-01-01 | 90A | Specified, as a date | NIVEN | GIVEN | 2/2 | 100.0% |
| nyt-1989-01-01 | 90D | Balm of ___ | NIVEAA | GILEAD | 2/2 | 100.0% |

Ranked by how many systems missed the entry, then by how confident the wrong answer was - a confidently wrong entry corrupts its crossings too.

## 7. Reproducibility

- git sha: `09547a8`
- model: `claude-sonnet-5`
- suite: `nyt:14` (14 puzzles)
- systems: `full`, `greedy-llm`
- seed: `0` (metrics seed `0`)
- started: 2026-09-02T07:14:15+00:00 / finished: 2026-09-02T07:14:16+00:00

```
xword eval run --suite "nyt:14" --systems full,greedy-llm --seed 0 --workers 1 --out reports/nyt-14
```
