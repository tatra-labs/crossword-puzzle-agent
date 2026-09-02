# Bundled puzzles

Ten original crosswords that ship with the repository. They exist so that
`xword demo` and the test suite work for anyone who clones this project without
first fetching anything: the NYT archive the fetcher can pull is copyrighted and
can never be committed here, so these are the offline stand-in, and they are the
fixed set the evaluation harness scores against.

Every file in this directory is an original work by `crossword-puzzle-agent` —
grid and clues both. No published puzzle's grid pattern, fill, or clue text was
copied. All ten are released under **CC0-1.0**: public domain dedication, no
attribution required, use them for anything.

## Inventory

| id | size | entries | difficulty | blocks | symmetric |
|---|---|---|---|---|---|
| `mini-01` | 5x5 | 10 | easy | 6 | yes |
| `mini-02` | 5x5 | 10 | easy | 8 | yes |
| `mini-03` | 5x5 | 10 | easy | 4 | yes |
| `mini-04` | 5x5 | 10 | easy | 4 | yes |
| `midi-01` | 7x7 | 14 | medium | 12 | yes |
| `midi-02` | 7x7 | 16 | medium | 13 | yes |
| `midi-03` | 7x7 | 16 | medium | 9 | yes |
| `maxi-01` | 9x9 | 30 | hard | 18 | yes |
| `maxi-02` | 9x9 | 32 | hard | 21 | yes |
| `maxi-03` | 11x11 | 44 | hard | 33 | yes |

Each grid uses a different black-square pattern. All ten have 180-degree
rotational symmetry, the standard construction convention; `meta.symmetric`
records the checked result rather than the intention.

## Format

One puzzle per file, `<id>.json`, in the repo's native `.xwj`/`.json` schema:

```json
{
  "id": "mini-01",
  "title": "First Steps",
  "author": "crossword-puzzle-agent",
  "date": "2026-09-01",
  "difficulty": "easy",
  "source": "original",
  "license": "CC0-1.0",
  "grid": ["##OAF", "#UNDO", "PRIDE", "EGOS#", "TEN##"],
  "clues": {
    "across": {"1": "Clumsy dolt", "4": "Ctrl+Z action"},
    "down":   {"1": "Vegetable with layers", "2": "Puts in, as an ingredient"}
  },
  "meta": {"symmetric": "true", "clue_types": "straight definition, ..."}
}
```

Rules the loader relies on:

- `grid` holds the **solution letters**. `#` is a black square; every other
  character is an open square, so the shape is derived from the same array.
- A puzzle with no known solution uses `"shape"` instead of `"grid"`, with any
  non-`#` character marking an open square. `Puzzle.solution` is then `None`.
  All ten puzzles here carry full solutions, since the harness scores against them.
- Answers are upper-case `A-Z` only: no spaces, punctuation, or rebus squares.
- Clue keys are entry **numbers as strings**, and they must match the numbers
  that `xword.core.grid.build_slots` derives from the shape. Numbering is not
  stored in the file; it is recomputed on load, so a clue keyed to a number that
  does not start an entry is simply dropped and the entry is left unclued.
- `meta` holds free-form **string** values only. These files use `size`,
  `blocks`, `entries`, `symmetric`, `clue_types`, and `notes`.

## Construction standards used here

- Every across and down run of two or more squares is a real word.
- No entry is shorter than three letters, and no square is unchecked: every open
  square belongs to both an across and a down entry, so every letter is
  confirmed by a crossing.
- No answer repeats within a puzzle, and no two clues in a puzzle are identical.
- Each clue has a single defensible answer at its length.
- Clue and answer agree in number and tense: a plural clue for a plural answer,
  a past-tense clue for a past-tense answer.

`meta.clue_types` names the kinds of clue a given puzzle contains. Across the
set they cover straight definition, fill-in-the-blank (`___`), abbreviation
(`for short`, `org.`), wordplay (clue ends in `?`), plural and tense agreement,
comparative agreement, and crosswordese. The evaluation harness's failure
taxonomy needs examples of each, so keep that spread when adding puzzles.

## Adding a puzzle

1. Draw a grid whose across and down runs are all real words of three letters or
   more, ideally with 180-degree rotational symmetry.
2. Write a clue for every entry, keyed by the number `build_slots` derives —
   print them first rather than guessing (see the command below).
3. Save it as `data/puzzles/bundled/<id>.json` in the schema above, with
   `"author"`, `"source"` and `"license"` set to something you are entitled to
   publish. Do not add anything under copyright: this directory is committed.
4. Verify it. Every puzzle must report `PROBLEMS []`:

```
PYTHONPATH=src python -c "
import json,glob
from xword.core.grid import make_puzzle, validate_puzzle
for p in sorted(glob.glob('data/puzzles/bundled/*.json')):
    d=json.load(open(p))
    rows=d['grid']
    pz=make_puzzle(d['id'], rows,
         {int(k):v for k,v in d['clues']['across'].items()},
         {int(k):v for k,v in d['clues']['down'].items()},
         solution_rows=rows)
    probs=validate_puzzle(pz)
    print(p, pz.height, 'x', pz.width, 'slots', len(pz.slots), 'PROBLEMS', probs)
    assert not probs, (p, probs)
    for s in pz.slots:
        print('   ', s.id, pz.solution[s.id], '=', s.clue)
"
```

`validate_puzzle` checks structure — entry lengths, missing clues, orphaned
squares, solution/shape agreement. It cannot tell you whether an answer is a
real word or whether a clue actually clues it, so read the printed entry list
yourself before committing.
