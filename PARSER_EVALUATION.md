# Parser Evaluation: Can the Fall Parser Handle Winter and Spring?

## TL;DR

**One parser, parameterized — not three separate parsers.** ~70% of `parse_record_book.py` works across all three seasons. The remaining ~30% is fall-specific and needs generalization rather than duplication.

## What's Shared Across All Three Seasons

All three record books (Fall 78pp, Winter 100pp, Spring 96pp) use the same organizational pattern:

1. **School records** — identical `Ch:/Fn:/Sf:/RU:` coding (Spring lacrosse adds `Qf:`)
2. **Championship finals tables** — same YEAR | CLASS | CHAMPION | COACH | FINALIST | COACH layout, both single-column and multi-column variants
3. **Sportsmanship awards** — same year + school format
4. **Individual results** — similar `Year Class Athlete-School-Mark` pattern, but sport-specific content

### Reusable components (no changes needed)

| Component | Lines | Notes |
|---|---|---|
| `load_pages()` | 133–135 | Works for any PDF |
| `parse_school_records()` | 188–253 | Same format everywhere |
| `is_school_records()` | 157–158 | `Ch: \d{4}` appears in all seasons |
| `is_sportsmanship()` | 175–178 | Works everywhere |
| `is_year_class_table()` | 161–163 | Matches winter/spring tables too |
| `is_multicolumn_results()` | 166–168 | Matches basketball, baseball, softball |
| `extract_championship_results()` | 259–284 | LLM prompt is generic enough |
| `extract_sportsmanship()` | 339–366 | Works everywhere |
| `ChampionshipResult` schema | 32–43 | Covers all team championships |
| CSV/JSON output helpers | 370–538 | Fully reusable |
| Chunking and dedup | 383–488 | Fully reusable |

## What's Fall-Specific

| Component | Why it doesn't work for winter/spring |
|---|---|
| `SECTIONS` dict (142–151) | Hardcoded to 8 fall sports and their page ranges |
| `is_individual_xc()` (171–172) | Looks for "MILES" — only matches cross country |
| `is_golf_results()` (181–182) | Looks for "Team Champion" — only matches golf |
| `extract_individual_xc()` (290–308) | Prompt tuned for XC (name, time, distance) |
| `extract_golf_results()` (314–333) | Prompt tuned for golf (strokes) |
| `IndividualChampion` schema (50–57) | Fields are XC-specific |
| `GolfResult` schema (64–76) | Golf-only |

## New Content Types With No Parser Support

| Season | Content | Example format |
|---|---|---|
| Winter | Indoor track event champions | `Year Class Athlete—School Mark` (e.g., "2025 4A Fred Colvin—Fairmont Heights 6.6") |
| Winter | Swimming event champions | `Year Class Athlete, School Time` (e.g., "2019 4A-3A Catherine Belyakov, Quince Orchard 2:01.41") |
| Winter | Wrestling weight-class champions | Champion + finalist per weight class per year |
| Winter | Swimming team championships | `Year Class Team—Score Coach Finalist Coach Site` |
| Spring | Outdoor track event champions | `Year Class Athlete-School-Mark` (same pattern as indoor track) |
| Spring | Tennis individual champions | `Year Class Athlete-School-Score` (e.g., "2025 4A Exodus Waite, Seneca Valley (6-1, 6-1)") |
| Spring | Lacrosse two-column format | `Class 4A-3A / Class 2A-1A` with score—coach on same line |

## Recommended Changes

### 1. Parameterize `SECTIONS` by season

Add section dicts for winter and spring. Detect season from filename or accept as CLI arg.

```python
FALL_SECTIONS = {
    "Girls Cross Country": (3, 13),
    "Boys Cross Country":  (13, 27),
    # ... existing
}
WINTER_SECTIONS = {
    "Girls Basketball":        (3, 11),
    "Boys Basketball":         (11, 21),
    "Girls Indoor Track":      (21, 40),
    "Boys Indoor Track":       (40, 60),
    "Girls Swimming & Diving": (60, 69),
    "Boys Swimming & Diving":  (69, 79),
    "Wrestling":               (79, 101),
}
SPRING_SECTIONS = {
    "Baseball":              (3, 10),
    "Girls Lacrosse":        (10, 15),
    "Boys Lacrosse":         (15, 21),
    "Softball":              (21, 29),
    "Tennis":                (29, 35),
    "Girls Track and Field": (35, 61),
    "Boys Track and Field":  (61, 95),
}
```

### 2. Generalize individual results extraction

- Rename `IndividualChampion` → `IndividualResult` with fields: `sport`, `event`, `year`, `classification`, `name`, `school`, `mark`
- Replace `is_individual_xc()` with `is_individual_results()` that detects `Year Class Athlete-School-Mark` pattern
- Replace `extract_individual_xc()` with generic `extract_individual_results()` that adapts to sport context

### 3. Add `Qf:` support to school records

One-line change to the regex in `parse_school_records()`.

### 4. Handle sport-specific formats

Keep golf as a special case. Add similar special cases for:
- Swimming team results (similar structure to golf — team champion + score)
- Wrestling brackets (unique format — weight classes with champion/finalist pairs)

### 5. Season-aware output

Output to season-specific subdirectories: `data/fall/`, `data/winter/`, `data/spring/`.
