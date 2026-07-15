# Maryland High Schools — MPSSAA Record Book Parser

Converts [MPSSAA](https://mpssaa.org/) Record Book PDFs (fall, winter, spring) into structured JSON and CSV data, with a built-in verification step so accuracy is checkable rather than assumed.

## What it does

`parse_record_book.py` reads an MPSSAA Record Book PDF and produces output files organized by season in `data/<season>/`:

| File | Contents |
|---|---|
| `record_book.json` | **Primary output.** All tables plus a `meta` block (see below). Built for the website. |
| `championship_results.csv` | One row per championship final per year per classification |
| `school_records.csv` | Wide: years each school appeared as champion/finalist/semifinalist/runner-up/quarterfinalist |
| `school_record_years.csv` | Long/tidy: one row per `(school, year, result)` — easier to query and chart |
| `individual_xc_champions.csv` | Individual state cross country champions (fall only) |
| `individual_results.csv` | Individual event champions — track, swimming, tennis (winter/spring) |
| `sportsmanship_awards.csv` | Sportsmanship award winners |
| `golf_results.csv` | Team and individual golf championship results (fall only) |
| `verification_report.json` | Result of the cross-checks (see [Verification](#verification)) |
| `school_name_map.json` | Every raw → canonical school-name change applied, for audit |
| `pages.jsonl` | Raw extracted text per PDF page, for provenance spot-checks |

`data/all.json` combines all three seasons into one file (each row tagged with `season`) for the website.

### Sports covered

| Fall (8) | Winter (7) | Spring (7) |
|---|---|---|
| Girls Cross Country | Girls Basketball | Baseball |
| Boys Cross Country | Boys Basketball | Girls Lacrosse |
| Field Hockey | Girls Indoor Track | Boys Lacrosse |
| Football | Boys Indoor Track | Softball |
| Golf | Girls Swimming & Diving | Tennis |
| Girls Soccer | Boys Swimming & Diving | Girls Track and Field |
| Boys Soccer | Wrestling | Boys Track and Field |
| Volleyball | | |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set your Anthropic API key (used for LLM-based extraction):

```bash
export ANTHROPIC_API_KEY=your-key-here
# or store it permanently:
uv run llm keys set anthropic
```

## Usage

```bash
uv run parse_record_book.py [PDF_PATH] [OUTPUT_DIR] [--season fall|winter|spring]
```

The season is auto-detected from the PDF filename; use `--season` to override when the
filename doesn't say. Output defaults to `data/<season>/`.

```bash
# Parse all three seasons
uv run parse_record_book.py pdfs/FallRecordBook2024.pdf
uv run parse_record_book.py "pdfs/Winter record book.pdf"
uv run parse_record_book.py "pdfs/Spring record book 2025.pdf"
```

Then verify each season (see below).

> **Note:** the `data/fall/` output currently committed predates the accuracy
> and schema changes described here. Regenerate it (and add winter/spring) with
> the commands above once an API key is available, then commit the fresh output
> and its `verification_report.json`.

## Verification

Accuracy is the priority, so the pipeline verifies itself two ways.

**1. Independent cross-check.** The parser extracts the same facts along two independent
paths — championship finals tables (via the LLM) and school records (via regex). If they
disagree, something is wrong. `verify_record_book.py` plays them against each other:

```bash
uv run verify_record_book.py data/fall
```

It reports, per sport, the champion years the championship table is missing versus the
school records (and vice versa), plus duplicate keys, year-continuity gaps (2020 COVID
exempt), and champion schools not found in the school records. It writes
`verification_report.json` and exits non-zero if any check fails — usable as a CI or
pre-commit gate. `parse_record_book.py` runs the same check automatically and embeds the
summary in `record_book.json`'s `meta.verification`.

**2. Row-level provenance.** Every output row carries a `source_pages` field (e.g. `"62-63"`),
and `pages.jsonl` holds the raw text of each PDF page. To spot-check any row, look up its
`source_pages` in `pages.jsonl` (or open the PDF to that page) and compare.

The parser also guards against silent LLM data loss: after extracting each year-keyed table
it compares the years returned against the years visible in the source text, and if coverage
falls short it re-extracts page-by-page and loudly reports any years still missing.

## JSON shape (for the website)

`record_book.json`:

```jsonc
{
  "meta": {
    "season": "fall",
    "source_pdf": "FallRecordBook2024.pdf",
    "generated_at": "2025-…",
    "parser_commit": "abc1234",
    "row_counts": { "championship_results": 970, "school_records": 667, … },
    "verification": { "errors": 0, "warnings": 12, "passed": true }
  },
  "championship_results": [ … ],
  "school_records": [ … ],   // year fields are real arrays here
  "individual_xc_champions": [ … ],
  "individual_results": [ … ],
  "sportsmanship_awards": [ … ],
  "golf_results": [ … ]
}
```

Every school field has a canonical value plus a url-safe `_slug` (e.g.
`champion_school` + `champion_school_slug`) for routing. Year lists are real arrays in
JSON; in the wide CSV they are `;`-joined (never a Python list repr).

## School name normalization

School names are canonicalized so the same school joins across tables: ALL-CAPS names from
school-records pages are title-cased to match the championship tables, and a hand-curated
alias map (`school_aliases.json`) merges known variants (`B-CC` → `Bethesda-Chevy Chase`,
etc.). Mixed-case names pass through untouched — there is **no fuzzy matching**, so no two
distinct schools are ever silently merged.

To extend it: run the verify script, look at the champion schools it lists as not found in
the school records, confirm against the PDF, and add `"variant (casefolded)": "Canonical Name"`
entries to `school_aliases.json`. Every change is recorded in `school_name_map.json`.

## How it works

The script classifies each PDF page by content and routes it to the appropriate handler:

| Page type | Handler |
|---|---|
| School records (`Ch: 1997, 1998 …`) | Regex — no API calls |
| Championship tables (`YEAR CLASS CHAMPION …` or multi-column `CLASS 4A CLASS 3A …`) | LLM |
| Individual event champions (track, swimming, tennis, XC) | LLM |
| Golf year-by-year results | LLM |
| Sportsmanship award lists | LLM |

LLM calls use `claude-haiku-4-5-20251001` via the [`llm`](https://llm.datasette.io/) library
with Pydantic schemas for structured output. Model output is validated through those schemas
before use (defaults applied, types coerced), pages are sent in small overlapping chunks, and
all tables are de-duplicated with conflicts reported.

Before parsing, the section page ranges are sanity-checked against the PDF; a new edition with
shifted page numbers fails loudly rather than silently mis-slicing.

## Testing

```bash
uv run pytest
```

Tests are deterministic (no LLM calls): page classifiers and the regex parser run against the
real PDFs in `pdfs/`, the verification checks and helpers run on synthetic fixtures, and one
end-to-end test drives the full pipeline with the LLM stubbed.

## Notes

- Football school records use uppercase codes (`CH:`, `RU:`, `SF:`, `QF:`) with region/classification suffixes like `(4AW)`. These are parsed correctly.
- Spring lacrosse school records include `Qf:` (quarterfinal) years, captured in `quarterfinal_years`.
- Wrestling weight-class individual champions are not yet parsed (team championship tables and school records are) — planned future work.
- Cancelled seasons (COVID 2020) are omitted.
