# Maryland High Schools — MPSSAA Record Book Parser

Converts [MPSSAA](https://mpssaa.org/) Record Book PDFs (fall, winter, spring) into structured CSV and JSON data.

## What it does

`parse_record_book.py` reads an MPSSAA Record Book PDF and produces output files organized by season in `data/<season>/`:

| File | Contents |
|---|---|
| `championship_results.csv` | One row per championship final per year per classification |
| `school_records.csv` | Years each school appeared as champion, finalist, semifinalist, or runner-up |
| `individual_xc_champions.csv` | Individual state cross country champions (fall only) |
| `individual_results.csv` | Individual event champions — track, swimming, tennis (winter/spring) |
| `sportsmanship_awards.csv` | Sportsmanship award winners |
| `golf_results.csv` | Team and individual golf championship results (fall only) |
| `record_book.json` | All of the above combined into a single JSON file |

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
uv run parse_record_book.py [PDF_PATH] [OUTPUT_DIR]
```

The season is auto-detected from the PDF filename. Defaults to `pdfs/FallRecordBook2024.pdf` → `data/fall/`.

```bash
# Parse all three seasons
uv run parse_record_book.py pdfs/FallRecordBook2024.pdf
uv run parse_record_book.py "pdfs/Winter record book.pdf"
uv run parse_record_book.py "pdfs/Spring record book 2025.pdf"
```

## How it works

The script classifies each PDF page by content and routes it to the appropriate handler:

| Page type | Handler |
|---|---|
| School records (`Ch: 1997, 1998 …`) | Regex — no API calls |
| Championship tables (`YEAR CLASS CHAMPION …` or multi-column `CLASS 4A CLASS 3A …`) | LLM |
| Individual event champions (track, swimming, tennis, XC) | LLM |
| Golf year-by-year results | LLM |
| Sportsmanship award lists | LLM |

LLM calls use `claude-haiku-4-5-20251001` via the [`llm`](https://llm.datasette.io/) library with Pydantic schemas for structured output. Pages are sent in chunks of up to four at a time, with one-page overlap to avoid splitting entries across chunk boundaries. Results are deduplicated before writing.

## Notes

- Football school records use uppercase codes (`CH:`, `RU:`, `SF:`, `QF:`) with region/classification suffixes like `(4AW)`. These are now parsed correctly.
- Spring lacrosse school records include `Qf:` (quarterfinal) years, captured in `quarterfinal_years`.
- Wrestling weight-class individual champions are not yet parsed (the team championship tables and school records are).
- Cancelled seasons (COVID 2020) are omitted.
