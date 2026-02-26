# Maryland High Schools — MPSSAA Record Book Parser

Converts the [MPSSAA](https://mpssaa.org/) Fall Record Book PDF into structured CSV and JSON data.

## What it does

`parse_record_book.py` reads `pdfs/FallRecordBook2024.pdf` (78 pages, 8 fall sports) and produces five output files in `data/`:

| File | Contents |
|---|---|
| `championship_results.csv` | One row per championship final per year per classification |
| `school_records.csv` | Years each school appeared as champion, finalist, or semifinalist |
| `individual_xc_champions.csv` | Individual state cross country champions |
| `sportsmanship_awards.csv` | Sportsmanship award winners |
| `golf_results.csv` | Team and individual golf championship results |
| `record_book.json` | All of the above combined into a single JSON file |

Sports covered: Girls Cross Country, Boys Cross Country, Field Hockey, Football, Golf, Girls Soccer, Boys Soccer, Volleyball.

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

Defaults to `pdfs/FallRecordBook2024.pdf` → `data/`.

## How it works

The script classifies each PDF page by content and routes it to the appropriate handler:

| Page type | Handler |
|---|---|
| School records (`Ch: 1997, 1998 …`) | Regex — no API calls |
| Championship tables (`YEAR CLASS CHAMPION …` or multi-column `CLASS 4A CLASS 3A …`) | LLM |
| Individual XC champions | LLM |
| Golf year-by-year results | LLM |
| Sportsmanship award lists | LLM |

LLM calls use `claude-haiku-4-5-20251001` via the [`llm`](https://llm.datasette.io/) library with Pydantic schemas for structured output. Pages are sent in chunks of up to four at a time, with one-page overlap to avoid splitting entries across chunk boundaries. Results are deduplicated before writing.

### Output schema

**`championship_results.csv`**

| Column | Notes |
|---|---|
| `sport` | e.g. `Girls Cross Country`, `Football` |
| `year` | 4-digit year |
| `classification` | `4A`, `3A`, `2A`, `1A`, `AA`, `Combined`, etc. |
| `champion_school` | School name; both names joined with ` & ` for co-champions |
| `champion_coach` | |
| `finalist_school` | Not available for all sports |
| `finalist_coach` | |
| `score` | Final game score, e.g. `2-0`, `28-14 OT` |
| `champion_undefeated` | `True` if champion entered the final with an undefeated record |
| `co_champion` | `True` if two schools shared the title |
| `notes` | Anything else notable |

**`school_records.csv`**

| Column | Notes |
|---|---|
| `sport` | |
| `school` | |
| `champion_years` | Python list of years, e.g. `[1997, 1998, 2002]` |
| `finalist_years` | |
| `semifinalist_years` | |
| `runner_up_years` | |

**`individual_xc_champions.csv`**

| Column | Notes |
|---|---|
| `sport` | `Girls Cross Country` or `Boys Cross Country` |
| `year` | |
| `classification` | |
| `name` | Athlete name |
| `school` | |
| `time` | Finishing time, e.g. `19:17` |
| `distance` | Course distance, e.g. `3.0 Miles` |

**`sportsmanship_awards.csv`**

| Column | Notes |
|---|---|
| `sport` | |
| `year` | |
| `classification` | `4A`–`1A` for cross country; `null` for single-winner sports |
| `school` | |

**`golf_results.csv`**

| Column | Notes |
|---|---|
| `year` | |
| `classification` | `Combined` (1971–1992), `1A/2A`, or `3A/4A` (1993+) |
| `team_champion_school` | |
| `team_score` | Total team strokes |
| `individual_winner_name` | |
| `individual_winner_school` | |
| `individual_score` | Total individual strokes |
| `individual_gender` | `male`, `female`, or blank |

## Notes

- Football school records use a different internal format (with playoff round codes and region suffixes like `4AW`) and are not parsed into `school_records.csv`. Championship finals for football are included in `championship_results.csv`.
- The Girls Soccer section includes pre-MPSSAA Field Ball championships (1946–1988) with `sport = Girls Soccer`.
- The Boys Soccer sportsmanship page covers both Boys and Girls soccer; both appear in `sportsmanship_awards.csv` with the correct sport label.
- Cancelled seasons (COVID 2020) are omitted.
