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

Set up the LLM backend. Extraction defaults to **GLM-5.2 served by a local
[Ollama](https://ollama.com) daemon** (`glm-5.2:cloud`, via the `llm-ollama`
plugin) — no API key needed:

```bash
ollama pull glm-5.2:cloud   # or however your Ollama instance provides it
ollama list                 # should show glm-5.2:cloud
```

An Anthropic model is kept as an optional fallback (`llm-anthropic`); to use it
instead, change `MODEL_ID` in `parse_record_book.py` and set a key:

```bash
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

### Annual update workflow

When a new record book PDF arrives:

```
1. drop new PDF into pdfs/
2. make routes    # zero-cost routing sanity check (no LLM calls)
3. make extract   # only changed pages hit the API; the rest is cache
4. make diff      # human reviews added/removed/changed vs last commit
5. make verify    # cross-path, continuity, era-floor, regression gates
6. git commit data/ + cache/
```

- **Cache** (`cache/extractions/`) keys each extracted page by the model + schema
  + prompt, so re-running extraction after a code change only re-calls the LLM
  for pages whose inputs actually changed. Commit the cache alongside `data/`.
- **Provenance** — every row carries `source_pdf`, `source_pages`,
  `extracted_at`, `extraction_model` so any row traces back to its PDF page.
- **Diff** (`diff_outputs.py`) reports added/removed/changed rows by natural
  key vs `HEAD`, written to `diff_report.md`. Git is the verified store — there
  is no auto-merge; removals are surfaced for human review, never merged over.
- **Verify** (`verify_record_book.py`) checks champion-year coverage across the
  two extraction paths, duplicate natural keys, year continuity (2020 exempt),
  era-floor anchor years (e.g. Boys Cross Country ≥ 1946, Golf ≥ 2024), and a
  regression guard against `HEAD`. Exits non-zero on errors.

## How it works

The script classifies each PDF page by content and routes it to the appropriate handler:

| Page type | Handler |
|---|---|
| School records (`Ch: 1997, 1998 …`) | Regex — no API calls |
| Championship tables (`YEAR CLASS CHAMPION …` or multi-column `CLASS 4A CLASS 3A …`) | LLM |
| Individual event champions (track, swimming, tennis, XC) | LLM |
| Golf year-by-year results | LLM |
| Sportsmanship award lists | LLM |

Section boundaries are detected at runtime from short `MPSSAA <Sport> Records`
divider pages (NaturalPDF extracts the large-font titles pypdf drops), rather
than hardcoded page indices, so a new edition that shifts page numbers still
slices correctly. The detected sport set is validated against a baseline map
and any added/renamed/dropped sport is a hard error.

LLM calls use GLM-5.2 via the [`llm`](https://llm.datasette.io/) library. GLM-5.2
does not honor Ollama's schema-structured output, so extraction uses
`json_object=True` with the Pydantic JSON schema embedded in the prompt.
Championship pages are sent two at a time; other tables one page at a time.
Results are deduplicated by natural key before writing.

## Web site (newsroom use)

`build_site.py` turns the committed `data/*/record_book.json` into a static
site under `site/` — one clip-file page per school, story-peg pages, and
embeddable widgets. No npm; Python + Jinja2, vanilla JS/CSS. Deployed to
GitHub Pages on push to `main` (`.github/workflows/deploy.yml`).

```bash
make site         # build site/
make serve        # build + preview at http://localhost:8000/
make check-links  # build + crawl for broken internal links
```

### For reporters

- **School clip files** (`/schools/<slug>/`) — a ready-to-paste fast-facts
  paragraph ("X has won N state championships across M sports…"), a print-safe
  championship timeline (inline SVG, no JS needed), and per-sport title tables
  with PDF page citations. Use the **Copy paragraph** button.
- **Story pegs** (`/pegs/`) — ranked angles: longest active title droughts,
  current win streaks, schools that have never won a title, first-title watch,
  and championship anniversaries (25/50/75/100 years).
- **Embeddable widgets** (`/embed/`) — drop-in `<iframe>` snippets for a
  school's timeline or title list. The `/embed/` builder page generates the
  snippet for any school; widgets are self-contained (inline CSS/SVG, no
  external requests) and accept `?theme=light|dark`.

Widgets are standalone HTML files, so an embed is one line:

```html
<iframe src="https://<org>.github.io/<repo>/embed/timeline/<slug>/"
        width="100%" height="300" frameborder="0"></iframe>
```

### School-name matching

The one hard data problem is school identity: `school_records` is ALLCAPS
regex output, championship/individual tables are mixed-case LLM output, and
names appear with abbreviations, em-dashes, county suffixes (`-AA`/`-B`), and
co-champion `" & "` joins. `build_site.py` normalizes names
(`normalize_school` + `slugify`), `web/aliases.csv` holds the curated
alias→canonical map, and a `canonical_display` map keeps proper mixed-case
display names. The build report (`site/site-build-report.txt`) lists any names
that appear in championship results but couldn't be matched to a school
record — mostly closed/segregated-era schools and individual-event artifacts —
so the gaps are documented rather than silent.

## Notes

- Football school records use uppercase codes (`CH:`, `RU:`, `SF:`, `QF:`) with region/classification suffixes like `(4AW)`. These are now parsed correctly.
- Spring lacrosse school records include `Qf:` (quarterfinal) years, captured in `quarterfinal_years`.
- Wrestling weight-class individual champions are not yet parsed (the team championship tables and school records are).
- Cancelled seasons (COVID 2020) are omitted.
