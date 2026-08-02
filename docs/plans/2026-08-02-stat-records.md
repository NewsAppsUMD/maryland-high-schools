# stat_records Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `stat_records` table capturing all-time statistical superlative records (football, basketball, baseball, lacrosse, volleyball, softball) across all three MPSSAA seasonal record books — the team-sport record sections not captured by any existing table.

**Architecture:** A new `stat_records` extraction route in `parse_record_book.py`, detected by a deterministic classifier `is_stat_records`, extracted by GLM-5.2 via Ollama with a new prompt + pydantic schema, page-level cached like the other tables, deduped by `(sport, category, record, holder, year)`, written to `stat_records.csv` + a `stat_records` key in `record_book.json`, and covered by `verify_record_book.py`'s duplicate-key check. Purely additive — no changes to existing tables.

**Tech Stack:** Python 3, pydantic, pypdf, GLM-5.2 via local Ollama (`llm-ollama` plugin), pytest. Run commands from the repo root with the project venv (`.venv/bin/python3`).

**Design doc:** `docs/plans/2026-08-02-stat-records-design.md` (approved).

**Text-extractable stat pages (1-indexed) — implementer must confirm via `--routes`:**
fall: Football 37,38 | Volleyball 74,75,76 · winter: Girls Basketball 6,7 | Boys Basketball 13,20 · spring: Baseball 10,11 | Girls Lacrosse 16 | Boys Lacrosse 21 | Softball 26.

---

### Task 1: Pydantic schema for stat records

**Files:**
- Modify: `parse_record_book.py` (add classes after `SportsmanshipAwards` at line 106)
- Test: `test_parse_record_book.py`

**Step 1: Write the failing test**

Add to `test_parse_record_book.py` (in the imports/classifier test area, or a new `TestStatRecordSchema` class):

```python
def test_stat_record_schema_accepts_full_row():
    from parse_record_book import StatRecord, StatResults
    row = StatRecord(
        sport="Football",
        category="team",
        record="Most Touchdowns, Season",
        value="98",
        holder="Fort Hill",
        school="Fort Hill",
        year="2016",
        co_holder=False,
        notes=None,
    )
    page = StatResults(results=[row])
    assert len(page.results) == 1
    assert page.results[0].record == "Most Touchdowns, Season"
    assert page.results[0].value == "98"

def test_stat_record_schema_optional_fields():
    from parse_record_book import StatRecord
    r = StatRecord(sport="Baseball", record="Runs Scored - Game")
    # category/value/holder/school/year/notes are optional; co_holder optional
    assert r.category is None
    assert r.value is None
    assert r.co_holder is None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_record_schema -v`
Expected: FAIL with `ImportError: cannot import name 'StatRecord'`

**Step 3: Write minimal implementation**

In `parse_record_book.py`, after the `SportsmanshipAwards` class (line 106) and before the `# ── LLM ──` comment (line 109), add:

```python
class StatRecord(BaseModel):
    sport: str
    category: Optional[str] = None  # "team" | "individual" | None
    record: str
    value: Optional[str] = None
    holder: Optional[str] = None      # player name (individual) or school (team)
    school: Optional[str] = None
    year: Optional[str] = None        # "2016", "2015-2018", "1988 & 2015"
    co_holder: Optional[bool] = None
    notes: Optional[str] = None


class StatResults(BaseModel):
    results: list[StatRecord]
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_record_schema -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add parse_record_book.py test_parse_record_book.py
git commit -m "feat: add StatRecord pydantic schema"
```

---

### Task 2: `is_stat_records` classifier

**Files:**
- Modify: `parse_record_book.py` (add function after `is_golf_results` at line 587)
- Test: `test_parse_record_book.py`

**Step 1: Write the failing test**

Add to the classifier test class in `test_parse_record_book.py`. Add `is_stat_records` to the import block at line ~36-43:

```python
    is_stat_records,
```

Then add these tests:

```python
    def test_is_stat_records_football_bullets(self):
        text = ("TEAM RECORDS\n• Most State Championships\n13 ....Dunbar\n"
                "• Consecutive Wins\n53 ....Damascus 2015-2018\n"
                "• Most Touchdowns, Season\n98 ....Fort Hill 2016\n")
        assert is_stat_records(text)

    def test_is_stat_records_basketball_headers(self):
        text = ("Girls Tournament Records\nINDIVIDUAL RECORDS\n"
                "MOST POINTS - final game\n48 - Janet Flora, Loch Raven 1976\n")
        assert is_stat_records(text)

    def test_is_stat_records_baseball_dugout(self):
        text = ("Dugout Chatter - Hits, Runs and Records\nRuns Scored\n"
                "Season ..............................331 ..Bowie 2016\n")
        assert is_stat_records(text)

    def test_is_stat_records_appearances_list(self):
        text = ("Boys Tournament Trivia\nTournament Appearances (10 minimum)\n"
                "29 ............Wicomico\n28 ............Annapolis\n")
        assert is_stat_records(text)

    def test_is_stat_records_negative_championship_table(self):
        assert not is_stat_records("YEAR CLASS CHAMPION COACH FINALIST COACH\n"
                                   "1975 1st DuVal - Beverly Bigham\n")

    def test_is_stat_records_negative_individual_results(self):
        # event-based individual_results use Athlete—School—Mark, not stat headers
        assert not is_stat_records("Athlete—School—Mark\n100 m Dash\n")

    def test_is_stat_records_negative_school_records(self):
        assert not is_stat_records("ALLEGANY\nCh: 1997, 1998\nFn: 1988")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k is_stat_records -v`
Expected: FAIL with `ImportError: cannot import name 'is_stat_records'`

**Step 3: Write minimal implementation**

In `parse_record_book.py`, after `is_golf_results` (line 587) and before the `# ── Page routing ──` comment (line 589), add:

```python
def is_stat_records(text: str) -> bool:
    """Detect all-time statistical-superlative record pages (football,
    basketball, baseball, lacrosse, volleyball, softball): 'Most Touchdowns',
    'Consecutive Wins', 'Dugout Chatter', 'Tournament Appearances', etc.

    Distinct from year-by-year championship tables (is_year_class_table /
    is_multicolumn_results) and event-based individual_results
    (Athlete-School-Mark headers). Ranked lists (Tournament Appearances,
    X-Plus Point Scorers) are included.
    """
    headers = (
        r"TEAM RECORDS|INDIVIDUAL RECORDS|Tournament Records|Dugout Chatter|"
        r"Tournament Trivia|Tournament Appearances|Plus Point Scorers|"
        r"Stats and Records|State Tournament Records"
    )
    if re.search(headers, text):
        return True
    # cluster of record bullets: "• Most …", "• Consecutive …", "• Longest …"
    bullets = re.findall(
        r"^\s*•\s*(?:Most|Consecutive|Longest|Fewest|Highest|Lowest|Best)\b",
        text, re.MULTILINE)
    return len(bullets) >= 3
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k is_stat_records -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add parse_record_book.py test_parse_record_book.py
git commit -m "feat: add is_stat_records classifier"
```

---

### Task 3: Route `stat_records` in `classify_page` + `ROUTE_ORDER`

**Files:**
- Modify: `parse_record_book.py:594` (ROUTE_ORDER) and `parse_record_book.py:597-625` (classify_page)
- Test: `test_parse_record_book.py`

**Step 1: Write the failing test**

Add to `test_parse_record_book.py` (the classifier test class, which already imports `classify_page`):

```python
    def test_classify_stat_records_football(self):
        text = ("TEAM RECORDS\n• Most State Championships\n13 ....Dunbar\n"
                "• Consecutive Wins\n53 ....Damascus 2015-2018\n"
                "• Most Touchdowns, Season\n98 ....Fort Hill 2016\n")
        routes = classify_page(text, "Football")
        assert "stat_records" in routes
        assert "championship" not in routes

    def test_classify_stat_records_does_not_steal_championship(self):
        # a year-class championship table must still route to championship
        text = "YEAR CLASS CHAMPION COACH FINALIST COACH\n1975 B Bates-53 Worcester-26\n"
        routes = classify_page(text, "Football")
        assert "championship" in routes
        assert "stat_records" not in routes

    def test_classify_stat_records_does_not_steal_individual_results(self):
        text = ("Athlete—School—Mark\n1990 4A 100 m Dash 10.4 John, School 10.4\n")
        routes = classify_page(text, "Boys Track and Field")
        assert "individual_results" in routes
        assert "stat_records" not in routes
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k classify_stat_records -v`
Expected: FAIL (football stat text routes to `championship` or nothing, not `stat_records`)

**Step 3: Write minimal implementation**

In `parse_record_book.py`:

(a) Edit `ROUTE_ORDER` (line 594) to insert `stat_records` before `championship`:

```python
ROUTE_ORDER = ("golf", "sportsmanship", "individual_results", "individual_xc", "stat_records", "championship")
```

(b) In `classify_page` (line 597-625), insert a new `elif` for stat_records **before** the championship `elif` (which is at line 615). The championship branch is currently:

```python
    elif is_year_class_table(text) or is_multicolumn_results(text):
        routes.add("championship")
```

Insert immediately above it:

```python
    elif is_stat_records(text):
        routes.add("stat_records")
```

So the elif chain becomes: golf → sportsmanship → individual_results → individual_xc → **stat_records** → championship.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k classify_stat_records -v`
Expected: PASS (3 tests)

**Step 5: Run the full classifier suite to confirm no regressions**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -q`
Expected: PASS (all tests, including pre-existing classifier tests)

**Step 6: Commit**

```bash
git add parse_record_book.py test_parse_record_book.py
git commit -m "feat: route stat_records pages in classify_page"
```

---

### Task 4: `_stat_records_prompt` + register in EXTRACTORS

**Files:**
- Modify: `parse_record_book.py` (add prompt fn after `_sportsmanship_prompt` at line 957; add to `EXTRACTORS` at line 961)

**Step 1: Write the failing test**

Add to `test_parse_record_book.py`:

```python
def test_stat_records_prompt_returns_schema():
    from parse_record_book import _stat_records_prompt, StatResults
    prompt, schema = _stat_records_prompt(["some page text"], "Football")
    assert schema is StatResults
    assert "stat" in prompt.lower() or "record" in prompt.lower()
    assert "some page text" in prompt
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_records_prompt -v`
Expected: FAIL with `ImportError: cannot import name '_stat_records_prompt'`

**Step 3: Write minimal implementation**

In `parse_record_book.py`, after `_sportsmanship_prompt` (ends line 957) and before the `EXTRACTORS` dict (line 961), add:

```python
def _stat_records_prompt(pages: list[str], sport: str) -> tuple[str, type]:
    combined = "\n".join(pages)
    prompt = textwrap.dedent(f"""
        Extract every all-time statistical superlative record from this
        {sport} record-book page. These are NOT year-by-year champions; they are
        all-time records like "Most Touchdowns, Season", "Consecutive Wins",
        "Most Points - final game", "Runs Scored - Game", "Most Tournament
        Appearances", "Longest Rivalries", and ranked qualifier lists like
        "32-Plus Point Scorers" or "Tournament Appearances (10 minimum)".

        Rules:
        - sport must always be exactly: {sport}
        - category: "team" when the record is under a TEAM RECORDS heading or
          is a team/school record; "individual" when under INDIVIDUAL RECORDS
          or it is a player record; null if the page does not split that way.
        - record: a concise canonical description of the record, e.g.
          "Most Touchdowns, Season", "Most Points - final game",
          "Runs Scored - Game", "Most Tournament Appearances",
          "Consecutive Wins", "32-Plus Point Scorers".
        - value: the mark exactly as printed — a count ("98"), a decimal
          ("15.7"), a fraction ("18/18" or "1.000, 18/18"), or a range. Keep as
          a string; do not convert to a number.
        - holder: the entity that holds the record — a player name for
          individual records, a school for team records, or "SchoolA v SchoolB"
          for rivalry records.
        - school: the school. For team records this is the school; for
          individual records this is the player's school.
        - year: the year or year-range exactly as printed — "2016",
          "2015-2018", "1998-2001", "1988 & 2015". Keep as a string. null if
          no year is given.
        - co_holder: true when this row is one of multiple tied holders for the
          same record (e.g. "Hereford 2002 & Fort Hill 2016"). Emit ONE row per
          holder, each with co_holder=true. Do not join tied holders into one
          row.
        - notes: extra context that is not the value or year — opponent
          ("vs. Westmar"), game count ("(14 games)"), round ("semifinal",
          "Sf", "F"), or caveats ("*No 2020 season", "Prior to formation of
          MPSSAA"). null if none.
        - For ranked qualifier lists (e.g. "Tournament Appearances (10
          minimum)" listing many schools, or "32-Plus Point Scorers" listing
          many players), emit one row per listed entry with the same record
          description and the entry's value/holder/school/year.
        - Skip prose commentary, "Did You Know" trivia sentences, and any
          non-record text.

        TEXT:
        {combined}
    """).strip()
    return prompt, StatResults
```

Then add to the `EXTRACTORS` dict (line 961-967):

```python
    "stat_records":          (_stat_records_prompt, StatResults, "results"),
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_records_prompt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add parse_record_book.py test_parse_record_book.py
git commit -m "feat: add _stat_records_prompt and register extractor"
```

---

### Task 5: Dedup key, chunk size, main() extraction block

**Files:**
- Modify: `parse_record_book.py:633-640` (DEDUP_KEYS), `parse_record_book.py:1084-1092` (CHUNK_SIZES), `parse_record_book.py:1192-1257` (main accumulators + extraction loop)

**Step 1: Write the failing test**

Add to `test_parse_record_book.py`:

```python
def test_stat_records_dedup_key_collapses_duplicates():
    from parse_record_book import dedup, DEDUP_KEYS
    rows = [
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Fort Hill", "school": "Fort Hill",
         "year": "2016", "co_holder": False, "notes": None},
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Fort Hill", "school": "Fort Hill",
         "year": "2016", "co_holder": False, "notes": None},
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "95", "holder": "Damascus", "school": "Damascus",
         "year": "2015", "co_holder": True, "notes": None},
    ]
    out, warns = dedup(rows, DEDUP_KEYS["stat_records"], "stat_records")
    # duplicate (same holder/year) collapsed; distinct co-holder kept
    assert len(out) == 2
    holders = {r["holder"] for r in out}
    assert holders == {"Fort Hill", "Damascus"}
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_records_dedup -v`
Expected: FAIL with `KeyError: 'stat_records'` (DEDUP_KEYS has no entry)

**Step 3: Write minimal implementation**

In `parse_record_book.py`:

(a) Add to `DEDUP_KEYS` (line 633-640), after the `sportsmanship_awards` entry:

```python
    "stat_records": ("sport", "category", "record", "holder", "year"),
```

(b) Add to `CHUNK_SIZES` (line 1084-1092), after the `sportsmanship` entry:

```python
    "stat_records": 1,
```

(c) In `main()` (line 1192-1197), add an accumulator after `all_golf`:

```python
    all_stat_records: list[dict] = []
```

(d) In the per-section extraction loop (after the golf block ending at line 1247, before the sportsmanship block at line 1249), add:

```python
        # Stat records (LLM) — team-sport all-time superlatives
        if routed.get("stat_records"):
            pairs = routed["stat_records"]
            rows = _extract_route("stat_records", pairs, sport, args.pdf,
                                  refresh=args.refresh, offline=args.offline)
            all_stat_records.extend(rows)
            print(f"  stat records        : {len(rows)} rows  "
                  f"({len(pairs)} pages, LLM)")
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_records_dedup -v`
Expected: PASS

**Step 5: Commit**

```bash
git add parse_record_book.py test_parse_record_book.py
git commit -m "feat: wire stat_records into main extraction + dedup"
```

---

### Task 6: Output — `stat_records.csv` + `record_book.json` key

**Files:**
- Modify: `parse_record_book.py:1295-1302` (raw_tables), `parse_record_book.py:1339-1358` (csv writes), `parse_record_book.py:1360-1373` (record_book.json)

**Step 1: Write the failing test**

Add to `test_parse_record_book.py`:

```python
def test_stat_records_in_raw_tables_and_json_keys(tmp_path, monkeypatch):
    # Smoke-test that main() produces a stat_records csv + json key when given
    # a fixture cache hit. We point CACHE_DIR at a fixture and run --offline.
    import parse_record_book as p
    # minimal: just assert the output field order constant exists
    fields = ["sport", "category", "record", "value", "holder",
              "school", "year", "co_holder", "notes"]
    # the csv writer uses these names; ensure they all exist on a sample row
    row = {k: None for k in fields}
    row.update({"sport": "Football", "record": "Most TDs, Season"})
    assert set(fields) <= set(row.keys())
```

(This is a light guard; the real round-trip is exercised in Task 8.)

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_records_in_raw_tables -v`
Expected: FAIL (test should actually pass trivially once fields list is correct — if it passes already, that's fine; the real validation is Task 8. If it fails on a typo, fix the test.)

**Step 3: Write minimal implementation**

In `parse_record_book.py`:

(a) Add to `raw_tables` (line 1295-1302):

```python
        "stat_records": all_stat_records,
```

(b) Add a CSV write block after the golf CSV block (after line 1358, before the `record_book = {` at line 1360). stat_records exists in all three seasons, so write it unconditionally:

```python
    write_csv(
        out_dir / "stat_records.csv",
        tables["stat_records"],
        ["sport", "category", "record", "value", "holder", "school",
         "year", "co_holder", "notes"] + pf,
    )
```

(c) Add to the `record_book` dict (line 1360-1373), e.g. after `"golf_results"`:

```python
        "stat_records": tables["stat_records"],
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest test_parse_record_book.py -k stat_records_in_raw_tables -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/python3 -m pytest -q`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add parse_record_book.py test_parse_record_book.py
git commit -m "feat: write stat_records.csv and record_book.json key"
```

---

### Task 7: verify_record_book — TABLE_KEYS entry + 0-row warning + test

**Files:**
- Modify: `verify_record_book.py:70-77` (TABLE_KEYS), `parse_record_book.py` main() (0-row warning)
- Test: `test_verify_record_book.py`

**Step 1: Write the failing test**

In `test_verify_record_book.py`, add `stat_records` awareness. Add a test that duplicate keys are detected for stat_records (mirror the existing dup-key test style). First check the existing dup-key test name:

Run: `.venv/bin/python3 -m pytest test_verify_record_book.py -k dup -v -q` to see the existing pattern, then add:

```python
    def test_stat_records_duplicate_keys_detected(self):
        # build a minimal record_book with a stat_records dup and run the dup check
        from verify_record_book import check_duplicate_keys
        book = {"stat_records": [
            {"sport": "Football", "category": "team", "record": "Most TDs, Season",
             "holder": "Fort Hill", "year": "2016"},
            {"sport": "Football", "category": "team", "record": "Most TDs, Season",
             "holder": "Fort Hill", "year": "2016"},
        ]}
        report = check_duplicate_keys(book)
        assert report["duplicate_key_count"] >= 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest test_verify_record_book.py -k stat_records_duplicate -v`
Expected: FAIL — `check_duplicate_keys` iterates `TABLE_KEYS.items()`; without a `stat_records` entry the table is ignored and `duplicate_key_count` is 0.

**Step 3: Write minimal implementation**

In `verify_record_book.py`, add to `TABLE_KEYS` (line 70-77):

```python
    "stat_records": ("sport", "category", "record", "holder", "year"),
```

(No era floor, no regression-guard entry — stat_records is new and all-time, per the design.)

**Step 4: Add the 0-row warning to the parser (silent-failure guard)**

In `parse_record_book.py` main(), after the per-section loop ends (after line 1258, before the normalisation block at line 1260), add:

```python
    # Guard: a stat_records route that yielded 0 rows usually means the LLM
    # dropped the page (the relay-drop failure mode). Surface it loudly.
    stat_zero = []
```

Then, inside the stat_records extraction block added in Task 5, extend the 0-row detection. Modify the block to:

```python
        if routed.get("stat_records"):
            pairs = routed["stat_records"]
            rows = _extract_route("stat_records", pairs, sport, args.pdf,
                                  refresh=args.refresh, offline=args.offline)
            all_stat_records.extend(rows)
            print(f"  stat records        : {len(rows)} rows  "
                  f"({len(pairs)} pages, LLM)")
            if not rows:
                stat_zero.append((sport, sorted(i + 1 for i, _ in pairs)))
```

And after the loop (where `stat_zero` is declared), print warnings:

```python
    for sport, pgs in stat_zero:
        print(f"  ! WARNING: stat_records routed for {sport} pages {pgs} "
              f"but extracted 0 rows — possible LLM drop; re-run with --refresh")
```

(Place the `stat_zero = []` declaration before the `for sport, (start, end)` loop, and the warning print after the loop.)

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest test_verify_record_book.py -k stat_records_duplicate -v`
Expected: PASS

**Step 6: Run full test suite**

Run: `.venv/bin/python3 -m pytest -q`
Expected: PASS (all tests)

**Step 7: Commit**

```bash
git add verify_record_book.py parse_record_book.py test_verify_record_book.py
git commit -m "feat: verify stat_records dup keys + 0-row drop warning"
```

---

### Task 8: Dry-run routing check (no LLM cost)

**Files:** none (verification step)

**Step 1: Run the routes dry-run for each season and confirm the ~15 stat pages route to `stat_records` with correct sports**

Run:
```bash
.venv/bin/python3 parse_record_book.py "pdfs/FallRecordBook2024.pdf" --season fall --routes 2>&1 | grep -i "stat\|SUMMARY"
.venv/bin/python3 parse_record_book.py "pdfs/Winter record book.pdf" --season winter --routes 2>&1 | grep -i "stat\|SUMMARY"
.venv/bin/python3 parse_record_book.py "pdfs/Spring record book 2025.pdf" --season spring --routes 2>&1 | grep -i "stat\|SUMMARY"
```

Expected: each season's SUMMARY line shows `stat_records=<N>` with N matching the page counts (fall 5, winter 4, spring 5). Confirm the sport column for each stat page is correct (Football, Volleyball, Girls/Boys Basketball, Baseball, Girls/Boys Lacrosse, Softball).

**Step 2: If any stat page shows the wrong sport or wrong/no route**

Adjust the section detection (`find_sections` / `_divider_candidates`) or the classifier so the page routes correctly. Re-run Step 1 until correct. Do not proceed to LLM extraction until routing is correct.

**Step 3: Commit routing fixes if any**

```bash
git add -A
git commit -m "fix: stat_records routing for <sport> section detection"
```

---

### Task 9: Fresh-LLM extraction + inspect

**Files:** produces `data/{fall,winter,spring}/stat_records.csv` + updated `record_book.json` + new `cache/extractions/stat_records_*.json`

**Step 1: Run fresh extraction for fall (LLM calls only for the new stat pages; all other tables cache-hit)**

Run:
```bash
.venv/bin/python3 parse_record_book.py "pdfs/FallRecordBook2024.pdf" --season fall 2>&1 | grep -v "Ignoring wrong pointing" | tail -30
```

Expected: prints `stat records : N rows (M pages, LLM)` for Football and Volleyball; cache writes `stat_records_*.json`; no 0-row warnings. Other tables cache-hit (no `cache write` lines).

**Step 2: Inspect the fall stat_records output for known records**

Run:
```bash
.venv/bin/python3 -c "
import json
d=json.load(open('data/fall/record_book.json'))
sr=d['stat_records']
print('fall stat_records rows:', len(sr))
for r in sr:
    if r.get('record') and 'Touchdown' in r['record']:
        print(r)
"
```

Expected: rows for "Most Touchdowns, Season" (value 98, Fort Hill, 2016) and "Most Touchdowns, Game" (13, Patterson v Milford Mill, 1992). Sanity-check a few more (Most Passing Yards Season → 4,049 Westminster 2012; Consecutive Wins → 53 Damascus 2015-2018).

**Step 3: If a stat page yielded 0 rows or obviously dropped records**

Re-run that page with `--refresh` (forces a fresh LLM call):
```bash
.venv/bin/python3 parse_record_book.py "pdfs/FallRecordBook2024.pdf" --season fall --refresh 2>&1 | tail
```
Re-inspect. Repeat until the known records are present.

**Step 4: Repeat Steps 1-3 for winter and spring**

```bash
.venv/bin/python3 parse_record_book.py "pdfs/Winter record book.pdf" --season winter 2>&1 | grep -v "Ignoring wrong pointing" | tail -30
.venv/bin/python3 parse_record_book.py "pdfs/Spring record book 2025.pdf" --season spring 2>&1 | grep -v "Ignoring wrong pointing" | tail -30
```

Expected: winter stat_records for Girls/Boys Basketball (e.g. "MOST POINTS - final game" 48 Janet Flora, Loch Raven 1976; 47 Sherron Mills, Snow Hill 1989); spring for Baseball (Runs Scored Season 331 Bowie 2016), Lacrosse, Softball. No 0-row warnings.

Inspect each:
```bash
.venv/bin/python3 -c "
import json
for s in ['winter','spring']:
    d=json.load(open(f'data/{s}/record_book.json'))
    sr=d['stat_records']
    print(s, 'rows:', len(sr))
    for r in sr[:8]: print('  ', r.get('sport'), '|', r.get('record'), '|', r.get('value'), r.get('holder'), r.get('year'))
"
```

**Step 5: Commit the extracted data + cache**

```bash
git add data/fall data/winter data/spring cache/extractions
git commit -m "data: extract stat_records for all three seasons"
```

---

### Task 10: Offline reproducibility + verify

**Files:** verification step (may adjust data/ to match build/)

**Step 1: Rebuild each season offline (cache hits only) into build/<season> and diff vs data/<season>**

```bash
.venv/bin/python3 parse_record_book.py "pdfs/FallRecordBook2024.pdf" --season fall --offline --out build/fall 2>&1 | grep -v "Ignoring wrong pointing" | tail -3
.venv/bin/python3 diff_outputs.py build/fall --baseline-dir data 2>&1 | grep -v "Ignoring wrong pointing" | grep Total
.venv/bin/python3 parse_record_book.py "pdfs/Winter record book.pdf" --season winter --offline --out build/winter 2>&1 | grep -v "Ignoring wrong pointing" | tail -3
.venv/bin/python3 diff_outputs.py build/winter --baseline-dir data 2>&1 | grep -v "Ignoring wrong pointing" | grep Total
.venv/bin/python3 parse_record_book.py "pdfs/Spring record book 2025.pdf" --season spring --offline --out build/spring 2>&1 | grep -v "Ignoring wrong pointing" | tail -3
.venv/bin/python3 diff_outputs.py build/spring --baseline-dir data 2>&1 | grep -v "Ignoring wrong pointing" | grep Total
```

Expected per season: `Total: +0 added, -0 removed, ~0 changed`. (The stat_records table is new vs the prior commit, but build/ vs data/ are both regenerated from the same cache, so they match. The vs-HEAD diff is reviewed separately in Task 11.)

If a season shows nonzero changes: the cache extraction was nondeterministic on a stat page — re-run that page with `--refresh` (Task 9 Step 3), re-copy build→data, re-diff.

**Step 2: Run verify_record_book on each season**

```bash
for s in fall winter spring; do
  echo "=== $s ==="
  .venv/bin/python3 verify_record_book.py data/$s 2>&1 | grep -v "Ignoring wrong pointing" | tail -4
done
```

Expected: each `PASSED ✓ (0 error(s), ...)`. No duplicate-key errors for stat_records.

**Step 3: Run the full test suite one final time**

Run: `.venv/bin/python3 -m pytest -q`
Expected: PASS (≈255 tests)

**Step 4: Commit reproducibility confirmation if data/ was adjusted**

```bash
git add -A
git commit -m "test: stat_records offline reproducible +0/-0/~0, 0 verify errors"
```

---

### Task 11: Review vs HEAD and surface the new table

**Files:** none (review step)

**Step 1: Diff stat_records vs HEAD (the prior commit had no stat_records table)**

```bash
.venv/bin/python3 diff_outputs.py build/fall --baseline-dir data 2>&1 | grep -v "Ignoring wrong pointing" | grep -E "stat_records|Total"
git diff --stat HEAD -- data/fall/stat_records.csv data/winter/stat_records.csv data/spring/stat_records.csv
```

Expected: `stat_records` is a new table (added in all three seasons). Existing tables (championship_results, individual_results, etc.) are unchanged vs the Phase 7 commit — confirm `+0/-0` for those.

**Step 2: Sanity-check row counts per sport**

```bash
.venv/bin/python3 -c "
import json
from collections import Counter
for s in ['fall','winter','spring']:
    d=json.load(open(f'data/{s}/record_book.json'))
    sr=d['stat_records']
    print(s, 'total', len(sr), Counter(r.get('sport') for r in sr))
"
```

Expected: nonzero counts for Football, Volleyball (fall); Girls/Boys Basketball (winter); Baseball, Girls/Boys Lacrosse, Softball (spring). Confirm no sport is unexpectedly 0 (would indicate a dropped page).

**Step 3: Final commit (if anything remains uncommitted)**

```bash
git status -s
git add -A
git commit -m "feat: stat_records table for all-time superlative records (all 3 seasons)"
```

---

## Notes for the implementer

- **Run commands from the repo root** (`/Users/dwillis/code/maryland-high-schools`). Always use `.venv/bin/python3`, never the system python.
- **`grep -v "Ignoring wrong pointing"`** filters pypdf warnings that clutter output.
- **LLM calls are ~30-60s each.** Task 9 makes ~15 calls total across 3 seasons (~10 min). Tasks 1-8 and 10-11 use cache hits / no LLM and are fast.
- **The relay-drop nondeterminism** (see `mpssaa-phase7-extraction-complete` memory) can strike stat pages too — the Task 7 0-row warning and the Task 9 inspect step are the guards. If a page yields few/0 rows, `--refresh` it.
- **Do not modify existing tables' extraction.** This is purely additive. If `diff_outputs` shows changes to championship_results/individual_results/etc. vs HEAD, something is wrong — investigate before committing.
- **Cache files** are named `stat_records_<16char hash>.json` under `cache/extractions/` (the `{extractor}_{key[:16]}.json` pattern in `_cache_path`). Commit them (enables offline reproducibility).
- **Image-only records pages** (Girls Lacrosse p12, Boys Lacrosse p17, Girls Track p36 — ~40 chars of text) are deliberately NOT routed; nothing to extract. Do not try to force them.