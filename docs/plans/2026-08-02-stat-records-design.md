# MPSSAA Statistical Superlative Records — Design

Date: 2026-08-02
Branch: `claude/mpssaa-pipeline`

## Problem

The MPSSAA record books contain "all-time statistical superlative" sections for
team sports — e.g. Football "TEAM RECORDS / INDIVIDUAL RECORDS" (most
touchdowns, most passing yards, longest rivalries), Basketball "Tournament
Records" (most points in a final, most rebounds), Baseball "Dugout Chatter"
(runs, hits, doubles), Lacrosse tournament records, Volleyball tournament
records, Softball state tournament records. These are **not** captured by any
existing table:

- `championship_results` holds year-by-year team champions.
- `school_records` holds champion-school honor rolls (one row per school per
  sport; `record_type`/`record` are `None` for all 670 fall rows).
- `individual_results` is event-based year-by-year champion records (track,
  swimming, tennis, wrestling) — fall has 0 rows because no fall sport
  produces that shape.
- `individual_xc_champions` / `golf_results` cover XC and golf.

So the superlative records are currently dropped. This design adds a new
`stat_records` table to capture them.

## Scope

All three seasons, superlative records only — excluding pages already captured
by other tables (tennis individual champions, championship tables, school
appearance summaries) and excluding pages whose text layer is image-only.

Text-extractable stat-record pages (~15 pages, 2-8k chars each):

| season | sport | pages (1-indexed) |
|---|---|---|
| fall | Football | 37, 38 |
| fall | Volleyball | 74, 75, 76 |
| winter | Girls Basketball | 6, 7 |
| winter | Boys Basketball | 13, 20 |
| spring | Baseball | 10, 11 |
| spring | Girls Lacrosse | 16 |
| spring | Boys Lacrosse | 21 |
| spring | Softball | 26 |

Out of scope (image-only text layer, ~40 chars): Girls Lacrosse p12, Boys
Lacrosse p17, Girls Track & Field p36 — left unextracted as today.

## Decisions (settled in brainstorming)

- **Table:** new `stat_records` table (record-centric; one row per holder),
  separate from `school_records` (which stays school-centric).
- **Extraction:** LLM (GLM-5.2 via Ollama) with a new prompt + pydantic schema,
  page-level cache, consistent with the existing pipeline. Chosen over
  regex/hybrid because the layout varies by sport (bullets, CAPS headers,
  dot-leaders) across three seasons.
- **Seasons:** all three.

## Schema

One row per record-holder. Ties / co-holders → multiple rows sharing
`(sport, record)`.

| field | type | notes |
|---|---|---|
| `sport` | str | Football, Girls Basketball, Baseball, Girls Lacrosse, Volleyball, … |
| `category` | `"team"` \| `"individual"` \| `None` | from TEAM/INDIVIDUAL RECORDS grouping; `None` when the page doesn't split that way (Dugout Chatter) |
| `record` | str | normalized description: "Most Points, Season", "Most Points - final game", "Runs Scored - Season", "Most Tournament Appearances", "Consecutive Wins", "32-Plus Point Scorers" |
| `value` | str | the mark, as-is: "13", "5,868", "49", "331", "1.000, 18/18" (heterogeneous counts/decimals/fractions → string) |
| `holder` | str \| None | player name (individual), school (team), or "City v Polytechnic" (rivalry) |
| `school` | str \| None | the school; team records → the school, individual → player's school |
| `year` | str \| None | "2016", "2015-2018", "1988 & 2015" (ranges/multiples → string) |
| `co_holder` | bool | true when one of multiple tied holders |
| `notes` | str \| None | context: "vs. Westmar", "(14 games)", "semifinal", "*No 2020 season", "Prior to formation of MPSSAA" |

**Dedup key:** `(sport, category, record, holder, year)` — ties produce distinct
holders, ranked lists produce distinct schools/players.
`TABLE_KEYS["stat_records"]` = that tuple.

## Routing & classifier

New classifier `is_stat_records(text)`. Signals (any match, when the page's
sport context is a team sport): headers `TEAM RECORDS` / `INDIVIDUAL RECORDS` /
`Tournament Records` / `Dugout Chatter` / `Tournament Trivia` / `Tournament
Appearances` / `Plus Point Scorers` / `Stats and Records` / `State Tournament
Records`, **or** a cluster of `• Most …` / `Consecutive …` / `Longest …`
bullet lines.

`ROUTE_ORDER`: insert `stat_records` after `individual_results` / `individual_xc`
and before `championship`, so pages already captured by
golf/sportsmanship/individual_results/individual_xc keep their routing and the
currently-unrouted stat pages don't fall through to `championship`. The stat
pages don't match `is_individual_results` (no `Athlete—School—Mark` header, no
`year class Name,School time` pattern); a test asserts no stat page
double-routes.

Sport context comes from `route_pages`'s existing section tracking; verify and
repair sport detection for the ~15 stat pages (e.g. p74-76 → Volleyball, p37-38
→ Football, p26 → Softball).

**Chunk size:** 1 page (dense, format-varied; tight schema; matches the
individual_results precedent).

## Extraction

New `_stat_records_prompt(pages, sport)` + pydantic `StatRecord` /
`StatResultsPage` schema, registered in the prompt table under route key
`"stat_results"` (cache filename prefix `stat_…`). GLM-5.2 via Ollama, same
`json_object` + schema-in-prompt pattern, page-level cache keyed identically to
the other tables. The prompt instructs the model to:

- emit one row per holder; split ties into separate rows with `co_holder=true`,
- normalize `record` to a concise canonical phrase,
- put ranked-list entries (Tournament Appearances, 32-Plus Point Scorers) each
  on their own row,
- leave `value` and `year` as raw strings.

## Output

`stat_results.csv` + a `stat_records` key in `record_book.json`, written the
same way as the other tables. `--offline` works (cache hits); `make
rebuild-offline` / `diff_outputs.py` cover it with no special handling.

## Verify

- Add `stat_records` to `TABLE_KEYS` → automatic duplicate-key check.
- No era floors (all-time records, not a year series).
- No regression guard vs HEAD (table is new; the pre-extraction snapshot +
  diff is the safety net, as for winter/spring).
- New check: warn if a stat page routed but yielded 0 rows (catches a
  relay-drop-style silent failure).

## Testing

- Unit tests for `is_stat_records`: true on the ~15 stat pages, false on
  championship / individual_results / tennis / school-appearance pages.
- Dedup-key test.
- A small fixture cache file round-tripped through the offline rebuild.
- Add `stat_records` to `test_verify_record_book.py` dup-key test.
- 245 → ~255 tests.

## Rollout

Implement → fresh-LLM extraction of the ~15 pages (~15 calls, ~10 min) →
inspect output for known football/basketball records → rebuild offline and
confirm `+0/−0/~0` → verify 0 errors → commit on `claude/mpssaa-pipeline`
(new table + code + tests + cache + data). Purely additive: no changes to
existing fall/winter/spring rows.

## Out of scope (YAGNI)

Image-only records pages (lacrosse/track title pages); numeric `value` column;
era floors; regression guard.