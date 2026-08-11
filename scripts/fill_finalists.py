#!/usr/bin/env python3
"""Back-fill ``finalist_school`` for head-to-head contest sports from
authoritatively-sourced state-final recaps.

The MPSSAA record-book PDFs print only champion + score + coach per class on
their modern "honor roll" pages — **the runner-up is not in the PDFs**. So the
opponent must be sourced, not re-extracted. This script reads a hand-curated
recap data file (``data/<season>/<sport>_finalists.json``) listing, per
(state-final) entry, the champion, score, runner-up (``finalist``), and a
source URL — each one verified against a published state-final recap.

Only entries present in that data file are written; **every other year's
``finalist_school`` stays blank**. There is no LLM step and no inference —
nothing is guessed. Coverage is extended year-by-year by adding entries to the
data file and re-running ``--apply``.

Each entry is validated before writing:

- **champion match** — the entry's ``champion`` must normalize-equal the
  championship row's ``champion_school`` for that (year, classification). This
  is the load-bearing check: it catches a finalist filed under the wrong year,
  class, or champion.
- **score match** — the entry's ``score`` must equal the row's ``score`` (extra
  confirmation the recap and the record-book row describe the same final).
- **finalist != champion** — a school cannot be both.
- **no reuse** — the same finalist school cannot be assigned to two
  classifications in the same year.
- **candidate-set warning** (soft) — the finalist *should* appear in the
  per-(sport, year) candidate set built from ``school_records.csv``
  (finalist/runner-up years). ``school_records`` is known-incomplete, so a miss
  is logged as a warning, not a rejection. (If it were hard, true entries like
  2024 4A/3A Mergenthaler Vo-Tech — absent from school_records but confirmed by
  four sources — would be wrongly dropped.)

Results are written into the **championship extraction cache**
(``cache/extractions/championship_*.json``), not the CSV/JSON directly — both
``championship_results.csv`` and ``record_book.json`` are regenerated from that
cache by ``parse_record_book.py`` (mirrors ``scripts/patch_season_relays.py``).
Provenance goes in the row-level ``notes`` field (the only field the pipeline
never re-stamps). ``--apply`` patches the cache then regenerates offline.

Usage:
  uv run scripts/fill_finalists.py --sport Football                          # dry-run
  uv run scripts/fill_finalists.py --sport Football --apply
  uv run scripts/fill_finalists.py --sport "Boys Basketball" --apply
  uv run scripts/fill_finalists.py --sport Football --year 2024              # audit one year
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

import parse_record_book as p          # noqa: E402
import build_site                        # noqa: E402
from verify_record_book import _iter_years  # noqa: E402

CACHE = p.CACHE_DIR
PDFS = {"fall": "pdfs/FallRecordBook2024.pdf",
        "winter": "pdfs/Winter record book.pdf",
        "spring": "pdfs/Spring record book 2025.pdf"}

SPORT_CONFIG = {
    "Football":        {"season": "fall",   "candidate_field": "runner_up_years"},
    "Boys Basketball": {"season": "winter", "candidate_field": "finalist_years"},
}
# Per-sport finalist year field, for sports beyond the pilot (the candidate
# field is sport-specific; school_records uses runner_up_years for Football,
# finalist_years for everything else).
FINALIST_FIELD = {
    "Football": "runner_up_years",
    "Boys Soccer": "finalist_years",
    "Girls Soccer": "finalist_years",
    "Field Hockey": "finalist_years",
    "Volleyball": "finalist_years",
    "Boys Basketball": "finalist_years",
    "Girls Basketball": "finalist_years",
    "Baseball": "finalist_years",
    "Boys Lacrosse": "finalist_years",
    "Girls Lacrosse": "finalist_years",
    "Softball": "finalist_years",
}

NOTE_RECAP = "finalist:recap-verified"
PRE_MPSSAA = "Pre-MPSSAA"
MANUAL_CORRECTION = ("finalist back-filled from authoritatively-sourced state-final "
                     "recaps (see data/<season>/<sport>_finalists.json)")


# ── Loading + candidate construction ──────────────────────────────────────────

def load_rows(season, sport):
    """Read championship_results.csv + school_records.csv for one sport.

    The CSV (not record_book.json) is the human-auditable input for the
    matching step; the JSON is regenerated from cache later. Returns
    (champ_rows, school_rows) filtered to the sport.
    """
    import csv
    base = Path("data") / season
    champ_rows, school_rows = [], []
    with (base / "championship_results.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["sport"] == sport:
                champ_rows.append(r)
    with (base / "school_records.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["sport"] == sport:
                school_rows.append(r)
    return champ_rows, school_rows


def year_list_has(value, year):
    """True if a semicolon/JSON year-list field contains ``year`` (int)."""
    return any(int(y) == int(year) for y in _iter_years(value))


def build_champion_index(champ_rows, normalize):
    """{year: {normalized champion school names across all classes}}.

    Used both to exclude champions from the candidate set and as a quick
    champion-identity reference.
    """
    idx = defaultdict(set)
    for r in champ_rows:
        try:
            year = int(r["year"])
        except (ValueError, TypeError):
            continue
        n = normalize(r["champion_school"])
        if n:
            idx[year].add(n)
    return idx


def candidate_keys(school_rows, year, candidate_field, normalize, champion_idx):
    """Set of normalized finalist/runner-up school keys for ``year`` minus
    champions. This is the soft cross-check set, not a write gate."""
    champs = champion_idx.get(int(year), set())
    out = set()
    for r in school_rows:
        if year_list_has(r.get(candidate_field, ""), year):
            key = normalize(r["school"])
            if key and key not in champs:
                out.add(key)
    return out


def _titlecase(name):
    """Title-case an ALLCAPS school name ('BETHESDA-CHEVY CHASE' -> 'Bethesda-Chevy Chase')."""
    if not name:
        return name
    return name.title()


# ── Recap data file ──────────────────────────────────────────────────────────

def recaps_path(season, sport):
    """Path to the curated recap data file for (season, sport)."""
    slug = sport.lower().replace(" ", "_").replace("/", "-")
    return Path("data") / season / f"{slug}_finalists.json"


def load_recaps(season, sport):
    """Read the curated recap entries. Returns a list of entry dicts, each:
    {year, classification, champion, score, finalist, source}."""
    path = recaps_path(season, sport)
    if not path.exists():
        return []
    blob = json.loads(path.read_text(encoding="utf-8"))
    return blob.get("entries", [])


# ── Validation ───────────────────────────────────────────────────────────────

def validate_recaps(entries, champ_rows, school_rows, normalize, candidate_field,
                    champion_idx, *, year_filter=None):
    """Validate curated recap entries against the championship rows.

    Returns (plan, rejected, warnings):
      plan     — list of (year, classification, finalist_display, note) for
                 entries that passed every hard gate. These get written.
      rejected — list of (entry, [reasons]) for entries that failed a hard gate.
      warnings — list of (year, classification, msg) for soft issues (written
                 anyway): finalist not in the school_records candidate set, or
                 the row was already filled with a different finalist.
    """
    # Index championship rows by (year, classification) for matching.
    row_by_key = {}
    for r in champ_rows:
        try:
            y = int(r["year"])
        except (ValueError, TypeError):
            continue
        row_by_key[(y, r["classification"])] = r

    plan = []
    rejected = []
    warnings = []

    # Track finalist reuse within a year (normalized key).
    used_per_year = defaultdict(set)
    # Track which (year, classification) we've already planned, to reject
    # duplicate entries in the data file.
    seen = set()

    for entry in entries:
        try:
            year = int(entry["year"])
        except (KeyError, ValueError, TypeError):
            rejected.append((entry, ["entry has no parseable year"]))
            continue
        if year_filter is not None and year != int(year_filter):
            continue

        cls = entry.get("classification")
        key = (year, cls)
        if key in seen:
            rejected.append((entry, [f"duplicate entry for {year} {cls}"]))
            continue
        seen.add(key)

        row = row_by_key.get(key)
        if row is None:
            rejected.append((entry, [f"no championship row for {year} {cls}"]))
            continue

        reasons = []
        champ_entry = entry.get("champion", "")
        champ_row = row.get("champion_school", "")
        if normalize(champ_entry) != normalize(champ_row):
            reasons.append(
                f"champion mismatch: entry {champ_entry!r} != row {champ_row!r}")

        score_entry = (entry.get("score") or "").strip()
        score_row = (row.get("score") or "").strip()
        if score_entry and score_row and score_entry != score_row:
            reasons.append(f"score mismatch: entry {score_entry!r} != row {score_row!r}")

        finalist = (entry.get("finalist") or "").strip()
        if not finalist:
            reasons.append("entry has no finalist")
        else:
            nf = normalize(finalist)
            if not nf:
                reasons.append(f"finalist {finalist!r} normalizes to empty")
            elif nf == normalize(champ_row):
                reasons.append(f"finalist == champion ({finalist!r})")
            elif nf in used_per_year[year]:
                reasons.append(f"finalist {finalist!r} reused across classes in {year}")
            else:
                used_per_year[year].add(nf)

        if reasons:
            rejected.append((entry, reasons))
            continue

        # Hard gates passed. Soft candidate-set cross-check (warning only).
        cand = candidate_keys(school_rows, year, candidate_field, normalize, champion_idx)
        nf = normalize(finalist)
        if cand and nf not in cand:
            warnings.append((year, cls,
                f"finalist {finalist!r} not in school_records candidate set "
                f"(school_records may be incomplete)"))
        # If there is no candidate set at all for the year, we cannot cross-check
        # — but the champion+score match is the load-bearing guarantee, so we
        # proceed silently rather than warn on every pre-candidate year.

        # Soft: row already filled with a *different* finalist (conflict).
        existing = (row.get("finalist_school") or "").strip()
        if existing and normalize(existing) != nf:
            warnings.append((year, cls,
                f"row already filled with {existing!r}; entry would overwrite "
                f"with {finalist!r}"))

        plan.append((year, cls, finalist, NOTE_RECAP))

    return plan, rejected, warnings


# ── Apply to cache ───────────────────────────────────────────────────────────

def apply_to_cache(plan, season, sport, apply_flag):
    """Patch every championship cache file row matching a planned assignment.

    ``plan``: list of (year, classification, finalist_display, note_kind).
    Patches all cache files containing the row (dedup later keeps a patched
    copy). Sets finalist_school, appends the provenance note (idempotently —
    only if the note is not already present), stamps meta.manual_correction.
    """
    src_pdf = PDFS[season]
    by_key = {}
    for year, classification, display, note_kind in plan:
        nclass = p._normalize_classification(classification)
        by_key[(int(year), nclass)] = (display, note_kind)

    files = sorted(glob.glob(str(CACHE / "championship_*.json")))
    patched = 0
    files_changed = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        meta = d.get("meta", {})
        if meta.get("source_pdf", "") != src_pdf:
            continue
        rows = d.get("rows", [])
        changed = False
        for r in rows:
            if r.get("sport") != sport:
                continue
            try:
                year = int(r.get("year"))
            except (ValueError, TypeError):
                continue
            nclass = p._normalize_classification(r.get("classification"))
            entry = by_key.get((year, nclass))
            if entry is None:
                continue
            display, note_kind = entry
            r["finalist_school"] = display
            # finalist_coach left untouched (null) — the PDFs don't carry it.
            existing = (r.get("notes") or "").strip()
            if note_kind not in existing:
                r["notes"] = f"{existing}; {note_kind}" if existing else note_kind
            changed = True
            patched += 1
        if changed:
            files_changed += 1
            if apply_flag:
                d.setdefault("meta", {})
                d["meta"]["manual_correction"] = MANUAL_CORRECTION
                with open(f, "w", encoding="utf-8") as fh:
                    json.dump(d, fh, indent=2, ensure_ascii=False)
    verb = "patched" if apply_flag else "would patch"
    print(f"  {verb} {patched} row(s) across {files_changed} cache file(s)")
    return patched


def regenerate(season):
    """Re-derive championship_results.csv + record_book.json from patched cache
    (offline, no LLM), then rebuild the site."""
    pdf = PDFS[season]
    print(f"\n→ regenerating {season} from cache (offline)…")
    subprocess.run(["uv", "run", "parse_record_book.py", pdf, "--season", season,
                    "--offline"], check=True)
    print("→ rebuilding site…")
    subprocess.run(["uv", "run", "build_site.py"], check=True)


# ── Orchestration ─────────────────────────────────────────────────────────────

def process_sport(sport, *, apply_flag, year_filter):
    cfg = SPORT_CONFIG.get(sport)
    if cfg is None:
        season = _season_for(sport)
        candidate_field = FINALIST_FIELD.get(sport)
        if candidate_field is None:
            print(f"  ! unknown sport {sport!r}; skipping")
            return None
    else:
        season = cfg["season"]
        candidate_field = cfg["candidate_field"]

    alias_map, _canonical_display = build_site.load_aliases()
    normalize = build_site.make_normalizer(alias_map)

    champ_rows, school_rows = load_rows(season, sport)
    champion_idx = build_champion_index(champ_rows, normalize)

    entries = load_recaps(season, sport)
    if not entries:
        print(f"\n{sport} ({season}): no recap entries found at "
              f"{recaps_path(season, sport)}")
        print("  nothing to apply")
        return {"sport": sport, "verified": 0, "rejected": 0, "warnings": 0}

    plan, rejected, warnings = validate_recaps(
        entries, champ_rows, school_rows, normalize, candidate_field,
        champion_idx, year_filter=year_filter)

    print(f"\n{sport} ({season}): {len(entries)} recap entries")
    print(f"  verified: {len(plan)}   (will be written)")
    print(f"  rejected: {len(rejected)}")
    print(f"  warnings: {len(warnings)}")
    if rejected:
        print("  rejections:")
        for entry, reasons in rejected:
            print(f"    {entry.get('year')} {entry.get('classification')} "
                  f"{entry.get('finalist','')!r} — {'; '.join(reasons)}")
    if warnings:
        print("  warnings:")
        for year, cls, msg in warnings:
            print(f"    {year} {cls} — {msg}")

    if plan:
        apply_to_cache(plan, season, sport, apply_flag)
        if apply_flag:
            regenerate(season)
    else:
        print("  nothing to apply")

    return {"sport": sport, "verified": len(plan), "rejected": len(rejected),
            "warnings": len(warnings)}


def _season_for(sport):
    """Best-effort season for a sport not in SPORT_CONFIG (pilot extras)."""
    fall = {"Football", "Boys Soccer", "Girls Soccer", "Field Hockey", "Volleyball"}
    winter = {"Boys Basketball", "Girls Basketball"}
    if sport in fall:
        return "fall"
    if sport in winter:
        return "winter"
    return "spring"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", choices=["fall", "winter", "spring"])
    ap.add_argument("--sport", required=True,
                    help="e.g. Football, 'Boys Basketball'")
    ap.add_argument("--year", type=int, default=None,
                    help="process only this year (for auditing)")
    ap.add_argument("--apply", action="store_true",
                    help="patch cache + regenerate (default: dry-run)")
    args = ap.parse_args(argv)

    season = args.season or (SPORT_CONFIG.get(args.sport, {}) or {}).get("season") or _season_for(args.sport)
    if args.season and args.sport in SPORT_CONFIG and SPORT_CONFIG[args.sport]["season"] != args.season:
        print(f"  ! --season {args.season} overrides config season for {args.sport}")

    result = process_sport(args.sport, apply_flag=args.apply, year_filter=args.year)
    if result is None:
        return 1
    if result["rejected"]:
        return 2  # non-zero so CI / a wrapper notices rejected entries
    return 0


if __name__ == "__main__":
    main()