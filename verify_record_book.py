"""Verify a parsed record book for internal consistency — no LLM calls.

Usage:
    uv run verify_record_book.py [DATA_DIR] [--allow-removals]

    DATA_DIR defaults to data/fall. Point it at any data/<season> directory.
    --allow-removals suppresses regression-guard errors (row-count shrink or
        HEAD natural key disappearing), for the case where a re-extraction
        legitimately drops rows. Still reported as a finding.

The parser extracts the same facts along two independent paths: championship
finals tables (via the LLM) and school records (via regex). This script plays
them against each other and reports every disagreement, so a silent extraction
loss like the Boys Soccer table (which once held only 15 of 55 champion years)
becomes a visible, actionable failure rather than quietly wrong data.

Checks:
  1. cross_path_champion_years — champion years in the championship/golf tables
     vs the champion years aggregated from school records, per sport.
  2. duplicate_keys — rows sharing a natural key within any table.
  3. continuity — gaps in champion-year coverage inside each sport's span.
  4. referential_schools — championship-table schools missing from school records.
  5. era_floors — must-include years per sport derived from the legacy extraction
     cache (e.g. Boys Cross Country must include 1946, Golf must include 2024).
     Guards against a re-extraction silently losing the earliest or latest year
     of a sport's known span.
  6. regression_guard — vs the record_book.json checked in at HEAD: error if any
     table's row count shrinks or any HEAD natural key disappears. Override with
     --allow-removals when a re-extraction legitimately drops rows.

Writes DATA_DIR/verification_report.json and prints a summary. Exits non-zero
if any check produces an error (warnings alone do not fail the run).
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

# 2020 state tournaments were cancelled (COVID); continuity ignores that gap.
COVID_YEAR = 2020
# Documented gaps in championship history — historical facts, not extraction
# losses. Continuity skips these so its warnings stay meaningful.
_WINTER_2021 = {2021}  # winter 2020-21 championships were cancelled (COVID)
KNOWN_GAPS: dict[str, set[int]] = {
    "Boys Basketball": _WINTER_2021,
    # Girls tournament ran 1947-49, paused, and resumed in 1973 (Title IX era);
    # the winter book shows 1949 -> 1973 with nothing between.
    "Girls Basketball": _WINTER_2021 | set(range(1950, 1973)),
    "Boys Indoor Track": _WINTER_2021,
    "Girls Indoor Track": _WINTER_2021,
    "Boys Swimming & Diving": _WINTER_2021,
    "Girls Swimming & Diving": _WINTER_2021,
    "Wrestling": _WINTER_2021,
    # No 1951 meet in the source book (1950 is followed by 1952).
    "Boys Cross Country": {1951},
}
# Notes value parse_record_book.py stamps on precursor tournaments run before
# MPSSAA sponsorship ("PRIOR TO MPSSAA SPONSORSHIP" / "PRE-MPSSAA" sections).
# Verify excludes these rows from continuity, cross-path, and referential checks.
PRE_MPSSAA = "Pre-MPSSAA"
# A sport's champion years must appear in both the championship/golf table and
# the school records; below this coverage ratio the cross-path check errors.
CROSS_PATH_THRESHOLD = 0.95

# Must-include years per sport, derived from the legacy extraction cache
# (.extraction_cache/, the last verified Haiku-era run). A re-extraction that
# loses the anchor year for a sport is almost certainly a regression, so it
# errors rather than slipping through as a warning.
#   Boys Cross Country: 436 rows spanning 1946-2024 -> 1946 is the floor.
#   Golf: 253 rows spanning 1971-2024 -> 2024 is the ceiling (the most recent
#   champion year the legacy run held; the golf regex regression dropped the
#   split-era pages, so re-extraction must recover through 2024).
ERA_FLOORS = {
    "Boys Cross Country": 1946,
    "Golf": 2024,
}

# Head-to-head contest sports whose `finalist_school` (the runner-up) should be
# populated for every MPSSAA-era final. The record-book PDFs don't print the
# runner-up on their honor-roll pages, so it is back-filled from school_records
# finalist/runner-up years + LLM class assignment (scripts/fill_finalists.py).
# Below this fill ratio the back-fill has clearly regressed. The floor starts
# conservative (0.80) and warn-only; once the post-pilot baseline is measured,
# raise it toward 0.95 and flip FINALIST_WARN_ONLY to False to enforce.
CONTEST_SPORT_FINALIST_FLOOR = {
    "Football": 0.80,
    "Boys Basketball": 0.80,
    "Boys Lacrosse": 0.80,
    "Girls Lacrosse": 0.80,
    # remaining contest sports are added here as they are back-filled:
    # Boys/Girls Soccer, Field Hockey, Volleyball, Girls Basketball,
    # Baseball, Softball.
}
FINALIST_WARN_ONLY = True

# Natural key per table. Same keys the parser dedups on, minus the cross-country
# `name` addition (co-champions legitimately share a year/classification, so
# the XC natural key for verification stays coarse).
TABLE_KEYS = {
    "championship_results": ("sport", "year", "classification", "champion_school"),
    "school_records": ("sport", "school"),
    "individual_xc_champions": ("sport", "year", "classification"),
    "individual_results": ("sport", "event", "year", "classification"),
    "golf_results": ("year", "classification", "individual_gender"),
    "sportsmanship_awards": ("sport", "year", "classification"),
    "stat_records": ("sport", "category", "record", "holder", "value", "year"),
}


def load_record_book(data_dir: Path) -> dict:
    """Load <data_dir>/record_book.json, or exit with guidance if absent."""
    path = data_dir / "record_book.json"
    if not path.exists():
        raise SystemExit(
            f"Error: {path} not found. Run parse_record_book.py first."
        )
    return json.loads(path.read_text())


def _as_int(value):
    """Best-effort int coercion; returns None on failure (None stays None)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _table(book: dict, name: str) -> list:
    """Return book[name] as a list, or [] if missing/not a list."""
    rows = book.get(name, [])
    return rows if isinstance(rows, list) else []


def _iter_years(value):
    """Yield ints from a champion_years-style field.

    The JSON keeps year lists as ``[1997, 1998]``; the CSV writer joins them
    with "; ". Normalise both so verification is robust to either leaking in.
    """
    if value is None:
        return
    if isinstance(value, str):
        for part in value.replace(",", ";").split(";"):
            part = part.strip()
            if part:
                year = _as_int(part)
                if year is not None:
                    yield year
        return
    try:
        for item in value:
            year = _as_int(item)
            if year is not None:
                yield year
    except TypeError:
        year = _as_int(value)
        if year is not None:
            yield year


def llm_champion_years(book: dict) -> dict:
    """Champion years per sport from the LLM extraction path.

    Rows tagged ``notes="Pre-MPSSAA"`` (precursor tournaments run before MPSSAA
    sponsorship) are excluded — they are not MPSSAA championships, so they
    should not anchor a sport's continuity span nor inflate cross-path coverage.
    """
    by_sport = collections.defaultdict(set)
    for r in _table(book, "championship_results"):
        if (r.get("notes") or "").startswith(PRE_MPSSAA):
            continue
        year = _as_int(r.get("year"))
        if year is not None and r.get("sport"):
            by_sport[r["sport"]].add(year)
    for r in _table(book, "golf_results"):
        year = _as_int(r.get("year"))
        if year is not None and r.get("team_champion_school"):
            by_sport["Golf"].add(year)
    return by_sport


def record_champion_years(book: dict) -> dict:
    """Champion years per sport from the regex path (school records)."""
    by_sport = collections.defaultdict(set)
    for r in _table(book, "school_records"):
        sport = r.get("sport")
        if not sport:
            continue
        for year in _iter_years(r.get("champion_years", [])):
            by_sport[sport].add(year)
    return by_sport


def check_cross_path(book: dict) -> dict:
    """Per sport, compare champion years from the table vs school records."""
    llm = llm_champion_years(book)
    records = record_champion_years(book)
    sports = sorted(set(llm) | set(records))
    per_sport = {}
    errors = 0
    for sport in sports:
        table_years = llm.get(sport, set())
        record_years = records.get(sport, set())
        missing_from_table = sorted(record_years - table_years)
        missing_from_records = sorted(table_years - record_years)
        coverage = (
            len(table_years & record_years) / len(record_years)
            if record_years
            else 1.0
        )
        is_error = bool(record_years) and coverage < CROSS_PATH_THRESHOLD
        if is_error:
            errors += 1
        per_sport[sport] = {
            "table_year_count": len(table_years),
            "record_year_count": len(record_years),
            "coverage": round(coverage, 3),
            "missing_from_championship_table": missing_from_table,
            "in_table_not_in_records": missing_from_records,
            "status": "error" if is_error else "ok",
        }
    return {"errors": errors, "sports": per_sport}


def _key_tuple(row: dict, key_fields: tuple) -> tuple:
    """Build a natural key, treating a missing field the same as an explicit null.

    record_book.json rows always carry every schema field (missing values
    serialize as JSON null), while an older committed copy may omit the key
    entirely. Without this, str(None) == "None" would diverge from the old
    row's "" and every such row would misreport as a regression.
    """
    return tuple(str(v) if (v := row.get(k)) is not None else "" for k in key_fields)


def check_duplicate_keys(book: dict) -> dict:
    """Flag rows sharing a natural key within any table."""
    result = {}
    errors = 0
    for table, key_fields in TABLE_KEYS.items():
        rows = _table(book, table)
        counts = collections.Counter(_key_tuple(r, key_fields) for r in rows)
        dupes = {" | ".join(k): n for k, n in counts.items() if n > 1}
        if dupes:
            errors += len(dupes)
        result[table] = {
            "row_count": len(rows),
            "duplicate_key_count": len(dupes),
            "duplicates": dict(sorted(dupes.items(), key=lambda kv: -kv[1])),
        }
    return {"errors": errors, "tables": result}


def check_continuity(book: dict) -> dict:
    """Warn on gaps in champion-year coverage inside each sport's span."""
    llm = llm_champion_years(book)
    records = record_champion_years(book)
    sports = sorted(set(llm) | set(records))
    per_sport = {}
    total_gaps = 0
    for sport in sports:
        years = llm.get(sport, set()) | records.get(sport, set())
        if len(years) < 2:
            continue
        lo = min(years)
        hi = max(years)
        exempt = KNOWN_GAPS.get(sport, set())
        gaps = [y for y in range(lo, hi + 1)
                if y not in years and y != COVID_YEAR and y not in exempt]
        if gaps:
            total_gaps += len(gaps)
            per_sport[sport] = {"span": [lo, hi], "missing_years": gaps}
    return {"warnings": total_gaps, "sports": per_sport}


def check_referential_schools(book: dict) -> dict:
    """Warn on championship-table champion schools missing from school records.

    Pre-MPSSAA precursor rows are excluded — their 1940s public-school winners
    are not expected to appear in the MPSSAA-era school_records table.

    Names on both sides go through the site's alias-aware normalizer
    (build_site + web/aliases.csv), so a championship-table short form like
    "Churchill" matches "WINSTON CHURCHILL" in school records instead of
    drowning the report in spelling-variant noise. Whole names are tried
    before tie-splitting ("Cambridge/South Dorchester" is one school).
    """
    from build_site import load_aliases, make_normalizer, split_cochampions

    aliases, _ = load_aliases()
    norm = make_normalizer(aliases)
    records_by_sport = collections.defaultdict(set)
    for r in _table(book, "school_records"):
        if r.get("sport") and r.get("school"):
            records_by_sport[r["sport"]].add(norm(r["school"]))
    per_sport = {}
    warnings = 0
    for r in _table(book, "championship_results"):
        if (r.get("notes") or "").startswith(PRE_MPSSAA):
            continue
        sport, school = r.get("sport"), r.get("champion_school")
        if not sport or not school:
            continue
        known = records_by_sport.get(sport, set())
        if norm(school) in known:
            continue
        for part in split_cochampions(school):
            if part and norm(part) not in known:
                per_sport.setdefault(sport, set()).add(part)
    out = {}
    for sport, schools in per_sport.items():
        warnings += len(schools)
        out[sport] = sorted(schools)
    return {"warnings": warnings, "sports": out}


def check_era_floors(book: dict) -> dict:
    """Error if a present sport's must-include year (from the legacy cache) is absent.

    A floor year missing from the championship/golf table means the extraction
    lost an anchor row — almost always a regression, so this errors rather
    than warning. Sports absent from the book are skipped: the ERA_FLOORS
    anchors are fall-sport specific (Boys Cross Country, Golf), and a season
    that does not contest that sport should not trigger its floor. A sport
    disappearing entirely is caught by the regression guard instead.
    """
    llm = llm_champion_years(book)
    present_sports = {r.get("sport") for r in _table(book, "championship_results")}
    if _table(book, "golf_results"):
        present_sports.add("Golf")
    missing = []
    for sport, floor in sorted(ERA_FLOORS.items()):
        if sport not in present_sports:
            continue
        years = llm.get(sport, set())
        if floor not in years:
            missing.append(
                {
                    "sport": sport,
                    "floor_year": floor,
                    "table_years": sorted(years),
                }
            )
    return {"errors": len(missing), "missing": missing}


def check_finalist_coverage(book: dict) -> dict:
    """Report finalist_school fill rate for head-to-head contest sports.

    The PDF honor-roll pages print only champion + score, so finalist_school is
    back-filled (scripts/fill_finalists.py). Eligible rows: championship_results
    rows for a floored contest sport with a populated score (the contest-sport
    proxy — score is ~100% filled), excluding Pre-MPSSAA precursor rows and
    KNOWN_GAPS years. Below the per-sport floor the check would error, downgraded
    to a warning while FINALIST_WARN_ONLY is set so the pilot run stays green.
    """
    per_sport = {}
    warnings = 0
    errors = 0
    for sport, floor in sorted(CONTEST_SPORT_FINALIST_FLOOR.items()):
        gaps = KNOWN_GAPS.get(sport, set())
        eligible = filled = 0
        missing = []
        for r in _table(book, "championship_results"):
            if r.get("sport") != sport:
                continue
            if (r.get("notes") or "").startswith(PRE_MPSSAA):
                continue
            try:
                year = int(r.get("year"))
            except (TypeError, ValueError):
                continue
            if year in gaps:
                continue
            # Score is ~100% filled for contest sports; rows without one are
            # pre-MPSSAA/precursor oddities, not back-fill targets.
            if not (r.get("score") or "").strip():
                continue
            eligible += 1
            if (r.get("finalist_school") or "").strip():
                filled += 1
            else:
                missing.append((year, r.get("classification")))
        if eligible == 0:
            continue
        coverage = filled / eligible
        status = "ok" if coverage >= floor else "below_floor"
        if status == "below_floor":
            if FINALIST_WARN_ONLY:
                warnings += 1
            else:
                errors += 1
        per_sport[sport] = {
            "eligible": eligible,
            "filled": filled,
            "coverage": coverage,
            "floor": floor,
            "status": status,
            "missing": sorted(missing),
        }
    return {"warnings": warnings, "errors": errors, "sports": per_sport}


def _load_head_book(data_dir: Path) -> tuple[dict | None, str | None]:
    """Load the record_book.json checked in at HEAD, for the regression guard.

    Returns (book, None) on success, (None, reason) if it could not be loaded
    (not a git repo, file not tracked, etc.) so the guard can skip cleanly.
    """
    season = data_dir.name
    git_path = f"data/{season}/record_book.json"
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{git_path}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, "HEAD record_book.json unavailable (not tracked or not a git repo)"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"HEAD record_book.json is not valid JSON: {exc}"


def check_regression(book: dict, head_book: dict | None, head_note: str | None,
                     allow_removals: bool) -> dict:
    """Error if row counts shrink or HEAD natural keys disappear.

    With --allow-removals the errors are suppressed (still reported as
    findings) for the case where a re-extraction legitimately drops rows.
    Skips cleanly when the HEAD book is unavailable.
    """
    if head_book is None:
        return {
            "errors": 0,
            "skipped": True,
            "skip_reason": head_note,
            "row_count_shrinks": {},
            "missing_head_keys": {},
            "allow_removals": allow_removals,
        }
    row_count_shrinks = {}
    missing_head_keys = {}
    errors = 0
    for table, key_fields in TABLE_KEYS.items():
        cur_rows = _table(book, table)
        head_rows = _table(head_book, table)
        cur_count = len(cur_rows)
        head_count = len(head_rows)
        if cur_count < head_count:
            row_count_shrinks[table] = {
                "head": head_count,
                "current": cur_count,
                "delta": cur_count - head_count,
            }
            if not allow_removals:
                errors += 1
        if not allow_removals:
            head_keys = {_key_tuple(r, key_fields) for r in head_rows}
            cur_keys = {_key_tuple(r, key_fields) for r in cur_rows}
            disappeared = sorted(head_keys - cur_keys)
            if disappeared:
                missing_head_keys[table] = disappeared
                errors += len(disappeared)
    return {
        "errors": errors,
        "skipped": False,
        "row_count_shrinks": row_count_shrinks,
        "missing_head_keys": missing_head_keys,
        "allow_removals": allow_removals,
    }


def build_report(book: dict, data_dir: Path, *, allow_removals: bool) -> dict:
    cross_path = check_cross_path(book)
    dup = check_duplicate_keys(book)
    continuity = check_continuity(book)
    referential = check_referential_schools(book)
    era = check_era_floors(book)
    finalist_cov = check_finalist_coverage(book)
    head_book, head_note = _load_head_book(data_dir)
    regression = check_regression(book, head_book, head_note, allow_removals)

    errors = (
        cross_path["errors"]
        + dup["errors"]
        + era["errors"]
        + regression["errors"]
        + finalist_cov["errors"]
    )
    warnings = (
        continuity["warnings"]
        + referential["warnings"]
        + finalist_cov["warnings"]
    )

    row_counts = {t: len(_table(book, t)) for t in TABLE_KEYS}
    return {
        "data_dir": str(data_dir),
        "source": "record_book.json",
        "row_counts": row_counts,
        "summary": {
            "errors": errors,
            "warnings": warnings,
            "passed": errors == 0,
        },
        "checks": {
            "cross_path_champion_years": cross_path,
            "duplicate_keys": dup,
            "continuity": continuity,
            "referential_schools": referential,
            "era_floors": era,
            "finalist_coverage": finalist_cov,
            "regression_guard": regression,
        },
    }


def _clip(seq, limit):
    """Return seq, or seq[:limit] + ['…'] when it is longer than limit."""
    seq = list(seq)
    if len(seq) <= limit:
        return seq
    return seq[:limit] + ["…"]


def print_summary(report: dict) -> None:
    print(f"Verification of {report['data_dir']}")
    print("  rows: " + ", ".join(f"{k}={v}" for k, v in report["row_counts"].items()))
    print()

    cp = report["checks"]["cross_path_champion_years"]
    print("Cross-path champion-year coverage (championship/golf table vs school records):")
    for sport, info in cp["sports"].items():
        marker = "✗" if info["status"] == "error" else "✓"
        print(
            f"  {marker} {sport:24s} table={info['table_year_count']:3d}"
            f"  records={info['record_year_count']:3d}"
            f"  coverage={info['coverage']:.0%}"
        )
        if info["missing_from_championship_table"]:
            yrs = info["missing_from_championship_table"]
            shown = _clip(yrs, 20)
            print(f"      missing from table: {shown}")
    print()

    dup = report["checks"]["duplicate_keys"]
    if dup["errors"]:
        print("Duplicate keys:")
        for table, info in dup["tables"].items():
            if info["duplicate_key_count"]:
                print(f"  ✗ {table}: {info['duplicate_key_count']} duplicated key(s)")
                for key, n in list(info["duplicates"].items())[:8]:
                    print(f"      {key}  ×{n}")
    else:
        print("Duplicate keys: none ✓")
    print()

    cont = report["checks"]["continuity"]
    print(f"Continuity gaps (warnings): {cont['warnings']}")
    for sport, info in list(cont["sports"].items())[:10]:
        yrs = info["missing_years"]
        shown = _clip(yrs, 15)
        print(f"  · {sport} {info['span'][0]}–{info['span'][1]}: missing {shown}")
    print()

    ref = report["checks"]["referential_schools"]
    print(f"\nChampion schools not found in school records (warnings): {ref['warnings']}")
    for sport, schools in list(ref["sports"].items())[:10]:
        shown = _clip(schools, 10)
        print(f"  · {sport}: {shown}")

    era = report["checks"]["era_floors"]
    if era["errors"]:
        print("\nEra floors (must-include years):")
        for m in era["missing"]:
            print(f"  ✗ {m['sport']} missing anchor year {m['floor_year']}")
    else:
        print("\nEra floors: all present ✓")

    fc = report["checks"]["finalist_coverage"]
    if fc["sports"]:
        label = "warnings" if FINALIST_WARN_ONLY else "errors"
        print(f"\nFinalist coverage (contest sports; {label}): {fc['warnings'] if FINALIST_WARN_ONLY else fc['errors']}")
        for sport, info in fc["sports"].items():
            marker = "✓" if info["status"] == "ok" else "✗"
            print(
                f"  {marker} {sport:18s} filled={info['filled']:3d}/{info['eligible']:3d}"
                f"  coverage={info['coverage']:.0%}  floor={info['floor']:.0%}"
            )
            shown = _clip([f"{y} {c}" for y, c in info["missing"]], 10)
            if info["missing"]:
                print(f"      missing: {shown}")

    reg = report["checks"]["regression_guard"]
    if reg.get("skipped"):
        print(f"\nRegression guard: skipped — {reg.get('skip_reason')}")
    elif reg["errors"]:
        label = " (suppressed by --allow-removals)" if reg["allow_removals"] else ""
        print(f"\nRegression guard: {reg['errors']} finding(s){label}")
        for table, info in reg["row_count_shrinks"].items():
            print(f"  ✗ {table} shrank {info['head']} -> {info['current']} (Δ{info['delta']})")
        for table, keys in reg["missing_head_keys"].items():
            shown = _clip(keys, 10)
            print(f"  ✗ {table} lost HEAD key(s): {shown}")
    else:
        print("\nRegression guard: no row-count or key loss vs HEAD ✓")

    s = report["summary"]
    print(f"\n{'PASSED ✓' if s['passed'] else 'FAILED ✗'}  ({s['errors']} error(s), {s['warnings']} warning(s))")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data") / "fall",
                        help="data/<season> directory holding record_book.json (default: data/fall)")
    parser.add_argument("--allow-removals", action="store_true",
                        help="suppress regression-guard errors when a re-extraction "
                             "legitimately drops rows or keys vs HEAD")
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    book = load_record_book(data_dir)
    report = build_report(book, data_dir, allow_removals=args.allow_removals)

    report_path = data_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nReport written to {report_path}")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())