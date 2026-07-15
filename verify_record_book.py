#!/usr/bin/env python3
"""Verify a parsed record book for internal consistency — no LLM calls.

Usage:
    uv run verify_record_book.py [DATA_DIR]

    DATA_DIR defaults to data/fall. Point it at any data/<season> directory.

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

Writes DATA_DIR/verification_report.json and prints a summary. Exits non-zero
if any check produces an error (warnings alone do not fail the run).
"""

import collections
import json
import sys
from pathlib import Path
from typing import Optional

COVID_YEAR = 2020  # cancelled season — never counts as a coverage gap

# Fraction of school-record champion years that must also appear in the
# championship/golf tables before a sport is considered an error.
CROSS_PATH_THRESHOLD = 0.95

TABLE_KEYS = {
    "championship_results": ("sport", "year", "classification", "champion_school"),
    "school_records": ("sport", "school"),
    "individual_xc_champions": ("sport", "year", "classification"),
    "individual_results": ("sport", "event", "year", "classification"),
    "golf_results": ("year", "classification", "individual_gender"),
    "sportsmanship_awards": ("sport", "year", "classification"),
}


def load_record_book(data_dir: Path) -> dict:
    path = data_dir / "record_book.json"
    if not path.exists():
        raise SystemExit(f"Error: {path} not found. Run parse_record_book.py first.")
    return json.loads(path.read_text())


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _table(book: dict, name: str) -> list[dict]:
    rows = book.get(name, [])
    return rows if isinstance(rows, list) else []


# ── Check 1: cross-path champion years ────────────────────────────────────────


def llm_champion_years(book: dict) -> dict[str, set[int]]:
    """Champion years per sport from the LLM path (championship + golf tables)."""
    by_sport: dict[str, set[int]] = collections.defaultdict(set)
    for r in _table(book, "championship_results"):
        year = _as_int(r.get("year"))
        if year is not None and r.get("sport"):
            by_sport[r["sport"]].add(year)
    # Golf champions live in their own table, not championship_results.
    for r in _table(book, "golf_results"):
        year = _as_int(r.get("year"))
        if year is not None and r.get("team_champion_school"):
            by_sport["Golf"].add(year)
    return by_sport


def record_champion_years(book: dict) -> dict[str, set[int]]:
    """Champion years per sport from the regex path (school records)."""
    by_sport: dict[str, set[int]] = collections.defaultdict(set)
    for r in _table(book, "school_records"):
        sport = r.get("sport")
        if not sport:
            continue
        for y in r.get("champion_years", []) or []:
            year = _as_int(y)
            if year is not None:
                by_sport[sport].add(year)
    return by_sport


def check_cross_path(book: dict) -> dict:
    llm = llm_champion_years(book)
    records = record_champion_years(book)
    sports = sorted(set(llm) | set(records))

    per_sport = {}
    errors = 0
    for sport in sports:
        table_years = llm.get(sport, set())
        record_years = records.get(sport, set())
        missing_from_table = sorted(record_years - table_years)  # regex knows, table lost
        missing_from_records = sorted(table_years - record_years)  # table only
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


# ── Check 2: duplicate keys ───────────────────────────────────────────────────


def check_duplicate_keys(book: dict) -> dict:
    result = {}
    errors = 0
    for table, key_fields in TABLE_KEYS.items():
        rows = _table(book, table)
        counts = collections.Counter(
            tuple(str(r.get(k, "")) for k in key_fields) for r in rows
        )
        dupes = {" | ".join(k): n for k, n in counts.items() if n > 1}
        if dupes:
            errors += len(dupes)
        result[table] = {
            "row_count": len(rows),
            "duplicate_key_count": len(dupes),
            "duplicates": dict(sorted(dupes.items(), key=lambda kv: -kv[1])),
        }
    return {"errors": errors, "tables": result}


# ── Check 3: continuity ───────────────────────────────────────────────────────


def check_continuity(book: dict) -> dict:
    """Report champion-year gaps inside each sport's observed span (COVID exempt)."""
    llm = llm_champion_years(book)
    records = record_champion_years(book)
    sports = sorted(set(llm) | set(records))

    per_sport = {}
    total_gaps = 0
    for sport in sports:
        years = llm.get(sport, set()) | records.get(sport, set())
        if len(years) < 2:
            continue
        lo, hi = min(years), max(years)
        gaps = [y for y in range(lo, hi + 1) if y not in years and y != COVID_YEAR]
        if gaps:
            total_gaps += len(gaps)
            per_sport[sport] = {"span": [lo, hi], "missing_years": gaps}
    return {"warnings": total_gaps, "sports": per_sport}


# ── Check 4: referential school consistency ──────────────────────────────────


def check_referential_schools(book: dict) -> dict:
    """Championship-table champion schools that never appear in school records.

    School records are keyed on the canonical school name (see normalize_school
    in parse_record_book). A champion school absent here usually signals an
    unmapped name variant rather than a missing record.
    """
    records_by_sport: dict[str, set[str]] = collections.defaultdict(set)
    for r in _table(book, "school_records"):
        if r.get("sport") and r.get("school"):
            records_by_sport[r["sport"]].add(r["school"])

    per_sport = {}
    warnings = 0
    for r in _table(book, "championship_results"):
        sport, school = r.get("sport"), r.get("champion_school")
        if not sport or not school:
            continue
        # Co-champion cells ("A & B") are checked component-by-component.
        for part in (p.strip() for p in school.split(" & ")):
            if part and part not in records_by_sport.get(sport, set()):
                per_sport.setdefault(sport, set()).add(part)

    out = {}
    for sport, schools in per_sport.items():
        warnings += len(schools)
        out[sport] = sorted(schools)
    return {"warnings": warnings, "sports": out}


# ── Report assembly ───────────────────────────────────────────────────────────


def build_report(book: dict, data_dir: Path) -> dict:
    cross_path = check_cross_path(book)
    dup = check_duplicate_keys(book)
    continuity = check_continuity(book)
    referential = check_referential_schools(book)

    errors = cross_path["errors"] + dup["errors"]
    warnings = continuity["warnings"] + referential["warnings"]

    return {
        "data_dir": str(data_dir),
        "source": "record_book.json",
        "row_counts": {t: len(_table(book, t)) for t in TABLE_KEYS},
        "summary": {"errors": errors, "warnings": warnings, "passed": errors == 0},
        "checks": {
            "cross_path_champion_years": cross_path,
            "duplicate_keys": dup,
            "continuity": continuity,
            "referential_schools": referential,
        },
    }


def print_summary(report: dict) -> None:
    print(f"Verification of {report['data_dir']}")
    print(f"  rows: " + ", ".join(f"{t}={n}" for t, n in report["row_counts"].items()))
    print()

    cp = report["checks"]["cross_path_champion_years"]
    print("Cross-path champion-year coverage (championship/golf table vs school records):")
    for sport, info in cp["sports"].items():
        marker = "✗" if info["status"] == "error" else "✓"
        print(
            f"  {marker} {sport:24s} table={info['table_year_count']:3d}  "
            f"records={info['record_year_count']:3d}  coverage={info['coverage']:.0%}"
        )
        if info["missing_from_championship_table"]:
            yrs = info["missing_from_championship_table"]
            shown = yrs if len(yrs) <= 20 else yrs[:20] + ["…"]
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
        shown = yrs if len(yrs) <= 15 else yrs[:15] + ["…"]
        print(f"  · {sport} {info['span'][0]}–{info['span'][1]}: missing {shown}")

    ref = report["checks"]["referential_schools"]
    print(f"\nChampion schools not found in school records (warnings): {ref['warnings']}")
    for sport, schools in list(ref["sports"].items())[:10]:
        shown = schools if len(schools) <= 10 else schools[:10] + ["…"]
        print(f"  · {sport}: {shown}")

    s = report["summary"]
    print(f"\n{'PASSED ✓' if s['passed'] else 'FAILED ✗'}  "
          f"({s['errors']} error(s), {s['warnings']} warning(s))")


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / "fall"
    book = load_record_book(data_dir)
    report = build_report(book, data_dir)

    report_path = data_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nReport written to {report_path}")

    sys.exit(0 if report["summary"]["passed"] else 1)


if __name__ == "__main__":
    main()
