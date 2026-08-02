"""Semantic diff of working-tree data/ vs HEAD by natural key.

Per table: added (in working tree, not HEAD), removed (in HEAD, not working
tree), changed (same key, different payload). Printed and written to
diff_report.md so a human can review what a re-extraction changed before
committing. No staging directory, no auto-merge — git is the verified store;
extraction regenerates fully (the cache makes that ~free) and removals are
surfaced for review, never merged over silently.

Usage:
    uv run diff_outputs.py [DATA_DIR ...] [--out diff_report.md]

    DATA_DIR defaults to data/fall data/winter data/spring. Point it at any
    data/<season> directory holding record_book.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from parse_record_book import DEDUP_KEYS, PROVENANCE_FIELDS

# Tables to diff, in a stable display order.
TABLE_ORDER = [
    "championship_results",
    "school_records",
    "individual_xc_champions",
    "individual_results",
    "golf_results",
    "sportsmanship_awards",
]


def load_book(data_dir: Path) -> dict:
    path = data_dir / "record_book.json"
    if not path.exists():
        raise SystemExit(f"Error: {path} not found. Run parse_record_book.py first.")
    return json.loads(path.read_text())


def load_head_book(data_dir: Path, baseline_dir: Path | None = None
                   ) -> tuple[dict | None, str | None]:
    """Load the baseline record_book.json for ``data_dir``.

    By default the baseline is the file at HEAD (``git show HEAD:data/<season>``).
    If ``baseline_dir`` is given, the baseline is read from
    ``<baseline_dir>/<season>/record_book.json`` on disk instead — used by the
    ``rebuild-offline`` CI check to compare a freshly rebuilt ``build/`` tree
    against the committed ``data/`` tree without touching git.

    Returns (book, None) on success, (None, reason) if unavailable.
    """
    season = data_dir.name
    if baseline_dir is not None:
        path = baseline_dir / season / "record_book.json"
        if not path.exists():
            return None, f"{path} not found"
        try:
            return json.loads(path.read_text()), None
        except json.JSONDecodeError as exc:
            return None, f"{path} is not valid JSON: {exc}"
    git_path = f"data/{season}/record_book.json"
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{git_path}"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, "not tracked at HEAD (or not a git repo)"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"HEAD record_book.json is not valid JSON: {exc}"


def _key(row: dict, key_fields: tuple) -> tuple:
    return tuple(str(row.get(k, "")) for k in key_fields)


def _payload(row: dict) -> dict:
    """Row content minus provenance, which legitimately changes on re-extraction."""
    return {k: v for k, v in row.items() if k not in PROVENANCE_FIELDS}


def diff_table(cur_rows: list[dict], head_rows: list[dict],
               key_fields: tuple) -> dict:
    """Added/removed/changed for one table by natural key."""
    cur_by_key: dict[tuple, dict] = {}
    for r in cur_rows:
        cur_by_key.setdefault(_key(r, key_fields), r)
    head_by_key: dict[tuple, dict] = {}
    for r in head_rows:
        head_by_key.setdefault(_key(r, key_fields), r)

    added, removed, changed = [], [], []
    for k, r in cur_by_key.items():
        if k not in head_by_key:
            added.append(r)
        elif _payload(r) != _payload(head_by_key[k]):
            changed.append((head_by_key[k], r))
    for k, r in head_by_key.items():
        if k not in cur_by_key:
            removed.append(r)

    return {"added": added, "removed": removed, "changed": changed}


def diff_season(data_dir: Path, baseline_dir: Path | None = None) -> dict:
    """Per-table diff for one season dir. ``head`` is None when no baseline exists."""
    cur = load_book(data_dir)
    head, head_note = load_head_book(data_dir, baseline_dir=baseline_dir)
    tables = {}
    for table in TABLE_ORDER:
        key_fields = DEDUP_KEYS[table]
        cur_rows = cur.get(table, []) if isinstance(cur.get(table), list) else []
        if head is None:
            # No baseline -> every current row is "added" against an empty HEAD.
            tables[table] = diff_table(cur_rows, [], key_fields)
        else:
            head_rows = head.get(table, []) if isinstance(head.get(table), list) else []
            tables[table] = diff_table(cur_rows, head_rows, key_fields)
    return {"data_dir": str(data_dir), "head_note": head_note, "tables": tables,
            "row_counts": {t: len(cur.get(t, []) if isinstance(cur.get(t), list) else [])
                            for t in TABLE_ORDER}}


def _counts(diff: dict) -> tuple[int, int, int]:
    return (len(diff["added"]), len(diff["removed"]), len(diff["changed"]))


def print_summary(report: dict) -> None:
    for season in report["seasons"]:
        print(f"\n{season['data_dir']}" +
              (f"  (no HEAD baseline: {season['head_note']})"
               if season["head_note"] else "  vs HEAD"))
        print(f"  {'table':<26} {'rows':>5} {'add':>5} {'rm':>5} {'chg':>5}")
        print("  " + "-" * 52)
        for table in TABLE_ORDER:
            d = season["tables"][table]
            rows = season["row_counts"][table]
            a, r, c = _counts(d)
            print(f"  {table:<26} {rows:>5} {a:>5} {r:>5} {c:>5}")
    tot_add = sum(sum(_counts(s["tables"][t])[0] for t in TABLE_ORDER)
                  for s in report["seasons"])
    tot_rm = sum(sum(_counts(s["tables"][t])[1] for t in TABLE_ORDER)
                for s in report["seasons"])
    tot_chg = sum(sum(_counts(s["tables"][t])[2] for t in TABLE_ORDER)
                 for s in report["seasons"])
    print(f"\nTotal: +{tot_add} added, -{tot_rm} removed, ~{tot_chg} changed")


def _key_str(row: dict, key_fields: tuple) -> str:
    return " | ".join(str(row.get(k, "")) for k in key_fields)


def build_markdown(report: dict) -> str:
    lines = ["# Record book diff report", ""]
    lines.append("Semantic diff of working-tree `data/` vs `HEAD`, by natural key. "
                 "Added = new natural key vs HEAD; removed = HEAD key absent now; "
                 "changed = same key, different payload (provenance fields ignored).")
    lines.append("")
    for season in report["seasons"]:
        lines.append(f"## {season['data_dir']}")
        if season["head_note"]:
            lines.append(f"_No HEAD baseline: {season['head_note']} — every row "
                         f"counts as added._")
        lines.append("")
        lines.append(f"| table | added | removed | changed |")
        lines.append(f"|---|---:|---:|---:|")
        for table in TABLE_ORDER:
            a, r, c = _counts(season["tables"][table])
            lines.append(f"| {table} | {a} | {r} | {c} |")
        lines.append("")
        for table in TABLE_ORDER:
            d = season["tables"][table]
            if not (d["added"] or d["removed"] or d["changed"]):
                continue
            key_fields = DEDUP_KEYS[table]
            lines.append(f"### {season['data_dir']} — {table}")
            if d["added"]:
                lines.append(f"**Added ({len(d['added'])})**")
                for r in d["added"][:15]:
                    lines.append(f"- + {_key_str(r, key_fields)}")
                if len(d["added"]) > 15:
                    lines.append(f"- … and {len(d['added']) - 15} more")
            if d["removed"]:
                lines.append(f"**Removed ({len(d['removed'])})**")
                for r in d["removed"][:15]:
                    lines.append(f"- - {_key_str(r, key_fields)}")
                if len(d["removed"]) > 15:
                    lines.append(f"- … and {len(d['removed']) - 15} more")
            if d["changed"]:
                lines.append(f"**Changed ({len(d['changed'])})**")
                for old, new in d["changed"][:10]:
                    lines.append(f"- ~ {_key_str(new, key_fields)}")
                    # Show which non-provenance fields differ.
                    diffs = {k: (old.get(k), new.get(k)) for k in
                             set(old) | set(new)
                             if k not in PROVENANCE_FIELDS and old.get(k) != new.get(k)}
                    for k, (o, n) in list(diffs.items())[:5]:
                        lines.append(f"    - `{k}`: {o!r} → {n!r}")
                if len(d["changed"]) > 10:
                    lines.append(f"- … and {len(d['changed']) - 10} more")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dirs", nargs="*",
                        type=Path,
                        default=[Path("data") / s for s in ("fall", "winter", "spring")],
                        help="data/<season> dirs to diff (default: all three)")
    parser.add_argument("--out", default="diff_report.md",
                        help="markdown report path (default: diff_report.md)")
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="compare against <baseline-dir>/<season>/record_book.json "
                             "on disk instead of git HEAD (used by rebuild-offline)")
    parser.add_argument("--fail-on-diff", action="store_true",
                        help="exit 1 if any added/removed/changed rows (CI gate)")
    args = parser.parse_args(argv)

    seasons = []
    for data_dir in args.data_dirs:
        if not (data_dir / "record_book.json").exists():
            print(f"skip {data_dir}: no record_book.json", file=sys.stderr)
            continue
        seasons.append(diff_season(data_dir, baseline_dir=args.baseline_dir))
    if not seasons:
        print("No data directories with record_book.json to diff.", file=sys.stderr)
        return 1
    report = {"seasons": seasons}
    print_summary(report)
    out = Path(args.out)
    out.write_text(build_markdown(report))
    print(f"\nReport written to {out}")
    if args.fail_on_diff:
        total = sum(
            sum(_counts(s["tables"][t])[i] for t in TABLE_ORDER for i in range(3))
            for s in seasons
        )
        if total:
            print(f"\nFAIL: {total} row difference(s) vs baseline.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())