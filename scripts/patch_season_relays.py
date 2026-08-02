#!/usr/bin/env python3
"""Recover individual_results rows a fresh-LLM rebuild dropped, by merging them
from older (orphaned) cache files into the canonical cache files.

The GLM extraction nondeterministically drops relay (and sometimes other) events
on dense individual_results pages — e.g. a page with 4x200 Relay + 55 m Hurdles
comes back with only the hurdles. The older orphan cache files for the same
source_pages still contain the dropped rows (identical row schema), so we merge
the missing rows back into the canonical cache file for each page. The pipeline's
event-name + classification normalisers and dedup then run as usual on
rebuild-offline, so the patch is reproducible.

Detection: compares the rebuilt data/<season> vs the pre-rebuild baseline
snapshot, finds (sport, event, year, classification) keys the baseline had that
the rebuild lacks (ignoring cosmetic event-name renames like 55m Dash -> 55 m),
and recovers those rows from orphan cache. Dry-run by default; --apply writes.

Usage:
  python scripts/patch_season_relays.py --season spring [--apply]
  python scripts/patch_season_relays.py --season winter --baseline /tmp/.../winter.json
"""
import argparse, glob, json, os, sys
from collections import defaultdict

os.environ.setdefault("OLLAMA_MODEL", "glm-5.2:cloud")
os.environ.setdefault("MODEL_ID", "glm-5.2:cloud")
sys.path.insert(0, ".")
import parse_record_book as p
import verify_record_book as vrb
from scripts.find_orphan_cache import expected_keys

CACHE = p.CACHE_DIR
PDFS = {"fall": "pdfs/FallRecordBook2024.pdf",
        "winter": "pdfs/Winter record book.pdf",
        "spring": "pdfs/Spring record book 2025.pdf"}
BASELINE_DIR = "/tmp/winter_spring_baseline"


def ne(v):
    return p._normalize_event_name(v)


def nc(v):
    return p._normalize_classification(v)


def coarse_key(r):
    """Coarse natural key (no school/mark) — a row is a REAL drop only if the
    rebuild has no row at all for this (sport, year, class, event). Comparing
    on school+mark would flag spelling variants (Fairmont Heights vs Fairmont
    Hts.) as missing and over-merge. Classification is normalized so the new
    Comb->Combined rename does not read as a drop."""
    return (r.get("sport"), r.get("year"), nc(r.get("classification")),
            ne(r.get("event")))


def row_key(r):
    return (r.get("sport"), r.get("year"), r.get("classification"),
            ne(r.get("event")), r.get("school"), r.get("mark"))


def load(f):
    d = json.load(open(f))
    return d, d.get("rows", d.get("data", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--baseline", default=None,
                    help="baseline json (default /tmp/winter_spring_baseline/<season>.json)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src_pdf = PDFS[args.season]
    baseline_path = args.baseline or os.path.join(BASELINE_DIR, f"{args.season}.json")
    base = json.load(open(baseline_path))
    new = json.load(open(f"data/{args.season}/record_book.json"))

    # coarse keys present in the rebuild
    new_coarse = {coarse_key(r) for r in new.get("individual_results", [])}
    # missing coarse keys = baseline had these, rebuild has none at all
    base_rows_by_coarse = defaultdict(list)
    for r in base.get("individual_results", []):
        base_rows_by_coarse[coarse_key(r)].append(r)
    missing_coarse = sorted(ck for ck in base_rows_by_coarse if ck not in new_coarse)

    by_se = defaultdict(int)
    for ck in missing_coarse:
        by_se[(ck[0], ck[3])] += 1
    print(f"REAL missing (coarse-absent) individual_results keys vs baseline: {len(missing_coarse)}")
    for (s, e), c in sorted(by_se.items()):
        print(f"  {c:5d}  {s} | {e}")

    if not missing_coarse:
        print("nothing to recover"); return

    canonical_names = {f for f in expected_keys() if f.startswith("individual_results")}
    # group cache files by (sport, source_pages)
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CACHE, "individual_results_*.json"))):
        name = os.path.basename(f)
        d, rows = load(f)
        meta = d.get("meta", {})
        if meta.get("source_pdf", "") != src_pdf:
            continue
        sports = {r.get("sport") for r in rows}
        sp = tuple(sorted(meta.get("source_pages", [])))
        is_canon = name in canonical_names
        for sport in sports:
            groups[(sport, sp)].append((name, rows, is_canon))

    # orphan row index by coarse key -> list of (row, source_pages)
    orphan_by_coarse = defaultdict(list)
    for (sport, sp), members in groups.items():
        for name, rows, is_canon in members:
            if is_canon:
                continue
            for r in rows:
                if r.get("sport") != sport:
                    continue
                orphan_by_coarse[coarse_key(r)].append((r, sp))

    total = 0
    plan = defaultdict(list)  # canonical_name -> list of rows
    for ck in missing_coarse:
        for row, sp in orphan_by_coarse.get(ck, []):
            canons = [m for m in groups.get((ck[0], sp), []) if m[2]]
            if not canons:
                continue
            plan[canons[0][0]].append(row)
    for target, rows_ in plan.items():
        seen = set(); add = []
        for r in rows_:
            k = row_key(r)
            if k in seen: continue
            seen.add(k); add.append(r)
        plan[target] = add
        total += len(add)

    print(f"\nrecovery plan: {total} rows into {len(plan)} canonical files")
    for target, rows_ in sorted(plan.items()):
        evs = sorted({ne(r.get("event")) for r in rows_})
        print(f"  +{len(rows_):4d}  {target}  events={evs}")

    if args.apply:
        for target, rows_ in plan.items():
            tp = os.path.join(CACHE, target)
            td, trows = load(tp)
            trows.extend(rows_)
            td["rows"] = trows
            td.setdefault("meta", {})
            td["meta"]["manual_correction"] = (
                "Rows recovered from prior cache extraction (orphan); fresh LLM "
                f"call had dropped these events on this page: "
                f"{sorted({ne(r.get('event')) for r in rows_})}")
            with open(tp, "w") as fh:
                json.dump(td, fh, indent=2, ensure_ascii=False)
        print(f"APPLIED {total} rows")
    else:
        print("(DRY-RUN)")


if __name__ == "__main__":
    main()