#!/usr/bin/env python3
"""Patch winter individual_results cache: recover relay rows the fresh LLM
extraction dropped, by merging them from the old (orphaned) cache files.

The winter rebuild's fresh LLM calls nondeterministically dropped entire relay
events on several pages (e.g. Boys Indoor Track page 56 returned only 55 m
Hurdles, dropping 4x200 + 4x400 relay). The older orphan cache files for the
same source_pages still contain those relay rows (same row schema), so we merge
the missing relay rows back into the canonical cache file for each page. The
pipeline's event-name + classification normalisers and dedup then run as usual
on rebuild-offline, so the patch is reproducible.

Dry-run by default (prints the plan). Apply with --apply.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

os.environ.setdefault("OLLAMA_MODEL", "glm-5.2:cloud")
os.environ.setdefault("MODEL_ID", "glm-5.2:cloud")
sys.path.insert(0, ".")
import parse_record_book as p
from scripts.find_orphan_cache import expected_keys

CACHE = p.CACHE_DIR

# (sport, canonical-event) pairs we need to recover from orphans.
MISSING = {
    ("Girls Indoor Track", "4x800 m Relay"),
    ("Girls Indoor Track", "4x400 m Relay"),
    ("Boys Indoor Track", "4x200 m Relay"),
    ("Boys Indoor Track", "4x800 m Relay"),
    ("Boys Indoor Track", "4x400 m Relay"),
    ("Girls Swimming & Diving", "200 Yard Medley Relay"),
    ("Girls Swimming & Diving", "200 Yard Freestyle Relay"),
}
AFFECTED_SPORTS = {s for s, _ in MISSING}


def norm_event(v):
    return p._normalize_event_name(v)


def row_key(r):
    return (r.get("sport"), r.get("year"), r.get("classification"),
            norm_event(r.get("event")), r.get("school"), r.get("mark"))


def load(f):
    d = json.load(open(f))
    return d, d.get("rows", d.get("data", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    canonical_names = {f for f in expected_keys() if f.startswith("individual_results")}
    files = sorted(glob.glob(os.path.join(CACHE, "individual_results_*.json")))

    # Group by (sport, source_pages) -> list of (name, rows, meta, is_canonical)
    groups = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        d, rows = load(f)
        meta = d.get("meta", {})
        sports = {r.get("sport") for r in rows}
        if not sports or not (sports & AFFECTED_SPORTS):
            continue
        if meta.get("source_pdf", "") != "pdfs/Winter record book.pdf":
            # only winter orphans are relevant; skip spring track/tennis orphans
            if not (sports & AFFECTED_SPORTS):
                continue
        sp = tuple(sorted(meta.get("source_pages", [])))
        for sport in (sports & AFFECTED_SPORTS):
            is_canon = name in canonical_names
            groups[(sport, sp)].append((name, rows, meta, is_canon))

    total_added = 0
    plan = []
    for (sport, sp), members in sorted(groups.items()):
        canons = [m for m in members if m[3]]
        orphans = [m for m in members if not m[3]]
        if not canons or not orphans:
            continue
        # rows for this sport only
        canon_rows = [r for n, rs, _, _ in canons for r in rs if r.get("sport") == sport]
        canon_keys = {row_key(r) for r in canon_rows}
        # union of orphan rows for this sport, restricted to missing relay events
        missing_events = {e for s, e in MISSING if s == sport}
        orphan_relay = [r for n, rs, _, _ in orphans for r in rs
                        if r.get("sport") == sport and norm_event(r.get("event")) in missing_events]
        to_add = [r for r in orphan_relay if row_key(r) not in canon_keys]
        # dedup to_add itself
        seen = set(); add_dedup = []
        for r in to_add:
            k = row_key(r)
            if k in seen: continue
            seen.add(k); add_dedup.append(r)
        if not add_dedup:
            continue
        # target = the canonical file(s) for this (sport, page); merge into the
        # one canonical file (there should be exactly one per page+sport)
        target = canons[0][0]
        plan.append((sport, sp, target, len(canon_rows), len(add_dedup),
                     sorted({norm_event(r.get("event")) for r in add_dedup})))
        total_added += len(add_dedup)
        if args.apply:
            td, trows = load(os.path.join(CACHE, target))
            trows.extend(add_dedup)
            td["rows"] = trows
            td.setdefault("meta", {})
            td["meta"]["manual_correction"] = (
                "Relay rows recovered from prior cache extraction "
                f"(orphan) for {sorted({norm_event(r.get('event')) for r in add_dedup})}; "
                "fresh LLM call had dropped these relay events on this page.")
            with open(os.path.join(CACHE, target), "w") as fh:
                json.dump(td, fh, indent=2, ensure_ascii=False)

    print(f"{'sport':22s} {'page':>6s} {'canon':>6s} {'add':>5s}  events  -> target")
    for sport, sp, target, nc, na, evs in plan:
        print(f"{sport:22s} {str(sp):>6s} {nc:6d} {na:5d}  {evs}  -> {target}")
    print(f"\nTOTAL rows to add: {total_added}  ({'APPLIED' if args.apply else 'DRY-RUN'})")


if __name__ == "__main__":
    main()