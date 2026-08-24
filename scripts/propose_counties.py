#!/usr/bin/env python3
"""Propose counties for schools not covered by scripts/fetch_msde_schools.py.

Many schools in the record book are closed, merged, or segregated-era and do
not appear on MSDE's current list, so the MSDE scraper can't place them. This
script asks the same LLM the extraction pipeline uses (see
``parse_record_book.llm_extract``) to propose a Maryland jurisdiction for each
still-unmapped school, using the school's name, its raw name variants, and the
sports/years it won titles in as location clues.

Nothing here is authoritative — output is a *proposal* CSV for human review,
never written to ``web/counties.csv`` directly. Low-confidence proposals are
emitted commented-out so a careless paste can't introduce a guess. Each row
carries the model's confidence and rationale as a comment.

Usage:
  uv run scripts/propose_counties.py --out proposals.csv
  uv run scripts/propose_counties.py --limit 20        # sample while iterating
  uv run scripts/propose_counties.py --batch 15
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, ".")

import build_site  # noqa: E402
import parse_record_book as p  # noqa: E402

try:  # pydantic ships with the `llm` dependency
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    print("pydantic is required (installed with the `llm` dependency).",
          file=sys.stderr)
    raise


class CountyProposal(BaseModel):
    school: str
    county: str
    confidence: str  # "high" | "medium" | "low"
    rationale: str


class CountyProposalBatch(BaseModel):
    proposals: list[CountyProposal]


def _clues(school) -> dict:
    """Compact location clues for one school to feed the model."""
    sports_years = sorted({
        (t.get("sport"), t.get("year")) for t in school.titles
        if t.get("sport")
    })
    return {
        "name": school.display_name,
        "variants": sorted(school.raw_variants)[:8],
        "titles": [f"{sp} {yr}" for sp, yr in sports_years][:12],
        "closed": school.closed,
    }


def _prompt(batch: list[dict]) -> str:
    jurisdictions = ", ".join(sorted(build_site.MD_JURISDICTIONS))
    import json
    return (
        "You are placing historical Maryland high schools into their county / "
        "jurisdiction. Each school below is closed, merged, or otherwise not on "
        "the current state school list. Use the name and any clues to identify "
        "the Maryland jurisdiction.\n\n"
        f"Valid jurisdictions (use EXACTLY one of these strings): {jurisdictions}.\n\n"
        "Set confidence to 'high' only when you are certain, 'medium' when the "
        "name strongly implies a place, and 'low' when guessing. Keep rationale "
        "to one short sentence. Return one proposal per input school.\n\n"
        f"Schools:\n{json.dumps(batch, indent=2)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        help="write proposal CSV here (default: stdout)")
    parser.add_argument("--batch", type=int, default=15,
                        help="schools per LLM call (default: 15)")
    parser.add_argument("--limit", type=int,
                        help="only process the first N unmapped schools")
    parser.add_argument("--model",
                        help=f"llm model id to use (default: {p.MODEL_ID})")
    args = parser.parse_args(argv)

    if args.model:
        p.MODEL_ID = args.model

    books = build_site.load_books()
    alias_map, canonical_display = build_site.load_aliases()
    normalize = build_site.make_normalizer(alias_map)
    registry, _ = build_site.build_school_index(books, normalize, canonical_display)
    existing_counties, _ = build_site.load_counties(normalize)

    unmapped = [s for s in registry.by_key.values()
                if s.normalized_key not in existing_counties]
    unmapped.sort(key=lambda s: s.display_name.lower())
    if args.limit:
        unmapped = unmapped[:args.limit]
    print(f"{len(unmapped)} unmapped schools to propose", file=sys.stderr)

    results: list[CountyProposal] = []
    by_name = {s.display_name: s for s in unmapped}
    for i in range(0, len(unmapped), args.batch):
        chunk = unmapped[i:i + args.batch]
        clues = [_clues(s) for s in chunk]
        print(f"  batch {i // args.batch + 1}: {len(chunk)} schools…",
              file=sys.stderr)
        data = p.llm_extract(_prompt(clues), CountyProposalBatch)
        for item in data.get("proposals", []):
            try:
                prop = CountyProposal(**item)
            except Exception as exc:  # noqa: BLE001 - skip malformed rows
                print(f"    skipped malformed proposal {item!r}: {exc}",
                      file=sys.stderr)
                continue
            results.append(prop)

    # Emit, sorted by confidence then name; low-confidence rows commented out.
    order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda r: (order.get(r.confidence.lower(), 3),
                                r.school.lower()))
    buf = io.StringIO()
    buf.write("# LLM-PROPOSED counties for unmapped (closed/historical) schools.\n")
    buf.write("# REVIEW BEFORE COMMITTING — these are guesses, not authoritative.\n")
    buf.write("# Low-confidence rows are commented out. Generated by\n")
    buf.write("# scripts/propose_counties.py.\n")
    writer = csv.writer(buf)
    writer.writerow(["school", "county"])
    for r in results:
        buf.write(f"# confidence={r.confidence}: {r.rationale}\n")
        row = io.StringIO()
        csv.writer(row).writerow([r.school, r.county])
        line = row.getvalue().rstrip("\r\n")
        valid = r.county in build_site.MD_JURISDICTIONS
        if r.confidence.lower() == "low" or not valid:
            if not valid:
                buf.write(f"# (invalid county {r.county!r}) ")
            buf.write(f"# {line}\n")
        else:
            buf.write(f"{line}\n")

    if args.out:
        args.out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Wrote {len(results)} proposals to {args.out}", file=sys.stderr)
    else:
        print(buf.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
