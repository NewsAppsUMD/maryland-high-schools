#!/usr/bin/env python3
"""Seed web/counties.csv from the MSDE report-card school list.

The Maryland State Department of Education publishes the current schools in each
jurisdiction at::

    https://reportcard.msde.maryland.gov/SchoolsList/Index?l=<code>

where ``<code>`` is a district code: counties 01–23 (alphabetical), Baltimore
City 30, the SEED School 32. Each page is server-rendered HTML with the schools
grouped under ``<h2>`` section headings (Elementary / Middle / High School /
Other); an entry looks like ``Fort Hill High (0405)``.

This script fetches the **High School** section for every jurisdiction (add
``--include-other`` to also take the "Other" section, which holds evening/
alternative high schools), normalizes each name through ``build_site``'s
normalizer, and matches it against the school universe built from the record
book. It writes a *proposal* CSV of ``school,county`` rows for schools that are
in the record book but not yet mapped in ``web/counties.csv`` — it never edits
``counties.csv`` directly. Review the proposal, then paste the good rows in.

Names MSDE lists that don't match any record-book school are printed as a
worklist: usually they want an ``aliases.csv`` row (preferred — it helps the
whole site) rather than a duplicate spelling in ``counties.csv``.

The live site returns 403 to non-browser clients; this script sends a browser
User-Agent, but if it is still blocked, save each district page from a real
browser as ``<code>.html`` in a directory and pass ``--from-html <dir>``.

Usage:
  uv run scripts/fetch_msde_schools.py                       # fetch + propose (stdout)
  uv run scripts/fetch_msde_schools.py --out proposals.csv   # write proposal CSV
  uv run scripts/fetch_msde_schools.py --include-other
  uv run scripts/fetch_msde_schools.py --from-html saved/    # parse saved HTML files
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, ".")

import build_site  # noqa: E402

# District code -> jurisdiction name (matching build_site.MD_JURISDICTIONS).
DISTRICTS: dict[str, str] = {
    "01": "Allegany", "02": "Anne Arundel", "03": "Baltimore County",
    "04": "Calvert", "05": "Caroline", "06": "Carroll", "07": "Cecil",
    "08": "Charles", "09": "Dorchester", "10": "Frederick", "11": "Garrett",
    "12": "Harford", "13": "Howard", "14": "Kent", "15": "Montgomery",
    "16": "Prince George's", "17": "Queen Anne's", "18": "St. Mary's",
    "19": "Somerset", "20": "Talbot", "21": "Washington", "22": "Wicomico",
    "23": "Worcester", "30": "Baltimore City",
    # 32 = The SEED School of Maryland (statewide); has no record-book history,
    # so it is intentionally omitted here.
}

BASE_URL = "https://reportcard.msde.maryland.gov/SchoolsList/Index?l={code}"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# The 12 known suffix-disambiguated schools carry a suffix that MSDE names lack.
# When an MSDE name in one of these districts fails to match, retry with the
# district's suffix appended so e.g. "Northeast" (l=02) matches "Northeast-AA".
DISTRICT_SUFFIX = {"Anne Arundel": "AA", "Prince George's": "PG",
                   "Baltimore City": "B", "Baltimore County": "BC"}


class SchoolListParser(HTMLParser):
    """Collect school names grouped by their ``<h2>`` section heading."""

    def __init__(self, wanted_sections: set[str]):
        super().__init__()
        self.wanted = wanted_sections
        self.section = ""          # current h2 text
        self._in_h2 = False
        self._h2_parts: list[str] = []
        self._in_link = False
        self._link_parts: list[str] = []
        self.names: list[str] = []  # names within wanted sections

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self._in_h2 = True
            self._h2_parts = []
        elif tag == "a":
            href = dict(attrs).get("href", "")
            if "ReportCardSchool" in href:
                self._in_link = True
                self._link_parts = []

    def handle_endtag(self, tag):
        if tag == "h2" and self._in_h2:
            self.section = "".join(self._h2_parts).strip()
            self._in_h2 = False
        elif tag == "a" and self._in_link:
            name = _clean_name("".join(self._link_parts))
            if name and self.section in self.wanted:
                self.names.append(name)
            self._in_link = False

    def handle_data(self, data):
        if self._in_h2:
            self._h2_parts.append(data)
        elif self._in_link:
            self._link_parts.append(data)


def _clean_name(text: str) -> str:
    """Collapse whitespace and strip the trailing ``(1234)`` school code."""
    import re
    name = " ".join(text.split())
    return re.sub(r"\s*\(\d+\)\s*$", "", name).strip()


def _fetch(code: str) -> str:
    req = urllib.request.Request(BASE_URL.format(code=code),
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed host)
        return resp.read().decode("utf-8", "replace")


def _load_html(code: str, from_html: Path | None) -> str | None:
    if from_html is not None:
        path = from_html / f"{code}.html"
        if not path.exists():
            print(f"  [{code}] {path} not found — skipping", file=sys.stderr)
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        return _fetch(code)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        print(f"  [{code}] fetch failed: {exc} — save the page as {code}.html "
              f"and use --from-html", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        help="write proposal CSV here (default: stdout)")
    parser.add_argument("--include-other", action="store_true",
                        help="also take the 'Other' section (evening/alt HS)")
    parser.add_argument("--from-html", type=Path,
                        help="parse saved <code>.html files instead of fetching")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between fetches (politeness; default 1.0)")
    args = parser.parse_args(argv)

    wanted = {"High School"}
    if args.include_other:
        wanted.add("Other")

    # Build the record-book school universe + existing county map.
    books = build_site.load_books()
    alias_map, canonical_display = build_site.load_aliases()
    normalize = build_site.make_normalizer(alias_map)
    registry, _ = build_site.build_school_index(books, normalize, canonical_display)
    existing_counties, _ = build_site.load_counties(normalize)

    proposals: dict[str, tuple[str, str]] = {}  # key -> (display, county)
    conflicts: list[str] = []
    unmatched: list[str] = []

    for code, county in DISTRICTS.items():
        html = _load_html(code, args.from_html)
        if html is None:
            continue
        p = SchoolListParser(wanted)
        p.feed(html)
        for msde_name in p.names:
            school = registry.lookup(msde_name)
            if school is None and county in DISTRICT_SUFFIX:
                school = registry.lookup(f"{msde_name}-{DISTRICT_SUFFIX[county]}")
            if school is None:
                unmatched.append(f"{msde_name} ({county})")
                continue
            key = school.normalized_key
            if key in existing_counties:
                if existing_counties[key] != county:
                    conflicts.append(
                        f"{school.display_name}: counties.csv={existing_counties[key]} "
                        f"MSDE={county}")
                continue  # already mapped — idempotent
            proposals[key] = (school.display_name, county)
        if args.from_html is None and args.delay:
            time.sleep(args.delay)

    # Emit proposals. Use csv.writer so names containing commas (e.g.
    # "Dr. Henry A. Wise, Jr.") are properly quoted.
    import csv
    import io
    rows = sorted(proposals.values(), key=lambda dc: (dc[1], dc[0].lower()))
    buf = io.StringIO()
    buf.write("# Proposed counties.csv rows from MSDE — review, then paste into\n")
    buf.write("# web/counties.csv. Generated by scripts/fetch_msde_schools.py.\n")
    writer = csv.writer(buf)
    writer.writerow(["school", "county"])
    for display, county in rows:
        writer.writerow([display, county])
    if args.out:
        args.out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"Wrote {len(rows)} proposed rows to {args.out}")
    else:
        print(buf.getvalue())

    print(f"\n{len(rows)} new proposals, {len(conflicts)} conflicts, "
          f"{len(unmatched)} unmatched MSDE names", file=sys.stderr)
    if conflicts:
        print("\nCONFLICTS (counties.csv disagrees with MSDE — resolve by hand):",
              file=sys.stderr)
        for c in conflicts:
            print(f"  {c}", file=sys.stderr)
    if unmatched:
        print("\nUNMATCHED MSDE names (add an aliases.csv row if it's a spelling "
              "variant of a record-book school):", file=sys.stderr)
        for u in sorted(unmatched):
            print(f"  {u}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
