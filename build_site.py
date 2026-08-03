"""Static site generator for the Maryland HS sports record-book data.

Builds a GitHub-Pages static site from ``data/{fall,winter,spring}/record_book.json``:
one clip-file page per school, an A-Z/search index, story-peg pages, and
embeddable iframe widgets. Output is written to ``site/`` (gitignored).

This file holds the data layer + pure logic. Templates live in ``web/templates``
and static assets in ``web/static``; rendering is wired in later batches.

Run:  uv run build_site.py            # build site/
       uv run build_site.py --report   # print build report only (no files)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
ALIASES_CSV = WEB_DIR / "aliases.csv"
OUT_DIR = ROOT / "site"
SEASONS = ("fall", "winter", "spring")

# Tables we pull from each season book.
TABLES = (
    "championship_results",
    "school_records",
    "individual_xc_champions",
    "individual_results",
    "sportsmanship_awards",
    "golf_results",
    "stat_records",
)

# Provenance fields carried on every row; surfaced as citations.
PROVENANCE_FIELDS = ("source_pdf", "source_pages", "extracted_at", "extraction_model")

# School-name suffixes/words stripped during normalization. County/class
# disambiguators like "-AA" / "-B" are NOT stripped (they distinguish schools).
_SUFFIX_WORDS = ("high", "school", "hs", "sr", "jr")

# Generic standalone-token abbreviation expansions applied during normalization
# (token-level, so "Mt." -> "mt" -> "mount"; never touches substrings).
_ABBREV = {
    "mt": "mount",
    "hts": "heights",
    "col": "colonel",
    "tech": "technical",
    "voc": "vocational",
    "vo": "vocational",
    "inst": "institute",
    "acad": "academy",
}

# Regex: any run of non-alphanumeric characters becomes a single space.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# A "word" of a school name, used to guess a display name from an ALLCAPS form.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


# ── School-name normalization ────────────────────────────────────────────────
def _base_normalize(name: str) -> str:
    """Aggressive casefold + punctuation collapse, but keep digits.

    "E. Roosevelt" -> "e roosevelt"; "Central—PG" -> "central pg";
    "NORTHEAST-AA" -> "northeast aa"; "Baltimore Poly. Inst." -> "baltimore poly inst".
    """
    if not name:
        return ""
    # Normalize unicode (decompose accents), then replace any remaining
    # non-ASCII char (em/en dashes, curly quotes, combining marks) with a SPACE
    # so it acts as a separator rather than being silently dropped — otherwise
    # "Central—PG" would become "centralpg" and fail to match "Central-PG".
    n = unicodedata.normalize("NFKD", name)
    n = "".join(ch if ord(ch) < 128 else " " for ch in n)
    n = n.lower()
    n = _NON_ALNUM.sub(" ", n).strip()
    # Drop TRAILING suffix words only: "Aberdeen High" / "Aberdeen HS" ->
    # "aberdeen", but "High Point" keeps "high" (it's part of the name, not a
    # trailing descriptor). Strip a trailing "high school" pair, then a single
    # trailing suffix word, repeatedly.
    tokens = n.split()
    while len(tokens) >= 2 and tokens[-2] in _SUFFIX_WORDS and tokens[-1] in _SUFFIX_WORDS:
        tokens.pop()
        tokens.pop()
    while tokens and tokens[-1] in _SUFFIX_WORDS:
        tokens.pop()
    # Expand generic standalone abbreviations ("mt" -> "mount") so "Mt. Hebron"
    # matches "MOUNT HEBRON". Token-level only.
    tokens = [_ABBREV.get(t, t) for t in tokens]
    return " ".join(tokens)


def load_aliases(path: Path = ALIASES_CSV) -> dict[str, str]:
    """Load ``alias,canonical`` rows into a normalized alias_key -> canonical_key map.

    Comment lines (starting with ``#``) and blank rows are skipped. Both sides
    are base-normalized so lookups are case/punctuation-insensitive.
    """
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            alias_key = _base_normalize(row[0])
            canon_key = _base_normalize(row[1])
            if alias_key and canon_key:
                mapping[alias_key] = canon_key
    return mapping


def make_normalizer(aliases: dict[str, str]):
    """Return a normalize_school(name) that applies the alias map after base normalization."""

    def normalize_school(name: str) -> str:
        key = _base_normalize(name)
        if not key:
            return ""
        # Chase alias chains (alias -> canonical -> canonical) in case of
        # indirect mappings; bounded because the graph is small.
        seen: set[str] = set()
        while key in aliases and key not in seen:
            seen.add(key)
            key = aliases[key]
        return key

    return normalize_school


def slugify(name: str) -> str:
    """URL-safe slug from a (normalized or display) name.

    "eleanor roosevelt" -> "eleanor-roosevelt";
    "northeast aa" -> "northeast-aa"; "bethesda chevy chase" -> "bethesda-chevy-chase".
    """
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    n = n.lower()
    n = re.sub(r"[^a-z0-9]+", "-", n).strip("-")
    return n


# Co-champion split: "A & B" -> ["A", "B"]. A bare " & " inside a single school
# name is unknown in this data, so splitting on " & " is safe here.
_COCHAMP_SEP = re.compile(r"\s*&\s*")


def split_cochampions(name: str) -> list[str]:
    """Split a co-champion school field into its constituent school names."""
    if not name:
        return []
    parts = [p.strip() for p in _COCHAMP_SEP.split(name)]
    return [p for p in parts if p]


def best_display_name(variants: list[str]) -> str:
    """Pick the nicest display name from the raw variants seen for one school.

    Prefer a mixed-case (not ALLCAPS, not alllower) form; fall back to a
    title-cased version of an ALLCAPS form with county/class suffixes restored.
    """
    mixed = [v for v in variants if v != v.upper() and v != v.lower()]
    if mixed:
        # Shortest mixed-case form tends to be the cleanest ("Bowie" over
        # "Bowie HS"), but prefer one without a trailing suffix word.
        mixed.sort(key=lambda v: (any(w in v.lower() for w in _SUFFIX_WORDS), len(v)))
        return mixed[0]
    # Only ALLCAPS forms available — title-case carefully.
    up = variants[0]
    return _titlecase_allcaps(up)


def _titlecase_allcaps(name: str) -> str:
    """Title-case an ALLCAPS school name, preserving short county/class suffixes.

    "NORTHEAST-AA" -> "Northeast-AA"; "MCDONOGH" -> "Mcdonogh" (best effort;
    perfect capitalization of names like "McDonogh" requires the alias map).
    """
    def fix_token(tok: str) -> str:
        # Preserve all-caps suffixes attached by a hyphen: AA, B, PG, LBV, etc.
        if "-" in tok:
            return "-".join(fix_token(p) for p in tok.split("-"))
        if len(tok) <= 3 and tok.isalpha():
            return tok.upper()  # AA, B, PG, LBV
        return tok.capitalize()

    return " ".join(fix_token(tok) for tok in name.split())


# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class School:
    """Unified per-school record across all seasons and tables."""
    slug: str
    display_name: str
    normalized_key: str
    raw_variants: set[str] = field(default_factory=set)
    # sport -> list of championship rows where this school is champion.
    titles: list[dict] = field(default_factory=list)
    finals: list[dict] = field(default_factory=list)        # championship rows as finalist
    semis: list[dict] = field(default_factory=list)         # school_records semifinalist (sport, years)
    runner_ups: list[dict] = field(default_factory=list)    # school_records runner_up
    school_record_rows: list[dict] = field(default_factory=list)  # school_records rows (sport, years)
    individual_champions: list[dict] = field(default_factory=list)  # XC + individual_results
    sportsmanship: list[dict] = field(default_factory=list)
    golf_team: list[dict] = field(default_factory=list)
    golf_individual: list[dict] = field(default_factory=list)
    stat_records: list[dict] = field(default_factory=list)


def load_books(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Load the three season record_book.json files."""
    books: dict[str, dict] = {}
    for season in SEASONS:
        path = data_dir / season / "record_book.json"
        if not path.exists():
            raise SystemExit(f"Error: {path} not found. Run the extraction pipeline first.")
        books[season] = json.loads(path.read_text(encoding="utf-8"))
    return books


def _cite(row: dict) -> dict:
    """Provenance citation for a row: source PDF + pages + a human label."""
    pdf = row.get("source_pdf", "")
    pages = row.get("source_pages", []) or []
    # "Fall Record Book 2024" style label from the filename.
    label = Path(pdf).stem if pdf else ""
    label = re.sub(r"(?i)record\s*book", "Record Book", label)
    return {
        "pdf": pdf,
        "pages": sorted(set(pages)),
        "label": label,
    }


class SchoolRegistry:
    """Holds every known school keyed by normalized name; dedupes display names."""

    def __init__(self, normalize_school):
        self.norm = normalize_school
        self.by_key: dict[str, School] = {}
        self._variants: dict[str, set[str]] = defaultdict(set)
        # names seen in championships/individual/etc that did NOT resolve to a
        # school_records school (the QA signal from verify_record_book).
        self.unmatched: dict[str, list[str]] = defaultdict(list)  # norm_key -> raw variants

    def _school_for(self, raw_name: str) -> School | None:
        key = self.norm(raw_name)
        if not key:
            return None
        if key not in self.by_key:
            return None
        s = self.by_key[key]
        s.raw_variants.add(raw_name)
        self._variants[key].add(raw_name)
        return s

    def register(self, raw_name: str) -> School:
        """Get-or-create a School for a raw name (used to seed from school_records)."""
        key = self.norm(raw_name)
        if not key:
            raise ValueError(f"cannot register empty name {raw_name!r}")
        s = self.by_key.get(key)
        if s is None:
            s = School(slug="", display_name="", normalized_key=key)
            self.by_key[key] = s
        s.raw_variants.add(raw_name)
        self._variants[key].add(raw_name)
        return s

    def finalize_names(self) -> None:
        """After all variants are collected, assign slugs + display names."""
        # Ensure display name is set from the best variant; the alias canonical
        # (mixed-case) is preferred if it was registered early as a variant.
        for key, school in self.by_key.items():
            if not school.display_name:
                school.display_name = best_display_name(sorted(self._variants[key]))
            if not school.slug:
                school.slug = slugify(key)
        # Resolve slug collisions deterministically by appending a short hash.
        seen: dict[str, str] = {}
        for key in sorted(self.by_key):
            school = self.by_key[key]
            slug = school.slug
            if slug in seen and seen[slug] != key:
                suffix = abs(hash(key)) % 1000
                school.slug = f"{slug}-{suffix}"
                slug = school.slug
            seen[slug] = key

    def lookup(self, raw_name: str) -> School | None:
        return self._school_for(raw_name)

    def note_unmatched(self, raw_name: str, context: str) -> None:
        """Record a name (from championships/individual/etc) with no school_records match."""
        key = self.norm(raw_name)
        if key and key not in self.by_key:
            self.unmatched[key].append(f"{raw_name} ({context})")


def build_school_index(books: dict[str, dict], normalize_school) -> tuple[SchoolRegistry, dict]:
    """Build the unified per-school model from all three season books.

    Returns (registry, report) where report carries counts + unmatched names.
    """
    registry = SchoolRegistry(normalize_school)

    # 1. Seed registry from school_records (the canonical school universe).
    sr_rows = 0
    for season, book in books.items():
        for row in book.get("school_records", []):
            school = registry.register(row["school"])
            school.school_record_rows.append({**row, "season": season})
            sr_rows += 1

    # 2. Championships. Split co-champions; attach to champion + finalist schools.
    champ_rows = 0
    for season, book in books.items():
        for row in book.get("championship_results", []):
            champ_rows += 1
            enriched = {**row, "season": season}
            champ_schools = split_cochampions(row.get("champion_school") or "")
            for nm in champ_schools:
                s = registry.lookup(nm)
                if s:
                    s.titles.append(enriched)
                else:
                    registry.note_unmatched(nm, f"{season} champion {row.get('sport')}")
            fin = row.get("finalist_school")
            if fin:
                for nm in split_cochampions(fin):
                    s = registry.lookup(nm)
                    if s:
                        s.finals.append(enriched)
                    else:
                        registry.note_unmatched(nm, f"{season} finalist {row.get('sport')}")

    # 3. Individual champions (XC + individual_results).
    indiv_rows = 0
    for season, book in books.items():
        for table in ("individual_xc_champions", "individual_results"):
            for row in book.get(table, []):
                indiv_rows += 1
                enriched = {**row, "season": season, "table": table}
                s = registry.lookup(row.get("school") or "")
                if s:
                    s.individual_champions.append(enriched)
                else:
                    registry.note_unmatched(row.get("school") or "", f"{season} {table}")

    # 4. Sportsmanship.
    sport_rows = 0
    for season, book in books.items():
        for row in book.get("sportsmanship_awards", []):
            sport_rows += 1
            enriched = {**row, "season": season}
            s = registry.lookup(row.get("school") or "")
            if s:
                s.sportsmanship.append(enriched)
            else:
                registry.note_unmatched(row.get("school") or "", f"{season} sportsmanship")

    # 5. Golf: team + individual champions are separate school fields.
    golf_rows = 0
    for season, book in books.items():
        for row in book.get("golf_results", []):
            golf_rows += 1
            enriched = {**row, "season": season}
            for nm in split_cochampions(row.get("team_champion_school") or ""):
                s = registry.lookup(nm)
                if s:
                    s.golf_team.append(enriched)
                else:
                    registry.note_unmatched(nm, f"{season} golf team")
            indiv_school = row.get("individual_winner_school")
            if indiv_school:
                s = registry.lookup(indiv_school)
                if s:
                    s.golf_individual.append(enriched)
                else:
                    registry.note_unmatched(indiv_school, f"{season} golf individual")

    # 6. stat_records (school may be None for team-vs-team records).
    stat_rows = 0
    for season, book in books.items():
        for row in book.get("stat_records", []):
            stat_rows += 1
            enriched = {**row, "season": season}
            s = registry.lookup(row.get("school") or "")
            if s:
                s.stat_records.append(enriched)
            # stat_records with school=None (e.g. "Dunbar v. Allegany") are not
            # attached to a school page; they're surfaced on peg/sport pages later.

    registry.finalize_names()

    report = {
        "seasons": list(books),
        "school_records_rows": sr_rows,
        "championship_rows": champ_rows,
        "individual_rows": indiv_rows,
        "sportsmanship_rows": sport_rows,
        "golf_rows": golf_rows,
        "stat_records_rows": stat_rows,
        "schools": len(registry.by_key),
        "unmatched_keys": len(registry.unmatched),
        "unmatched": _bucket_unmatched(registry, normalize_school),
    }
    return registry, report


def _bucket_unmatched(registry: SchoolRegistry, normalize_school) -> dict:
    """Group unmatched names into actionable buckets for curation.

    - ``junk``: very short normalized keys (<=3 chars) — almost always
      extraction artifacts from individual_results (e.g. "ABE", "ALL", "BOO").
    - ``near_known``: close to a known school_records name — likely a typo or
      spelling variant worth an alias row.
    - ``unresolved``: no near match — likely closed/segregated-era schools or
      genuine unknowns; listed for manual review.
    """
    import difflib
    known = sorted(registry.by_key)
    buckets: dict[str, list] = {"junk": [], "near_known": [], "unresolved": []}
    for key, variants in registry.unmatched.items():
        entries = sorted(set(variants))
        if len(key) <= 3:
            buckets["junk"].append({"key": key, "variants": entries})
            continue
        close = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
        if close:
            buckets["near_known"].append({"key": key, "suggest": close[0], "variants": entries})
        else:
            buckets["unresolved"].append({"key": key, "variants": entries})
    return buckets


def schools_index_json(registry: SchoolRegistry) -> list[dict]:
    """Compact A-Z/search index: one entry per school."""
    out = []
    for school in sorted(registry.by_key.values(), key=lambda s: s.display_name.lower()):
        out.append({
            "slug": school.slug,
            "name": school.display_name,
            "titles": len(school.titles),
            "finals": len(school.finals),
            "individual_champions": len(school.individual_champions),
            "sportsmanship": len(school.sportsmanship),
        })
    return out


def render_report(report: dict) -> str:
    """Human-readable build report, including unmatched names for curation."""
    lines = [
        "School site build report",
        "=" * 60,
        f"Seasons:            {', '.join(report['seasons'])}",
        f"school_records rows:{report['school_records_rows']}",
        f"championship rows:  {report['championship_rows']}",
        f"individual rows:    {report['individual_rows']}",
        f"sportsmanship rows: {report['sportsmanship_rows']}",
        f"golf rows:          {report['golf_rows']}",
        f"stat_records rows:  {report['stat_records_rows']}",
        f"schools:            {report['schools']}",
        f"unmatched keys:     {report['unmatched_keys']}",
        "",
    ]
    um = report["unmatched"]
    n_junk = len(um["junk"])
    n_near = len(um["near_known"])
    n_unr = len(um["unresolved"])
    lines.append(f"unmatched breakdown: {n_junk} junk / {n_near} near-known / {n_unr} unresolved")
    lines.append("")

    if um["near_known"]:
        lines.append("Near-known (likely typo/variant — add an alias row):")
        lines.append("-" * 60)
        for e in um["near_known"]:
            lines.append(f"  [{e['key']}] ~ {e['suggest']}")
            for v in e["variants"][:3]:
                lines.append(f"    {v}")
        lines.append("")

    if um["unresolved"]:
        lines.append("Unresolved (closed/segregated-era schools or unknowns — review):")
        lines.append("-" * 60)
        for e in um["unresolved"][:80]:
            lines.append(f"  [{e['key']}]")
            for v in e["variants"][:2]:
                lines.append(f"    {v}")
        if len(um["unresolved"]) > 80:
            lines.append(f"  … and {len(um['unresolved']) - 80} more")
        lines.append("")

    if n_junk:
        lines.append(f"Junk ({n_junk} short tokens, mostly individual_results artifacts — ignored):")
        samples = ", ".join(e["key"] for e in um["junk"][:20])
        lines.append(f"  e.g. {samples}")
        lines.append("")

    if not (n_junk or n_near or n_unr):
        lines.append("No unmatched school names — aliases.csv covers everything.")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true",
                        help="print the build report only; write no files")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                        help=f"output directory (default: {OUT_DIR})")
    args = parser.parse_args(argv)

    books = load_books()
    aliases = load_aliases()
    normalize_school = make_normalizer(aliases)
    registry, report = build_school_index(books, normalize_school)

    print(render_report(report))

    if args.report:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "schools-index.json").write_text(
        json.dumps(schools_index_json(registry), indent=2), encoding="utf-8")
    (args.out / "site-build-report.txt").write_text(render_report(report), encoding="utf-8")
    print(f"\nWrote {args.out / 'schools-index.json'} ({len(registry.by_key)} schools).")
    return 0


if __name__ == "__main__":
    sys.exit(main())