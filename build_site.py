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
import shutil
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


def load_aliases(path: Path = ALIASES_CSV) -> tuple[dict[str, str], dict[str, str]]:
    """Load ``alias,canonical`` rows.

    Returns ``(alias_map, canonical_display)``:
    - ``alias_map``: normalized alias_key -> normalized canonical_key (used by
      the normalizer so alias forms resolve to one school).
    - ``canonical_display``: normalized canonical_key -> canonical display name
      (mixed-case, as written in the CSV), so a school reached via an
      abbreviation uses the curated full name instead of the short alias form.
    Comment lines (starting with ``#``) and blank rows are skipped.
    """
    alias_map: dict[str, str] = {}
    canonical_display: dict[str, str] = {}
    if not path.exists():
        return alias_map, canonical_display
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            alias_key = _base_normalize(row[0])
            canon_key = _base_normalize(row[1])
            canon_display = row[1].strip()
            if alias_key and canon_key:
                alias_map[alias_key] = canon_key
            if canon_key and canon_display:
                # Last write wins; keep the first-seen display for a canonical.
                canonical_display.setdefault(canon_key, canon_display)
    return alias_map, canonical_display


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

    def __init__(self, normalize_school, canonical_display: dict[str, str] | None = None):
        self.norm = normalize_school
        self.canonical_display = canonical_display or {}
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
        # Prefer the curated canonical display name (from aliases.csv) over the
        # heuristic, so "E. Roosevelt" doesn't win over "Eleanor Roosevelt".
        for key, school in self.by_key.items():
            if not school.display_name:
                school.display_name = (self.canonical_display.get(key)
                                       or best_display_name(sorted(self._variants[key])))
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


def build_school_index(books: dict[str, dict], normalize_school,
                       canonical_display: dict[str, str] | None = None
                       ) -> tuple[SchoolRegistry, dict]:
    """Build the unified per-school model from all three season books.

    Returns (registry, report) where report carries counts + unmatched names.
    """
    registry = SchoolRegistry(normalize_school, canonical_display)

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


# ── SVG timeline renderer ────────────────────────────────────────────────────
# Build-time inline SVG: fast, print-friendly, works without JS. One
# swimlane per sport the school has won a title in, decade gridlines, a dot per
# title year. Pure function — deterministic, no randomness.

_SPORT_PALETTE = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f",
    "#aec7e8", "#ffbb78", "#98df8a", "#f7b6d2", "#c5b0d5",
)


def _sport_color(sport: str) -> str:
    return _SPORT_PALETTE[hash(sport) % len(_SPORT_PALETTE)]


def _esc(text: str) -> str:
    """XML-escape text for safe inline SVG."""
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_timeline_svg(titles: list[dict], *, width: int = 760,
                        row_height: int = 20, padding: int = 44) -> str:
    """Render an all-sport championship timeline as inline SVG.

    ``titles`` is a list of dicts each with ``year`` (int) and ``sport`` (str).
    Returns a self-contained ``<svg>...</svg>`` string. Decade gridlines are
    drawn at every 10-year boundary within the year range.
    """
    if not titles:
        return ('<svg class="timeline" width="100%" height="48" '
                'role="img" aria-label="No championship titles">'
                '<text x="6" y="20" class="timeline-empty">No titles yet</text></svg>')

    years = [int(t["year"]) for t in titles if t.get("year") is not None]
    if not years:
        return ('<svg class="timeline" width="100%" height="48" '
                'role="img" aria-label="No championship titles">'
                '<text x="6" y="20" class="timeline-empty">No titles yet</text></svg>')

    year_min = min(years)
    year_max = max(years)
    # Pad the range so a single-title school still gets a usable axis.
    if year_max - year_min < 10:
        year_min = (year_min // 10) * 10
        year_max = ((year_max // 10) + 1) * 10

    # One swimlane per sport, sorted alphabetically for stable layout.
    sports = sorted({t["sport"] for t in titles})
    sport_y = {s: padding + i * row_height for i, s in enumerate(sports)}
    # Size the left padding from the longest sport label so right-anchored
    # labels (text-anchor="end" at x=left_pad-8) don't overflow the viewBox
    # and get clipped. ~6.2px per character at the 11px label font.
    max_label_chars = max(len(s) for s in sports)
    left_pad = max(padding, int(max_label_chars * 6.2) + 12)
    right_pad = padding
    plot_w = width - left_pad - right_pad
    inner_h = padding + len(sports) * row_height + padding // 2
    height = inner_h

    def x(year: int) -> float:
        return left_pad + (year - year_min) / (year_max - year_min) * plot_w

    parts = [
        f'<svg class="timeline" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="Championship timeline {year_min} to {year_max}">'
    ]

    # Decade gridlines + labels. Skip decades before year_min so the leftmost
    # gridline sits at the plot's left edge, not in the sport-label zone.
    decade = (year_min // 10) * 10
    if decade < year_min:
        decade += 10
    d = decade
    while d <= year_max:
        gx = x(d)
        parts.append(f'<line class="tl-grid" x1="{gx:.1f}" y1="{padding-6}" '
                     f'x2="{gx:.1f}" y2="{inner_h - padding//2}" />')
        parts.append(f'<text class="tl-axis" x="{gx:.1f}" y="{height-6}" '
                     f'text-anchor="middle">{d}</text>')
        d += 10

    # Sport labels (left) + swimlane dots.
    for sport in sports:
        y = sport_y[sport] + row_height / 2
        parts.append(f'<text class="tl-sport" x="{left_pad-8}" y="{y+4:.1f}" '
                     f'text-anchor="end">{_esc(sport)}</text>')
        color = _sport_color(sport)
        for t in titles:
            if t["sport"] != sport or t.get("year") is None:
                continue
            cx = x(int(t["year"]))
            parts.append(f'<circle class="tl-dot" cx="{cx:.1f}" cy="{y:.1f}" '
                         f'r="4.5" fill="{color}"><title>{_esc(sport)} '
                         f'{int(t["year"])}</title></circle>')

    parts.append('</svg>')
    return "\n".join(parts)


# ── Citation helper ──────────────────────────────────────────────────────────
def pdf_label(pdf: str) -> str:
    """Human label for a source PDF: 'FallRecordBook2024.pdf' -> 'Fall Record Book 2024'."""
    stem = Path(pdf).stem if pdf else ""
    # Split camelCase: insert a space before each uppercase letter past the first.
    stem = re.sub(r"(?<!^)(?=[A-Z])", " ", stem)
    # Separate a trailing 4-digit year: 'Book 2024' (already split) or 'Book2024'.
    stem = re.sub(r"(?<=\D)(\d{4})$", r" \1", stem)
    # Normalize 'record book' (any casing/spacing) to 'Record Book'.
    stem = re.sub(r"(?i)record\s*book", "Record Book", stem)
    return re.sub(r"\s+", " ", stem).strip()


def cite_str(row: dict) -> str:
    """'p. 37, Fall Record Book 2024' or 'pp. 37–38, Fall Record Book 2024'."""
    pdf = row.get("source_pdf", "")
    pages = sorted(set(row.get("source_pages") or []))
    label = pdf_label(pdf)
    if not pages:
        return label
    if len(pages) == 1:
        return f"p. {pages[0]}, {label}"
    # Contiguous range vs scattered pages.
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pp. {pages[0]}–{pages[-1]}, {label}"
    return f"pp. {', '.join(str(p) for p in pages)}, {label}"


# ── Per-school page data ─────────────────────────────────────────────────────
def _last_title_str(titles: list[dict]) -> str | None:
    """Short form for the fact grid: '4A Boys Basketball 2022'."""
    if not titles:
        return None
    t = max(titles, key=lambda r: (r.get("year") or 0, r.get("sport") or ""))
    cls = t.get("classification") or ""
    return f"{cls} {t['sport']} {t['year']}".strip()


def _last_title_phrase(titles: list[dict]) -> str | None:
    """Prose form for the fast-facts paragraph: '4A boys basketball in 2022'."""
    if not titles:
        return None
    t = max(titles, key=lambda r: (r.get("year") or 0, r.get("sport") or ""))
    cls = t.get("classification") or ""
    sport = (t.get("sport") or "").lower()
    prefix = f"{cls} {sport}".strip()
    return f"{prefix} in {t.get('year')}"


def fast_facts_paragraph(name: str, school: School) -> str:
    """Deterministic, paste-ready summary paragraph for a school clip file."""
    n_titles = len(school.titles)
    n_finals = len(school.finals)
    n_indiv = len(school.individual_champions)
    n_sport = len(school.sportsmanship)
    sports_won = {t["sport"] for t in school.titles}

    def pl(n: int, sing: str, plur: str) -> str:
        return f"{n} {sing if n == 1 else plur}"

    extras: list[str] = []
    if n_indiv:
        extras.append(f"produced {pl(n_indiv, 'individual state champion', 'individual state champions')}")
    if n_sport:
        extras.append(f"won {pl(n_sport, 'sportsmanship award', 'sportsmanship awards')}")
    extra_str = (" It has " + " and ".join(extras) + ".") if extras else ""

    if n_titles:
        phrase = _last_title_phrase(school.titles)
        return (f"{name} has won {pl(n_titles, 'state championship', 'state championships')} "
                f"across {pl(len(sports_won), 'sport', 'sports')}, most recently {phrase}."
                + extra_str)
    if n_finals:
        return (f"{name} has reached {pl(n_finals, 'state final', 'state finals')} "
                f"without a title." + extra_str)
    return f"{name} has no recorded state championship appearances.{extra_str}"


def _grouped_by_sport(rows: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r.get("sport") or ""].append(r)
    return g


def school_page_data(school: School) -> dict:
    """Assemble the context dict for the school clip-file template."""
    titles_by_sport = _grouped_by_sport(school.titles)
    rec_by_sport = _grouped_by_sport(school.school_record_rows)
    indiv_by_sport = _grouped_by_sport(school.individual_champions)
    sport_by_sport = _grouped_by_sport(school.sportsmanship)
    stat_by_sport = _grouped_by_sport(school.stat_records)

    all_sports = sorted(set(titles_by_sport) | set(rec_by_sport) | set(indiv_by_sport)
                        | set(sport_by_sport) | set(stat_by_sport))

    sports = []
    for sport in all_sports:
        titles = sorted(titles_by_sport.get(sport, []), key=lambda r: (r.get("year") or 0))
        # Opponent = finalist school (fall/winter); spring rows lack it.
        title_rows = [{
            "year": r.get("year"), "classification": r.get("classification"),
            "score": r.get("score"), "opponent": r.get("finalist_school"),
            "co_champion": r.get("co_champion"),
            "source_pdf": r.get("source_pdf"), "source_pages": r.get("source_pages"),
        } for r in titles]

        rec = rec_by_sport.get(sport, [{}])[0] if rec_by_sport.get(sport) else {}
        record_years = {
            "champion_years": sorted(rec.get("champion_years") or []),
            "finalist_years": sorted(rec.get("finalist_years") or []),
            "semifinalist_years": sorted(rec.get("semifinalist_years") or []),
            "runner_up_years": sorted(rec.get("runner_up_years") or []),
        } if rec else {"champion_years": [], "finalist_years": [],
                       "semifinalist_years": [], "runner_up_years": []}

        sports.append({
            "sport": sport,
            "titles": title_rows,
            "record_years": record_years,
            "individual": sorted(indiv_by_sport.get(sport, []),
                                 key=lambda r: (r.get("year") or 0)),
            "sportsmanship": sorted(sport_by_sport.get(sport, []),
                                    key=lambda r: (r.get("year") or 0)),
            "stat_records": stat_by_sport.get(sport, []),
        })

    return {
        "school": {
            "name": school.display_name,
            "slug": school.slug,
            "total_titles": len(school.titles),
            "total_finals": len(school.finals),
            "last_title": _last_title_str(school.titles),
            "sportsmanship_count": len(school.sportsmanship),
        },
        "fast_facts": fast_facts_paragraph(school.display_name, school),
        "timeline_svg": render_timeline_svg(school.titles),
        "sports": sports,
    }


def school_json(school: School) -> dict:
    """Machine-readable per-school record (powers embeds; Phase 2 API)."""
    return {
        "slug": school.slug,
        "name": school.display_name,
        "totals": {
            "titles": len(school.titles),
            "finals": len(school.finals),
            "individual_champions": len(school.individual_champions),
            "sportsmanship": len(school.sportsmanship),
            "golf_team": len(school.golf_team),
            "golf_individual": len(school.golf_individual),
            "stat_records": len(school.stat_records),
        },
        "last_title": _last_title_str(school.titles),
        "titles": school.titles,
        "finals": school.finals,
        "individual_champions": school.individual_champions,
        "sportsmanship": school.sportsmanship,
        "golf_team": school.golf_team,
        "golf_individual": school.golf_individual,
        "stat_records": school.stat_records,
    }


# ── Peg computations (story angles) ──────────────────────────────────────────
# Pure functions over the unified school index: droughts, streaks, never-won,
# first-title watch, and round-number anniversaries. All deterministic.

_ANNIVERSARY_YEARS = (25, 50, 75, 100)


def _longest_consecutive_run(years: list[int]) -> tuple[int, int | None, int | None]:
    """Return (length, start, end) of the longest run of consecutive years."""
    yrs = sorted(set(y for y in years if y is not None))
    if not yrs:
        return 0, None, None
    best_len, best_start, best_end = 1, yrs[0], yrs[0]
    cur_start, cur_len = yrs[0], 1
    for i in range(1, len(yrs)):
        if yrs[i] == yrs[i - 1] + 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start, best_end = cur_len, cur_start, yrs[i - 1]
            cur_start, cur_len = yrs[i], 1
    if cur_len > best_len:
        best_len, best_start, best_end = cur_len, cur_start, yrs[-1]
    return best_len, best_start, best_end


def _latest_year(registry: SchoolRegistry) -> int:
    years = [t.get("year") for s in registry.by_key.values() for t in s.titles if t.get("year")]
    years += [f.get("year") for s in registry.by_key.values() for f in s.finals if f.get("year")]
    years = [y for y in years if y is not None]
    return max(years) if years else 2025


def compute_droughts(registry: SchoolRegistry, current_year: int | None = None) -> dict:
    """Active title droughts: years since a school's last title.

    ``overall`` is per school (last title in any sport); ``per_sport`` is per
    (school, sport). Sorted by drought length descending.
    """
    if current_year is None:
        current_year = _latest_year(registry)
    overall, per_sport = [], []
    for s in registry.by_key.values():
        if not s.titles:
            continue
        last = max(t["year"] for t in s.titles if t.get("year") is not None)
        overall.append({"school": s.display_name, "slug": s.slug,
                        "last_title": last, "drought": current_year - last})
        by_sport: dict[str, list[int]] = defaultdict(list)
        for t in s.titles:
            by_sport[t["sport"]].append(t["year"])
        for sport, yrs in by_sport.items():
            last = max(yrs)
            per_sport.append({"school": s.display_name, "slug": s.slug, "sport": sport,
                              "last_title": last, "drought": current_year - last})
    overall.sort(key=lambda r: (-r["drought"], r["school"]))
    per_sport.sort(key=lambda r: (-r["drought"], r["sport"], r["school"]))
    return {"current_year": current_year, "overall": overall, "per_sport": per_sport}


def compute_streaks(registry: SchoolRegistry) -> dict:
    """Longest consecutive-title runs per (school, sport) + active streaks.

    A streak is consecutive calendar years with a title in the same sport.
    ``active`` streaks are those ending at the latest title year in the data.
    """
    latest = _latest_year(registry)
    runs, active = [], []
    for s in registry.by_key.values():
        by_sport: dict[str, list[int]] = defaultdict(list)
        for t in s.titles:
            by_sport[t["sport"]].append(t["year"])
        for sport, yrs in by_sport.items():
            length, start, end = _longest_consecutive_run(yrs)
            if length >= 2:
                runs.append({"school": s.display_name, "slug": s.slug, "sport": sport,
                             "length": length, "start": start, "end": end})
            if yrs and max(yrs) == latest and length >= 2 and end == latest:
                active.append({"school": s.display_name, "slug": s.slug, "sport": sport,
                               "length": length, "start": start, "end": end})
    runs.sort(key=lambda r: (-r["length"], r["sport"], r["school"]))
    active.sort(key=lambda r: (-r["length"], r["sport"], r["school"]))
    return {"latest_year": latest, "dynasties": runs, "active": active}


def compute_never_won(registry: SchoolRegistry) -> dict:
    """Schools that have reached a final but never won, and schools with no
    championship history at all."""
    reached_final_no_title = []
    no_title_any_sport = []
    no_history = []
    for s in registry.by_key.values():
        has_title = bool(s.titles)
        # finalist_years comes from school_records (all seasons).
        finalist_years = {y for r in s.school_record_rows
                          for y in (r.get("finalist_years") or [])}
        has_final = bool(s.finals) or bool(finalist_years)
        if not has_title and has_final:
            no_title_any_sport.append({
                "school": s.display_name, "slug": s.slug,
                "finals": len(s.finals) + len(finalist_years),
            })
        if not has_title and not has_final:
            no_history.append({"school": s.display_name, "slug": s.slug})
        # per-sport: reached a final in a sport, never won that sport
        rec_by_sport = {r["sport"]: r for r in s.school_record_rows}
        for sport, rec in rec_by_sport.items():
            fy = rec.get("finalist_years") or []
            cy = rec.get("champion_years") or []
            if fy and not cy:
                reached_final_no_title.append({
                    "school": s.display_name, "slug": s.slug, "sport": sport,
                    "finalist_years": sorted(fy),
                })
    no_title_any_sport.sort(key=lambda r: (-r["finals"], r["school"]))
    reached_final_no_title.sort(key=lambda r: (r["sport"], r["school"]))
    no_history.sort(key=lambda r: r["school"])
    return {"reached_final_no_title": reached_final_no_title,
            "no_title_any_sport": no_title_any_sport, "no_history": no_history}


def compute_first_title_watch(registry: SchoolRegistry) -> dict:
    """Most-recent-season finalists who have never won a title in that sport."""
    latest = _latest_year(registry)
    candidates = []
    for s in registry.by_key.values():
        titles_by_sport = {t["sport"] for t in s.titles}
        for f in s.finals:
            if f.get("year") != latest:
                continue
            sport = f.get("sport")
            if sport and sport not in titles_by_sport:
                candidates.append({
                    "school": s.display_name, "slug": s.slug, "sport": sport,
                    "year": latest, "classification": f.get("classification"),
                })
    # De-dup (a school could appear twice across seasons in the same sport).
    seen, unique = set(), []
    for c in candidates:
        key = (c["slug"], c["sport"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    unique.sort(key=lambda r: (r["sport"], r["school"]))
    return {"latest_year": latest, "candidates": unique}


def compute_anniversaries(registry: SchoolRegistry, current_year: int | None = None) -> dict:
    """Round-number title anniversaries (25/50/75/100 yrs) falling in the
    latest championship year. Each entry carries its PDF citation.

    The record books carry years but no dates, so entries are keyed by
    anniversary year rather than calendar week; the page presents them as
    'this season's round-number anniversaries.'
    """
    if current_year is None:
        current_year = _latest_year(registry)
    targets = {current_year - a: a for a in _ANNIVERSARY_YEARS}
    out = []
    for s in registry.by_key.values():
        for t in s.titles:
            y = t.get("year")
            if y in targets:
                out.append({
                    "school": s.display_name, "slug": s.slug, "sport": t["sport"],
                    "year": y, "anniversary": targets[y],
                    "classification": t.get("classification"),
                    "citation": cite_str(t),
                })
    out.sort(key=lambda r: (-r["anniversary"], r["sport"], r["school"]))
    return {"current_year": current_year, "anniversaries": out}


def compute_pegs(registry: SchoolRegistry) -> dict:
    """All story-peg computations in one bundle."""
    return {
        "droughts": compute_droughts(registry),
        "streaks": compute_streaks(registry),
        "never_won": compute_never_won(registry),
        "first_title_watch": compute_first_title_watch(registry),
        "anniversaries": compute_anniversaries(registry),
    }


# ── Rendering ────────────────────────────────────────────────────────────────
def _jinja_env() -> "jinja2.Environment":
    import jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(WEB_DIR / "templates")),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        trim_blocks=True, lstrip_blocks=True,
    )
    env.filters["cite"] = cite_str
    return env


def build_site(registry: SchoolRegistry, report: dict, out_dir: Path) -> dict:
    """Render the full static site to ``out_dir``. Returns page counts."""
    env = _jinja_env()
    root = _relative_root()  # path prefix from a page back to site root
    schools = sorted(registry.by_key.values(), key=lambda s: s.display_name.lower())

    # Clean output so merged-away schools don't leave stale pages behind.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Static assets.
    static_src = WEB_DIR / "static"
    static_dst = out_dir / "static"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(static_src, static_dst)

    # Home page.
    home = env.get_template("home.html")
    (out_dir / "index.html").write_text(home.render(
        root=root, school_count=len(schools),
        totals={
            "championships": report["championship_rows"],
            "schools": len(schools),
            "individual": report["individual_rows"],
            "stat_records": report["stat_records_rows"],
        }), encoding="utf-8")

    # Schools index page + JSON.
    index_tmpl = env.get_template("index.html")
    schools_dir = out_dir / "schools"
    schools_dir.mkdir(parents=True, exist_ok=True)
    (schools_dir / "index.html").write_text(
        index_tmpl.render(root="../", schools=schools_index_json(registry)),
        encoding="utf-8")

    # Per-school pages + JSON. Pages live at site/schools/{slug}/index.html,
    # so their root back to the site root is ../../
    school_tmpl = env.get_template("school.html")
    for school in schools:
        page_dir = schools_dir / school.slug
        page_dir.mkdir(parents=True, exist_ok=True)
        ctx = school_page_data(school)
        (page_dir / "index.html").write_text(
            school_tmpl.render(root="../../", **ctx), encoding="utf-8")
        (schools_dir / f"{school.slug}.json").write_text(
            json.dumps(school_json(school), indent=2), encoding="utf-8")

    # Story pegs.
    peg_pages = build_pegs(registry, out_dir, root)

    # Embeddable widgets + builder.
    embed_pages = build_embeds(registry, out_dir, root)

    return {
        "schools": len(schools),
        "pages": len(schools) + 2 + peg_pages + embed_pages,
        "json_files": len(schools) + 1 + 5,     # per-school + index + 5 peg JSON
    }


def build_pegs(registry: SchoolRegistry, out_dir: Path, root: str) -> int:
    """Render the story-peg pages + JSON. Returns number of pages rendered."""
    env = _jinja_env()
    pegs = compute_pegs(registry)
    pegs_dir = out_dir / "pegs"
    pegs_dir.mkdir(parents=True, exist_ok=True)

    # Peg index (one level deep: site/pegs/index.html -> root ../).
    (pegs_dir / "index.html").write_text(
        env.get_template("pegs/index.html").render(
            root="../",
            droughts=len(pegs["droughts"]["overall"]),
            streaks=len(pegs["streaks"]["dynasties"]),
            watch=len(pegs["first_title_watch"]["candidates"]),
            anniversaries=len(pegs["anniversaries"]["anniversaries"]),
        ), encoding="utf-8")

    # One page + one JSON each. (page_root is ../../ for /pegs/<name>/index.html)
    pages = [
        ("droughts", "droughts.html", pegs["droughts"]),
        ("streaks", "streaks.html", pegs["streaks"]),
        ("never-won", "never_won.html", pegs["never_won"]),
        ("first-title-watch", "first_title.html", pegs["first_title_watch"]),
        ("anniversaries", "anniversaries.html", pegs["anniversaries"]),
    ]
    for slug, tmpl, data in pages:
        sub = pegs_dir / slug
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "index.html").write_text(
            env.get_template(f"pegs/{tmpl}").render(root="../../", data=data),
            encoding="utf-8")
        (pegs_dir / f"{slug.replace('-', '_')}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8")
    return len(pages) + 1  # 5 peg pages + index


def build_embeds(registry: SchoolRegistry, out_dir: Path, root: str) -> int:
    """Render self-contained iframe embed pages + the builder page.

    Each embed inlines its data at build time (no external requests), so it
    works inside any CMS that allows iframes. ``?theme=light|dark`` is read
    by a tiny inline script in each embed.
    """
    env = _jinja_env()
    embed_dir = out_dir / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)
    schools = sorted(registry.by_key.values(), key=lambda s: s.display_name.lower())
    anniversaries = compute_anniversaries(registry)

    timeline_tmpl = env.get_template("embed/timeline.html")
    titles_tmpl = env.get_template("embed/titles.html")

    for school in schools:
        # /embed/timeline/{slug}/index.html  -> school page is ../../../schools/{slug}/
        tl_dir = embed_dir / "timeline" / school.slug
        tl_dir.mkdir(parents=True, exist_ok=True)
        (tl_dir / "index.html").write_text(timeline_tmpl.render(
            school={"name": school.display_name, "slug": school.slug,
                    "total_titles": len(school.titles)},
            timeline_svg=render_timeline_svg(school.titles),
            school_url=f"../../../schools/{school.slug}/index.html",
        ), encoding="utf-8")

        ti_dir = embed_dir / "titles" / school.slug
        ti_dir.mkdir(parents=True, exist_ok=True)
        sports_count = len({t["sport"] for t in school.titles})
        last = _last_title_str(school.titles)
        (ti_dir / "index.html").write_text(titles_tmpl.render(
            school={"name": school.display_name, "slug": school.slug,
                    "total_titles": len(school.titles), "last_title": last},
            sports_count=sports_count,
            school_url=f"../../../schools/{school.slug}/index.html",
        ), encoding="utf-8")

    # Anniversaries widget (no per-school variant).
    ann_dir = embed_dir / "anniversaries"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "index.html").write_text(
        env.get_template("embed/anniversaries.html").render(
            data=anniversaries, school_base="../../"),
        encoding="utf-8")

    # Builder page (one level deep: site/embed/index.html -> root ../).
    (embed_dir / "index.html").write_text(
        env.get_template("embed/builder.html").render(
            root="../",
            schools_json=json.dumps(schools_index_json(registry))),
        encoding="utf-8")

    return len(schools) * 2 + 2  # timeline + titles per school, + anniversaries + builder


def _relative_root() -> str:
    """Path prefix from a top-level page to the site root (empty for flat)."""
    return "./"


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
    alias_map, canonical_display = load_aliases()
    normalize_school = make_normalizer(alias_map)
    registry, report = build_school_index(books, normalize_school, canonical_display)

    print(render_report(report))

    if args.report:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    counts = build_site(registry, report, args.out)
    # Top-level schools-index.json (the search dataset) + build report.
    (args.out / "schools-index.json").write_text(
        json.dumps(schools_index_json(registry), indent=2), encoding="utf-8")
    (args.out / "site-build-report.txt").write_text(render_report(report), encoding="utf-8")
    print(f"\nBuilt {counts['pages']} pages and {counts['json_files']} JSON files "
          f"to {args.out}/ ({counts['schools']} schools).")
    return 0


if __name__ == "__main__":
    sys.exit(main())