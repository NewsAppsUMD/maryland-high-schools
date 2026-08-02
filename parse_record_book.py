#!/usr/bin/env python3
"""
Parse MPSSAA Record Book PDFs (fall, winter, spring) into structured CSV / JSON data.

Usage:
    uv run parse_record_book.py [PDF] [--out DIR] [--season fall|winter|spring]
                               [--routes] [--refresh] [--offline]

Defaults:
    PDF_PATH   = pdfs/FallRecordBook2024.pdf
    OUTPUT_DIR = data/<season>   (auto-detected from the PDF filename)

Extraction runs against GLM-5.2 served by a local Ollama daemon (the
``glm-5.2:cloud`` model) via the ``llm-ollama`` plugin — no Anthropic key needed.
``--routes`` is a zero-cost dry run (page→classifier table, no LLM calls).
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import llm
from pypdf import PdfReader
from pydantic import BaseModel

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class ChampionshipResult(BaseModel):
    sport: str
    year: int
    classification: str
    champion_school: str
    champion_coach: Optional[str] = None
    finalist_school: Optional[str] = None
    finalist_coach: Optional[str] = None
    score: Optional[str] = None
    champion_undefeated: bool = False
    co_champion: bool = False
    notes: Optional[str] = None


class ChampionshipResults(BaseModel):
    results: list[ChampionshipResult]


class IndividualChampion(BaseModel):
    sport: str
    year: int
    classification: str
    name: str
    school: str
    time: Optional[str] = None
    distance: Optional[str] = None


class IndividualChampions(BaseModel):
    champions: list[IndividualChampion]


class IndividualResult(BaseModel):
    sport: str
    event: str
    year: int
    classification: str
    name: str
    school: str
    mark: Optional[str] = None


class IndividualResults(BaseModel):
    results: list[IndividualResult]


class GolfResult(BaseModel):
    year: int
    classification: str  # "Combined", "1A/2A", or "3A/4A"
    team_champion_school: Optional[str] = None
    team_score: Optional[int] = None
    individual_winner_name: Optional[str] = None
    individual_winner_school: Optional[str] = None
    individual_score: Optional[int] = None
    individual_gender: Optional[str] = None  # "male" or "female" or None


class GolfResults(BaseModel):
    results: list[GolfResult]


class SportsmanshipAward(BaseModel):
    sport: str
    year: int
    classification: Optional[str] = None
    school: str


class SportsmanshipAwards(BaseModel):
    awards: list[SportsmanshipAward]


class StatRecord(BaseModel):
    sport: str
    category: Optional[str] = None  # "team" | "individual" | None
    record: str
    value: Optional[str] = None
    holder: Optional[str] = None      # player name (individual) or school (team)
    school: Optional[str] = None
    year: Optional[str] = None        # "2016", "2015-2018", "1988 & 2015"
    co_holder: Optional[bool] = None
    notes: Optional[str] = None


class StatResults(BaseModel):
    results: list[StatRecord]


# ── LLM ───────────────────────────────────────────────────────────────────────
# Extraction runs against GLM-5.2 served by a local Ollama daemon (cloud tag).
# Reached via the llm-ollama plugin; the model_id is the Ollama model name.

MODEL_ID = "glm-5.2:cloud"

# First year MPSSAA sponsored each sport's state tournament (the cut-point the
# record books mark with "TOURNAMENTS UNDER THE DIRECTION OF MPSSAA"). Rows from
# the preceding "PRIOR TO MPSSAA SPONSORSHIP" / "PRE-MPSSAA" / "SEGREGATED SCHOOL"
# precursor sections are tagged notes="Pre-MPSSAA" in main(). Verified from each
# sport's PDF section header. Sports not listed either have no precursor section
# or start at MPSSAA's 1946 founding (e.g. Boys Cross Country, the era-floor
# anchor). NB: a handful of "SEGREGATED SCHOOL TOURNAMENTS" rows run in PARALLEL
# with early MPSSAA years (Boys Basketball 1948/1950/1951, Girls Basketball 1948)
# and so fall at/after the cut-point — the year rule does not tag them; they need
# section-header detection (a prompt-based pass) to tag as a distinct category.
MPSSAA_ERA_STARTS: dict[str, int] = {
    "Boys Soccer": 1969,
    "Girls Soccer": 1989,
    "Field Hockey": 1975,
    "Volleyball": 1975,
    "Boys Basketball": 1947,
    "Girls Basketball": 1947,
    "Boys Track and Field": 1952,
    "Girls Track and Field": 1975,
}

_model: Optional[llm.Model] = None


def get_model() -> llm.Model:
    global _model
    if _model is None:
        _model = llm.get_model(MODEL_ID)
    return _model


def _schema_to_instruction(schema) -> str:
    """Render a Pydantic schema as a JSON-Schema block to embed in the prompt.

    GLM-5.2 (cloud) does not honor Ollama's schema-structured ``format`` (it
    answers conversationally), so we describe the schema in the prompt text and
    constrain output to JSON via ``json_object=True`` (Ollama ``format="json"``).
    """
    return json.dumps(schema.model_json_schema(), indent=2)


def _strip_fences(text: str) -> str:
    """Strip ```json ... ``` markdown fences if the model wrapped its output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def llm_extract(prompt: str, schema, retries: int = 2) -> dict:
    """Call the LLM and return parsed JSON matching ``schema`` (a Pydantic class).

    Default path (GLM-5.2 via Ollama): uses JSON mode (``json_object=True``)
    with the JSON schema described in the prompt, because GLM-5.2 cloud ignores
    Ollama's schema-structured ``format`` and answers conversationally.

    Fallback path: if the provider does not support JSON mode (e.g. Anthropic
    via ``llm-anthropic``), switches to ``schema=schema`` structured output.
    """
    model = get_model()
    schema_instruction = (
        "Return ONLY valid JSON, no prose and no markdown fences. "
        "Your output must match this JSON schema exactly:\n"
        f"{_schema_to_instruction(schema)}"
    )
    json_prompt = f"{schema_instruction}\n\n{prompt}"

    text = ""
    response = None
    use_schema = False  # fallback for providers without JSON mode
    for attempt in range(1, retries + 1):
        try:
            if use_schema:
                response = model.prompt(prompt, schema=schema, stream=False)
            else:
                response = model.prompt(json_prompt, json_object=True, stream=False)
        except (NotImplementedError, ValueError) as exc:
            if not use_schema:
                # Provider lacks JSON mode — retry over the schema= path.
                use_schema = True
                continue
            raise
        text = _strip_fences(response.text())
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Some models/states return the payload via a tool call instead.
        if response.tool_calls():
            args = response.tool_calls()[0].arguments
            if isinstance(args, dict):
                return args
            return json.loads(args)

        if attempt < retries:
            print(f"    (retry {attempt}/{retries} — LLM returned no usable JSON)")

    raise RuntimeError(
        f"LLM returned no usable JSON after {retries} attempts.\n"
        f"text={text!r}\n"
        f"response_json={getattr(response, 'response_json', None)}"
    )


# ── PDF extraction ────────────────────────────────────────────────────────────


_QUOTE_NORMALIZE = str.maketrans({
    "‘": "'", "’": "'",  # curly single quotes -> straight apostrophe
    "“": '"', "”": '"',  # curly double quotes -> straight quote
})


def _normalize_quotes(text: str) -> str:
    """Fold curly quotes to ASCII so names match across the regex and LLM paths.

    PDF text extraction preserves whatever quote glyph the source used, but the
    same school name (e.g. "Queen Anne's") must produce an identical string
    whether it comes through parse_school_records() or an LLM extractor, or
    every downstream join/dedup on school name silently splits into two rows.
    """
    return text.translate(_QUOTE_NORMALIZE)


def load_pages(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    return [_normalize_quotes(page.extract_text() or "") for page in reader.pages]


# ── Section maps ──────────────────────────────────────────────────────────────
# PDF 0-based page indices as [start, end) slices.

FALL_SECTIONS: dict[str, tuple[int, int]] = {
    "Girls Cross Country": (3, 13),
    "Boys Cross Country":  (13, 27),
    "Field Hockey":        (27, 35),
    "Football":            (35, 47),
    "Golf":                (47, 53),
    "Girls Soccer":        (53, 59),
    "Boys Soccer":         (59, 66),
    "Volleyball":          (66, 77),
}

WINTER_SECTIONS: dict[str, tuple[int, int]] = {
    # Girls Basketball starts at idx 2 (its divider page); the segregated-school
    # tournament results (1948+, journalistically significant) begin on idx 3.
    "Girls Basketball":        (2, 11),
    "Boys Basketball":         (11, 21),
    "Girls Indoor Track":      (21, 40),
    "Boys Indoor Track":       (40, 60),
    "Girls Swimming & Diving": (60, 69),
    "Boys Swimming & Diving":  (69, 79),
    "Wrestling":               (79, 100),
}

SPRING_SECTIONS: dict[str, tuple[int, int]] = {
    "Baseball":              (3, 11),
    "Girls Lacrosse":        (11, 16),
    "Boys Lacrosse":         (16, 22),
    "Softball":              (22, 29),
    "Tennis":                (29, 35),
    "Girls Track and Field": (35, 61),
    "Boys Track and Field":  (61, 95),
}

SEASON_SECTIONS = {
    "fall": FALL_SECTIONS,
    "winter": WINTER_SECTIONS,
    "spring": SPRING_SECTIONS,
}

# ── Heading-based section detection (edition drift) ───────────────────────────
# The maps above are the verified baseline for the current PDFs. At runtime we
# detect sport divider pages instead of trusting the hardcoded indices, so a
# new edition that shifts page numbers still slices correctly. The detected
# sport set/order is validated against the baseline (SEASON_SECTIONS) and any
# drift — a new, renamed, or dropped sport — is a hard error, not a silently
# wrong slice. The baseline maps are retained as the fixture the detection is
# asserted to reproduce (see test_parse_record_book.py).

# Divider pages are short: the NaturalPDF text is "MPSSAA\n<Sport>\nRecords\n<n>"
# (~25-45 chars). pypdf drops the large-font sport title and returns only
# "MPSSAA", which is why section detection uses NaturalPDF. The length cap
# excludes table-of-contents pages (~250+ chars) and content pages whose
# running header also starts with "MPSSAA ... Records" (~600-13000 chars).
DIVIDER_RE = re.compile(r"^\s*MPSSAA\s+(.+?)\s+Records?\b", re.DOTALL)
DIVIDER_MAX_LEN = 80

# Trailing back-matter: each PDF ends with a "DISTRICTS OF THE ..." county map
# page that is not data. The final sport section ends just before it (fall p78,
# spring p96; winter has none and ends on a sportsmanship page).
BACK_MATTER_RE = re.compile(r"\bDISTRICTS OF THE\b")


def _normalize_sport(name: str) -> str:
    """Canonical comparison form: whitespace collapsed, '&' -> 'and', lowercased.

    The PDF titles use 'Girls Track & Field' while the baseline map uses
    'Girls Track and Field'; swimming is '&' in both. Normalising both sides
    makes the match robust to that inconsistency.
    """
    return re.sub(r"\s+", " ", name.replace("&", "and")).strip().lower()


def _normalize_classification(value):
    """Canonicalise a ``classification`` string.

    The championship prompt asks the LLM for the raw class label ("1A", "2A",
    "B", "Combined", ...) but on a fresh extraction the model sometimes copies
    the column header verbatim — "CLASS 1A" — or appends a stray asterisk from
    a PDF footnote marker — "B*". Both are the same class as the bare label;
    leaving them as-is would fragment the natural key (sport, year,
    classification, ...) and break downstream joins by classification.

    Strip a leading "CLASS " prefix and trailing "*" characters, and collapse
    surrounding whitespace. None/empty pass through unchanged so nullable
    classification columns (sportsmanship, golf) are unaffected.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return s
    s = re.sub(r"^class\s+", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip("*").strip()
    # The model sometimes abbreviates the "Combined" class label as "Comb" on a
    # page (a one-off shortening, not a distinct class); expand it so the natural
    # key matches the "Combined" rows on every other page.
    if s.lower() == "comb":
        s = "Combined"
    return s


def _normalize_event_name(value):
    """Canonicalise an ``event`` string for individual_results.

    The model emits the same short track event under several spellings across
    chunks — "55m", "55 m", "55m Dash" — which fragments the natural key
    (sport, year, classification, event) and invents phantom duplicate events.
    Collapse the metric-dash variants to the plain "NN m" form already used by
    the longer distance events ("1600 m", "500 m"): drop a redundant trailing
    "Dash" and insert a space before a lowercase "m" that directly follows a
    digit. Capitalised "Meter(s)" forms (spring track) and "Hurdles"/relay
    events are left untouched, so this is a no-op for fall (no individual
    results) and spring. None/empty pass through unchanged.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return s
    s = re.sub(r"\s+dash$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"(\d)m\b", r"\1 m", s)
    return s


def _tag_pre_mpssaa(rows: list[dict]) -> None:
    """Stamp ``notes="Pre-MPSSAA"`` on precursor-tournament rows in place.

    Rows whose sport has a known MPSSAA-era start (``MPSSAA_ERA_STARTS``) and
    whose year precedes it are public-school precursor tournaments run before
    MPSSAA sponsorship — not MPSSAA championships. Tag them so they stay
    distinguishable; ``verify_record_book`` then excludes tagged rows from its
    continuity/cross-path/referential checks. Existing notes are preserved by
    prefixing (e.g. ``"Pre-MPSSAA; forfeit"``) so a forfeit/default marker is
    not lost.
    """
    for r in rows:
        start = MPSSAA_ERA_STARTS.get(r.get("sport"))
        if start is None or r.get("year") is None or r["year"] >= start:
            continue
        existing = (r.get("notes") or "").strip()
        if not existing:
            r["notes"] = "Pre-MPSSAA"
        elif not existing.startswith("Pre-MPSSAA"):
            r["notes"] = f"Pre-MPSSAA; {existing}"


def detect_dividers(page_texts) -> list[tuple[int, str]]:
    """Find short 'MPSSAA <Sport> Records' divider pages.

    ``page_texts`` is per-page text from NaturalPDF, which extracts the
    large-font divider titles that pypdf drops. Returns [(page_index,
    sport_name), ...] in page order.
    """
    out = []
    for i, t in enumerate(page_texts):
        t = (t or "").strip()
        if len(t) > DIVIDER_MAX_LEN:
            continue
        m = DIVIDER_RE.match(t)
        if not m:
            continue
        sport = re.sub(r"\s+", " ", m.group(1)).strip()
        out.append((i, sport))
    return out


def _book_end(page_texts, start: int) -> int:
    """Index of the trailing districts-map back-matter page, else len(pages)."""
    for i in range(start, len(page_texts)):
        if BACK_MATTER_RE.search(page_texts[i] or ""):
            return i
    return len(page_texts)


def find_sections(page_texts, season: str) -> dict[str, tuple[int, int]]:
    """Detect sport divider pages and build {sport: (start, end)} slices.

    Replaces hardcoded index use with runtime detection. Validates the
    detected sport set and order against the season's baseline map
    (SEASON_SECTIONS) and raises ValueError on drift. Section keys are the
    canonical baseline names so downstream prompt/classifier code is unchanged.
    """
    if season not in SEASON_SECTIONS:
        raise ValueError(f"Unknown season {season!r}; expected one of {list(SEASON_SECTIONS)}")
    expected_order = list(SEASON_SECTIONS[season].keys())
    norm_to_canonical = {_normalize_sport(k): k for k in expected_order}

    dividers = detect_dividers(page_texts)
    if not dividers:
        raise ValueError(f"No sport divider pages detected in {season} PDF")
    book_end = _book_end(page_texts, dividers[0][0])

    detected = []
    for idx, sport in dividers:
        key = norm_to_canonical.get(_normalize_sport(sport))
        if key is None:
            raise ValueError(
                f"Detected unknown sport {sport!r} on PDF page {idx + 1}; not in "
                f"expected {season} sports {expected_order}. Edition drift?"
            )
        detected.append((idx, key))

    detected_names = [k for _, k in detected]
    missing = [k for k in expected_order if k not in set(detected_names)]
    if missing:
        raise ValueError(f"Missing expected {season} sport divider(s): {missing}")
    if detected_names != expected_order:
        raise ValueError(
            f"Detected {season} sport order {detected_names} != expected {expected_order}"
        )

    sections = {}
    for i, (start, key) in enumerate(detected):
        end = detected[i + 1][0] if i + 1 < len(detected) else book_end
        sections[key] = (start, end)
    return sections


def _divider_candidates(pypdf_pages) -> list[int]:
    """pypdf page indices that might be dividers: short pages mentioning MPSSAA.

    Used to limit NaturalPDF extraction (which is ~10s for a full PDF) to the
    ~8 pages that actually need it. pypdf returns only 'MPSSAA' on divider
    pages (it drops the large-font title), but that's enough to find them.
    """
    return [
        i for i, t in enumerate(pypdf_pages)
        if 0 < len((t or "").strip()) < DIVIDER_MAX_LEN and "MPSSAA" in (t or "")
    ]


def load_page_titles(pdf_path: str, candidate_indices, pypdf_pages) -> list[str]:
    """Per-page text for section detection.

    Candidate divider pages are extracted with NaturalPDF (which captures the
    large-font sport title); all other pages reuse the already-loaded pypdf
    text (only the back-matter scan reads them). Keeps NaturalPDF to ~8 pages
    instead of the full PDF.
    """
    import natural_pdf as npdf

    pdf = npdf.PDF(pdf_path)
    texts = list(pypdf_pages)
    for i in candidate_indices:
        texts[i] = pdf.pages[i].extract_text() or texts[i]
    return texts


def detect_season(pdf_path: str, override: Optional[str] = None) -> str:
    """Detect the season from the PDF filename, or use an explicit override.

    Raises ValueError if no season can be determined — silently defaulting to
    fall would let a mis-named PDF clobber data/fall/.
    """
    if override:
        if override not in SEASON_SECTIONS:
            raise ValueError(f"Unknown season {override!r}; expected one of {list(SEASON_SECTIONS)}")
        return override
    name = Path(pdf_path).stem.lower()
    for season in ("fall", "winter", "spring"):
        if season in name:
            return season
    raise ValueError(
        f"Could not detect season from filename {pdf_path!r}; pass --season fall|winter|spring"
    )


# ── Page classifiers ──────────────────────────────────────────────────────────


def is_school_records(text: str) -> bool:
    return bool(re.search(r"\bCh:\s*\d{4}", text, re.IGNORECASE))


def is_year_class_table(text: str) -> bool:
    """Championship table with a YEAR CLASS … CHAMPION header (e.g. indoor track: 'Year Class Team Champion').

    Accepts either whitespace or a slash between YEAR and CLASS: the Wrestling
    Dual Meet Championship header reads "YEAR/CLASS CHAMPION".
    """
    return bool(re.search(r"YEAR[/\s]+CLASS\s+(?:\w+\s+)*CHAMPION", text, re.IGNORECASE))


def is_multicolumn_results(text: str) -> bool:
    """Multi-column champion table (football, soccer, volleyball, lacrosse, etc.)."""
    return bool(
        re.search(r"CLASS\s+(?:4A|AA)\s+CLASS\s+(?:3A|A)", text)
        or re.search(r"Class\s+\dA-\dA\s+Class\s+\dA-\dA", text)
    )


def is_individual_xc(text: str) -> bool:
    return bool(re.search(r"\d+\.\d+\s+MILES?|3\.0 MILES?\s", text, re.IGNORECASE))


def is_individual_results(text: str) -> bool:
    """Detect individual event champion pages (track, swimming, tennis, XC)."""
    return bool(
        re.search(r"Athlete[-—]School[-—](?:Mark|Score)", text)
        or re.search(r"(?:Singles|Doubles)\s+Champion", text)
        or re.search(r"\d{4}\s+\d[A-Z](?:-\d[A-Z])?\s+[\w\s]+,\s+[\w\s]+\d+:\d+", text)
        or re.search(r"\d+\.\d+\s+MILES?", text, re.IGNORECASE)
    )


def is_wrestling_weightclass(text: str) -> bool:
    r"""Detect a Wrestling weight-class champion page.

    These pages have no table header. The signal is a standalone weight-class
    number on its own line (98-285, optionally followed by "(con't.)" for a
    continuation page) IMMEDIATELY followed by a 2-digit year and a
    "Name, School" champion line, e.g.:
        106
        12 Austin Shaffer, Southern-Garrett
    Requiring the year+name to follow the weight number excludes pages whose
    only standalone numbers are page-number headers, and excludes
    team-championship tables (whose numbers are inline scores). Gated on the
    section's sport being Wrestling by the caller.
    """
    return bool(re.search(
        r"(?:^|\n)\s*\d{2,3}\s*(?:\(con['’]?t\.?\))?\s*\n"
        r"\s*\d{2}\s+[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+)*,\s+[A-Z]",
        text))


def is_sportsmanship(text: str) -> bool:
    # Require the heading near the top of the page to avoid incidental mentions
    return bool(re.search(r"SPORTSMANSHIP AWARD", text[:500], re.IGNORECASE)) and bool(
        re.search(r"\b(20|19)\d{2}\b", text)
    )


def is_golf_results(text: str) -> bool:
    # Match golf format: "Team Champion......School (score)" and the split-era
    # "Team Champion 1A/2A..........School (score)". The dot-leader requirement
    # excludes swimming's "Year Class Team Champion Coach" header (no dots).
    return bool(re.search(r"Team Champion\b[^\n]*\.{3,}", text))


# ── Page routing ──────────────────────────────────────────────────────────────
# Classifiers a page can be routed to. ``school_records`` is independent — a
# page may carry both school-record stats and a championship table; every other
# route is mutually exclusive, evaluated in priority order below.

ROUTE_ORDER = ("golf", "sportsmanship", "individual_results", "individual_xc", "championship")


def classify_page(text: str, sport: str) -> set[str]:
    """Return the set of extraction routes a page should be sent down.

    Golf is only considered when the section's sport is Golf, so a stray
    "Team Champion..." line in another sport's pages can't misroute.
    """
    routes: set[str] = set()
    if is_school_records(text):
        routes.add("school_records")

    if sport == "Golf" and is_golf_results(text):
        routes.add("golf")
    elif is_sportsmanship(text):
        routes.add("sportsmanship")
    elif is_individual_results(text) and "Cross Country" not in sport:
        routes.add("individual_results")
    elif is_individual_xc(text):
        routes.add("individual_xc")
    elif is_year_class_table(text) or is_multicolumn_results(text):
        routes.add("championship")
    # Wrestling weight-class champion pages have no table header, so the
    # classifiers above miss them. Route them to individual_results with the
    # sport-aware wrestling prompt (event = weight class). A page already
    # routed to a mutually-exclusive route is left alone.
    if (sport == "Wrestling" and is_wrestling_weightclass(text)
            and not (routes & {"championship", "sportsmanship", "golf",
                               "individual_xc", "individual_results"})):
        routes.add("individual_results")
    return routes


# ── Deduplication ────────────────────────────────────────────────────────────
# Per-table natural keys. Recovered from the lost verify_record_book.py pyc and
# adjusted: individual_xc_champions adds ``name`` so real co-champions (same
# year/class, different runner) are both kept instead of one being dropped.

DEDUP_KEYS: dict[str, tuple[str, ...]] = {
    "championship_results": ("sport", "year", "classification", "champion_school"),
    "school_records": ("sport", "school"),
    "individual_xc_champions": ("sport", "year", "classification", "name"),
    "individual_results": ("sport", "event", "year", "classification"),
    "golf_results": ("year", "classification", "individual_gender"),
    "sportsmanship_awards": ("sport", "year", "classification", "school"),
}


def dedup(rows: list[dict], key_fields: tuple[str, ...], table_name: str = "") -> tuple[list[dict], list[str]]:
    """Drop rows sharing a natural key, keeping the first occurrence.

    Returns (unique_rows, warnings). A warning is emitted when two rows share a
    key but differ in payload — that signals an extraction inconsistency (or a
    real tie the key doesn't distinguish) and should be reviewed by hand.
    """
    seen: dict[tuple, dict] = {}
    unique: list[dict] = []
    warnings: list[str] = []
    for r in rows:
        key = tuple(str(r.get(f, "")) for f in key_fields)
        if key in seen:
            if seen[key] != r:
                warnings.append(
                    f"{table_name}: same key {key} with differing payloads; kept first"
                )
            continue
        seen[key] = r
        unique.append(r)
    return unique, warnings


# ── School records (regex) ────────────────────────────────────────────────────


def parse_school_records(pages_text: list[str], sport: str) -> list[dict]:
    """
    Extract Ch/Fn/Sf/RU years from school record pages using line-by-line parsing.
    Handles: 'Ch: 1997, 1998' and wrapped years across multiple lines.
    """
    combined = "\n".join(pages_text)
    records: list[dict] = []

    # A school name: either all-caps (e.g. "ALLEGANY") or mixed-case with
    # parenthetical stats (e.g. "Aberdeen (16, 7-15)" in Football records)
    school_re = re.compile(
        r"^(?!YEAR|CLASS|MPSSAA|HONOR ROLL|TOURNAMENTS|STATE|PREVIOUS|PUBLIC|SOCCER)"
        r"(?:[A-Z][A-Z\s\.\-\'/&]+$|[A-Z][A-Za-z\s\.\-\'/&]+\(\d)"
    )
    status_start_re = re.compile(r"^(Ch|Fn|Sf|RU|QF|RS|RR\d?|CH|SF|RU):")

    def get_years(block: list[str], code: str) -> list[int]:
        combined_block = " ".join(block)
        # Grab everything after "Code:" until the next code or end.
        # Allow parenthetical classifications like "(2AE)" between years.
        parts = re.findall(rf"(?i)\b{code}:\s*([\d,\s\n\(\)A-Za-z]+?)(?=\b(?:Ch|Fn|Sf|RU|QF|RS|RR\d?|CH|SF)\b:|$)", combined_block)
        years: list[int] = []
        for part in parts:
            years.extend(int(y) for y in re.findall(r"\d{4}", part))
        return sorted(set(years))

    current_school: Optional[str] = None
    current_block: list[str] = []

    def flush(school: str, block: list[str]) -> Optional[dict]:
        ch = get_years(block, "Ch")
        fn = get_years(block, "Fn")
        sf = get_years(block, "Sf")
        ru = get_years(block, "RU")
        qf = get_years(block, "Qf")
        if ch or fn or ru:
            rec = {
                "sport": sport,
                "school": school,
                "champion_years": ch,
                "finalist_years": fn,
                "semifinalist_years": sf,
                "runner_up_years": ru,
            }
            if qf:
                rec["quarterfinal_years"] = qf
            return rec
        return None

    for raw_line in combined.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip standalone page numbers
        if re.match(r"^\d+$", line):
            continue
        if school_re.match(line) and len(line) >= 3:
            if current_school and current_block:
                rec = flush(current_school, current_block)
                if rec:
                    records.append(rec)
            # Strip parenthetical stats like "(16, 7-15)" from Football records
            current_school = re.sub(r"\s*\([\d,\s\-]+\)\s*$", "", line).strip()
            current_block = []
        elif current_school:
            current_block.append(line)

    if current_school and current_block:
        rec = flush(current_school, current_block)
        if rec:
            records.append(rec)

    return records


# ── Championship results (LLM) ────────────────────────────────────────────────


def _clean_dot_leaders(text: str) -> str:
    """Replace dot leaders (......) with a single tab for cleaner LLM input."""
    return re.sub(r"\.{3,}", "\t", text)


def _clean_dot_leaders(text: str) -> str:
    """Replace dot leaders (......) with a single tab for cleaner LLM input."""
    return re.sub(r"\.{3,}", "\t", text)


# ── Prompt builders ───────────────────────────────────────────────────────────
# Each builder returns (prompt, schema). The prompt embeds the chunk's page text,
# so the cache key (below) captures the exact text a row was extracted from.

def _championship_prompt(pages: list[str], sport: str) -> tuple[str, type]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(_clean_dot_leaders(p) for p in pages)

    # Wrestling has TWO distinct team championships in the same section. The
    # individual-tournament team champion (1970-2017, "Honor Roll of Team
    # Champions" / "Tournament History" pages) and the Dual Meet Championship
    # (1994-2025, "Dual Meet Championships" pages) ran concurrently for
    # 1994-2017, so a year can carry two different champion schools. Tag the
    # dual-meet rows in `notes` so the two titles stay distinguishable in
    # championship_results.
    if sport == "Wrestling":
        wrestling_note = (
            "- Wrestling note: there are TWO separate team championships in this section. "
            "Pages titled \"Dual Meet\" are the MPSSAA Dual Meet Championship — for every row "
            "extracted from those pages, set notes=\"Dual Meet Championship\". Pages titled "
            "\"Honor Roll of Team Champions\" or \"Tournament History\" are the individual "
            "State Tournament team championship — leave notes null for those rows. Both use "
            "sport=\"Wrestling\"."
        )
    else:
        wrestling_note = ""

    prompt = textwrap.dedent(f"""
        Extract every state championship final result from this MPSSAA {sport} record book text.

        Rules:
        - One row per championship final per year per classification.
        - score: the final game score string (e.g. "2-0", "3-1 OT", "28-14").  Null if not shown.
        - If the champion's name is preceded by *, set champion_undefeated=true and strip the *.
        - TIE or co-champion: set co_champion=true; put both school names in champion_school
          (e.g. "James M. Bennett & Linganore").
        - Equal scores (genuine tie): when two schools post the SAME winning score in the
          same year/class (e.g. "Winston Churchill—98 ... Central-PG—98") and there is NO
          tie-breaker note (no "tie-breaker", "defeated", "playoff", or similar wording
          that says how first place was decided), they are CO-CHAMPIONS: set
          co_champion=true and join both names in champion_school with " & "
          (e.g. "Winston Churchill & Central-PG"); leave finalist_school null.
        - Tie-breaker-decided tie: when two schools post the same score but a note says a
          tie-breaker decided first place (e.g. "Damascus—84 ... Catonsville—84 (sixth girl
          tie-breaker used to determine first place)"), there is ONE champion — the
          first-listed school (the tie-breaker winner) goes in champion_school with
          co_champion=false, and the other school goes in finalist_school. Keep the note.
        - classification: use the raw text value (e.g. "4A", "AA", "Combined", "1A/2A", "A", "B").
        - If a coach is not listed, set the field to null.
        - In multi-column tables (CLASS 4A  CLASS 3A  CLASS 2A  CLASS 1A), each column is a
          separate row with its own classification.  The school name and score are on one line;
          the coach name is on the following line directly below that column.
        - sport must always be exactly: {sport}
        - Do not invent data. Skip section headers, stats, and non-championship content.
        - The text may use dot leaders (......) between fields. Ignore the dots and extract the data.
        {wrestling_note}

        TEXT:
        {combined}
    """).strip()
    return prompt, ChampionshipResults


def _individual_xc_prompt(pages: list[str], sport: str) -> tuple[str, type]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(_clean_dot_leaders(p) for p in pages)
    prompt = textwrap.dedent(f"""
        Extract every individual state cross country champion from this MPSSAA {sport} text.

        Rules:
        - One row per champion: year, classification, name, school, time, distance.
        - distance: course distance (e.g. "3.0 Miles", "2.5 Miles"). Null if not stated.
        - classification: raw value (e.g. "4A", "1A", "AA", "ABC", "Combined").
        - time: the finishing time string (e.g. "19:17", "15:34.98").
        - sport must always be exactly: {sport}
        - Skip team results, records tables, and top-ten lists (only include individual
          champions, i.e. the first-place finisher each year per class).

        TEXT:
        {combined}
    """).strip()
    return prompt, IndividualChampions


def _individual_results_prompt(pages: list[str], sport: str) -> tuple[str, type]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(pages)
    if sport == "Wrestling":
        # Boys and girls wrestling share weight-class numbers (e.g. both have a
        # "120"), and the IndividualResults schema has no gender field, so the
        # dedup key (sport, event, year, classification) would collide and drop
        # one. The girls section is a single page carrying the header "Girls
        # Individual Champions 2020-2025"; every other weight-class page is boys.
        # Prefix the event with the gender so boys and girls keys stay distinct.
        is_girls = bool(re.search(r"Girls Individual Champions", combined))
        section = "Girls" if is_girls else "Boys"
        prompt = textwrap.dedent(f"""
            Extract every individual state champion from this MPSSAA Wrestling record book.

            This text is from the {section.upper()} wrestling section. It is organized by
            WEIGHT CLASS, not by a table header. Read it carefully.

            Rules:
            - event: the gender prefix "{section} " followed by the weight class, shown
              as a standalone number on its own line (e.g. "{section} 106", "{section} 113",
              "{section} 145", "{section} 285"). A line like "145 (con't.)" means that
              weight class continues from the previous page — the event is still
              "{section} 145". If the heaviest class is labeled "Heavyweight" instead of
              a number, use "{section} Heavyweight".
            - Under each weight class, years appear as 2-DIGIT numbers immediately
              followed by the champion, e.g. "12 Austin Shaffer, Southern-Garrett".
              Convert 2-digit years to 4-digit: years 70-99 mean 1970-1999; years
              00-25 mean 2000-2025.
            - Each year lists TWO people: the CHAMPION on the year line, then the
              RUNNER-UP on the following line (which has NO year number). Keep ONLY the
              champion (the name on the year line). Do NOT extract the runner-up.
            - classification: wrestling is single-class — always "Combined".
            - mark: null (there is no mark/score).
            - name: athlete name only (no school). school: the champion's school.
            - Skip any year where the champion is replaced by a COVID cancellation
              note (e.g. "The season was cancelled due to the COVID-19 pandemic.") —
              emit NO row for that year.
            - sport must always be exactly: Wrestling.
            - One row per champion: sport, event, year, classification, name, school, mark.

            TEXT:
            {combined}
        """).strip()
        return prompt, IndividualResults
    prompt = textwrap.dedent(f"""
        Extract every individual event state champion from this MPSSAA {sport} text.

        Rules:
        - One row per champion: sport, event, year, classification, name, school, mark.
        - event: the specific event name (e.g. "55m Dash", "200 Yard Freestyle", "Boys Singles",
          "Shot Put", "High Jump"). Extract from section headers like "Event: 55m Dash".
        - mark: the performance value (time, distance, score). Examples: "6.6", "11:03.93",
          "5-09", "38-09 1/4", "1:46.72", "(6-3, 6-3)". Null if not stated.
        - classification: raw value (e.g. "4A", "3A", "2A", "1A", "4A-3A", "3A-2A-1A").
        - name: athlete name only (no school). For doubles/relay, join names with " & ".
        - sport must always be exactly: {sport}
        - Skip all-time records lists (those showing the single best performance ever).
          Only include year-by-year state champions (the winner each year per class).
        - Skip cancelled seasons (COVID etc.).

        TEXT:
        {combined}
    """).strip()
    return prompt, IndividualResults


def _golf_prompt(pages: list[str], sport: str) -> tuple[str, type]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(_clean_dot_leaders(p) for p in pages)
    prompt = textwrap.dedent("""
        Extract every annual Golf state championship result from this MPSSAA Golf record book.

        Rules:
        - classification: "Combined" (1971-1992 single champion era), "1A/2A", or "3A/4A"
          (split era from 1993 onward).
        - team_score: total team strokes (integer). Null if not shown.
        - individual_score: individual total strokes (integer). Null if not shown.
        - When years have both a male and female individual winner, produce two rows per
          classification: one with individual_gender="male" and one with individual_gender="female".
        - If only one individual winner, individual_gender may be null.
        - Do not invent data.

        TEXT:
        {combined}
    """).strip().format(combined=combined)
    return prompt, GolfResults


def _sportsmanship_prompt(pages: list[str], sport: str) -> tuple[str, type]:
    combined = "\n".join(pages)

    # The Boys Soccer page lists both Boys and Girls soccer winners
    # (e.g. "2002 Boys—Winston Churchill  2002 Girls—Century")
    dual = bool(re.search(r"\bBoys[—–-]", combined) and re.search(r"\bGirls[—–-]", combined))
    if dual:
        sport_instruction = (
            'sport: set to "Boys Soccer" for entries labelled "Boys" '
            'and "Girls Soccer" for entries labelled "Girls".'
        )
    else:
        sport_instruction = f"sport must always be exactly: {sport}"

    prompt = textwrap.dedent(f"""
        Extract every sportsmanship award winner from this text.

        Rules:
        - One row per winner: sport, year, classification, school.
        - classification: "4A", "3A", "2A", "1A", or null if no classification is given.
        - {sport_instruction}
        - If a year is listed as cancelled (COVID etc.), skip it.
        - Co-winners: when a single year lists two schools as joint winners
          (e.g. "2003—Dulaney & North County" or "2022—Century & Damascus"),
          emit ONE row with the school field joining both names with " & "
          (e.g. school="Dulaney & North County"). Do NOT split them into
          separate rows — the award is one award with two co-winners.

        TEXT:
        {combined}
    """).strip()
    return prompt, SportsmanshipAwards


# route → (prompt builder, schema, result-key within the parsed JSON)
EXTRACTORS: dict[str, tuple] = {
    "championship":        (_championship_prompt, ChampionshipResults, "results"),
    "individual_xc":        (_individual_xc_prompt, IndividualChampions, "champions"),
    "individual_results":  (_individual_results_prompt, IndividualResults, "results"),
    "golf":                 (_golf_prompt, GolfResults, "results"),
    "sportsmanship":       (_sportsmanship_prompt, SportsmanshipAwards, "awards"),
}


# ── Extraction cache & provenance ─────────────────────────────────────────────
# A committed, page-level cache so re-runs are free and a worse extraction can't
# silently clobber a better one. Keyed on model + schema + prompt (which embeds the
# page text), so schema/prompt/model changes auto-invalidate and text-based keying
# survives edition page shifts. Provenance (source_pdf, source_pages, extracted_at,
# extraction_model) is stamped on every row from the current run's location.

CACHE_DIR = Path("cache/extractions")
PROVENANCE_FIELDS = ["source_pdf", "source_pages", "extracted_at", "extraction_model"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_key(model_id: str, schema, prompt: str) -> str:
    """sha256 over model + schema JSON + prompt (which embeds the page text)."""
    schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x1f")
    h.update(schema_json.encode("utf-8"))
    h.update(b"\x1f")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def _cache_path(extractor: str, key: str) -> Path:
    return CACHE_DIR / f"{extractor}_{key[:16]}.json"


def _stamp(rows: list[dict], source_pdf, source_pages: list[int],
           extracted_at: str, model: str) -> None:
    """Add provenance fields to every row (in place)."""
    for r in rows:
        r["source_pdf"] = str(source_pdf)
        r["source_pages"] = source_pages
        r["extracted_at"] = extracted_at
        r["extraction_model"] = model


def cached_extract(extractor: str, pages: list[str], sport: str,
                  source_pdf, source_pages: list[int], *,
                  refresh: bool = False, offline: bool = False) -> list[dict]:
    """Return provenance-stamped rows for one chunk, using the cache when possible.

    - Cache hit (and not --refresh): rows are read from disk; provenance is
      re-stamped from this run's source_pdf/pages (text-based keying means a hit
      can come from a different edition, so location provenance must follow the
      current run, not the original).
    - Cache miss + --offline: raise (CI proves data == f(cache) with no API key).
    - Cache miss otherwise: call the LLM, write the raw rows to cache, then stamp.
    """
    build_prompt, schema, result_key = EXTRACTORS[extractor]
    prompt, _schema = build_prompt(pages, sport)
    key = _cache_key(MODEL_ID, schema, prompt)
    path = _cache_path(extractor, key)

    extracted_at = _now()
    if not refresh and path.exists():
        blob = json.loads(path.read_text())
        rows = blob.get("rows", [])
        extracted_at = blob.get("meta", {}).get("extracted_at", extracted_at)
    elif offline:
        raise RuntimeError(
            f"cache miss for {extractor} ({path.name}); --offline forbids an API call"
        )
    else:
        data = llm_extract(prompt, schema)
        rows = data.get(result_key, [])
        extracted_at = _now()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "meta": {
                "extractor": extractor,
                "model_id": MODEL_ID,
                "source_pdf": str(source_pdf),
                "source_pages": source_pages,
                "extracted_at": extracted_at,
            },
            "rows": rows,
        }, indent=2))
        print(f"    · cache write {path.name}")

    _stamp(rows, source_pdf, source_pages, extracted_at, MODEL_ID)
    return rows


# ── CSV helpers ───────────────────────────────────────────────────────────────


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for k, v in row.items():
                if isinstance(v, list):
                    # Year lists etc. → "1997; 1998" instead of Python repr "[1997, 1998]"
                    out[k] = "; ".join(str(x) for x in v)
                else:
                    out[k] = v
            writer.writerow(out)
    print(f"  ✓ {len(rows):4d} rows  →  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

CHUNK = 2  # default max pages per LLM call — smaller chunks improve Haiku reliability

# Dense year-list pages (individual results, XC, golf) are where Haiku drops
# rows; extract those one page at a time. Championship tables are safer at 2.
CHUNK_SIZES = {
    "championship": 2,
    "individual_xc": 1,
    "individual_results": 1,
    "golf": 1,
    # Sportsmanship: all pages in one call so the dual Boys/Girls detection in
    # the prompt sees every page of the sport together.
    "sportsmanship": 64,
}


def chunked(lst: list, size: int, overlap: int = 0) -> list[list]:
    """Split list into non-overlapping chunks.

    Overlap is supported (for diagnostics) but defaults to 0: the old overlap=1
    default extracted every interior page twice and seeded duplicates.
    """
    if not lst:
        return []
    step = max(size - overlap, 1)
    return [lst[i : i + size] for i in range(0, len(lst), step)]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse MPSSAA record book PDFs into structured CSV/JSON.",
    )
    p.add_argument("pdf", nargs="?", default="pdfs/FallRecordBook2024.pdf",
                   help="PDF path (default: pdfs/FallRecordBook2024.pdf)")
    p.add_argument("--out", default=None,
                   help="output directory (default: data/<season>)")
    p.add_argument("--season", default=None,
                   help="force season fall|winter|spring (overrides filename detection)")
    p.add_argument("--routes", action="store_true",
                   help="dry-run: print a page→classifier table and exit (no LLM cost)")
    p.add_argument("--refresh", action="store_true",
                   help="bypass the extraction cache and re-call the LLM")
    p.add_argument("--offline", action="store_true",
                   help="error if a cache miss would require an LLM call (CI mode)")
    return p.parse_args(argv)


def route_pages(pages: list[str], sport: str, start: int, end: int) -> dict[str, list[tuple[int, str]]]:
    """Classify every page in a section, returning {route: [(page_index, text), ...]}.

    The 0-based page index is carried through so provenance can record the real
    PDF page a row came from. A page may appear under ``school_records`` and one
    other route (dual-content pages).
    """
    routes: dict[str, list[tuple[int, str]]] = {}
    for idx in range(start, end):
        text = pages[idx]
        for route in classify_page(text, sport):
            routes.setdefault(route, []).append((idx, text))
    return routes


def print_routes(pages: list[str], sections: dict[str, tuple[int, int]]) -> None:
    """Dry-run routing report: page → classifier(s), per sport section."""
    print(f"{'idx':>4} {'p#':>3}  {'sport':<24} routes")
    print("-" * 64)
    for sport, (start, end) in sections.items():
        per_route: dict[str, int] = {}
        for offset, text in enumerate(pages[start:end]):
            idx = start + offset
            routes = classify_page(text, sport)
            label = ", ".join(sorted(routes)) if routes else "—"
            print(f"{idx:>4} {idx + 1:>3}  {sport:<24} {label}")
            for r in routes:
                per_route[r] = per_route.get(r, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(per_route.items())) or "no data pages"
        print(f"{'':>4} {'':>3}  {'SUMMARY':<24} {summary}\n")


def _extract_route(route: str, pairs: list[tuple[int, str]], sport: str, pdf,
                   *, refresh: bool, offline: bool) -> list[dict]:
    """Chunk (page_index, text) pairs and extract each via the cache."""
    rows: list[dict] = []
    for chunk in chunked(pairs, CHUNK_SIZES.get(route, 1)):
        texts = [t for _, t in chunk]
        page_nums = sorted({i + 1 for i, _ in chunk})  # 1-based for provenance
        rows.extend(cached_extract(route, texts, sport, pdf, page_nums,
                                   refresh=refresh, offline=offline))
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    season = detect_season(args.pdf, override=args.season)
    out_dir = Path(args.out) if args.out else Path("data") / season

    print(f"Loading {args.pdf} … (season: {season})")
    pages = load_pages(args.pdf)
    print(f"  {len(pages)} pages loaded.")

    # Detect section boundaries from divider-page titles (NaturalPDF) instead
    # of trusting the hardcoded index map, so a new edition that shifts pages
    # still slices correctly. Validates the detected sport set against the
    # baseline map and raises on drift.
    candidates = _divider_candidates(pages)
    page_texts = load_page_titles(args.pdf, candidates, pages)
    sections = find_sections(page_texts, season)
    print(f"  detected {len(sections)} sport sections via divider pages.\n")

    if args.routes:
        print_routes(pages, sections)
        return

    all_championship: list[dict] = []
    all_school_records: list[dict] = []
    all_individual_xc: list[dict] = []
    all_individual: list[dict] = []
    all_sportsmanship: list[dict] = []
    all_golf: list[dict] = []

    for sport, (start, end) in sections.items():
        print(f"── {sport}  (PDF indices {start}–{end - 1}) ──")
        routed = route_pages(pages, sport, start, end)

        # School records (regex, no LLM, no cache)
        if routed.get("school_records"):
            pairs = routed["school_records"]
            recs = parse_school_records([t for _, t in pairs], sport)
            sr_pages = sorted({i + 1 for i, _ in pairs})
            _stamp(recs, args.pdf, sr_pages, _now(), "regex")
            print(f"  school records     : {len(recs)} schools  "
                  f"({len(pairs)} pages, regex)")
            all_school_records.extend(recs)

        # Championship results (LLM)
        if routed.get("championship"):
            pairs = routed["championship"]
            rows = _extract_route("championship", pairs, sport, args.pdf,
                                  refresh=args.refresh, offline=args.offline)
            all_championship.extend(rows)
            print(f"  championship table : {len(rows)} rows  "
                  f"({len(pairs)} pages, LLM)")

        # Individual XC (LLM) — cross country only
        if routed.get("individual_xc") and "Cross Country" in sport:
            pairs = routed["individual_xc"]
            rows = _extract_route("individual_xc", pairs, sport, args.pdf,
                                   refresh=args.refresh, offline=args.offline)
            all_individual_xc.extend(rows)
            print(f"  individual XC      : {len(rows)} rows  "
                  f"({len(pairs)} pages, LLM)")

        # Individual event results (LLM) — track, swimming, tennis
        if routed.get("individual_results"):
            pairs = routed["individual_results"]
            rows = _extract_route("individual_results", pairs, sport, args.pdf,
                                  refresh=args.refresh, offline=args.offline)
            all_individual.extend(rows)
            print(f"  individual events  : {len(rows)} rows  "
                  f"({len(pairs)} pages, LLM)")

        # Golf (LLM)
        if routed.get("golf"):
            pairs = routed["golf"]
            rows = _extract_route("golf", pairs, sport, args.pdf,
                                  refresh=args.refresh, offline=args.offline)
            all_golf.extend(rows)
            print(f"  golf results       : {len(rows)} rows  "
                  f"({len(pairs)} pages, LLM)")

        # Sportsmanship (LLM) — all pages in one call (dual Boys/Girls detection)
        if routed.get("sportsmanship"):
            pairs = routed["sportsmanship"]
            rows = _extract_route("sportsmanship", pairs, sport, args.pdf,
                                  refresh=args.refresh, offline=args.offline)
            all_sportsmanship.extend(rows)
            print(f"  sportsmanship      : {len(rows)} awards  "
                  f"({len(pairs)} pages, LLM)")

        print()

    # ── Normalise classification labels (post-extraction, pre-dedup) ────────
    # The LLM occasionally emits "CLASS 1A" or "B*" for the class label; the
    # cache stores that raw form, and we canonicalise here so both the data
    # run and `make rebuild-offline` produce identical bare labels ("1A", "B").
    for rows in (all_championship, all_individual_xc, all_individual,
                 all_golf, all_sportsmanship):
        for r in rows:
            if "classification" in r:
                r["classification"] = _normalize_classification(r["classification"])

    # ── Normalise event names (post-extraction, pre-dedup) ──────────────────
    # The model emits the same short track event as "55m" / "55 m" / "55m Dash"
    # on different chunks, fragmenting the (sport, year, classification, event)
    # natural key. Canonicalise to plain "55 m" (matching "1600 m" etc.) so the
    # winter individual_results table doesn't carry phantom duplicate events.
    for r in all_individual:
        if "event" in r:
            r["event"] = _normalize_event_name(r["event"])

    # ── Tag pre-MPSSAA precursor tournaments (post-extraction, pre-dedup) ──
    # The record books lead several sports' championship pages with a section
    # headed "TOURNAMENTS PRIOR TO MPSSAA SPONSORSHIP" / "PRE-MPSSAA" /
    # "NOT SPONSORED BY MPSSAA" — public-school precursor tournaments run
    # before MPSSAA took over. The LLM extracts them into championship_results
    # alongside the real MPSSAA championships. They are not MPSSAA
    # championships, so tag each precursor row with notes="Pre-MPSSAA" to keep
    # it distinguishable (verify then excludes tagged rows from its continuity
    # and referential checks, since precursor gaps and unlisted schools are
    # expected). Done deterministically by the sport's verified MPSSAA-era
    # start year (the "UNDER THE DIRECTION OF MPSSAA" cut-point in the PDF)
    # rather than by re-extraction, which risks the LLM dropping MPSSAA-era
    # rows on these dense shared pages. Mirrors the ERA_FLOORS precedent.
    _tag_pre_mpssaa(all_championship)

    # ── De-duplicate all six tables by natural key ───────────────────────────
    raw_tables = {
        "championship_results": all_championship,
        "school_records": all_school_records,
        "individual_xc_champions": all_individual_xc,
        "individual_results": all_individual,
        "golf_results": all_golf,
        "sportsmanship_awards": all_sportsmanship,
    }
    tables: dict[str, list[dict]] = {}
    warnings: list[str] = []
    for name, rows in raw_tables.items():
        tables[name], warns = dedup(rows, DEDUP_KEYS[name], name)
        warnings.extend(warns)
    for w in warnings:
        print(f"  ! dedup {w}")
    print("Dedup: " + ", ".join(
        f"{n} {len(raw_tables[n])}→{len(tables[n])}" for n in raw_tables) + "\n")

    # ── Write outputs (provenance columns trail each table) ──────────────────
    pf = PROVENANCE_FIELDS
    write_csv(
        out_dir / "championship_results.csv",
        tables["championship_results"],
        ["sport", "year", "classification",
         "champion_school", "champion_coach",
         "finalist_school", "finalist_coach",
         "score", "champion_undefeated", "co_champion", "notes"] + pf,
    )
    write_csv(
        out_dir / "school_records.csv",
        tables["school_records"],
        ["sport", "school", "champion_years", "finalist_years",
         "semifinalist_years", "runner_up_years", "quarterfinal_years"] + pf,
    )
    write_csv(
        out_dir / "individual_results.csv",
        tables["individual_results"],
        ["sport", "event", "year", "classification", "name", "school", "mark"] + pf,
    )
    write_csv(
        out_dir / "sportsmanship_awards.csv",
        tables["sportsmanship_awards"],
        ["sport", "year", "classification", "school"] + pf,
    )
    # XC and golf are fall-only sports. Outside fall they are always empty, so
    # skip writing their CSVs there to avoid littering data/winter|spring/ with
    # header-only files. The record_book.json below still carries every table
    # (as an empty list) so verify's regression guard and the diff stay stable.
    if season == "fall" or tables["individual_xc_champions"]:
        write_csv(
            out_dir / "individual_xc_champions.csv",
            tables["individual_xc_champions"],
            ["sport", "year", "classification", "name", "school", "time",
             "distance"] + pf,
        )
    if season == "fall" or tables["golf_results"]:
        write_csv(
            out_dir / "golf_results.csv",
            tables["golf_results"],
            ["year", "classification",
             "team_champion_school", "team_score",
             "individual_winner_name", "individual_winner_school",
             "individual_score", "individual_gender"] + pf,
        )

    record_book = {
        "meta": {
            "source_pdf": str(args.pdf),
            "season": season,
            "extraction_model": MODEL_ID,
            "generated_at": _now(),
        },
        "championship_results": tables["championship_results"],
        "school_records": tables["school_records"],
        "individual_xc_champions": tables["individual_xc_champions"],
        "individual_results": tables["individual_results"],
        "sportsmanship_awards": tables["sportsmanship_awards"],
        "golf_results": tables["golf_results"],
    }
    json_path = out_dir / "record_book.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record_book, indent=2, default=str))
    print(f"  ✓ record_book.json  →  {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
