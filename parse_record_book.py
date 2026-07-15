#!/usr/bin/env python3
"""
Parse MPSSAA Record Book PDFs (fall, winter, spring) into structured CSV / JSON data.

Usage:
    uv run parse_record_book.py [PDF_PATH] [OUTPUT_DIR]

Defaults:
    PDF_PATH   = pdfs/FallRecordBook2024.pdf
    OUTPUT_DIR = data/<season>   (auto-detected from PDF filename)

Requires:
    ANTHROPIC_API_KEY environment variable   (or `llm keys set anthropic`)
"""

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import llm
from pypdf import PdfReader
from pydantic import BaseModel, ValidationError

import verify_record_book

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


# ── LLM ───────────────────────────────────────────────────────────────────────

MODEL_ID = "anthropic/claude-haiku-4-5-20251001"
_model: Optional[llm.Model] = None


def get_model() -> llm.Model:
    global _model
    if _model is None:
        _model = llm.get_model(MODEL_ID)
    return _model


def _raw_response_dict(response) -> Optional[dict]:
    """Pull the model's structured output out of an llm response, or None.

    Handles both delivery paths without validating: JSON in the text body
    (Sonnet/Opus structured output) and the tool-call arguments fallback
    (Haiku's schema-as-tool mode).
    """
    text = response.text().strip()
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    if response.tool_calls():
        args = response.tool_calls()[0].arguments
        if isinstance(args, dict):
            return args
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            pass

    return None


class TruncationError(Exception):
    """The model hit its output-token limit, so the response is incomplete.

    Retrying the identical prompt just truncates again; the caller must shrink
    the input (fewer pages / split the text) and try the pieces separately.
    """


# Explicit output-token ceiling. The model already defaults to this; setting it
# here documents intent. Raising it only reduces how often a dense page has to
# be split — it is not a fix on its own, because any fixed cap can be exceeded.
MAX_TOKENS = 8192


def _was_truncated(response) -> bool:
    """True if the model stopped because it hit its output-token limit.

    Truncation drops trailing rows silently, which is exactly the failure mode
    behind the missing championship years. Anthropic reports this as
    stop_reason="max_tokens"; we search defensively since the `llm` library's
    stored JSON shape varies by version.
    """
    raw = getattr(response, "response_json", None)
    candidates = raw if isinstance(raw, list) else [raw]
    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in ("stop_reason", "finish_reason"):
            if item.get(key) in ("max_tokens", "length"):
                return True
    return False


def llm_extract(prompt: str, schema, retries: int = 2) -> dict:
    """Call the LLM with a Pydantic schema, validate the result, and return a dict.

    The raw model output is validated through the Pydantic `schema` before being
    returned, so field defaults are applied (e.g. champion_undefeated=False) and
    types are coerced (e.g. year "2024" → 2024). Output that does not conform to
    the schema triggers a retry and, if it never conforms, a loud error — bad
    rows never reach the CSV/JSON silently.

    Raises TruncationError if the model hit its output-token limit, so the caller
    can split the input rather than retry an identical (and identically
    truncated) prompt.
    """
    model = get_model()
    last_text = ""
    last_error = ""

    for attempt in range(1, retries + 1):
        response = model.prompt(prompt, schema=schema, stream=False, max_tokens=MAX_TOKENS)

        # Check truncation first: a truncated response is incomplete no matter
        # what it parsed to, and retrying the same prompt cannot help.
        if _was_truncated(response):
            raise TruncationError("model hit the output-token limit (stop_reason=max_tokens)")

        last_text = response.text().strip()
        data = _raw_response_dict(response)
        if data is not None:
            try:
                return schema.model_validate(data).model_dump()
            except ValidationError as exc:
                last_error = str(exc)
                if attempt < retries:
                    print(f"    (retry {attempt}/{retries} — schema validation failed)")
                    continue
        elif attempt < retries:
            print(f"    (retry {attempt}/{retries} — LLM returned no content)")

    raise RuntimeError(
        f"LLM returned no schema-valid content after {retries} attempts.\n"
        f"text={last_text!r}\n"
        f"validation_error={last_error!r}\n"
        f"response_json={response.response_json}"
    )


# ── PDF extraction ────────────────────────────────────────────────────────────


def load_pages(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


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
    "Girls Basketball":        (3, 11),
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


def detect_season(pdf_path: str) -> str:
    """Infer the season from the PDF filename.

    Raises ValueError when the filename matches no known season. Silently
    defaulting to "fall" would mis-slice a winter/spring PDF with the wrong
    page ranges and quietly produce wrong data; callers can pass an explicit
    --season instead.
    """
    name = Path(pdf_path).stem.lower()
    for season in ("fall", "winter", "spring"):
        if season in name:
            return season
    raise ValueError(
        f"Cannot detect season from filename {Path(pdf_path).name!r}. "
        f"Expected one of 'fall', 'winter', 'spring' in the name, "
        f"or pass --season explicitly."
    )


def _normalize_text(text: str) -> str:
    """Lowercase, treat '&' as 'and', and collapse whitespace (incl. line wraps)."""
    return re.sub(r"\s+", " ", text.replace("&", " and ")).lower()


def _section_keyword(sport: str) -> str:
    """The distinctive part of a sport name, minus any Girls/Boys prefix."""
    return _normalize_text(re.sub(r"^(?:Girls|Boys)\s+", "", sport))


def check_section_map(pages: list[str], sections: dict[str, tuple[int, int]]) -> list[str]:
    """Return a list of problems if the page ranges don't line up with `pages`.

    Each sport's name must appear somewhere within its assigned page range; if
    it doesn't, the hardcoded ranges are almost certainly stale for this PDF
    edition (page numbers shifted), which would silently mis-slice every sport.
    This is a cheap guard that turns a silent-wrong-data failure into a loud,
    actionable error.
    """
    problems: list[str] = []
    n = len(pages)
    for sport, (start, end) in sections.items():
        if start >= n:
            problems.append(
                f"{sport}: range {start}–{end - 1} starts past the last page ({n - 1})"
            )
            continue
        window = _normalize_text(" ".join(pages[start:end]))
        if _section_keyword(sport) not in window:
            problems.append(
                f"{sport}: name not found in pages {start}–{end - 1} "
                f"(page ranges may be stale for this PDF edition)"
            )
    return problems


# ── Page classifiers ──────────────────────────────────────────────────────────


def is_school_records(text: str) -> bool:
    return bool(re.search(r"\bCh:\s*\d{4}", text, re.IGNORECASE))


def is_year_class_table(text: str) -> bool:
    """Championship table with a YEAR CLASS … CHAMPION header (e.g. indoor track: 'Year Class Team Champion')."""
    return bool(re.search(r"YEAR\s+CLASS\s+(?:\w+\s+)*CHAMPION", text, re.IGNORECASE))


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


def is_sportsmanship(text: str) -> bool:
    # Require the heading near the top of the page to avoid incidental mentions
    return bool(re.search(r"SPORTSMANSHIP AWARD", text[:500], re.IGNORECASE)) and bool(
        re.search(r"\b(20|19)\d{2}\b", text)
    )


def is_golf_results(text: str) -> bool:
    # Golf lists "Team Champion......School (score)" in the Combined era and
    # "Team Champion 1A/2A......School (score)" in the split era (1993+), so allow
    # a short classification label between the phrase and the dot leaders. The
    # required dot leaders still exclude swimming's "Year Class Team Champion
    # Coach" header, which has none.
    return bool(re.search(r"Team Champion[^\n.]{0,10}\.{3,}", text))


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


def extract_championship_results(
    pages: list[str], sport: str
) -> list[dict]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(_clean_dot_leaders(p) for p in pages)
    prompt = textwrap.dedent(f"""
        Extract every state championship final result from this MPSSAA {sport} record book text.

        Rules:
        - One row per championship final per year per classification.
        - score: the final game score string (e.g. "2-0", "3-1 OT", "28-14").  Null if not shown.
        - If the champion's name is preceded by *, set champion_undefeated=true and strip the *.
        - TIE or co-champion: set co_champion=true; put both school names in champion_school
          (e.g. "James M. Bennett & Linganore").
        - classification: use the raw text value (e.g. "4A", "AA", "Combined", "1A/2A", "A", "B").
        - If a coach is not listed, set the field to null.
        - In multi-column tables (CLASS 4A  CLASS 3A  CLASS 2A  CLASS 1A), each column is a
          separate row with its own classification.  The school name and score are on one line;
          the coach name is on the following line directly below that column.
        - sport must always be exactly: {sport}
        - Do not invent data. Skip section headers, stats, and non-championship content.
        - The text may use dot leaders (......) between fields. Ignore the dots and extract the data.

        TEXT:
        {combined}
    """).strip()
    data = llm_extract(prompt, ChampionshipResults)
    return data.get("results", [])


# ── Individual XC champions (LLM) ────────────────────────────────────────────


def extract_individual_xc(pages: list[str], sport: str) -> list[dict]:
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
    data = llm_extract(prompt, IndividualChampions)
    return data.get("champions", [])


# ── Individual event results (LLM) ───────────────────────────────────────────


def extract_individual_results(pages: list[str], sport: str) -> list[dict]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(pages)
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
    data = llm_extract(prompt, IndividualResults)
    return data.get("results", [])


# ── Golf results (LLM) ────────────────────────────────────────────────────────


def extract_golf_results(pages: list[str]) -> list[dict]:
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
    data = llm_extract(prompt, GolfResults)
    return data.get("results", [])


# ── Sportsmanship awards (LLM) ────────────────────────────────────────────────


def extract_sportsmanship(pages: list[str], sport: str) -> list[dict]:
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

        TEXT:
        {combined}
    """).strip()
    data = llm_extract(prompt, SportsmanshipAwards)
    return data.get("awards", [])


# ── School name normalization ─────────────────────────────────────────────────

ALIASES_PATH = Path(__file__).with_name("school_aliases.json")

# School fields to canonicalize per table. Each gets its value replaced with the
# canonical name and a companion "<field>_slug" added.
SCHOOL_FIELDS = {
    "championship_results": ["champion_school", "finalist_school"],
    "school_records": ["school"],
    "individual_xc_champions": ["school"],
    "individual_results": ["school"],
    "golf_results": ["team_champion_school", "individual_winner_school"],
    "sportsmanship_awards": ["school"],
}


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    """Load the school alias map (casefolded variant → canonical name)."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        k.strip().casefold(): v
        for k, v in raw.items()
        if not k.startswith("_")
    }


def _titlecase_school(name: str) -> str:
    """Title-case an ALL-CAPS name, preserving Mc-, hyphen, and slash parts."""
    def fix_word(word: str) -> str:
        parts = re.split(r"([-'/.])", word)
        return "".join(p if p in "-'/." else p.capitalize() for p in parts)

    result = " ".join(fix_word(w) for w in name.split())
    return re.sub(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), result)


def normalize_school(raw: Optional[str], aliases: dict[str, str]) -> str:
    """Return the canonical name for a school as printed in a record book.

    Order of precedence, all accuracy-preserving:
      1. Explicit alias map (case-insensitive) — the only place two differently
         spelled names are ever merged, and every entry is hand-curated.
      2. ALL-CAPS names (how school-records pages print them) are title-cased so
         "ALLEGANY" matches the championship table's "Allegany".
      3. Mixed-case names pass through unchanged — they are already as the record
         book printed them, so we never risk corrupting them.
    Co-champion cells ("A & B") are normalized component-by-component.
    """
    name = re.sub(r"\s+", " ", (raw or "").strip())
    if not name:
        return ""
    if " & " in name:
        return " & ".join(normalize_school(p, aliases) for p in name.split(" & "))
    canonical = aliases.get(name.casefold())
    if canonical:
        return canonical
    if name.isupper():
        return _titlecase_school(name)
    return name


def slugify_school(name: str) -> str:
    """URL-safe slug for website routing, e.g. 'Bethesda-Chevy Chase' → 'bethesda-chevy-chase'."""
    s = name.casefold().replace("&", " and ").replace("'", "").replace(".", "")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def canonicalize_rows(
    rows: list[dict],
    fields: list[str],
    aliases: dict[str, str],
    name_map: dict[str, str],
) -> None:
    """Canonicalize each school field in-place and add a '<field>_slug' companion.

    Records every raw→canonical change in `name_map` for the audit file, so all
    name changes are inspectable in one place.
    """
    for row in rows:
        for field in fields:
            raw = row.get(field)
            if not raw:
                continue
            canonical = normalize_school(raw, aliases)
            if canonical != raw:
                name_map[raw] = canonical
            row[field] = canonical
            row[f"{field}_slug"] = slugify_school(canonical)


# ── School-record year serialization ─────────────────────────────────────────

# (list field on a school-records row, result label for the long format)
YEAR_FIELDS = [
    ("champion_years", "champion"),
    ("finalist_years", "finalist"),
    ("semifinalist_years", "semifinalist"),
    ("runner_up_years", "runner_up"),
    ("quarterfinal_years", "quarterfinal"),
]


def school_records_long(records: list[dict]) -> list[dict]:
    """Explode wide school-record rows into one (school, year, result) row each.

    Tidy long format is far easier to query, join, and chart than the wide
    file's year lists — e.g. "every champion in 2011" is a simple filter.
    """
    rows: list[dict] = []
    for r in records:
        for field, result in YEAR_FIELDS:
            for year in r.get(field, []) or []:
                rows.append(
                    {
                        "sport": r.get("sport", ""),
                        "school": r.get("school", ""),
                        "school_slug": r.get("school_slug", ""),
                        "year": year,
                        "result": result,
                        "source_pages": r.get("source_pages", ""),
                    }
                )
    return sorted(
        rows, key=lambda x: (x["sport"], x["school"], x["year"], x["result"])
    )


def _years_as_strings(records: list[dict]) -> list[dict]:
    """Copy of school-record rows with year lists joined by ';' for CSV output.

    JSON keeps the real arrays; only the wide CSV needs a flat scalar so it
    never serializes a Python list repr like "[1997, 1998]".
    """
    out: list[dict] = []
    for r in records:
        row = dict(r)
        for field, _ in YEAR_FIELDS:
            value = row.get(field)
            if isinstance(value, list):
                row[field] = ";".join(str(y) for y in value)
        out.append(row)
    return out


# ── CSV helpers ───────────────────────────────────────────────────────────────


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {len(rows):4d} rows  →  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

CHUNK = 2  # max pages per LLM call — smaller chunks improve Haiku reliability

# Row-level provenance fields, excluded from dedupe conflict detection (the same
# entry from two overlapping chunks legitimately carries different source_pages).
PROVENANCE_FIELDS = {"source_pages"}


def chunked(lst: list, size: int, overlap: int = 1) -> list[list]:
    """Split list into overlapping chunks."""
    if not lst:
        return []
    step = max(size - overlap, 1)
    return [lst[i : i + size] for i in range(0, len(lst), step)]


_ROW_YEAR_RE = re.compile(r"\s*(19[2-9]\d|20[0-2]\d)\b")


def _table_years_in_text(text: str) -> set[int]:
    """Years that begin a line — i.e. the year label of a championship-table row.

    Anchoring to line starts ignores incidental years in prose, records, and
    trivia, giving a clean lower bound on how many distinct years the table
    covers.
    """
    return {
        int(m.group(1))
        for line in text.splitlines()
        if (m := _ROW_YEAR_RE.match(line))
    }


def _years_in_rows(rows: list[dict]) -> set[int]:
    years: set[int] = set()
    for r in rows:
        try:
            years.add(int(r.get("year")))
        except (TypeError, ValueError):
            pass
    return years


def source_label(indices) -> str:
    """Render a set of 0-based PDF page indices as a compact source string.

    A single page → "62"; a contiguous or spanning chunk → "62-63". This is the
    row-level provenance stamped onto every extracted row so it can be checked
    against pages.jsonl (and the PDF) without re-parsing.
    """
    lo, hi = min(indices), max(indices)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _stamp(rows: list[dict], src: str) -> list[dict]:
    for r in rows:
        r.setdefault("source_pages", src)
    return rows


MIN_SPLIT_LINES = 6  # don't try to split a fragment smaller than this


def extract_resilient(texts: list[str], extract_fn, label: str) -> list[dict]:
    """Call extract_fn(texts); on truncation, shrink the input and retry.

    A dense page can produce more JSON than the model's output-token limit, so
    the response is truncated and unusable. Rather than crash the whole run, we
    split the input in half and extract the pieces separately, recursing until
    each piece fits: first by splitting the page list, then (for a single page)
    by splitting its lines. A fragment that still overflows at minimal size is
    logged loudly and skipped — the completeness guard and verify_record_book.py
    surface any resulting gap. Partial data beats aborting with none.
    """
    try:
        return extract_fn(texts)
    except (TruncationError, RuntimeError) as exc:
        if len(texts) > 1:
            mid = len(texts) // 2
            print(f"  ⚠ {label}: output too large for {len(texts)} pages — splitting")
            return (
                extract_resilient(texts[:mid], extract_fn, label)
                + extract_resilient(texts[mid:], extract_fn, label)
            )

        lines = texts[0].splitlines()
        if len(lines) >= MIN_SPLIT_LINES:
            mid = len(lines) // 2
            print(f"  ⚠ {label}: single page too large — splitting its text in half")
            return (
                extract_resilient(["\n".join(lines[:mid])], extract_fn, label)
                + extract_resilient(["\n".join(lines[mid:])], extract_fn, label)
            )

        print(f"  ⚠ {label}: fragment still overflows at minimal size — skipping ({exc})")
        return []


def extract_chunks_complete(
    indexed_pages: list[tuple[int, str]],
    extract_fn,
    label: str,
    threshold: float = 0.9,
) -> list[dict]:
    """Run `extract_fn` over chunked pages, then guard against silent row loss.

    `indexed_pages` is a list of (pdf_page_index, page_text) pairs; every row
    returned is stamped with a `source_pages` provenance string.

    The LLM occasionally drops most of a dense table (e.g. Boys Soccer captured
    only 15 of 76 championship years). We compare the distinct years the model
    returned against the distinct years visible at the start of table lines. If
    coverage is below `threshold`, we re-extract one page at a time (smaller
    inputs are far more reliable) and merge; duplicate rows are removed later by
    dedupe(). Any years still missing after the retry are reported loudly so a
    human can check them against the PDF — accuracy is never sacrificed silently.
    """
    rows: list[dict] = []
    for chunk in chunked(indexed_pages, CHUNK):
        indices = [i for i, _ in chunk]
        texts = [t for _, t in chunk]
        rows.extend(_stamp(extract_resilient(texts, extract_fn, label), source_label(indices)))

    expected: set[int] = set()
    for _, text in indexed_pages:
        expected |= _table_years_in_text(text)
    if not expected:
        return rows

    got = _years_in_rows(rows)
    covered = got & expected
    if len(covered) / len(expected) >= threshold:
        return rows

    print(
        f"  ⚠ {label}: captured only {len(covered)}/{len(expected)} table years "
        f"({len(covered) / len(expected):.0%}) — re-extracting page-by-page"
    )
    for idx, text in indexed_pages:
        rows.extend(_stamp(extract_resilient([text], extract_fn, label), str(idx)))

    still_missing = sorted(expected - _years_in_rows(rows))
    if still_missing:
        print(
            f"  ⚠ INCOMPLETE {label}: {len(still_missing)} year(s) not captured "
            f"after retry — verify against PDF: {still_missing}"
        )
    return rows


def dedupe(
    rows: list[dict], key_fields: tuple[str, ...], label: str = ""
) -> list[dict]:
    """Drop rows sharing a key, keeping the first occurrence.

    Chunks overlap by one page, so the same entry can be extracted twice; this
    removes those repeats. When two rows share a key but disagree on any other
    field, a CONFLICT warning names both — a silent first-wins choice could
    otherwise hide a real extraction discrepancy, and accuracy is the priority.
    """
    seen: dict[tuple, dict] = {}
    unique: list[dict] = []
    conflicts = 0

    for r in rows:
        key = tuple(r.get(k, "") for k in key_fields)
        if key not in seen:
            seen[key] = r
            unique.append(r)
            continue

        first = seen[key]
        differing = sorted(
            k
            for k in set(first) | set(r)
            if k not in key_fields
            and k not in PROVENANCE_FIELDS
            and first.get(k) != r.get(k)
        )
        if differing:
            conflicts += 1
            kept = ", ".join(f"{k}={first.get(k)!r}" for k in differing)
            dropped = ", ".join(f"{k}={r.get(k)!r}" for k in differing)
            print(f"  ⚠ CONFLICT [{label}] key={key}")
            print(f"      kept:    {kept}")
            print(f"      dropped: {dropped}")

    removed = len(rows) - len(unique)
    if removed:
        note = f" ({conflicts} with conflicting fields)" if conflicts else ""
        print(f"  {label}: {len(rows)} → {len(unique)} after dedup{note}")
    return unique


TABLE_NAMES = [
    "championship_results",
    "school_records",
    "individual_xc_champions",
    "individual_results",
    "sportsmanship_awards",
    "golf_results",
]


def git_commit() -> Optional[str]:
    """Short git commit of the parser, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_meta(season: str, pdf_path: str, tables: dict, report: dict) -> dict:
    """Self-describing metadata block for record_book.json."""
    return {
        "season": season,
        "source_pdf": Path(pdf_path).name,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "parser_commit": git_commit(),
        "row_counts": {name: len(tables.get(name, [])) for name in TABLE_NAMES},
        "verification": report.get("summary", {}),
    }


def write_combined_json(data_root: Path) -> Optional[Path]:
    """Merge every data/<season>/record_book.json into a single data/all.json.

    Each row is tagged with its season so the website can load one file and
    filter/group client-side. Reflects whatever seasons currently exist on disk.
    """
    combined: dict[str, list] = {name: [] for name in TABLE_NAMES}
    seasons: list[str] = []
    for season in SEASON_SECTIONS:
        book_path = data_root / season / "record_book.json"
        if not book_path.exists():
            continue
        seasons.append(season)
        book = json.loads(book_path.read_text())
        for name in TABLE_NAMES:
            for row in book.get(name, []):
                combined[name].append({"season": season, **row})
    if not seasons:
        return None

    out = {
        "meta": {
            "seasons": seasons,
            "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "row_counts": {name: len(rows) for name, rows in combined.items()},
        },
        **combined,
    }
    all_path = data_root / "all.json"
    all_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  ✓ combined {len(seasons)} season(s)  →  {all_path}")
    return all_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse an MPSSAA Record Book PDF into structured CSV / JSON."
    )
    parser.add_argument(
        "pdf_path", nargs="?", default="pdfs/FallRecordBook2024.pdf",
        help="Path to the record book PDF (default: pdfs/FallRecordBook2024.pdf)",
    )
    parser.add_argument(
        "out_dir", nargs="?", default=None,
        help="Output directory (default: data/<season>)",
    )
    parser.add_argument(
        "--season", choices=sorted(SEASON_SECTIONS), default=None,
        help="Override season detection (otherwise inferred from the filename).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf_path
    try:
        season = args.season or detect_season(pdf_path)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}")
    sections = SEASON_SECTIONS[season]
    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / season

    print(f"Loading {pdf_path} … (season: {season})")
    pages = load_pages(pdf_path)
    print(f"  {len(pages)} pages loaded.\n")

    section_problems = check_section_map(pages, sections)
    if section_problems:
        detail = "\n  - ".join(section_problems)
        raise SystemExit(
            f"Error: section page ranges do not match this PDF (season={season}):\n"
            f"  - {detail}\n"
            f"The hardcoded ranges in {season.upper()}_SECTIONS are likely stale for "
            f"this edition. Update them before parsing to avoid silently wrong data."
        )

    all_championship: list[dict] = []
    all_school_records: list[dict] = []
    all_individual_xc: list[dict] = []
    all_individual: list[dict] = []
    all_sportsmanship: list[dict] = []
    all_golf: list[dict] = []

    for sport, (start, end) in sections.items():
        sport_pages = pages[start:end]
        print(f"── {sport}  (PDF indices {start}–{end-1}) ──")

        # Classified pages keep their PDF page index for row-level provenance.
        school_record_pages: list[tuple[int, str]] = []
        championship_pages: list[tuple[int, str]] = []
        individual_xc_pages: list[tuple[int, str]] = []
        individual_event_pages: list[tuple[int, str]] = []
        sportsmanship_pages: list[tuple[int, str]] = []
        golf_pages: list[tuple[int, str]] = []

        for offset, text in enumerate(sport_pages):
            idx = start + offset
            classified = False
            # School records checked first — regex, no LLM cost. A page can
            # also match a second classifier below (dual-content pages).
            if is_school_records(text):
                school_record_pages.append((idx, text))
                classified = True
            if is_golf_results(text):
                golf_pages.append((idx, text))
            elif is_sportsmanship(text):
                sportsmanship_pages.append((idx, text))
            elif is_individual_results(text) and "Cross Country" not in sport:
                individual_event_pages.append((idx, text))
            elif is_individual_xc(text):
                individual_xc_pages.append((idx, text))
            elif is_year_class_table(text) or is_multicolumn_results(text):
                championship_pages.append((idx, text))
            elif not classified:
                pass  # section header, stats, ads → skip

        # School records (regex)
        if school_record_pages:
            src = source_label([i for i, _ in school_record_pages])
            recs = _stamp(
                parse_school_records([t for _, t in school_record_pages], sport), src
            )
            print(f"  school records     : {len(recs)} schools  ({len(school_record_pages)} pages, regex)")
            all_school_records.extend(recs)

        # Championship results (LLM) — guarded against silent row loss
        if championship_pages:
            results = extract_chunks_complete(
                championship_pages,
                lambda pgs: extract_championship_results(pgs, sport),
                f"{sport} championship",
            )
            all_championship.extend(results)
            print(f"  championship table : {len(results)} rows  ({len(championship_pages)} pages, LLM)")

        # Individual XC (LLM) — fall cross country only
        if individual_xc_pages and "Cross Country" in sport:
            champs = extract_chunks_complete(
                individual_xc_pages,
                lambda pgs: extract_individual_xc(pgs, sport),
                f"{sport} individual XC",
            )
            all_individual_xc.extend(champs)
            print(f"  individual XC      : {len(champs)} rows  ({len(individual_xc_pages)} pages, LLM)")

        # Individual event results (LLM) — track, swimming, tennis
        if individual_event_pages:
            results = extract_chunks_complete(
                individual_event_pages,
                lambda pgs: extract_individual_results(pgs, sport),
                f"{sport} individual events",
            )
            all_individual.extend(results)
            print(f"  individual events  : {len(results)} rows  ({len(individual_event_pages)} pages, LLM)")

        # Golf (LLM)
        if golf_pages:
            results = extract_chunks_complete(
                golf_pages, lambda pgs: extract_golf_results(pgs), f"{sport} golf"
            )
            all_golf.extend(results)
            print(f"  golf results       : {len(results)} rows  ({len(golf_pages)} pages, LLM)")

        # Sportsmanship (LLM) — not chunked, but still guarded against truncation
        if sportsmanship_pages:
            src = source_label([i for i, _ in sportsmanship_pages])
            awards = _stamp(
                extract_resilient(
                    [t for _, t in sportsmanship_pages],
                    lambda pgs: extract_sportsmanship(pgs, sport),
                    f"{sport} sportsmanship",
                ),
                src,
            )
            print(f"  sportsmanship      : {len(awards)} awards  ({len(sportsmanship_pages)} pages, LLM)")
            all_sportsmanship.extend(awards)

        print()

    # De-duplicate every LLM-extracted table (chunks overlap by one page).
    print("De-duplicating LLM-extracted tables:")
    unique_championship = dedupe(
        all_championship,
        ("sport", "year", "classification", "champion_school"),
        "championship_results",
    )
    all_individual_xc = dedupe(
        all_individual_xc, ("sport", "year", "classification"), "individual_xc_champions"
    )
    all_individual = dedupe(
        all_individual,
        ("sport", "event", "year", "classification"),
        "individual_results",
    )
    all_golf = dedupe(
        all_golf, ("year", "classification", "individual_gender"), "golf_results"
    )
    all_sportsmanship = dedupe(
        all_sportsmanship, ("sport", "year", "classification"), "sportsmanship_awards"
    )
    print()

    # ── Canonicalize school names ───────────────────────────────────────────────
    # Give every school field a single canonical spelling (+ url-safe slug) so
    # the same school joins across tables and the website can route by slug.
    aliases = load_aliases()
    name_map: dict[str, str] = {}
    tables = {
        "championship_results": unique_championship,
        "school_records": all_school_records,
        "individual_xc_champions": all_individual_xc,
        "individual_results": all_individual,
        "golf_results": all_golf,
        "sportsmanship_awards": all_sportsmanship,
    }
    for table_name, rows in tables.items():
        canonicalize_rows(rows, SCHOOL_FIELDS[table_name], aliases, name_map)
    print(f"School names: {len(name_map)} raw variant(s) mapped to a canonical form.\n")

    # ── Write outputs ─────────────────────────────────────────────────────────
    # Every table carries a trailing `source_pages` column so any row can be
    # traced back to the exact PDF page(s) it came from (see pages.jsonl).
    write_csv(
        out_dir / "championship_results.csv",
        unique_championship,
        [
            "sport", "year", "classification",
            "champion_school", "champion_school_slug", "champion_coach",
            "finalist_school", "finalist_school_slug", "finalist_coach",
            "score", "champion_undefeated", "co_champion", "notes",
            "source_pages",
        ],
    )
    write_csv(
        out_dir / "school_records.csv",
        _years_as_strings(all_school_records),  # year lists → "1997;1998", never a list repr
        ["sport", "school", "school_slug", "champion_years", "finalist_years",
         "semifinalist_years", "runner_up_years", "quarterfinal_years",
         "source_pages"],
    )
    write_csv(
        out_dir / "school_record_years.csv",
        school_records_long(all_school_records),  # tidy long format: one row per year
        ["sport", "school", "school_slug", "year", "result", "source_pages"],
    )
    write_csv(
        out_dir / "individual_xc_champions.csv",
        all_individual_xc,
        ["sport", "year", "classification", "name", "school", "school_slug",
         "time", "distance", "source_pages"],
    )
    if all_individual:
        write_csv(
            out_dir / "individual_results.csv",
            all_individual,
            ["sport", "event", "year", "classification", "name", "school",
             "school_slug", "mark", "source_pages"],
        )
    write_csv(
        out_dir / "sportsmanship_awards.csv",
        all_sportsmanship,
        ["sport", "year", "classification", "school", "school_slug", "source_pages"],
    )
    write_csv(
        out_dir / "golf_results.csv",
        all_golf,
        [
            "year", "classification",
            "team_champion_school", "team_champion_school_slug", "team_score",
            "individual_winner_name",
            "individual_winner_school", "individual_winner_school_slug",
            "individual_score", "individual_gender",
            "source_pages",
        ],
    )

    # Audit file: every raw → canonical school-name change applied above.
    name_map_path = out_dir / "school_name_map.json"
    name_map_path.write_text(json.dumps(dict(sorted(name_map.items())), indent=2))
    print(f"  ✓ school_name_map.json  →  {name_map_path}")

    # Raw extracted page text, so a row's source_pages can be checked against
    # the PDF's text without re-running the (slow) PDF extraction.
    pages_path = out_dir / "pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(pages):
            f.write(json.dumps({"page": i, "text": text}) + "\n")
    print(f"  ✓ {len(pages)} pages  →  {pages_path}")

    record_book = {
        "championship_results": unique_championship,
        "school_records": all_school_records,
        "individual_xc_champions": all_individual_xc,
        "individual_results": all_individual,
        "sportsmanship_awards": all_sportsmanship,
        "golf_results": all_golf,
    }

    # Self-describing metadata for the website / downstream consumers, including
    # an embedded verification summary so a consumer can tell at a glance
    # whether the data passed the cross-checks.
    report = verify_record_book.build_report(record_book, out_dir)
    (out_dir / "verification_report.json").write_text(json.dumps(report, indent=2))
    record_book = {
        "meta": build_meta(season, pdf_path, record_book, report),
        **record_book,
    }
    json_path = out_dir / "record_book.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record_book, indent=2, default=str))
    print(f"  ✓ record_book.json  →  {json_path}")

    write_combined_json(out_dir.parent)
    print("\nDone.")


if __name__ == "__main__":
    main()
