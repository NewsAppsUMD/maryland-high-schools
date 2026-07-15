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

import csv
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import llm
from pypdf import PdfReader
from pydantic import BaseModel, ValidationError

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


def llm_extract(prompt: str, schema, retries: int = 2) -> dict:
    """Call the LLM with a Pydantic schema, validate the result, and return a dict.

    The raw model output is validated through the Pydantic `schema` before being
    returned, so field defaults are applied (e.g. champion_undefeated=False) and
    types are coerced (e.g. year "2024" → 2024). Output that does not conform to
    the schema triggers a retry and, if it never conforms, a loud error — bad
    rows never reach the CSV/JSON silently.
    """
    model = get_model()
    last_text = ""
    last_error = ""

    for attempt in range(1, retries + 1):
        response = model.prompt(prompt, schema=schema, stream=False)
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
    name = Path(pdf_path).stem.lower()
    for season in ("fall", "winter", "spring"):
        if season in name:
            return season
    return "fall"


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
    # Match golf format: "Team Champion......School (score)" — exclude swimming "Year Class Team Champion Coach"
    return bool(re.search(r"Team Champion\s*\.{3,}", text))


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


def chunked(lst: list, size: int, overlap: int = 1) -> list[list]:
    """Split list into overlapping chunks."""
    if not lst:
        return []
    step = max(size - overlap, 1)
    return [lst[i : i + size] for i in range(0, len(lst), step)]


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
            if k not in key_fields and first.get(k) != r.get(k)
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


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdfs/FallRecordBook2024.pdf"
    season = detect_season(pdf_path)
    sections = SEASON_SECTIONS[season]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data") / season

    print(f"Loading {pdf_path} … (season: {season})")
    pages = load_pages(pdf_path)
    print(f"  {len(pages)} pages loaded.\n")

    all_championship: list[dict] = []
    all_school_records: list[dict] = []
    all_individual_xc: list[dict] = []
    all_individual: list[dict] = []
    all_sportsmanship: list[dict] = []
    all_golf: list[dict] = []

    for sport, (start, end) in sections.items():
        sport_pages = pages[start:end]
        print(f"── {sport}  (PDF indices {start}–{end-1}) ──")

        school_record_pages: list[str] = []
        championship_pages: list[str] = []
        individual_xc_pages: list[str] = []
        individual_event_pages: list[str] = []
        sportsmanship_pages: list[str] = []
        golf_pages: list[str] = []

        for text in sport_pages:
            classified = False
            # School records checked first — regex, no LLM cost. A page can
            # also match a second classifier below (dual-content pages).
            if is_school_records(text):
                school_record_pages.append(text)
                classified = True
            if is_golf_results(text):
                golf_pages.append(text)
            elif is_sportsmanship(text):
                sportsmanship_pages.append(text)
            elif is_individual_results(text) and "Cross Country" not in sport:
                individual_event_pages.append(text)
            elif is_individual_xc(text):
                individual_xc_pages.append(text)
            elif is_year_class_table(text) or is_multicolumn_results(text):
                championship_pages.append(text)
            elif not classified:
                pass  # section header, stats, ads → skip

        # School records (regex)
        if school_record_pages:
            recs = parse_school_records(school_record_pages, sport)
            print(f"  school records     : {len(recs)} schools  ({len(school_record_pages)} pages, regex)")
            all_school_records.extend(recs)

        # Championship results (LLM)
        if championship_pages:
            total = 0
            for chunk in chunked(championship_pages, CHUNK):
                results = extract_championship_results(chunk, sport)
                total += len(results)
                all_championship.extend(results)
            print(f"  championship table : {total} rows  ({len(championship_pages)} pages, LLM)")

        # Individual XC (LLM) — fall cross country only
        if individual_xc_pages and "Cross Country" in sport:
            total = 0
            for chunk in chunked(individual_xc_pages, CHUNK):
                champs = extract_individual_xc(chunk, sport)
                total += len(champs)
                all_individual_xc.extend(champs)
            print(f"  individual XC      : {total} rows  ({len(individual_xc_pages)} pages, LLM)")

        # Individual event results (LLM) — track, swimming, tennis
        if individual_event_pages:
            total = 0
            for chunk in chunked(individual_event_pages, CHUNK):
                results = extract_individual_results(chunk, sport)
                total += len(results)
                all_individual.extend(results)
            print(f"  individual events  : {total} rows  ({len(individual_event_pages)} pages, LLM)")

        # Golf (LLM)
        if golf_pages:
            total = 0
            for chunk in chunked(golf_pages, CHUNK):
                results = extract_golf_results(chunk)
                total += len(results)
                all_golf.extend(results)
            print(f"  golf results       : {total} rows  ({len(golf_pages)} pages, LLM)")

        # Sportsmanship (LLM)
        if sportsmanship_pages:
            awards = extract_sportsmanship(sportsmanship_pages, sport)
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

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_csv(
        out_dir / "championship_results.csv",
        unique_championship,
        [
            "sport", "year", "classification",
            "champion_school", "champion_coach",
            "finalist_school", "finalist_coach",
            "score", "champion_undefeated", "co_champion", "notes",
        ],
    )
    write_csv(
        out_dir / "school_records.csv",
        all_school_records,
        ["sport", "school", "champion_years", "finalist_years",
         "semifinalist_years", "runner_up_years", "quarterfinal_years"],
    )
    write_csv(
        out_dir / "individual_xc_champions.csv",
        all_individual_xc,
        ["sport", "year", "classification", "name", "school", "time", "distance"],
    )
    if all_individual:
        write_csv(
            out_dir / "individual_results.csv",
            all_individual,
            ["sport", "event", "year", "classification", "name", "school", "mark"],
        )
    write_csv(
        out_dir / "sportsmanship_awards.csv",
        all_sportsmanship,
        ["sport", "year", "classification", "school"],
    )
    write_csv(
        out_dir / "golf_results.csv",
        all_golf,
        [
            "year", "classification",
            "team_champion_school", "team_score",
            "individual_winner_name", "individual_winner_school",
            "individual_score", "individual_gender",
        ],
    )

    record_book = {
        "championship_results": unique_championship,
        "school_records": all_school_records,
        "individual_xc_champions": all_individual_xc,
        "individual_results": all_individual,
        "sportsmanship_awards": all_sportsmanship,
        "golf_results": all_golf,
    }
    json_path = out_dir / "record_book.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record_book, indent=2, default=str))
    print(f"  ✓ record_book.json  →  {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
