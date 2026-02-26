#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "llm>=0.28",
#   "llm-anthropic>=0.24",
#   "pypdf>=6.0",
# ]
# ///
"""
Parse the MPSSAA Fall Record Book PDF into structured CSV / JSON data.

Usage:
    uv run parse_record_book.py [PDF_PATH] [OUTPUT_DIR]

Defaults:
    PDF_PATH   = pdfs/FallRecordBook2024.pdf
    OUTPUT_DIR = data

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


def llm_extract(prompt: str, schema) -> dict:
    """Call the LLM with a Pydantic schema and return the parsed dict."""
    model = get_model()
    response = model.prompt(prompt, schema=schema, stream=False)

    # claude-sonnet-4-6 / claude-opus-4-6: structured output → response.text() is JSON
    text = response.text().strip()
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Haiku (tool-based schema fallback): result is in tool_calls[0].arguments
    if response.tool_calls:
        args = response.tool_calls[0].arguments
        if isinstance(args, dict):
            return args
        return json.loads(args)

    raise RuntimeError(
        f"LLM returned no usable content.\n"
        f"text={text!r}\n"
        f"response_json={response.response_json}"
    )


# ── PDF extraction ────────────────────────────────────────────────────────────


def load_pages(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


# ── Section map ───────────────────────────────────────────────────────────────
# Document page numbers → PDF 0-based index is (doc_page + 1).
# Slices below are [first_idx : last_idx+1] i.e. [doc_page_start+1 : doc_page_end+2].

SECTIONS: dict[str, tuple[int, int]] = {
    "Girls Cross Country": (3, 13),   # doc pages 2-11
    "Boys Cross Country":  (13, 27),  # doc pages 12-25
    "Field Hockey":        (27, 35),  # doc pages 26-33
    "Football":            (35, 47),  # doc pages 34-45
    "Golf":                (47, 53),  # doc pages 46-51
    "Girls Soccer":        (53, 59),  # doc pages 52-57
    "Boys Soccer":         (59, 66),  # doc pages 58-64
    "Volleyball":          (66, 77),  # doc pages 65-75
}


# ── Page classifiers ──────────────────────────────────────────────────────────


def is_school_records(text: str) -> bool:
    return bool(re.search(r"\bCh:\s*\d{4}", text))


def is_year_class_table(text: str) -> bool:
    """Cross country / field hockey style with explicit YEAR CLASS header."""
    return bool(re.search(r"YEAR\s+CLASS\s+CHAMPION", text, re.IGNORECASE))


def is_multicolumn_results(text: str) -> bool:
    """Football / soccer / volleyball multi-column champion table."""
    return bool(re.search(r"CLASS\s+(?:4A|AA)\s+CLASS\s+(?:3A|A)", text))


def is_individual_xc(text: str) -> bool:
    return bool(re.search(r"\d+\.\d+\s+MILES?|3\.0 MILES?\s", text, re.IGNORECASE))


def is_sportsmanship(text: str) -> bool:
    return bool(re.search(r"SPORTSMANSHIP AWARD", text, re.IGNORECASE)) and bool(
        re.search(r"\b(20|19)\d{2}\b", text)
    )


def is_golf_results(text: str) -> bool:
    return bool(re.search(r"Team Champion", text))


# ── School records (regex) ────────────────────────────────────────────────────


def parse_school_records(pages_text: list[str], sport: str) -> list[dict]:
    """
    Extract Ch/Fn/Sf/RU years from school record pages using line-by-line parsing.
    Handles: 'Ch: 1997, 1998' and wrapped years across multiple lines.
    """
    combined = "\n".join(pages_text)
    records: list[dict] = []

    # A school name is an all-caps line that isn't a table header or page number
    school_re = re.compile(
        r"^(?!YEAR|CLASS|MPSSAA|HONOR ROLL|TOURNAMENTS|STATE|PREVIOUS|PUBLIC|SOCCER)"
        r"[A-Z][A-Z\s\.\-\'/&]+$"
    )
    status_start_re = re.compile(r"^(Ch|Fn|Sf|RU|QF|RS|RR\d?|CH|SF|RU):")

    def get_years(block: list[str], code: str) -> list[int]:
        combined_block = " ".join(block)
        # Grab everything after "Code:" until the next code or end
        parts = re.findall(rf"(?i)\b{code}:\s*([\d,\s\n]+)", combined_block)
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
        if ch or fn or ru:
            return {
                "sport": sport,
                "school": school,
                "champion_years": ch,
                "finalist_years": fn,
                "semifinalist_years": sf,
                "runner_up_years": ru,
            }
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
            current_school = line
            current_block = []
        elif current_school:
            current_block.append(line)

    if current_school and current_block:
        rec = flush(current_school, current_block)
        if rec:
            records.append(rec)

    return records


# ── Championship results (LLM) ────────────────────────────────────────────────


def extract_championship_results(
    pages: list[str], sport: str
) -> list[dict]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(pages)
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

        TEXT:
        {combined}
    """).strip()
    data = llm_extract(prompt, ChampionshipResults)
    return data.get("results", [])


# ── Individual XC champions (LLM) ────────────────────────────────────────────


def extract_individual_xc(pages: list[str], sport: str) -> list[dict]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(pages)
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


# ── Golf results (LLM) ────────────────────────────────────────────────────────


def extract_golf_results(pages: list[str]) -> list[dict]:
    combined = "\n\n--- PAGE BREAK ---\n\n".join(pages)
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

CHUNK = 4  # max pages per LLM call


def chunked(lst: list, size: int, overlap: int = 1) -> list[list]:
    """Split list into overlapping chunks."""
    if not lst:
        return []
    step = max(size - overlap, 1)
    return [lst[i : i + size] for i in range(0, len(lst), step)]


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdfs/FallRecordBook2024.pdf"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")

    print(f"Loading {pdf_path} …")
    pages = load_pages(pdf_path)
    print(f"  {len(pages)} pages loaded.\n")

    all_championship: list[dict] = []
    all_school_records: list[dict] = []
    all_individual: list[dict] = []
    all_sportsmanship: list[dict] = []
    all_golf: list[dict] = []

    for sport, (start, end) in SECTIONS.items():
        sport_pages = pages[start:end]
        print(f"── {sport}  (PDF indices {start}–{end-1}) ──")

        school_record_pages: list[str] = []
        championship_pages: list[str] = []
        individual_pages: list[str] = []
        sportsmanship_pages: list[str] = []
        golf_pages: list[str] = []

        for text in sport_pages:
            if is_golf_results(text):
                golf_pages.append(text)
            elif is_sportsmanship(text):
                sportsmanship_pages.append(text)
            elif is_individual_xc(text):
                individual_pages.append(text)
            elif is_year_class_table(text) or is_multicolumn_results(text):
                championship_pages.append(text)
            elif is_school_records(text):
                school_record_pages.append(text)
            # else: section header, stats, ads → skip

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

        # Individual XC (LLM)
        if individual_pages and "Cross Country" in sport:
            total = 0
            for chunk in chunked(individual_pages, CHUNK):
                champs = extract_individual_xc(chunk, sport)
                total += len(champs)
                all_individual.extend(champs)
            print(f"  individual XC      : {total} rows  ({len(individual_pages)} pages, LLM)")

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

    # De-duplicate championship results
    seen: set[tuple] = set()
    unique_championship: list[dict] = []
    for r in all_championship:
        key = (
            r.get("sport", ""),
            r.get("year", ""),
            r.get("classification", ""),
            r.get("champion_school", ""),
        )
        if key not in seen:
            seen.add(key)
            unique_championship.append(r)
    print(
        f"Championship results: {len(all_championship)} extracted → "
        f"{len(unique_championship)} after dedup\n"
    )

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
         "semifinalist_years", "runner_up_years"],
    )
    write_csv(
        out_dir / "individual_xc_champions.csv",
        all_individual,
        ["sport", "year", "classification", "name", "school", "time", "distance"],
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
        "individual_xc_champions": all_individual,
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
