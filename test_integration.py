"""End-to-end pipeline test with the LLM stubbed (no API key needed).

Runs the full main() against the real Fall PDF, replacing only the LLM
extraction functions with canned data, and asserts that every output artifact
is produced with the expected shape: canonical school names + slugs, row-level
provenance, the school-records long format, self-describing JSON meta, the
combined all.json, and the verification report. This exercises the wiring that
unit tests can't, and needs no network access.
"""

import json
import sys

import pytest

import parse_record_book as p

FALL_PDF = "pdfs/FallRecordBook2024.pdf"


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace each LLM extractor with deterministic canned rows."""
    def champ(pages, sport):
        return [{"sport": sport, "year": 2024, "classification": "4A",
                 "champion_school": "TEST CHAMPION", "champion_undefeated": False,
                 "co_champion": False}]

    def xc(pages, sport):
        return [{"sport": sport, "year": 2024, "classification": "4A",
                 "name": "Jane Doe", "school": "TEST CHAMPION"}]

    def indiv(pages, sport):
        return [{"sport": sport, "event": "55m", "year": 2024,
                 "classification": "4A", "name": "Jane Doe", "school": "TEST CHAMPION"}]

    def golf(pages):
        return [{"year": 2024, "classification": "3A/4A",
                 "team_champion_school": "TEST CHAMPION", "individual_gender": None}]

    def sportsmanship(pages, sport):
        return [{"sport": sport, "year": 2024, "classification": "4A",
                 "school": "TEST CHAMPION"}]

    monkeypatch.setattr(p, "extract_championship_results", champ)
    monkeypatch.setattr(p, "extract_individual_xc", xc)
    monkeypatch.setattr(p, "extract_individual_results", indiv)
    monkeypatch.setattr(p, "extract_golf_results", golf)
    monkeypatch.setattr(p, "extract_sportsmanship", sportsmanship)


def test_full_pipeline_produces_all_artifacts(tmp_path, monkeypatch, stub_llm):
    out_dir = tmp_path / "fall"
    monkeypatch.setattr(sys, "argv", ["parse", FALL_PDF, str(out_dir)])
    p.main()

    # All expected files exist
    for name in [
        "championship_results.csv", "school_records.csv",
        "school_record_years.csv", "individual_xc_champions.csv",
        "sportsmanship_awards.csv", "golf_results.csv",
        "pages.jsonl", "school_name_map.json",
        "record_book.json", "verification_report.json",
    ]:
        assert (out_dir / name).exists(), f"missing {name}"
    assert (tmp_path / "all.json").exists()  # combined written to data root

    book = json.loads((out_dir / "record_book.json").read_text())

    # meta block is present and self-describing
    meta = book["meta"]
    assert meta["season"] == "fall"
    assert meta["source_pdf"] == "FallRecordBook2024.pdf"
    assert meta["row_counts"]["school_records"] > 50  # regex path ran for real
    assert "verification" in meta

    # Canonical names + slugs + provenance on real (regex) school-record rows
    rec = book["school_records"][0]
    assert "school_slug" in rec
    assert "source_pages" in rec
    assert isinstance(rec["champion_years"], list)  # JSON keeps arrays

    # Canonicalization applied to stubbed LLM rows too
    champ_row = book["championship_results"][0]
    assert champ_row["champion_school"] == "Test Champion"  # ALL-CAPS title-cased
    assert champ_row["champion_school_slug"] == "test-champion"
    assert "source_pages" in champ_row

    # Long-format school-records CSV has one row per year
    import csv
    with (out_dir / "school_record_years.csv").open() as f:
        long_rows = list(csv.DictReader(f))
    assert long_rows
    assert set(long_rows[0]) >= {"sport", "school", "year", "result", "source_pages"}
    assert {r["result"] for r in long_rows} <= {
        "champion", "finalist", "semifinalist", "runner_up", "quarterfinal"
    }


def test_wide_csv_has_no_list_reprs(tmp_path, monkeypatch, stub_llm):
    out_dir = tmp_path / "fall"
    monkeypatch.setattr(sys, "argv", ["parse", FALL_PDF, str(out_dir)])
    p.main()
    text = (out_dir / "school_records.csv").read_text()
    assert "[" not in text and "]" not in text  # no "[1997, 1998]" list reprs
