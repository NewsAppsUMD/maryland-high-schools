"""Tests for scripts/fill_finalists.py — the recap-validated back-fill.

Covers: candidate-set construction, the recap validation gate (champion +
score match, finalist != champion, no reuse within a year, duplicate-entry and
no-row rejection, soft candidate-set + overwrite warnings), and helpers. No LLM
calls, no file I/O except load_recaps reading a temp file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root and scripts/ importable (the script itself does
# sys.path.insert(0, ".") + imports parse_record_book / build_site, which work
# when pytest runs from the repo root).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_site                      # noqa: E402
import fill_finalists as ff            # noqa: E402

# A normalizer with no curated aliases: straight _base_normalize behaviour.
norm = build_site.make_normalizer({})


# ── year_list_has ─────────────────────────────────────────────────────────────

class TestYearListHas:
    def test_semicolon_string(self):
        assert ff.year_list_has("1975; 2009; 2010", 2009) is True

    def test_miss(self):
        assert ff.year_list_has("1975; 2009", 2010) is False

    def test_int_list(self):
        assert ff.year_list_has([1975, 2009], 1975) is True

    def test_empty(self):
        assert ff.year_list_has("", 2009) is False
        assert ff.year_list_has(None, 2009) is False


# ── build_champion_index / candidate_keys ─────────────────────────────────────

def _champ(year, school, classification="4A", finalist="", notes="", score="21-14"):
    return {"sport": "Football", "year": str(year), "classification": classification,
            "champion_school": school, "finalist_school": finalist, "notes": notes,
            "score": score}


def _school(name, years):
    return {"sport": "Football", "school": name, "runner_up_years": years,
            "finalist_years": ""}


class TestCandidateSet:
    def test_includes_finalist_for_year(self):
        schools = [_school("BEL AIR", "1975; 2009")]
        idx = ff.build_champion_index([_champ(1975, "FORT HILL", "A")], norm)
        cand = ff.candidate_keys(schools, 1975, "runner_up_years", norm, idx)
        assert cand == {norm("BEL AIR")}

    def test_excludes_same_year_champion(self):
        schools = [_school("BEL AIR", "1975")]
        champs = [_champ(1975, "BEL AIR", "AA"), _champ(1975, "FORT HILL", "A")]
        idx = ff.build_champion_index(champs, norm)
        cand = ff.candidate_keys(schools, 1975, "runner_up_years", norm, idx)
        assert cand == set()

    def test_normalizes_allcaps_vs_mixed_case(self):
        schools = [_school("BETHESDA-CHEVY CHASE", "1990; 1991")]
        idx = ff.build_champion_index([_champ(1990, "Bethesda-Chevy Chase", "4A")], norm)
        cand = ff.candidate_keys(schools, 1990, "runner_up_years", norm, idx)
        assert cand == set()  # champion in 1990 -> excluded
        cand2 = ff.candidate_keys(schools, 1991, "runner_up_years", norm, idx)
        assert cand2 == {norm("BETHESDA-CHEVY CHASE")}

    def test_wrong_year_excluded(self):
        schools = [_school("BEL AIR", "1975; 2009")]
        idx = ff.build_champion_index([], norm)
        assert ff.candidate_keys(schools, 2010, "runner_up_years", norm, idx) == set()


# ── _titlecase ─────────────────────────────────────────────────────────────────

class TestTitlecase:
    def test_allcaps_with_hyphen(self):
        assert ff._titlecase("BETHESDA-CHEVY CHASE") == "Bethesda-Chevy Chase"

    def test_allcaps_simple(self):
        assert ff._titlecase("FORT HILL") == "Fort Hill"


# ── recaps_path ────────────────────────────────────────────────────────────────

class TestRecapsPath:
    def test_football(self):
        assert ff.recaps_path("fall", "Football") == Path("data/fall/football_finalists.json")

    def test_boys_basketball_slug(self):
        assert ff.recaps_path("winter", "Boys Basketball") == \
            Path("data/winter/boys_basketball_finalists.json")


# ── validate_recaps (the validation gate) ──────────────────────────────────────

def _entry(year, classification, champion, finalist, score="21-14"):
    return {"year": str(year), "classification": classification,
            "champion": champion, "score": score, "finalist": finalist,
            "source": "https://example/recap"}


class TestValidateRecaps:
    def _rows(self):
        # Two classes in 1975: A Fort Hill, AA Parkdale.
        return [_champ(1975, "FORT HILL", "A", score="21-14"),
                _champ(1975, "PARKDALE", "AA", score="35-7")]

    def _idx(self, rows):
        return ff.build_champion_index(rows, norm)

    def test_accepts_valid_entries(self):
        rows = self._rows()
        entries = [_entry(1975, "A",  "Fort Hill", "Bel Air",  "21-14"),
                   _entry(1975, "AA", "Parkdale",  "Northwood", "35-7")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1975"), _school("NORTHWOOD", "1975")],
            norm, "runner_up_years", self._idx(rows))
        assert rejected == []
        assert warnings == []
        assert plan == [(1975, "A", "Bel Air", ff.NOTE_RECAP),
                        (1975, "AA", "Northwood", ff.NOTE_RECAP)]

    def test_rejects_champion_mismatch(self):
        rows = self._rows()
        entries = [_entry(1975, "A", "Wrong School", "Bel Air", "21-14")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1975")], norm,
            "runner_up_years", self._idx(rows))
        assert plan == []
        assert any("champion mismatch" in r for r in rejected[0][1])

    def test_rejects_score_mismatch(self):
        rows = self._rows()
        entries = [_entry(1975, "A", "Fort Hill", "Bel Air", "99-0")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1975")], norm,
            "runner_up_years", self._idx(rows))
        assert plan == []
        assert any("score mismatch" in r for r in rejected[0][1])

    def test_rejects_finalist_equals_champion(self):
        rows = self._rows()
        entries = [_entry(1975, "A", "Fort Hill", "Fort Hill", "21-14")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [], norm, "runner_up_years", self._idx(rows))
        assert plan == []
        assert any("== champion" in r for r in rejected[0][1])

    def test_rejects_reuse_across_classes(self):
        rows = self._rows()
        entries = [_entry(1975, "A",  "Fort Hill", "Bel Air", "21-14"),
                   _entry(1975, "AA", "Parkdale",  "Bel Air", "35-7")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1975")], norm,
            "runner_up_years", self._idx(rows))
        # First is accepted, second rejected for reuse.
        assert len(plan) == 1
        assert any("reused" in r for r in rejected[0][1])

    def test_rejects_no_matching_row(self):
        rows = self._rows()
        entries = [_entry(1999, "A", "Fort Hill", "Bel Air", "21-14")]  # no 1999 row
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1999")], norm,
            "runner_up_years", self._idx(rows))
        assert plan == []
        assert any("no championship row" in r for r in rejected[0][1])

    def test_rejects_duplicate_entry(self):
        rows = self._rows()
        entries = [_entry(1975, "A", "Fort Hill", "Bel Air", "21-14"),
                   _entry(1975, "A", "Fort Hill", "Bel Air", "21-14")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1975")], norm,
            "runner_up_years", self._idx(rows))
        assert len(plan) == 1
        assert any("duplicate" in r for r in rejected[0][1])

    def test_warns_but_writes_when_not_in_candidate_set(self):
        # school_records has no record of Bel Air in 1975 -> candidate set is
        # empty for Bel Air, but champion+score match so the entry is still
        # written (school_records is known-incomplete).
        rows = self._rows()
        entries = [_entry(1975, "A", "Fort Hill", "Bel Air", "21-14")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("NORTHWOOD", "1975")], norm,  # only Northwood
            "runner_up_years", self._idx(rows))
        assert rejected == []
        assert len(plan) == 1
        assert any("not in school_records candidate set" in w[2] for w in warnings)

    def test_warns_on_overwrite_of_existing_finalist(self):
        rows = [_champ(1975, "FORT HILL", "A", finalist="Someone Else", score="21-14")]
        entries = [_entry(1975, "A", "Fort Hill", "Bel Air", "21-14")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1975")], norm,
            "runner_up_years", self._idx(rows))
        assert rejected == []
        assert any("already filled" in w[2] for w in warnings)

    def test_year_filter_restricts_entries(self):
        rows = [_champ(1975, "FORT HILL", "A", score="21-14"),
                _champ(1976, "FORT HILL", "A", score="14-7")]
        entries = [_entry(1975, "A", "Fort Hill", "Bel Air", "21-14"),
                   _entry(1976, "A", "Fort Hill", "Oakdale", "14-7")]
        plan, rejected, warnings = ff.validate_recaps(
            entries, rows, [_school("BEL AIR", "1975"), _school("OAKDALE", "1976")],
            norm, "runner_up_years", self._idx(rows), year_filter=1976)
        assert len(plan) == 1
        assert plan[0][0] == 1976


# ── load_recaps (reads the committed football data file) ───────────────────────

class TestLoadRecaps:
    def test_reads_committed_football_file(self):
        # The committed data/fall/football_finalists.json should load and parse.
        entries = ff.load_recaps("fall", "Football")
        assert len(entries) >= 28  # 6 years of finals (4-6 classes each)
        for e in entries:
            assert {"year", "classification", "champion", "score", "finalist", "source"} <= set(e)

    def test_missing_file_returns_empty(self):
        assert ff.load_recaps("fall", "Nonexistent Sport") == []