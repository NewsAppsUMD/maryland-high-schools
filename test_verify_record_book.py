"""Tests for verify_record_book.py — cross-check, dedup, continuity, referential.

All checks are deterministic and run on small synthetic record-book dicts;
no LLM calls and no PDF reads.
"""

import json

import pytest

import verify_record_book as v


def _book(**tables):
    base = {name: [] for name in v.TABLE_KEYS}
    base.update(tables)
    return base


# ── Check 1: cross-path champion years ────────────────────────────────────────


class TestCrossPath:
    def test_full_coverage_is_ok(self):
        book = _book(
            championship_results=[
                {"sport": "Soccer", "year": 2023, "classification": "4A", "champion_school": "X"},
                {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Y"},
            ],
            school_records=[
                {"sport": "Soccer", "school": "X", "champion_years": [2023]},
                {"sport": "Soccer", "school": "Y", "champion_years": [2024]},
            ],
        )
        result = v.check_cross_path(book)
        assert result["errors"] == 0
        assert result["sports"]["Soccer"]["coverage"] == 1.0

    def test_detects_missing_table_years(self):
        # Records know 4 champion years; the table has only 1 → coverage 25%.
        book = _book(
            championship_results=[
                {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Y"},
            ],
            school_records=[
                {"sport": "Soccer", "school": "Y",
                 "champion_years": [2021, 2022, 2023, 2024]},
            ],
        )
        result = v.check_cross_path(book)
        assert result["errors"] == 1
        info = result["sports"]["Soccer"]
        assert info["status"] == "error"
        assert info["missing_from_championship_table"] == [2021, 2022, 2023]

    def test_golf_years_come_from_golf_table(self):
        # Golf champions are in golf_results, not championship_results.
        book = _book(
            golf_results=[
                {"year": 1995, "classification": "3A/4A", "team_champion_school": "Magruder",
                 "individual_gender": None},
            ],
            school_records=[
                {"sport": "Golf", "school": "Magruder", "champion_years": [1995]},
            ],
        )
        result = v.check_cross_path(book)
        assert result["sports"]["Golf"]["coverage"] == 1.0
        assert result["errors"] == 0


# ── Check 2: duplicate keys ───────────────────────────────────────────────────


class TestDuplicateKeys:
    def test_flags_duplicate_xc_rows(self):
        book = _book(
            individual_xc_champions=[
                {"sport": "GXC", "year": 2016, "classification": "4A", "name": "A"},
                {"sport": "GXC", "year": 2016, "classification": "4A", "name": "A"},
            ]
        )
        result = v.check_duplicate_keys(book)
        assert result["errors"] == 1
        assert result["tables"]["individual_xc_champions"]["duplicate_key_count"] == 1

    def test_no_duplicates(self):
        book = _book(
            individual_xc_champions=[
                {"sport": "GXC", "year": 2016, "classification": "4A", "name": "A"},
                {"sport": "GXC", "year": 2016, "classification": "3A", "name": "B"},
            ]
        )
        assert v.check_duplicate_keys(book)["errors"] == 0


# ── Check 3: continuity ───────────────────────────────────────────────────────


class TestContinuity:
    def test_covid_year_not_a_gap(self):
        book = _book(
            school_records=[
                {"sport": "Soccer", "school": "X", "champion_years": [2019, 2021]},
            ]
        )
        result = v.check_continuity(book)
        assert "Soccer" not in result["sports"]  # 2020 gap is exempt

    def test_reports_real_gap(self):
        book = _book(
            school_records=[
                {"sport": "Soccer", "school": "X", "champion_years": [2010, 2013]},
            ]
        )
        result = v.check_continuity(book)
        assert result["sports"]["Soccer"]["missing_years"] == [2011, 2012]


# ── Check 4: referential schools ──────────────────────────────────────────────


class TestReferential:
    def test_flags_school_absent_from_records(self):
        book = _book(
            championship_results=[
                {"sport": "Soccer", "year": 2024, "classification": "4A",
                 "champion_school": "Ghost School"},
            ],
            school_records=[
                {"sport": "Soccer", "school": "Real School", "champion_years": [2024]},
            ],
        )
        result = v.check_referential_schools(book)
        assert result["warnings"] == 1
        assert result["sports"]["Soccer"] == ["Ghost School"]

    def test_co_champions_checked_per_component(self):
        book = _book(
            championship_results=[
                {"sport": "FH", "year": 2024, "classification": "4A",
                 "champion_school": "Real School & Ghost School"},
            ],
            school_records=[
                {"sport": "FH", "school": "Real School", "champion_years": [2024]},
            ],
        )
        result = v.check_referential_schools(book)
        assert result["sports"]["FH"] == ["Ghost School"]  # only the missing half

    def test_matching_school_is_clean(self):
        book = _book(
            championship_results=[
                {"sport": "Soccer", "year": 2024, "classification": "4A",
                 "champion_school": "Real School"},
            ],
            school_records=[
                {"sport": "Soccer", "school": "Real School", "champion_years": [2024]},
            ],
        )
        assert v.check_referential_schools(book)["warnings"] == 0


# ── Report assembly + exit semantics ──────────────────────────────────────────


class TestReport:
    def test_clean_book_passes(self, tmp_path):
        book = _book(
            championship_results=[
                {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "X"},
            ],
            school_records=[
                {"sport": "Soccer", "school": "X", "champion_years": [2024]},
            ],
        )
        report = v.build_report(book, tmp_path)
        assert report["summary"]["passed"] is True
        assert report["summary"]["errors"] == 0

    def test_lossy_book_fails(self, tmp_path):
        book = _book(
            championship_results=[
                {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "X"},
            ],
            school_records=[
                {"sport": "Soccer", "school": "X",
                 "champion_years": [2020, 2021, 2022, 2023, 2024]},
            ],
        )
        report = v.build_report(book, tmp_path)
        assert report["summary"]["passed"] is False
        assert report["summary"]["errors"] >= 1

    def test_report_written_and_loadable(self, tmp_path):
        book = _book()
        (tmp_path / "record_book.json").write_text(json.dumps(book))
        loaded = v.load_record_book(tmp_path)
        report = v.build_report(loaded, tmp_path)
        assert "checks" in report and "row_counts" in report
