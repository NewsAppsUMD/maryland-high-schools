"""Tests for verify_record_book.py — consistency checks against synthetic record books.

No LLM calls, no git, no real data files. Each test builds a minimal
record_book dict in memory and asserts what the check reports.
"""

import json
import subprocess
from pathlib import Path

import pytest

import verify_record_book as vrb
from verify_record_book import (
    ERA_FLOORS,
    TABLE_KEYS,
    _as_int,
    _iter_years,
    _key_tuple,
    _load_head_book,
    _table,
    build_report,
    check_continuity,
    check_cross_path,
    check_duplicate_keys,
    check_era_floors,
    check_referential_schools,
    check_regression,
    llm_champion_years,
    record_champion_years,
)


# ---------------------------------------------------------------------------
# Minimal synthetic book used across tests.
# Boys Cross Country: table and records agree, span 2010-2013 (no gaps).
# Golf: table and records agree.
# ---------------------------------------------------------------------------

def _book():
    return {
        "championship_results": [
            {"sport": "Boys Cross Country", "year": 2010, "classification": "Combined",
             "champion_school": "Northwood"},
            {"sport": "Boys Cross Country", "year": 2011, "classification": "Combined",
             "champion_school": "Atholton"},
            {"sport": "Boys Cross Country", "year": 2012, "classification": "Combined",
             "champion_school": "Northwood"},
            {"sport": "Boys Cross Country", "year": 2013, "classification": "Combined",
             "champion_school": "Bel Air"},
        ],
        "golf_results": [
            {"year": 2024, "classification": "Combined", "team_champion_school": "Magruder",
             "individual_gender": "Boys"},
        ],
        "school_records": [
            {"sport": "Boys Cross Country", "school": "Northwood",
             "champion_years": [2010, 2012]},
            {"sport": "Boys Cross Country", "school": "Atholton", "champion_years": [2011]},
            {"sport": "Boys Cross Country", "school": "Bel Air", "champion_years": [2013]},
        ],
        "individual_xc_champions": [],
        "individual_results": [],
        "sportsmanship_awards": [],
    }


class TestAsInt:
    def test_int_passthrough(self):
        assert _as_int(2010) == 2010

    def test_string(self):
        assert _as_int("2010") == 2010

    def test_none(self):
        assert _as_int(None) is None

    def test_garbage(self):
        assert _as_int("n/a") is None


class TestIterYears:
    def test_list_of_ints(self):
        assert list(_iter_years([2010, 2011])) == [2010, 2011]

    def test_semicolon_string(self):
        assert list(_iter_years("2010; 2011")) == [2010, 2011]

    def test_comma_string(self):
        assert list(_iter_years("2010,2011")) == [2010, 2011]

    def test_none(self):
        assert list(_iter_years(None)) == []

    def test_single_int(self):
        assert list(_iter_years(2010)) == [2010]


class TestTable:
    def test_missing_key(self):
        assert _table({}, "nope") == []

    def test_non_list(self):
        assert _table({"x": "not a list"}, "x") == []

    def test_list(self):
        assert _table({"x": [1, 2]}, "x") == [1, 2]


class TestChampionYears:
    def test_llm_champion_years_includes_golf(self):
        book = _book()
        llm = llm_champion_years(book)
        assert llm["Boys Cross Country"] == {2010, 2011, 2012, 2013}
        assert 2024 in llm["Golf"]

    def test_llm_champion_years_skips_golf_without_team_champion(self):
        book = _book()
        book["golf_results"].append(
            {"year": 2023, "classification": "Combined", "team_champion_school": "",
             "individual_gender": "Boys"}
        )
        assert 2023 not in llm_champion_years(book).get("Golf", set())

    def test_record_champion_years_string_format(self):
        """champion_years may arrive as the CSV '; '-joined string."""
        book = {"school_records": [
            {"sport": "Golf", "school": "Magruder", "champion_years": "2023; 2024"},
        ]}
        assert record_champion_years(book)["Golf"] == {2023, 2024}

    def test_llm_champion_years_excludes_pre_mpssaa(self):
        """Pre-MPSSAA precursor rows (notes starting 'Pre-MPSSAA') don't count."""
        book = {"championship_results": [
            {"sport": "Boys Soccer", "year": 1920, "classification": "Combined",
             "champion_school": "Catonsville", "notes": "Pre-MPSSAA"},
            {"sport": "Boys Soccer", "year": 1947, "classification": "A",
             "champion_school": "Kenwood", "notes": "Pre-MPSSAA; forfeit"},
            {"sport": "Boys Soccer", "year": 2024, "classification": "4A",
             "champion_school": "Montgomery Blair"},
        ], "golf_results": [], "school_records": []}
        assert llm_champion_years(book)["Boys Soccer"] == {2024}


class TestCrossPath:
    def test_agreement_is_ok(self):
        report = check_cross_path(_book())
        assert report["errors"] == 0
        assert report["sports"]["Boys Cross Country"]["status"] == "ok"
        assert report["sports"]["Boys Cross Country"]["coverage"] == 1.0

    def test_under_extraction_errors(self):
        book = _book()
        # Drop half the championship rows -> coverage drops below threshold.
        book["championship_results"] = book["championship_results"][:2]
        report = check_cross_path(book)
        assert report["errors"] == 1
        assert report["sports"]["Boys Cross Country"]["status"] == "error"

    def test_empty_records_is_ok(self):
        """A sport with table years but no school-record years doesn't error."""
        book = _book()
        book["championship_results"].append(
            {"sport": "Field Hockey", "year": 2000, "classification": "Combined",
             "champion_school": "Eastern"}
        )
        report = check_cross_path(book)
        # Field Hockey has table years, no record years -> coverage 1.0, ok.
        assert report["sports"]["Field Hockey"]["status"] == "ok"
        assert report["errors"] == 0


class TestDuplicateKeys:
    def test_no_duplicates(self):
        report = check_duplicate_keys(_book())
        assert report["errors"] == 0
        assert report["tables"]["championship_results"]["duplicate_key_count"] == 0

    def test_duplicates_flagged(self):
        book = _book()
        book["championship_results"].append(
            {"sport": "Boys Cross Country", "year": 2010, "classification": "Combined",
             "champion_school": "Northwood"}
        )
        report = check_duplicate_keys(book)
        assert report["errors"] == 1
        d = report["tables"]["championship_results"]["duplicates"]
        assert "Boys Cross Country | 2010 | Combined | Northwood" in d

    def test_xc_co_champions_share_coarse_key(self):
        """XC key is (sport, year, classification) — co-champions collide."""
        book = {"individual_xc_champions": [
            {"sport": "Boys Cross Country", "year": 2016, "classification": "1A",
             "name": "Runner A"},
            {"sport": "Boys Cross Country", "year": 2016, "classification": "1A",
             "name": "Runner B"},
        ]}
        report = check_duplicate_keys(book)
        assert report["tables"]["individual_xc_champions"]["duplicate_key_count"] == 1

    def test_stat_records_duplicate_keys_detected(self):
        book = {"stat_records": [
            {"sport": "Football", "category": "team", "record": "Most TDs, Season",
             "holder": "Fort Hill", "year": "2016"},
            {"sport": "Football", "category": "team", "record": "Most TDs, Season",
             "holder": "Fort Hill", "year": "2016"},
        ]}
        report = check_duplicate_keys(book)
        assert report["tables"]["stat_records"]["duplicate_key_count"] == 1
        assert report["errors"] == 1


class TestContinuity:
    def test_no_gaps(self):
        report = check_continuity(_book())
        assert report["warnings"] == 0

    def test_gap_reported(self):
        book = _book()
        # Remove 2011 from both paths -> gap in 2010-2013 span.
        book["championship_results"] = [r for r in book["championship_results"]
                                         if r["year"] != 2011]
        book["school_records"] = [r for r in book["school_records"]
                                   if r["school"] != "Atholton"]
        report = check_continuity(book)
        assert report["warnings"] == 1
        assert 2011 in report["sports"]["Boys Cross Country"]["missing_years"]

    def test_covid_year_exempt(self):
        book = _book()
        # Span 2019-2021 with 2020 missing -> COVID exemption, no warning.
        book["championship_results"] = [
            {"sport": "Boys Cross Country", "year": 2019, "classification": "Combined",
             "champion_school": "Northwood"},
            {"sport": "Boys Cross Country", "year": 2021, "classification": "Combined",
             "champion_school": "Northwood"},
        ]
        book["school_records"] = [
            {"sport": "Boys Cross Country", "school": "Northwood",
             "champion_years": [2019, 2021]},
        ]
        report = check_continuity(book)
        assert report["warnings"] == 0

    def test_single_year_no_span(self):
        book = {"championship_results": [
            {"sport": "Golf", "year": 2024, "classification": "Combined",
             "champion_school": "X"},
        ], "school_records": [], "golf_results": []}
        report = check_continuity(book)
        assert report["warnings"] == 0

    def test_pre_mpssaa_rows_dont_anchor_span(self):
        """A pre-MPSSAA precursor year must not extend a sport's continuity span."""
        book = {"championship_results": [
            {"sport": "Field Hockey", "year": 1946, "classification": "A",
             "champion_school": "Towson", "notes": "Pre-MPSSAA"},
            {"sport": "Field Hockey", "year": 2023, "classification": "1A",
             "champion_school": "Severna Park"},
            {"sport": "Field Hockey", "year": 2024, "classification": "1A",
             "champion_school": "Severna Park"},
        ], "school_records": [], "golf_results": []}
        report = check_continuity(book)
        # Span would be 1946-2024 (78 gaps) if 1946 counted; instead no gap.
        assert report["warnings"] == 0
        assert "Field Hockey" not in report["sports"]


class TestReferentialSchools:
    def test_no_missing(self):
        report = check_referential_schools(_book())
        assert report["warnings"] == 0

    def test_missing_school_warned(self):
        book = _book()
        book["championship_results"].append(
            {"sport": "Boys Cross Country", "year": 2014, "classification": "Combined",
             "champion_school": "Mystery School"}
        )
        report = check_referential_schools(book)
        assert report["warnings"] == 1
        assert "Mystery School" in report["sports"]["Boys Cross Country"]

    def test_co_champion_split_on_ampersand(self):
        book = _book()
        book["championship_results"].append(
            {"sport": "Boys Cross Country", "year": 2014, "classification": "Combined",
             "champion_school": "Northwood & Mystery School"}
        )
        report = check_referential_schools(book)
        # Northwood is known; Mystery School is not -> 1 warning.
        assert report["warnings"] == 1
        assert report["sports"]["Boys Cross Country"] == ["Mystery School"]

    def test_pre_mpssaa_schools_not_warned(self):
        """Pre-MPSSAA precursor schools aren't expected in school_records."""
        book = _book()
        book["championship_results"].append(
            {"sport": "Field Hockey", "year": 1946, "classification": "A",
             "champion_school": "Kenwood", "notes": "Pre-MPSSAA"}
        )
        report = check_referential_schools(book)
        assert report["warnings"] == 0


class TestEraFloors:
    def test_present_passes(self):
        # _book has Boys XC 1946? No — it starts at 2010. Need a 1946 row.
        book = _book()
        book["championship_results"].append(
            {"sport": "Boys Cross Country", "year": 1946, "classification": "Combined",
             "champion_school": "Old School"}
        )
        assert check_era_floors(book)["errors"] == 0

    def test_missing_floor_errors(self):
        report = check_era_floors(_book())
        # _book lacks 1946 Boys XC and has 2024 golf -> only XC errors.
        missing_sports = {m["sport"] for m in report["missing"]}
        assert "Boys Cross Country" in missing_sports
        assert "Golf" not in missing_sports
        assert report["errors"] == 1

    def test_constants_match_legacy_cache(self):
        assert ERA_FLOORS["Boys Cross Country"] == 1946
        assert ERA_FLOORS["Golf"] == 2024

    def test_absent_sport_skipped(self):
        # A season with no Boys XC or Golf (e.g. winter/spring) must not trip
        # the fall-sport era floors — the anchors only apply to present sports.
        book = _book()
        book["championship_results"] = [
            {"sport": "Wrestling", "year": 2024, "classification": "Combined",
             "champion_school": "Damascus"}
        ]
        book["golf_results"] = []
        assert check_era_floors(book)["errors"] == 0


class TestRegression:
    def test_no_changes_no_errors(self):
        head = _book()
        report = check_regression(_book(), head, None, allow_removals=False)
        assert report["errors"] == 0
        assert report["skipped"] is False

    def test_row_count_shrink_errors(self):
        head = _book()
        cur = _book()
        cur["championship_results"] = cur["championship_results"][:2]
        report = check_regression(cur, head, None, allow_removals=False)
        # 1 for the table shrink + 2 for the two disappeared natural keys.
        assert report["errors"] == 3
        assert report["row_count_shrinks"]["championship_results"]["delta"] == -2
        assert len(report["missing_head_keys"]["championship_results"]) == 2

    def test_row_count_shrink_suppressed_by_allow_removals(self):
        head = _book()
        cur = _book()
        cur["championship_results"] = cur["championship_results"][:2]
        report = check_regression(cur, head, None, allow_removals=True)
        assert report["errors"] == 0
        assert report["row_count_shrinks"]  # still reported

    def test_missing_head_key_errors(self):
        head = _book()
        cur = _book()
        # Change one row's champion_school so its natural key disappears.
        cur["championship_results"][0]["champion_school"] = "Somebody Else"
        report = check_regression(cur, head, None, allow_removals=False)
        assert report["errors"] == 1
        assert report["missing_head_keys"]["championship_results"]

    def test_added_rows_ok(self):
        """Growing a table is not a regression."""
        head = _book()
        cur = _book()
        cur["championship_results"].append(
            {"sport": "Boys Cross Country", "year": 2015, "classification": "Combined",
             "champion_school": "Newcomer"}
        )
        report = check_regression(cur, head, None, allow_removals=False)
        assert report["errors"] == 0

    def test_skipped_when_head_unavailable(self):
        report = check_regression(_book(), None, "no git", allow_removals=False)
        assert report["skipped"] is True
        assert report["errors"] == 0
        assert "no git" in report["skip_reason"]


class TestLoadHeadBook:
    def test_returns_none_when_not_tracked(self, tmp_path, monkeypatch):
        # tmp_path is not a git repo -> git show fails.
        monkeypatch.chdir(tmp_path)
        book, note = _load_head_book(Path("data/fall"))
        assert book is None
        assert note is not None

    def test_loads_real_head(self, monkeypatch):
        """The real repo has data/fall/record_book.json at HEAD."""
        monkeypatch.chdir(Path(__file__).parent)
        book, note = _load_head_book(Path("data/fall"))
        assert book is not None
        assert note is None
        assert "championship_results" in book


class TestBuildReport:
    def test_summary_passes_when_clean(self):
        # Make _book satisfy era floors + cross-path + no dupes.
        book = _book()
        book["championship_results"].append(
            {"sport": "Boys Cross Country", "year": 1946, "classification": "Combined",
             "champion_school": "Old School"}
        )
        book["school_records"].append(
            {"sport": "Boys Cross Country", "school": "Old School",
             "champion_years": [1946]}
        )
        # Golf 2024 already present in _book; add a school record for it too.
        book["school_records"].append(
            {"sport": "Golf", "school": "Magruder", "champion_years": [2024]}
        )
        report = build_report(book, Path("data/fall"), allow_removals=False)
        # Regression guard compares against the REAL HEAD book (different rows),
        # so it may report. Assert only the in-memory checks we control.
        assert report["checks"]["cross_path_champion_years"]["errors"] == 0
        assert report["checks"]["duplicate_keys"]["errors"] == 0
        assert report["checks"]["era_floors"]["errors"] == 0

    def test_failed_summary_exits_nonzero(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "fall"
        (data_dir).mkdir()
        (data_dir / "record_book.json").write_text(json.dumps(_book()))
        # _book is missing 1946 -> era floor error -> nonzero exit.
        rc = vrb.main([str(data_dir)])
        assert rc == 1
        out = (data_dir / "verification_report.json")
        assert out.exists()
        report = json.loads(out.read_text())
        assert report["summary"]["passed"] is False
        assert report["summary"]["errors"] >= 1

    def test_clean_book_passes_with_zero_errors(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "fall"
        data_dir.mkdir()
        book = _book()
        book["championship_results"].append(
            {"sport": "Boys Cross Country", "year": 1946, "classification": "Combined",
             "champion_school": "Old School"}
        )
        book["school_records"].extend([
            {"sport": "Boys Cross Country", "school": "Old School",
             "champion_years": [1946]},
            {"sport": "Golf", "school": "Magruder", "champion_years": [2024]},
        ])
        (data_dir / "record_book.json").write_text(json.dumps(book))
        rc = vrb.main([str(data_dir)])
        assert rc == 0

    def test_allow_removals_flag_suppresses_regression(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "fall"
        data_dir.mkdir()
        head = _book()
        cur = _book()
        cur["championship_results"] = cur["championship_results"][:2]
        (data_dir / "record_book.json").write_text(json.dumps(cur))
        # No HEAD book in tmp_path git -> regression skipped, but era floors
        # still fail (no 1946). So exit is nonzero regardless. Verify the flag
        # is threaded through by calling check_regression directly instead.
        report = check_regression(cur, head, None, allow_removals=True)
        assert report["allow_removals"] is True
        assert report["errors"] == 0


class TestTableKeys:
    def test_all_tables_present(self):
        expected = {
            "championship_results", "school_records", "individual_xc_champions",
            "individual_results", "golf_results", "sportsmanship_awards",
            "stat_records",
        }
        assert set(TABLE_KEYS) == expected

    def test_key_tuple(self):
        row = {"sport": "Golf", "year": 2024, "classification": "Combined"}
        assert _key_tuple(row, ("sport", "year")) == ("Golf", "2024")

    def test_key_tuple_missing_field_is_empty_string(self):
        assert _key_tuple({}, ("sport",)) == ("",)

    def test_key_tuple_null_field_matches_missing_field(self):
        # record_book.json always serializes every schema field, so an absent
        # value shows up as an explicit JSON null (Python None) rather than a
        # missing key. An older committed copy of the same row predates the
        # field and omits it outright. Both must hash to the same key, or the
        # regression guard misreports every such row as a lost natural key.
        current_row = {"sport": "Golf", "year": 2024, "individual_gender": None}
        legacy_row = {"sport": "Golf", "year": 2024}
        key_fields = ("sport", "year", "individual_gender")
        assert _key_tuple(current_row, key_fields) == _key_tuple(legacy_row, key_fields)