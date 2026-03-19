"""Tests for parse_record_book.py — page classifiers, regex parser, and helpers.

These tests use text extracted from the actual Fall, Winter, and Spring PDFs
to verify that classifiers and parsers work across all three seasons.
No LLM calls are made; only deterministic (regex/logic) code is tested.
"""

import pytest
from pypdf import PdfReader

from parse_record_book import (
    chunked,
    is_golf_results,
    is_individual_xc,
    is_multicolumn_results,
    is_school_records,
    is_sportsmanship,
    is_year_class_table,
    parse_school_records,
)

# ── Helpers to load real PDF pages ────────────────────────────────────────────

FALL_PDF = "pdfs/FallRecordBook2024.pdf"
WINTER_PDF = "pdfs/Winter record book.pdf"
SPRING_PDF = "pdfs/Spring record book 2025.pdf"


@pytest.fixture(scope="module")
def fall_pages():
    return [p.extract_text() or "" for p in PdfReader(FALL_PDF).pages]


@pytest.fixture(scope="module")
def winter_pages():
    return [p.extract_text() or "" for p in PdfReader(WINTER_PDF).pages]


@pytest.fixture(scope="module")
def spring_pages():
    return [p.extract_text() or "" for p in PdfReader(SPRING_PDF).pages]


# ── chunked() ────────────────────────────────────────────────────────────────


class TestChunked:
    def test_empty(self):
        assert chunked([], 4) == []

    def test_single_element(self):
        assert chunked(["a"], 4) == [["a"]]

    def test_exact_fit(self):
        # Default overlap=1, so step=3 for size=4 → two chunks
        assert chunked([1, 2, 3, 4], 4) == [[1, 2, 3, 4], [4]]

    def test_exact_fit_no_overlap(self):
        assert chunked([1, 2, 3, 4], 4, overlap=0) == [[1, 2, 3, 4]]

    def test_overlap(self):
        result = chunked([1, 2, 3, 4, 5, 6], 4, overlap=1)
        assert result == [[1, 2, 3, 4], [4, 5, 6]]

    def test_no_overlap(self):
        result = chunked([1, 2, 3, 4, 5, 6], 3, overlap=0)
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_overlap_larger_than_chunk(self):
        # overlap >= size → step=max(1,1)=1, sliding window with trailing partial
        result = chunked([1, 2, 3], 2, overlap=2)
        assert result == [[1, 2], [2, 3], [3]]


# ── Page classifiers on synthetic text ────────────────────────────────────────


class TestClassifiersSynthetic:
    """Test classifiers against hand-crafted strings."""

    def test_is_school_records_positive(self):
        assert is_school_records("ALLEGANY\nCh: 1997, 1998\nFn: 1988")

    def test_is_school_records_negative(self):
        assert not is_school_records("YEAR CLASS CHAMPION COACH")

    def test_is_year_class_table_positive(self):
        assert is_year_class_table("YEAR CLASS CHAMPION COACH FINALIST COACH")

    def test_is_year_class_table_team_champion_variant(self):
        assert is_year_class_table("Year Class Team Champion Coach 2nd Place Coach Site")

    def test_is_year_class_table_negative(self):
        assert not is_year_class_table("ALLEGANY\nCh: 1997")

    def test_is_multicolumn_positive_4A_3A(self):
        assert is_multicolumn_results("CLASS 4A CLASS 3A CLASS 2A CLASS 1A")

    def test_is_multicolumn_positive_AA_A(self):
        assert is_multicolumn_results("CLASS AA CLASS A CLASS B CLASS C")

    def test_is_multicolumn_negative(self):
        assert not is_multicolumn_results("YEAR CLASS CHAMPION COACH")

    def test_is_individual_xc_positive(self):
        assert is_individual_xc("15:07.0  2.5 MILES")

    def test_is_individual_xc_positive_3_miles(self):
        assert is_individual_xc("3.0 MILES  some text")

    def test_is_individual_xc_negative(self):
        assert not is_individual_xc("YEAR CLASS CHAMPION COACH")

    def test_is_sportsmanship_positive(self):
        assert is_sportsmanship("SPORTSMANSHIP AWARD\n2024 Allegany")

    def test_is_sportsmanship_needs_year(self):
        assert not is_sportsmanship("SPORTSMANSHIP AWARD without a year")

    def test_is_golf_positive(self):
        assert is_golf_results("Team Champion......Magruder (610)")

    def test_is_golf_negative(self):
        assert not is_golf_results("YEAR CLASS CHAMPION COACH")


# ── Page classifiers on real Fall PDF pages ──────────────────────────────────


class TestClassifiersFall:
    def test_school_records_page(self, fall_pages):
        # Page 4 (0-indexed) has Girls XC school records
        assert is_school_records(fall_pages[4])

    def test_year_class_championship_table(self, fall_pages):
        # Page 7 has Girls XC championship table
        assert is_year_class_table(fall_pages[7]) or is_year_class_table(fall_pages[9])

    def test_multicolumn_football(self, fall_pages):
        # Page 38 has football multi-column results
        assert is_multicolumn_results(fall_pages[38])

    def test_individual_xc(self, fall_pages):
        # Page 10 has individual XC champions
        assert is_individual_xc(fall_pages[10])

    def test_golf_results(self, fall_pages):
        # Page 50 has golf results
        assert is_golf_results(fall_pages[50])

    def test_sportsmanship(self, fall_pages):
        # Page 65 has soccer sportsmanship awards
        assert is_sportsmanship(fall_pages[65])


# ── Page classifiers on real Winter PDF pages ────────────────────────────────


class TestClassifiersWinter:
    def test_school_records_basketball(self, winter_pages):
        # Page 8 has Girls Basketball school records
        assert is_school_records(winter_pages[8])

    def test_school_records_boys_basketball(self, winter_pages):
        # Page 16 has Boys Basketball school records
        assert is_school_records(winter_pages[16])

    def test_multicolumn_basketball(self, winter_pages):
        # Page 4 has Girls Basketball multi-column results
        assert is_multicolumn_results(winter_pages[4])

    def test_year_class_indoor_track(self, winter_pages):
        # Page 22 has indoor track championship table
        assert is_year_class_table(winter_pages[22])

    def test_sportsmanship_basketball(self, winter_pages):
        # Page 7 has Girls Basketball sportsmanship
        assert is_sportsmanship(winter_pages[7])

    def test_sportsmanship_boys_basketball(self, winter_pages):
        # Page 20 has Boys Basketball sportsmanship
        assert is_sportsmanship(winter_pages[20])

    def test_sportsmanship_wrestling(self, winter_pages):
        # Page 99 has wrestling sportsmanship
        assert is_sportsmanship(winter_pages[99])

    def test_swimming_has_year_class_table(self, winter_pages):
        # Page 62 has swimming team championship table
        assert is_year_class_table(winter_pages[62])


# ── Page classifiers on real Spring PDF pages ────────────────────────────────


class TestClassifiersSpring:
    def test_school_records_baseball(self, spring_pages):
        # Page 7 has Baseball school records
        assert is_school_records(spring_pages[7])

    def test_school_records_softball(self, spring_pages):
        # Page 26 has Softball school records
        assert is_school_records(spring_pages[26])

    def test_multicolumn_baseball(self, spring_pages):
        # Page 5 has Baseball multi-column results
        assert is_multicolumn_results(spring_pages[5])

    def test_year_class_girls_track(self, spring_pages):
        # Page 36 has Girls Track championship table
        assert is_year_class_table(spring_pages[36])

    def test_sportsmanship_softball(self, spring_pages):
        # Page 28 has Softball sportsmanship
        assert is_sportsmanship(spring_pages[28])

    def test_lacrosse_school_records(self, spring_pages):
        # Page 13 has Girls Lacrosse school records (with Qf: entries)
        assert is_school_records(spring_pages[13])

    def test_boys_lacrosse_school_records(self, spring_pages):
        # Page 18 has Boys Lacrosse school records
        assert is_school_records(spring_pages[18])


# ── Classifier exclusivity ───────────────────────────────────────────────────


class TestClassifierExclusivity:
    """Pages should generally match at most one primary classifier."""

    def test_school_records_not_championship(self, fall_pages):
        page = fall_pages[4]  # school records page
        assert is_school_records(page)
        assert not is_multicolumn_results(page)
        assert not is_individual_xc(page)
        assert not is_golf_results(page)

    def test_multicolumn_not_school_records(self, fall_pages):
        page = fall_pages[38]  # football multi-column
        assert is_multicolumn_results(page)
        assert not is_golf_results(page)
        assert not is_individual_xc(page)


# ── parse_school_records() ───────────────────────────────────────────────────


class TestParseSchoolRecords:
    """Test the regex-based school record parser."""

    def test_simple_record(self):
        text = "ALLEGANY\nCh: 1997, 1998\nFn: 1988"
        records = parse_school_records([text], "Test Sport")
        assert len(records) == 1
        r = records[0]
        assert r["sport"] == "Test Sport"
        assert r["school"] == "ALLEGANY"
        assert r["champion_years"] == [1997, 1998]
        assert r["finalist_years"] == [1988]

    def test_multiple_schools(self):
        text = (
            "ALLEGANY\n"
            "Ch: 1997, 1998\n"
            "Fn: 1988\n"
            "ATHOLTON\n"
            "Ch: 1987, 1988, 1989\n"
            "Fn: 1986\n"
        )
        records = parse_school_records([text], "Girls Cross Country")
        assert len(records) == 2
        assert records[0]["school"] == "ALLEGANY"
        assert records[1]["school"] == "ATHOLTON"

    def test_wrapped_years(self):
        text = (
            "BETHESDA-CHEVY CHASE\n"
            "Ch: 1976, 1977, 1978, 2002, 2011,\n"
            "2012, 2024\n"
            "Fn: 1979, 2013, 2014\n"
        )
        records = parse_school_records([text], "Girls Cross Country")
        assert len(records) == 1
        assert 2024 in records[0]["champion_years"]
        assert 1976 in records[0]["champion_years"]
        assert len(records[0]["champion_years"]) == 7

    def test_semifinalist_and_runner_up(self):
        text = (
            "SCHOOL NAME\n"
            "Ch: 2000\n"
            "Sf: 2001, 2002\n"
            "RU: 2003\n"
        )
        records = parse_school_records([text], "Football")
        assert len(records) == 1
        assert records[0]["semifinalist_years"] == [2001, 2002]
        assert records[0]["runner_up_years"] == [2003]

    def test_no_records_yields_empty(self):
        text = "YEAR CLASS CHAMPION COACH FINALIST COACH\n2024 4A Churchill John Doe"
        records = parse_school_records([text], "Soccer")
        assert records == []

    def test_fall_pdf_school_records(self, fall_pages):
        # Pages 4-5 have Girls XC school records
        records = parse_school_records(fall_pages[4:6], "Girls Cross Country")
        assert len(records) > 10  # Should find many schools
        schools = {r["school"] for r in records}
        assert "ALLEGANY" in schools
        assert "ATHOLTON" in schools

    def test_winter_pdf_school_records(self, winter_pages):
        # Pages 8-9 have Girls Basketball school records
        records = parse_school_records(winter_pages[8:10], "Girls Basketball")
        assert len(records) > 10
        schools = {r["school"] for r in records}
        assert "ABERDEEN" in schools

    def test_spring_pdf_school_records(self, spring_pages):
        # Pages 7-8 have Baseball school records
        records = parse_school_records(spring_pages[7:9], "Baseball")
        assert len(records) > 10
        schools = {r["school"] for r in records}
        assert "ALLEGANY" in schools
        assert "ARUNDEL" in schools

    def test_fall_allegany_xc_years(self, fall_pages):
        records = parse_school_records(fall_pages[4:6], "Girls Cross Country")
        allegany = [r for r in records if r["school"] == "ALLEGANY"][0]
        assert allegany["champion_years"] == [1997, 1998]
        assert allegany["finalist_years"] == [1988]

    def test_winter_aberdeen_basketball_years(self, winter_pages):
        records = parse_school_records(winter_pages[8:10], "Girls Basketball")
        aberdeen = [r for r in records if r["school"] == "ABERDEEN"][0]
        assert aberdeen["champion_years"] == [2012, 2013]
        assert aberdeen["finalist_years"] == [2011]

    def test_spring_arundel_baseball_years(self, spring_pages):
        records = parse_school_records(spring_pages[7:9], "Baseball")
        arundel = [r for r in records if r["school"] == "ARUNDEL"][0]
        assert 1976 in arundel["champion_years"]
        assert 1977 in arundel["champion_years"]
        assert 2006 in arundel["champion_years"]


# ── Championship dedup logic ─────────────────────────────────────────────────


class TestChampionshipDedup:
    """Test the dedup logic used in main() extracted here for unit testing."""

    @staticmethod
    def dedup(results: list[dict]) -> list[dict]:
        seen: set[tuple] = set()
        unique: list[dict] = []
        for r in results:
            key = (
                r.get("sport", ""),
                r.get("year", ""),
                r.get("classification", ""),
                r.get("champion_school", ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def test_no_dupes(self):
        results = [
            {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill"},
            {"sport": "Soccer", "year": 2024, "classification": "3A", "champion_school": "Broadneck"},
        ]
        assert len(self.dedup(results)) == 2

    def test_removes_exact_dupes(self):
        row = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill"}
        results = [row, dict(row)]  # same data
        assert len(self.dedup(results)) == 1

    def test_preserves_first_occurrence(self):
        row1 = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill", "score": "2-0"}
        row2 = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill", "score": "2-1"}
        result = self.dedup([row1, row2])
        assert len(result) == 1
        assert result[0]["score"] == "2-0"

    def test_different_years_kept(self):
        results = [
            {"sport": "Soccer", "year": 2023, "classification": "4A", "champion_school": "Churchill"},
            {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill"},
        ]
        assert len(self.dedup(results)) == 2

    def test_empty(self):
        assert self.dedup([]) == []


# ── Cross-season classifier coverage ─────────────────────────────────────────


class TestCrossSeasonCoverage:
    """Verify that key page types are correctly identified in each season."""

    def _count_matches(self, pages, classifier, start, end):
        return sum(1 for p in pages[start:end] if classifier(p))

    def test_fall_has_school_records(self, fall_pages):
        # Girls XC section (pages 3-13)
        count = self._count_matches(fall_pages, is_school_records, 3, 13)
        assert count >= 2, f"Expected >=2 school record pages in fall GXC, got {count}"

    def test_winter_has_school_records(self, winter_pages):
        # Girls Basketball section (pages 3-11)
        count = self._count_matches(winter_pages, is_school_records, 3, 11)
        assert count >= 1, f"Expected >=1 school record pages in winter GBB, got {count}"

    def test_spring_has_school_records(self, spring_pages):
        # Baseball section (pages 3-10)
        count = self._count_matches(spring_pages, is_school_records, 3, 10)
        assert count >= 1, f"Expected >=1 school record pages in spring baseball, got {count}"

    def test_fall_has_championship_tables(self, fall_pages):
        # Football section should have multi-column results
        count = self._count_matches(fall_pages, is_multicolumn_results, 35, 47)
        assert count >= 1, f"Expected >=1 multi-column pages in fall football, got {count}"

    def test_winter_has_championship_tables(self, winter_pages):
        # Girls Basketball should have multi-column results
        count = self._count_matches(winter_pages, is_multicolumn_results, 3, 11)
        assert count >= 1, f"Expected >=1 multi-column pages in winter GBB, got {count}"

    def test_spring_has_championship_tables(self, spring_pages):
        # Baseball should have multi-column results
        count = self._count_matches(spring_pages, is_multicolumn_results, 3, 10)
        assert count >= 1, f"Expected >=1 multi-column pages in spring baseball, got {count}"

    def test_all_seasons_have_sportsmanship(self, fall_pages, winter_pages, spring_pages):
        fall_count = sum(1 for p in fall_pages if is_sportsmanship(p))
        winter_count = sum(1 for p in winter_pages if is_sportsmanship(p))
        spring_count = sum(1 for p in spring_pages if is_sportsmanship(p))
        assert fall_count >= 1, "Fall should have sportsmanship pages"
        assert winter_count >= 1, "Winter should have sportsmanship pages"
        assert spring_count >= 1, "Spring should have sportsmanship pages"
