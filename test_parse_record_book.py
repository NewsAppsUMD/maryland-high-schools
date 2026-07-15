"""Tests for parse_record_book.py — page classifiers, regex parser, and helpers.

These tests use text extracted from the actual Fall, Winter, and Spring PDFs
to verify that classifiers and parsers work across all three seasons.
No LLM calls are made; only deterministic (regex/logic) code is tested.
"""

import json

import pytest
from pypdf import PdfReader

import parse_record_book
from parse_record_book import (
    ChampionshipResults,
    _table_years_in_text,
    _was_truncated,
    _years_in_rows,
    check_section_map,
    chunked,
    dedupe,
    detect_season,
    build_meta,
    extract_chunks_complete,
    extract_resilient,
    FALL_SECTIONS,
    school_records_long,
    source_label,
    SPRING_SECTIONS,
    TABLE_NAMES,
    WINTER_SECTIONS,
    write_combined_json,
    _years_as_strings,
    is_golf_results,
    is_individual_results,
    is_individual_xc,
    is_multicolumn_results,
    is_school_records,
    is_sportsmanship,
    is_year_class_table,
    llm_extract,
    parse_school_records,
    TruncationError,
)


# ── Fake LLM response plumbing (no network / no API key) ──────────────────────


class _FakeResponse:
    """Mimics the subset of the llm response object that llm_extract() reads."""

    def __init__(self, *, text="", tool_args=None, response_json=None):
        self._text = text
        self._tool_args = tool_args
        self.response_json = response_json or {}

    def text(self):
        return self._text

    def tool_calls(self):
        if self._tool_args is None:
            return []
        call = type("Call", (), {"arguments": self._tool_args})()
        return [call]


class _FakeModel:
    """Returns queued responses in order for each .prompt() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def prompt(self, prompt, schema=None, stream=False, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


@pytest.fixture
def patch_model(monkeypatch):
    def _install(responses):
        model = _FakeModel(responses)
        monkeypatch.setattr(parse_record_book, "_model", model)
        monkeypatch.setattr(parse_record_book, "get_model", lambda: model)
        return model

    return _install

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

    def test_is_golf_positive_split_era(self):
        # Split era (1993+) puts the classification before the dot leaders.
        assert is_golf_results("Team Champion 1A/2A..........Cambridge-South Dorchester (695)")
        assert is_golf_results("Team Champion 3A/4A ............Winston Churchill (637)")

    def test_is_golf_negative(self):
        assert not is_golf_results("YEAR CLASS CHAMPION COACH")

    def test_is_golf_negative_swimming_header(self):
        # Swimming's team-champ header has no dot leaders and must not match golf.
        assert not is_golf_results("Year Class Team Champion Coach Finalist Coach Site")

    def test_is_individual_results_track_header(self):
        assert is_individual_results("Event: 55m Dash\nYear Class Athlete-School-Mark\n2025 4A Fred Colvin—Fairmont Heights 6.6")

    def test_is_individual_results_tennis(self):
        assert is_individual_results("Year Boys Singles Champion Boys Doubles Champions\n1975 John Olson, Bowie 6-7")

    def test_is_individual_results_swimming(self):
        assert is_individual_results("Girls 200 Yard Freestyle\n2024 4A-3A Andrea Dworak, James Hubert Blake 1:46.72")

    def test_is_individual_results_negative(self):
        assert not is_individual_results("ALLEGANY\nCh: 1997, 1998")

    def test_is_multicolumn_lacrosse(self):
        assert is_multicolumn_results("Class 4A-3A Class 2A-1A\n1990 Severna Park 12-10")

    def test_is_sportsmanship_rejects_incidental_mention(self):
        # Sportsmanship mentioned deep in text should NOT match
        padding = "x" * 600
        assert not is_sportsmanship(padding + "\nSPORTSMANSHIP AWARD\n2024 Allegany")

    def test_is_school_records_uppercase(self):
        assert is_school_records("Aberdeen (16, 7-15)\nCH: 2003 (2AE)")

    def test_detect_season_fall(self):
        assert detect_season("pdfs/FallRecordBook2024.pdf") == "fall"

    def test_detect_season_winter(self):
        assert detect_season("pdfs/Winter record book.pdf") == "winter"

    def test_detect_season_spring(self):
        assert detect_season("pdfs/Spring record book 2025.pdf") == "spring"

    def test_detect_season_unknown_raises(self):
        # Silently defaulting to fall would mis-slice a non-fall PDF.
        with pytest.raises(ValueError):
            detect_season("pdfs/unknown.pdf")


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
        # Page 50 (Combined era) and pages 51-52 (split era) are all golf results.
        assert is_golf_results(fall_pages[50])
        assert is_golf_results(fall_pages[51])
        assert is_golf_results(fall_pages[52])

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

    def test_individual_results_indoor_track(self, winter_pages):
        # Page 35 has indoor track individual event champions
        assert is_individual_results(winter_pages[35])

    def test_individual_results_swimming(self, winter_pages):
        # Page 65 has swimming individual event champions
        assert is_individual_results(winter_pages[65])

    def test_school_records_football_uppercase(self, fall_pages):
        # Page 40 has Football school records with CH: format
        assert is_school_records(fall_pages[40])


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

    def test_multicolumn_lacrosse(self, spring_pages):
        # Page 12 has Girls Lacrosse two-column championship results
        assert is_multicolumn_results(spring_pages[12])

    def test_individual_results_tennis(self, spring_pages):
        # Page 30 has Tennis individual event results
        assert is_individual_results(spring_pages[30])

    def test_individual_results_track(self, spring_pages):
        # Page 50 has Girls Track individual event champions
        assert is_individual_results(spring_pages[50])

    def test_sportsmanship_not_incidental(self, spring_pages):
        # Page 12 mentions sportsmanship in trivia but is NOT a sportsmanship page
        assert not is_sportsmanship(spring_pages[12])


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

    def test_football_uppercase_codes_with_classifications(self):
        text = (
            "Allegany (29, 43-21)\n"
            "CH: 1978 (B), 1980 (B), 1983 (B),\n"
            "1988 (2A), 1989 (2A)\n"
            "RU: 1985 (B), 1986 (B)\n"
            "SF: 1982 (B), 1994 (2A)\n"
            "QF: 1987 (B), 1998 (1A)\n"
        )
        records = parse_school_records([text], "Football")
        assert len(records) == 1
        r = records[0]
        assert r["school"] == "Allegany"
        assert r["champion_years"] == [1978, 1980, 1983, 1988, 1989]
        assert r["runner_up_years"] == [1985, 1986]
        assert r["semifinalist_years"] == [1982, 1994]
        assert r["quarterfinal_years"] == [1987, 1998]

    def test_fall_football_school_records(self, fall_pages):
        records = parse_school_records(fall_pages[40:46], "Football")
        assert len(records) > 50
        allegany = [r for r in records if r["school"] == "Allegany"][0]
        assert 1978 in allegany["champion_years"]
        assert len(allegany["champion_years"]) == 8

    def test_spring_lacrosse_quarterfinals(self, spring_pages):
        records = parse_school_records(spring_pages[13:15], "Girls Lacrosse")
        bel_air = [r for r in records if "BEL AIR" in r["school"]]
        assert len(bel_air) == 1
        assert bel_air[0].get("quarterfinal_years", []) == [2024, 2025]


# ── dedupe() helper ──────────────────────────────────────────────────────────

CHAMP_KEY = ("sport", "year", "classification", "champion_school")


class TestDedupe:
    """Test the shared dedupe() helper used for all LLM-extracted tables."""

    def test_no_dupes(self):
        results = [
            {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill"},
            {"sport": "Soccer", "year": 2024, "classification": "3A", "champion_school": "Broadneck"},
        ]
        assert len(dedupe(results, CHAMP_KEY)) == 2

    def test_removes_exact_dupes(self):
        row = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill"}
        results = [row, dict(row)]  # same data
        assert len(dedupe(results, CHAMP_KEY)) == 1

    def test_preserves_first_occurrence(self):
        row1 = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill", "score": "2-0"}
        row2 = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill", "score": "2-1"}
        result = dedupe([row1, row2], CHAMP_KEY)
        assert len(result) == 1
        assert result[0]["score"] == "2-0"

    def test_different_years_kept(self):
        results = [
            {"sport": "Soccer", "year": 2023, "classification": "4A", "champion_school": "Churchill"},
            {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill"},
        ]
        assert len(dedupe(results, CHAMP_KEY)) == 2

    def test_empty(self):
        assert dedupe([], CHAMP_KEY) == []

    def test_conflict_is_reported(self, capsys):
        # Same key, differing score → must print a CONFLICT warning.
        row1 = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill", "score": "2-0"}
        row2 = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill", "score": "2-1"}
        result = dedupe([row1, row2], CHAMP_KEY, label="championship_results")
        assert len(result) == 1
        out = capsys.readouterr().out
        assert "CONFLICT" in out
        assert "2-0" in out and "2-1" in out

    def test_exact_dupe_is_not_flagged_as_conflict(self, capsys):
        row = {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill", "score": "2-0"}
        dedupe([row, dict(row)], CHAMP_KEY, label="championship_results")
        assert "CONFLICT" not in capsys.readouterr().out

    def test_xc_overlap_dupes_removed(self):
        # Reproduces the chunk-overlap duplication seen in individual_xc output.
        rows = [
            {"sport": "Girls Cross Country", "year": 2016, "classification": "4A", "name": "A", "school": "X"},
            {"sport": "Girls Cross Country", "year": 2016, "classification": "4A", "name": "A", "school": "X"},
            {"sport": "Girls Cross Country", "year": 2016, "classification": "3A", "name": "B", "school": "Y"},
        ]
        result = dedupe(rows, ("sport", "year", "classification"))
        assert len(result) == 2


# ── llm_extract() schema validation ──────────────────────────────────────────


class TestLlmExtractValidation:
    """llm_extract must validate raw model output through the Pydantic schema."""

    def test_applies_defaults_and_coerces_types_from_text(self, patch_model):
        # Year arrives as a string; boolean flags omitted entirely.
        raw = '{"results": [{"sport": "Boys Soccer", "year": "2024", '
        raw += '"classification": "4A", "champion_school": "Churchill"}]}'
        patch_model([_FakeResponse(text=raw)])
        out = llm_extract("prompt", ChampionshipResults)
        row = out["results"][0]
        assert row["year"] == 2024  # coerced str → int
        assert row["champion_undefeated"] is False  # default applied
        assert row["co_champion"] is False

    def test_reads_tool_call_arguments_fallback(self, patch_model):
        args = {"results": [{"sport": "Football", "year": 1999,
                             "classification": "1A", "champion_school": "Dunbar"}]}
        patch_model([_FakeResponse(tool_args=args)])
        out = llm_extract("prompt", ChampionshipResults)
        assert out["results"][0]["champion_school"] == "Dunbar"

    def test_retries_then_succeeds_on_invalid_first_response(self, patch_model):
        bad = _FakeResponse(text='{"results": [{"sport": "X"}]}')  # missing required fields
        good_args = {"results": [{"sport": "Volleyball", "year": 2010,
                                  "classification": "2A", "champion_school": "Glenelg"}]}
        model = patch_model([bad, _FakeResponse(tool_args=good_args)])
        out = llm_extract("prompt", ChampionshipResults, retries=2)
        assert model.calls == 2
        assert out["results"][0]["champion_school"] == "Glenelg"

    def test_raises_when_never_valid(self, patch_model):
        bad = _FakeResponse(text='{"results": [{"sport": "X"}]}')
        patch_model([bad, _FakeResponse(text='{"results": [{"sport": "Y"}]}')])
        with pytest.raises(RuntimeError):
            llm_extract("prompt", ChampionshipResults, retries=2)

    def test_truncation_raises_truncation_error(self, patch_model):
        # stop_reason=max_tokens must raise TruncationError, not RuntimeError,
        # and must not retry the identical prompt.
        model = patch_model([_FakeResponse(tool_args={}, response_json={"stop_reason": "max_tokens"})])
        with pytest.raises(TruncationError):
            llm_extract("prompt", ChampionshipResults, retries=2)
        assert model.calls == 1  # no pointless retry of the same prompt


class TestExtractResilient:
    """extract_resilient must split truncating input instead of crashing."""

    def test_no_split_when_ok(self):
        calls = []
        def fn(texts):
            calls.append(list(texts))
            return [{"year": 2024}]
        rows = extract_resilient(["page a", "page b"], fn, "t")
        assert rows == [{"year": 2024}]
        assert calls == [["page a", "page b"]]  # single call, no split

    def test_splits_page_list_on_truncation(self):
        # Truncates on the 2-page call, succeeds on each single page.
        def fn(texts):
            if len(texts) > 1:
                raise TruncationError("too big")
            return [{"page": texts[0]}]
        rows = extract_resilient(["A", "B"], fn, "t")
        assert {r["page"] for r in rows} == {"A", "B"}

    def test_splits_single_page_lines_on_truncation(self):
        # A single dense page truncates; splitting its lines lets each half through.
        page = "\n".join(f"line{i}" for i in range(10))
        def fn(texts):
            if len(texts[0].splitlines()) > 5:
                raise TruncationError("too big")
            return [{"n": len(texts[0].splitlines())}]
        rows = extract_resilient([page], fn, "t")
        assert sum(r["n"] for r in rows) == 10  # all lines covered across halves

    def test_skips_when_irreducible(self, capsys):
        # Always truncates, even at minimal size → warn and return [], never raise.
        def fn(texts):
            raise TruncationError("always")
        rows = extract_resilient(["one\ntwo"], fn, "stubborn")
        assert rows == []
        assert "skipping" in capsys.readouterr().out

    def test_runtime_error_also_handled(self):
        def fn(texts):
            if len(texts) > 1:
                raise RuntimeError("no content")
            return [{"ok": texts[0]}]
        rows = extract_resilient(["A", "B"], fn, "t")
        assert len(rows) == 2


# ── Section-map sanity check (Fix 4) ─────────────────────────────────────────


class TestCheckSectionMap:
    def test_fall_ranges_are_valid(self, fall_pages):
        assert check_section_map(fall_pages, FALL_SECTIONS) == []

    def test_winter_ranges_are_valid(self, winter_pages):
        assert check_section_map(winter_pages, WINTER_SECTIONS) == []

    def test_spring_ranges_are_valid(self, spring_pages):
        assert check_section_map(spring_pages, SPRING_SECTIONS) == []

    def test_detects_swapped_ranges(self, fall_pages):
        # Point "Football" at the Golf pages — the name won't be found there.
        bad = {"Football": (47, 53)}  # golf's real range
        problems = check_section_map(fall_pages, bad)
        assert len(problems) == 1
        assert "Football" in problems[0]

    def test_detects_range_past_end(self, fall_pages):
        bad = {"Football": (999, 1005)}
        problems = check_section_map(fall_pages, bad)
        assert len(problems) == 1
        assert "past the last page" in problems[0]

    def test_handles_ampersand_and_wrapped_names(self, winter_pages):
        # "Girls Swimming & Diving" renders as "Swimming &\nDiving" in the PDF.
        assert check_section_map(winter_pages, {"Girls Swimming & Diving": (60, 69)}) == []


# ── Completeness guard (Fix 3) ───────────────────────────────────────────────


class TestTableYearsInText:
    def test_line_start_years_only(self):
        text = "1974 Parkdale 28-8\n1975 Arundel 13-7\nMost points in 1985 Finals were 59"
        # 1985 is mid-line prose, must be ignored
        assert _table_years_in_text(text) == {1974, 1975}

    def test_indented_year(self):
        assert _table_years_in_text("   2001 Churchill") == {2001}

    def test_years_in_rows_coerces(self):
        rows = [{"year": "2001"}, {"year": 2002}, {"year": None}, {}]
        assert _years_in_rows(rows) == {2001, 2002}


class TestWasTruncated:
    def test_max_tokens_dict(self):
        assert _was_truncated(_FakeResponse(response_json={"stop_reason": "max_tokens"}))

    def test_length_finish_reason(self):
        assert _was_truncated(_FakeResponse(response_json={"finish_reason": "length"}))

    def test_list_shape(self):
        assert _was_truncated(_FakeResponse(response_json=[{"stop_reason": "max_tokens"}]))

    def test_normal_stop(self):
        assert not _was_truncated(_FakeResponse(response_json={"stop_reason": "end_turn"}))

    def test_missing_json(self):
        assert not _was_truncated(_FakeResponse(response_json=None))


class TestExtractChunksComplete:
    """The guard must re-extract page-by-page when year coverage falls short."""

    def test_no_reextract_when_complete(self):
        # One page listing three years; extractor returns all three.
        pages = [(0, "2001 A\n2002 B\n2003 C")]
        calls = []

        def extract_fn(pgs):
            calls.append(len(pgs))
            return [{"year": 2001}, {"year": 2002}, {"year": 2003}]

        rows = extract_chunks_complete(pages, extract_fn, "test")
        assert len(rows) == 3
        assert calls == [1]  # only the initial chunk pass, no re-extract

    def test_reextract_recovers_dropped_years(self, capsys):
        # Two pages, 10 table years; first pass "drops" all but 2 (simulated
        # truncation). Page-by-page pass returns everything.
        page1 = "\n".join(f"{y} School{y}" for y in range(2001, 2006))
        page2 = "\n".join(f"{y} School{y}" for y in range(2006, 2011))
        pages = [(0, page1), (1, page2)]

        state = {"first_multi_page_call": True}

        def extract_fn(pgs):
            combined = "\n".join(pgs)
            years = sorted(_table_years_in_text(combined))
            if len(pgs) > 1 and state["first_multi_page_call"]:
                state["first_multi_page_call"] = False
                return [{"year": years[0]}, {"year": years[1]}]  # drop the rest
            return [{"year": y} for y in years]

        rows = extract_chunks_complete(pages, extract_fn, "test champ")
        recovered = _years_in_rows(rows)
        assert recovered == set(range(2001, 2011))  # all years recovered
        assert "re-extracting page-by-page" in capsys.readouterr().out

    def test_no_year_pages_pass_through(self):
        pages = [(0, "header text with no leading years")]
        rows = extract_chunks_complete(pages, lambda pgs: [{"x": 1}], "test")
        assert rows == [{"x": 1, "source_pages": "0"}]


# ── Provenance (Fix 6) ────────────────────────────────────────────────────────


class TestProvenance:
    def test_source_label_single_page(self):
        assert source_label([62]) == "62"

    def test_source_label_span(self):
        assert source_label([62, 63]) == "62-63"

    def test_source_label_unordered(self):
        assert source_label([63, 62]) == "62-63"

    def test_chunks_stamped_with_page_range(self):
        pages = [(62, "2001 A"), (63, "2002 B")]
        rows = extract_chunks_complete(pages, lambda pgs: [{"year": 2001}], "t")
        # Single 2-page chunk (CHUNK=2, overlap=1 -> one chunk of both pages)
        assert rows[0]["source_pages"] == "62-63"

    def test_dedupe_ignores_source_pages_conflict(self, capsys):
        # Same entry from two overlapping chunks: identical except source_pages.
        row1 = {"sport": "S", "year": 2016, "classification": "4A",
                "name": "A", "source_pages": "10-11"}
        row2 = {"sport": "S", "year": 2016, "classification": "4A",
                "name": "A", "source_pages": "11-12"}
        result = dedupe([row1, row2], ("sport", "year", "classification"))
        assert len(result) == 1
        assert "CONFLICT" not in capsys.readouterr().out  # provenance diff is not a conflict

    def test_dedupe_still_flags_real_conflict_with_provenance(self, capsys):
        row1 = {"sport": "S", "year": 2016, "classification": "4A",
                "name": "A", "source_pages": "10-11"}
        row2 = {"sport": "S", "year": 2016, "classification": "4A",
                "name": "B", "source_pages": "11-12"}  # different name = real conflict
        dedupe([row1, row2], ("sport", "year", "classification"), "t")
        assert "CONFLICT" in capsys.readouterr().out


# ── School-record CSV shape (Fix 9) ──────────────────────────────────────────


class TestSchoolRecordSerialization:
    RECS = [
        {"sport": "GXC", "school": "Allegany", "school_slug": "allegany",
         "champion_years": [1997, 1998], "finalist_years": [1988],
         "semifinalist_years": [], "runner_up_years": [], "source_pages": "4-5"},
    ]

    def test_long_format_one_row_per_year(self):
        long = school_records_long(self.RECS)
        assert len(long) == 3  # two champion + one finalist
        champs = [r for r in long if r["result"] == "champion"]
        assert sorted(r["year"] for r in champs) == [1997, 1998]
        assert all(r["school_slug"] == "allegany" for r in long)
        assert all("source_pages" in r for r in long)

    def test_wide_years_joined_not_list_repr(self):
        wide = _years_as_strings(self.RECS)
        assert wide[0]["champion_years"] == "1997;1998"
        assert "[" not in wide[0]["champion_years"]  # never a python list repr

    def test_wide_does_not_mutate_original(self):
        _years_as_strings(self.RECS)
        assert self.RECS[0]["champion_years"] == [1997, 1998]  # JSON keeps arrays

    def test_quarterfinal_years_handled_when_present(self):
        recs = [{"sport": "GLax", "school": "Bel Air", "champion_years": [],
                 "quarterfinal_years": [2024, 2025]}]
        long = school_records_long(recs)
        assert [r["year"] for r in long if r["result"] == "quarterfinal"] == [2024, 2025]


# ── Website JSON structure (Fix 8) ───────────────────────────────────────────


class TestBuildMeta:
    def test_meta_fields(self):
        tables = {n: [] for n in TABLE_NAMES}
        tables["championship_results"] = [{"a": 1}, {"a": 2}]
        report = {"summary": {"errors": 0, "warnings": 3, "passed": True}}
        meta = build_meta("fall", "pdfs/FallRecordBook2024.pdf", tables, report)
        assert meta["season"] == "fall"
        assert meta["source_pdf"] == "FallRecordBook2024.pdf"
        assert meta["row_counts"]["championship_results"] == 2
        assert meta["verification"]["passed"] is True
        # generated_at must be ISO-parseable
        import datetime as _dt
        assert _dt.datetime.fromisoformat(meta["generated_at"])


class TestWriteCombinedJson:
    def test_merges_seasons_and_tags_rows(self, tmp_path):
        (tmp_path / "fall").mkdir()
        (tmp_path / "winter").mkdir()
        (tmp_path / "fall" / "record_book.json").write_text(
            json.dumps({"championship_results": [{"sport": "Soccer", "year": 2024}]})
        )
        (tmp_path / "winter" / "record_book.json").write_text(
            json.dumps({"championship_results": [{"sport": "Basketball", "year": 2023}]})
        )
        path = write_combined_json(tmp_path)
        combined = json.loads(path.read_text())
        assert combined["meta"]["seasons"] == ["fall", "winter"]
        seasons = {r["season"] for r in combined["championship_results"]}
        assert seasons == {"fall", "winter"}

    def test_no_seasons_returns_none(self, tmp_path):
        assert write_combined_json(tmp_path) is None


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
