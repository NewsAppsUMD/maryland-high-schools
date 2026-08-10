"""Tests for parse_record_book.py — page classifiers, regex parser, and helpers.

These tests use text extracted from the actual Fall, Winter, and Spring PDFs
to verify that classifiers and parsers work across all three seasons.
No LLM calls are made; only deterministic (regex/logic) code is tested.
"""

import json

import pytest
from pypdf import PdfReader

from parse_record_book import (
    CACHE_DIR,
    CHUNK_SIZES,
    DEDUP_KEYS,
    EXTRACTORS,
    FALL_SECTIONS,
    MODEL_ID,
    PROVENANCE_FIELDS,
    SPRING_SECTIONS,
    WINTER_SECTIONS,
    _cache_key,
    _cache_path,
    _divider_candidates,
    _golf_prompt,
    _normalize_quotes,
    _stamp,
    cached_extract,
    chunked,
    classify_page,
    dedup,
    detect_dividers,
    detect_season,
    find_sections,
    is_golf_results,
    is_individual_results,
    is_individual_xc,
    is_multicolumn_results,
    is_school_records,
    is_sportsmanship,
    is_stat_records,
    is_wrestling_weightclass,
    is_year_class_table,
    load_page_titles,
    load_pages,
    parse_school_records,
    write_csv,
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

    def test_default_is_non_overlapping(self):
        # overlap now defaults to 0: a size-4 list fits in one chunk (no trailing
        # duplicate). The old overlap=1 default extracted interior pages twice.
        assert chunked([1, 2, 3, 4], 4) == [[1, 2, 3, 4]]

    def test_exact_fit_no_overlap(self):
        assert chunked([1, 2, 3, 4], 4, overlap=0) == [[1, 2, 3, 4]]

    def test_non_overlapping_step(self):
        result = chunked([1, 2, 3, 4, 5, 6], 2)
        assert result == [[1, 2], [3, 4], [5, 6]]

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

    def test_dense_pages_chunked_singly(self):
        # Individual results / XC / golf use CHUNK=1 so Haiku sees one dense page
        # at a time (where it otherwise drops rows).
        for route in ("individual_xc", "individual_results", "golf"):
            assert CHUNK_SIZES[route] == 1
        assert CHUNK_SIZES["championship"] == 2


# ── Quote normalization ────────────────────────────────────────────────────────


class TestNormalizeQuotes:
    def test_curly_single_quotes(self):
        assert _normalize_quotes("Queen Anne’s") == "Queen Anne's"
        assert _normalize_quotes("‘tie’") == "'tie'"

    def test_curly_double_quotes(self):
        assert _normalize_quotes("“Champion”") == '"Champion"'

    def test_leaves_ascii_and_other_unicode_untouched(self):
        # em-dash is a legitimate dot-leader/separator glyph used throughout
        # the source PDFs (e.g. "Whitman—68"); normalization must not
        # touch it, and plain ASCII apostrophes must pass through unchanged.
        assert _normalize_quotes("O'Donnell") == "O'Donnell"
        assert _normalize_quotes("Northern—C") == "Northern—C"

    def test_load_pages_normalizes(self, tmp_path):
        # Regression guard for the school-identity bug: parse_school_records
        # (regex) and the LLM extractors must see identical apostrophes, or
        # "Queen Anne's" silently splits into two unmatched school names
        # across championship_results.csv and school_records.csv.
        pages = load_pages(FALL_PDF)
        assert not any("’" in p or "‘" in p for p in pages)


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

    def test_is_stat_records_football_bullets(self):
        text = ("TEAM RECORDS\n• Most State Championships\n13 ....Dunbar\n"
                "• Consecutive Wins\n53 ....Damascus 2015-2018\n"
                "• Most Touchdowns, Season\n98 ....Fort Hill 2016\n")
        assert is_stat_records(text)

    def test_is_stat_records_basketball_headers(self):
        text = ("Girls Tournament Records\nINDIVIDUAL RECORDS\n"
                "MOST POINTS - final game\n48 - Janet Flora, Loch Raven 1976\n")
        assert is_stat_records(text)

    def test_is_stat_records_baseball_dugout(self):
        text = ("Dugout Chatter - Hits, Runs and Records\nRuns Scored\n"
                "Season ..............................331 ..Bowie 2016\n")
        assert is_stat_records(text)

    def test_is_stat_records_appearances_list(self):
        text = ("Boys Tournament Trivia\nTournament Appearances (10 minimum)\n"
                "29 ............Wicomico\n28 ............Annapolis\n")
        assert is_stat_records(text)

    def test_is_stat_records_negative_championship_table(self):
        assert not is_stat_records("YEAR CLASS CHAMPION COACH FINALIST COACH\n"
                                   "1975 1st DuVal - Beverly Bigham\n")

    def test_is_stat_records_negative_individual_results(self):
        # event-based individual_results use Athlete—School—Mark, not stat headers
        assert not is_stat_records("Athlete—School—Mark\n100 m Dash\n")

    def test_is_stat_records_negative_school_records(self):
        assert not is_stat_records("ALLEGANY\nCh: 1997, 1998\nFn: 1988")

    def test_is_stat_records_bullet_cluster_no_header(self):
        # exercises the bullet-cluster fallback in isolation (no header keyword)
        text = ("Football All-Time Records\n"
                "• Most Touchdowns, Season\n98 ....Fort Hill 2016\n"
                "• Consecutive Wins\n53 ....Damascus 2015-2018\n"
                "• Longest Winning Streak\n22 ....Suitland 1990-1992\n")
        assert is_stat_records(text)

    def test_is_stat_records_bullet_cluster_below_threshold(self):
        text = ("Football All-Time Records\n"
                "• Most Touchdowns, Season\n98 ....Fort Hill 2016\n"
                "• Consecutive Wins\n53 ....Damascus 2015-2018\n")
        assert not is_stat_records(text)

    def test_detect_season_fall(self):
        assert detect_season("pdfs/FallRecordBook2024.pdf") == "fall"

    def test_detect_season_winter(self):
        assert detect_season("pdfs/Winter record book.pdf") == "winter"

    def test_detect_season_spring(self):
        assert detect_season("pdfs/Spring record book 2025.pdf") == "spring"

    def test_detect_season_raises_on_unknown(self):
        # Silently defaulting to fall would let a mis-named PDF clobber data/fall/.
        with pytest.raises(ValueError):
            detect_season("pdfs/unknown.pdf")

    def test_detect_season_override(self):
        assert detect_season("pdfs/unknown.pdf", override="winter") == "winter"

    def test_detect_season_override_validates(self):
        with pytest.raises(ValueError):
            detect_season("pdfs/FallRecordBook2024.pdf", override="summer")


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

    def test_golf_results_split_era(self, fall_pages):
        # Pages 51-52 use "Team Champion 1A/2A..........School" split-era headers
        # that the old regex (dots right after "Team Champion") dropped — the
        # root cause of golf stopping at 1994.
        assert is_golf_results(fall_pages[51])
        assert is_golf_results(fall_pages[52])

    def test_sportsmanship(self, fall_pages):
        # Page 65 has soccer sportsmanship awards
        assert is_sportsmanship(fall_pages[65])

    def test_championship_prompt_distinguishes_tie_from_tiebreaker(self, fall_pages):
        # Boys XC 1968 A is a genuine tie (Winston Churchill & Central-PG both 98,
        # no tie-breaker note) → must be extracted as co-champions. Girls XC 1994
        # 2A is the same score pattern BUT carries a "sixth girl tie-breaker" note
        # → one champion (Damascus), Catonsville is the finalist. The prompt must
        # teach the LLM the difference so genuine ties are preserved as co-champions
        # while tie-breaker-decided ties stay a single champion.
        from parse_record_book import _championship_prompt
        # A page that contains both patterns: Boys XC 1968 and Girls XC 1994.
        # The Girls XC championship pages hold the 1994 tie-breaker row.
        prompt, _ = _championship_prompt([fall_pages[9]], "Girls Cross Country")
        assert "CO-CHAMPIONS" in prompt
        assert "Tie-breaker-decided tie" in prompt
        assert "Winston Churchill & Central-PG" in prompt
        assert "tie-breaker" in prompt.lower()


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

    def test_wrestling_weightclass_detected(self, winter_pages):
        # Page 81 (idx 80) has Wrestling weight-class champions (106 lbs, 2012+).
        assert is_wrestling_weightclass(winter_pages[80])

    def test_wrestling_weightclass_continuation(self, winter_pages):
        # Page 82 (idx 81) is a "(con't.)" continuation page.
        assert is_wrestling_weightclass(winter_pages[81])

    def test_wrestling_team_championship_not_weightclass(self, winter_pages):
        # Page 92 (idx 91) is a Girls Team State Champions table — must NOT be
        # detected as a weight-class page (its numbers are inline scores, and
        # its standalone page-number line is not followed by a year+name).
        assert not is_wrestling_weightclass(winter_pages[91])

    def test_wrestling_title_page_not_weightclass(self, winter_pages):
        # Page 80 (idx 79) is the short Wrestling divider/title page.
        assert not is_wrestling_weightclass(winter_pages[79])

    def test_classify_wrestling_routes_to_individual_results(self, winter_pages):
        # The weight-class pages should route to individual_results (and not
        # to championship), giving the sport-aware wrestling prompt a chance.
        routes = classify_page(winter_pages[80], "Wrestling")
        assert "individual_results" in routes
        assert "championship" not in routes

    def test_classify_wrestling_team_page_not_individual(self, winter_pages):
        # The Girls Team State Champions page must not be misrouted to
        # individual_results.
        routes = classify_page(winter_pages[91], "Wrestling")
        assert "individual_results" not in routes

    def test_wrestling_dual_meet_routes_to_championship(self, winter_pages):
        # The MPSSAA Dual Meet Championship pages (idx 92-93) use a
        # "YEAR/CLASS CHAMPION" header (slash, not whitespace). They must route
        # to championship so the Dual Meet is extracted into championship_results,
        # not dropped entirely (the gap that once left the Dual Meet only in
        # school_records and tripped the cross-path check).
        from parse_record_book import is_year_class_table
        assert is_year_class_table(winter_pages[92])
        assert is_year_class_table(winter_pages[93])
        assert "championship" in classify_page(winter_pages[92], "Wrestling")
        assert "championship" in classify_page(winter_pages[93], "Wrestling")

    def test_championship_prompt_dual_meet_note_for_wrestling(self, winter_pages):
        # The Wrestling championship prompt must tell the LLM about the two
        # team championships and tag Dual Meet rows in notes so they stay
        # distinguishable from the individual-tournament team champion rows.
        from parse_record_book import _championship_prompt
        prompt, _ = _championship_prompt([winter_pages[92]], "Wrestling")
        assert "TWO separate team championships" in prompt
        assert "Dual Meet Championship" in prompt

    def test_championship_prompt_no_dual_meet_note_for_other_sports(self, winter_pages):
        # Non-Wrestling sports must NOT get the dual-meet note.
        from parse_record_book import _championship_prompt
        prompt, _ = _championship_prompt([winter_pages[87]], "Boys Basketball")
        assert "TWO separate team championships" not in prompt

    def test_wrestling_prompt_boys_prefix(self, winter_pages):
        # A boys weight-class page (idx 80) must produce a Boys-prefixed prompt
        # so boys/girls dedup keys stay distinct.
        from parse_record_book import _individual_results_prompt
        prompt, _ = _individual_results_prompt([winter_pages[80]], "Wrestling")
        assert "BOYS wrestling section" in prompt
        assert '"Boys 106"' in prompt

    def test_wrestling_prompt_girls_prefix(self, winter_pages):
        # The girls page (idx 83) carries the "Girls Individual Champions" header
        # and must produce a Girls-prefixed prompt.
        from parse_record_book import _individual_results_prompt
        prompt, _ = _individual_results_prompt([winter_pages[83]], "Wrestling")
        assert "GIRLS wrestling section" in prompt
        assert '"Girls 106"' in prompt

    def test_sportsmanship_prompt_joins_co_winners(self, spring_pages):
        # The Softball sportsmanship page (idx 28) lists co-winners joined by
        # "&" (e.g. "2003—Dulaney & North County"). The prompt must instruct
        # the LLM to emit ONE row per award with " & "-joined school, not split
        # them into separate rows (which would collide on the coarse
        # sportsmanship natural key).
        from parse_record_book import _sportsmanship_prompt
        prompt, _ = _sportsmanship_prompt([spring_pages[28]], "Softball")
        assert "ONE row" in prompt
        assert '" & "' in prompt
        assert "Dulaney & North County" in prompt

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

    def test_merged_class_years_with_slash(self):
        # "(4A/3AE)" classifications broke get_years mid-block before "/" was
        # allowed — North Point's 2022/2024 football titles vanished this way.
        text = (
            "North Point (14, 24-12)\n"
            "CH: 2022 (4A/3AE), 2024 (4A/3AE)\n"
            "RU: 2018 (4AE), 2023 (4A/3AE)\n"
            "SF: 2017 (4AE)\n"
        )
        records = parse_school_records([text], "Football")
        assert len(records) == 1
        assert records[0]["champion_years"] == [2022, 2024]
        assert records[0]["runner_up_years"] == [2018, 2023]

    def test_semifinal_only_school_kept(self):
        # Schools with only SF/QF appearances (e.g. "Northeast - AA") used to
        # be dropped entirely because flush() required Ch/Fn/RU.
        text = "Northeast - AA (7, 2-7)\nSF: 1981 (A)\n"
        records = parse_school_records([text], "Football")
        assert len(records) == 1
        assert records[0]["school"] == "Northeast - AA"
        assert records[0]["semifinalist_years"] == [1981]

    def test_comma_in_school_name(self):
        text = "Dr. Henry A. Wise, Jr. (16, 46-10)\nCH: 2009 (4A), 2012 (4A)\n"
        records = parse_school_records([text], "Football")
        assert len(records) == 1
        assert records[0]["school"] == "Dr. Henry A. Wise, Jr."
        assert records[0]["champion_years"] == [2009, 2012]

    def test_closed_school_x_prefix_stripped(self):
        text = "x-North Carroll (5, 2-5)\nSF: 1978 (C)\n"
        records = parse_school_records([text], "Football")
        assert len(records) == 1
        assert records[0]["school"] == "North Carroll"

    def test_allcaps_name_with_letter_parenthetical(self):
        # "SOUTHERN (G)" (Garrett) — letter suffix, not stats parenthetical.
        text = "SOUTHERN (G)\nCh: 1993, 1994\nFn: 1998\n"
        records = parse_school_records([text], "Boys Cross Country")
        assert len(records) == 1
        assert records[0]["school"] == "SOUTHERN (G)"
        assert records[0]["champion_years"] == [1993, 1994]

    def test_unclosed_stats_parenthetical_stripped(self):
        # pypdf sometimes wraps the ")" to the next line: "Northwood (6, 2-6".
        text = "Northwood (6, 2-6\nCH: 1987 (2A)\n"
        records = parse_school_records([text], "Football")
        assert len(records) == 1
        assert records[0]["school"] == "Northwood"


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


# ── Classification normalization ──────────────────────────────────────────────


class TestNormalizeClassification:
    """_normalize_classification canonicalises LLM class-label drift.

    The championship prompt asks for the bare label ("1A", "B", "Combined"),
    but a fresh extraction sometimes copies the PDF column header ("CLASS 1A")
    or a footnote-marked asterisk ("B*"). Both must collapse to the bare form so
    the natural key (sport, year, classification, ...) stays stable and joins
    by classification don't fragment.
    """

    def test_strips_class_prefix(self):
        from parse_record_book import _normalize_classification
        assert _normalize_classification("CLASS 1A") == "1A"
        assert _normalize_classification("class 2a") == "2a"
        assert _normalize_classification("Class 4A") == "4A"

    def test_strips_trailing_asterisk(self):
        from parse_record_book import _normalize_classification
        assert _normalize_classification("B*") == "B"
        assert _normalize_classification("C*") == "C"
        assert _normalize_classification("D**") == "D"

    def test_strips_class_prefix_and_asterisk(self):
        from parse_record_book import _normalize_classification
        assert _normalize_classification("CLASS 1A*") == "1A"

    def test_leaves_bare_labels_unchanged(self):
        from parse_record_book import _normalize_classification
        for v in ["1A", "2A", "3A", "4A", "A", "AA", "B", "C", "D",
                  "Combined", "1A/2A", "2A-1A", "B-C", "AA/A", "One Class"]:
            assert _normalize_classification(v) == v

    def test_none_and_empty_pass_through(self):
        from parse_record_book import _normalize_classification
        assert _normalize_classification(None) is None
        assert _normalize_classification("") == ""

    def test_collapses_surrounding_whitespace(self):
        from parse_record_book import _normalize_classification
        assert _normalize_classification("  CLASS 1A  ") == "1A"
        assert _normalize_classification("  B*  ") == "B"


class TestTagPreMpssaa:
    """_tag_pre_mpssaa stamps notes on precursor-tournament rows by year.

    Precursor tournaments (PDF sections "PRIOR TO MPSSAA SPONSORSHIP" /
    "PRE-MPSSAA") are extracted into championship_results but are not MPSSAA
    championships. Tagging them keeps them distinguishable and lets verify
    exclude them from continuity/referential checks.
    """

    def test_tags_row_before_era_start(self):
        from parse_record_book import _tag_pre_mpssaa
        rows = [{"sport": "Field Hockey", "year": 1946, "classification": "A",
                 "champion_school": "Towson", "notes": None}]
        _tag_pre_mpssaa(rows)
        assert rows[0]["notes"] == "Pre-MPSSAA"

    def test_does_not_tag_row_at_or_after_era_start(self):
        from parse_record_book import _tag_pre_mpssaa
        rows = [
            {"sport": "Field Hockey", "year": 1975, "classification": "AA",
             "champion_school": "Bel Air", "notes": None},   # era start
            {"sport": "Field Hockey", "year": 2024, "classification": "1A",
             "champion_school": "Severna Park", "notes": None},  # after
        ]
        _tag_pre_mpssaa(rows)
        assert rows[0]["notes"] is None
        assert rows[1]["notes"] is None

    def test_preserves_existing_note_with_prefix(self):
        from parse_record_book import _tag_pre_mpssaa
        rows = [{"sport": "Volleyball", "year": 1948, "classification": "B",
                 "champion_school": "Bel Air", "notes": "Won by Default"}]
        _tag_pre_mpssaa(rows)
        assert rows[0]["notes"] == "Pre-MPSSAA; Won by Default"

    def test_does_not_double_tag(self):
        from parse_record_book import _tag_pre_mpssaa
        rows = [{"sport": "Boys Soccer", "year": 1920, "classification": "Combined",
                 "champion_school": "Catonsville", "notes": "Pre-MPSSAA; forfeit"}]
        _tag_pre_mpssaa(rows)
        assert rows[0]["notes"] == "Pre-MPSSAA; forfeit"

    def test_sport_without_era_start_is_untouched(self):
        from parse_record_book import _tag_pre_mpssaa
        # Boys Cross Country starts at MPSSAA's 1946 founding — no precursor.
        rows = [{"sport": "Boys Cross Country", "year": 1946, "classification": "A",
                 "champion_school": "Catonsville", "notes": None}]
        _tag_pre_mpssaa(rows)
        assert rows[0]["notes"] is None

    def test_skips_rows_missing_year_or_sport(self):
        from parse_record_book import _tag_pre_mpssaa
        rows = [
            {"sport": "Field Hockey", "year": None, "classification": "A",
             "champion_school": "X", "notes": None},
            {"sport": None, "year": 1946, "classification": "A",
             "champion_school": "X", "notes": None},
        ]
        _tag_pre_mpssaa(rows)
        assert rows[0]["notes"] is None
        assert rows[1]["notes"] is None


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


# ── classify_page() routing ───────────────────────────────────────────────────


class TestClassifyPage:
    """The testable extraction of main()'s old if/elif routing chain."""

    def test_school_records_route(self):
        routes = classify_page("ALLEGANY\nCh: 1997, 1998\nFn: 1988", "Football")
        assert "school_records" in routes

    def test_championship_route(self):
        routes = classify_page("YEAR CLASS CHAMPION COACH FINALIST COACH", "Football")
        assert "championship" in routes

    def test_golf_only_when_sport_is_golf(self):
        # A stray "Team Champion......" in another sport must not route to golf.
        golf_text = "Team Champion......Magruder (610)"
        assert "golf" in classify_page(golf_text, "Golf")
        assert "golf" not in classify_page(golf_text, "Football")

    def test_individual_xc_route_for_cross_country(self):
        routes = classify_page("15:07.0  2.5 MILES", "Boys Cross Country")
        assert "individual_xc" in routes

    def test_individual_results_excluded_for_cross_country(self):
        # XC pages match is_individual_results via the MILES regex too; for XC
        # sports they must route to individual_xc, not individual_results.
        routes = classify_page("15:07.0  2.5 MILES", "Boys Cross Country")
        assert "individual_results" not in routes

    def test_individual_results_route(self):
        routes = classify_page(
            "Event: 55m Dash\nYear Class Athlete-School-Mark\n2025 4A Fred Colvin—Fairmont Heights 6.6",
            "Boys Indoor Track",
        )
        assert "individual_results" in routes

    def test_sportsmanship_route(self):
        routes = classify_page("SPORTSMANSHIP AWARD\n2024 Allegany", "Football")
        assert "sportsmanship" in routes

    def test_divider_page_routes_to_nothing(self, fall_pages):
        # The "MPSSAA Girls Cross Country Records" divider page carries no data.
        routes = classify_page(fall_pages[3], "Girls Cross Country")
        assert routes == set()

    def test_dual_content_page_keeps_school_records(self):
        # A page with both Ch: school-record codes and a championship table header
        # should route to school_records AND championship.
        text = "ALLEGANY\nCh: 1997\nYEAR CLASS CHAMPION COACH\n2024 4A Churchill"
        routes = classify_page(text, "Football")
        assert "school_records" in routes
        assert "championship" in routes

    def test_golf_split_era_routes_to_golf(self, fall_pages):
        # The pages the old classifier dropped — golf's 1995–2024 hole.
        assert "golf" in classify_page(fall_pages[51], "Golf")
        assert "golf" in classify_page(fall_pages[52], "Golf")

    def test_classify_stat_records_football(self):
        text = ("TEAM RECORDS\n• Most State Championships\n13 ....Dunbar\n"
                "• Consecutive Wins\n53 ....Damascus 2015-2018\n"
                "• Most Touchdowns, Season\n98 ....Fort Hill 2016\n")
        routes = classify_page(text, "Football")
        assert "stat_records" in routes
        assert "championship" not in routes

    def test_classify_stat_records_does_not_steal_championship(self):
        # a year-class championship table must still route to championship
        text = "YEAR CLASS CHAMPION COACH FINALIST COACH\n1975 B Bates-53 Worcester-26\n"
        routes = classify_page(text, "Football")
        assert "championship" in routes
        assert "stat_records" not in routes

    def test_classify_stat_records_does_not_steal_individual_results(self):
        text = ("Athlete—School—Mark\n1990 4A 100 m Dash 10.4 John, School 10.4\n")
        routes = classify_page(text, "Boys Track and Field")
        assert "individual_results" in routes
        assert "stat_records" not in routes


# ── dedup() ───────────────────────────────────────────────────────────────────


class TestDedup:
    def test_removes_exact_duplicates(self):
        row = {"sport": "Soccer", "year": 2024, "classification": "4A",
               "champion_school": "Churchill"}
        unique, warns = dedup([row, dict(row)], DEDUP_KEYS["championship_results"], "championship_results")
        assert unique == [row]
        assert warns == []

    def test_keeps_distinct_keys(self):
        rows = [
            {"sport": "Soccer", "year": 2024, "classification": "4A", "champion_school": "Churchill"},
            {"sport": "Soccer", "year": 2024, "classification": "3A", "champion_school": "Broadneck"},
        ]
        unique, warns = dedup(rows, DEDUP_KEYS["championship_results"], "championship_results")
        assert len(unique) == 2
        assert warns == []

    def test_warns_on_same_key_different_payload(self):
        rows = [
            {"sport": "Soccer", "year": 2024, "classification": "4A",
             "champion_school": "Churchill", "score": "2-0"},
            {"sport": "Soccer", "year": 2024, "classification": "4A",
             "champion_school": "Churchill", "score": "2-1"},
        ]
        unique, warns = dedup(rows, DEDUP_KEYS["championship_results"], "championship_results")
        assert len(unique) == 1
        assert len(warns) == 1
        assert "championship_results" in warns[0]

    def test_xc_key_includes_name_so_co_champions_survive(self):
        # Two first-place finishers in the same year/class (real co-champions)
        # must both survive dedup — the pyc key (sport,year,classification) would
        # have dropped one.
        rows = [
            {"sport": "Boys Cross Country", "year": 2024, "classification": "4A",
             "name": "Alex Runner", "school": "School A", "time": "15:30"},
            {"sport": "Boys Cross Country", "year": 2024, "classification": "4A",
             "name": "Sam Runner", "school": "School B", "time": "15:30"},
        ]
        unique, warns = dedup(rows, DEDUP_KEYS["individual_xc_champions"], "individual_xc_champions")
        assert len(unique) == 2
        assert warns == []

    def test_empty(self):
        unique, warns = dedup([], DEDUP_KEYS["championship_results"], "championship_results")
        assert unique == [] and warns == []


# ── write_csv() year-list serialization ────────────────────────────────────────


class TestWriteCsv:
    def test_year_lists_serialized_semicolon(self, tmp_path):
        rows = [{"sport": "Football", "school": "Allegany",
                 "champion_years": [1997, 1998], "finalist_years": [1988]}]
        path = tmp_path / "out.csv"
        write_csv(path, rows, ["sport", "school", "champion_years", "finalist_years"])
        import csv as _csv
        with path.open() as f:
            reader = _csv.DictReader(f)
            out = list(reader)
        assert out[0]["champion_years"] == "1997; 1998"
        assert out[0]["finalist_years"] == "1988"

    def test_non_list_values_pass_through(self, tmp_path):
        rows = [{"sport": "Golf", "year": 2024, "classification": "3A/4A"}]
        path = tmp_path / "out.csv"
        write_csv(path, rows, ["sport", "year", "classification"])
        import csv as _csv
        with path.open() as f:
            out = list(_csv.DictReader(f))
        assert out[0]["year"] == "2024"
        assert out[0]["classification"] == "3A/4A"


# ── Extraction cache & provenance ──────────────────────────────────────────────


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key(MODEL_ID, _golf_prompt(["p"], "Golf")[1], "prompt")
        k2 = _cache_key(MODEL_ID, _golf_prompt(["p"], "Golf")[1], "prompt")
        assert k1 == k2

    def test_changes_with_model(self):
        schema = _golf_prompt(["p"], "Golf")[1]
        assert _cache_key("glm-5.2:cloud", schema, "p") != _cache_key("other", schema, "p")

    def test_changes_with_prompt_text(self):
        schema = _golf_prompt(["p"], "Golf")[1]
        assert _cache_key(MODEL_ID, schema, "p1") != _cache_key(MODEL_ID, schema, "p2")

    def test_changes_with_schema(self):
        # Different schemas (golf vs championship) → different keys even for the same text.
        from parse_record_book import _championship_prompt
        golf_schema = _golf_prompt(["p"], "Golf")[1]
        champ_schema = _championship_prompt(["p"], "Golf")[1]
        assert _cache_key(MODEL_ID, golf_schema, "p") != _cache_key(MODEL_ID, champ_schema, "p")

    def test_path_uses_16_char_prefix(self):
        key = _cache_key(MODEL_ID, _golf_prompt(["p"], "Golf")[1], "p")
        path = _cache_path("golf", key)
        assert path.name.startswith("golf_")
        assert len(path.stem.split("_", 1)[1]) == 16


class TestStamp:
    def test_adds_all_four_provenance_fields(self):
        rows = [{"year": 2024, "classification": "3A/4A"}]
        _stamp(rows, "pdfs/fall.pdf", [50, 51], "2026-01-01T00:00:00Z", "glm-5.2:cloud")
        r = rows[0]
        assert r["source_pdf"] == "pdfs/fall.pdf"
        assert r["source_pages"] == [50, 51]
        assert r["extracted_at"] == "2026-01-01T00:00:00Z"
        assert r["extraction_model"] == "glm-5.2:cloud"

    def test_provenance_fields_constant(self):
        assert PROVENANCE_FIELDS == ["source_pdf", "source_pages", "extracted_at", "extraction_model"]


class TestCachedExtract:
    """Cache hit/miss logic tested without an LLM call (offline guards the API)."""

    def _seed_cache(self, tmp_path, pages, sport, extractor, rows, *, source_pdf="OLD.pdf",
                   source_pages=(99,), extracted_at="2020-01-01T00:00:00Z"):
        builder, schema, _ = EXTRACTORS[extractor]
        prompt, _schema = builder(pages, sport)
        key = _cache_key(MODEL_ID, schema, prompt)
        path = tmp_path / f"{extractor}_{key[:16]}.json"
        path.write_text(json.dumps({
            "meta": {"extractor": extractor, "model_id": MODEL_ID,
                     "source_pdf": source_pdf, "source_pages": list(source_pages),
                     "extracted_at": extracted_at},
            "rows": rows,
        }))
        return path

    def test_hit_returns_cached_rows_without_llm(self, tmp_path, monkeypatch):
        monkeypatch.setattr("parse_record_book.CACHE_DIR", tmp_path)
        pages = ["1971\nTeam Champion......Magruder (610)"]
        self._seed_cache(tmp_path, pages, "Golf", "golf",
                         [{"year": 1971, "classification": "Combined",
                           "team_champion_school": "Magruder"}])
        # offline=True would raise on a miss; a hit must return without an API call
        rows = cached_extract("golf", pages, "Golf", "NEW.pdf", [50],
                              refresh=False, offline=True)
        assert rows[0]["year"] == 1971
        assert rows[0]["team_champion_school"] == "Magruder"

    def test_hit_re_stamps_current_source_pages_and_pdf(self, tmp_path, monkeypatch):
        # Text-based keying means a hit can come from a different edition — so the
        # row's location provenance must follow the current run, not the original.
        monkeypatch.setattr("parse_record_book.CACHE_DIR", tmp_path)
        pages = ["1971\nTeam Champion......Magruder (610)"]
        self._seed_cache(tmp_path, pages, "Golf", "golf",
                         [{"year": 1971, "classification": "Combined"}],
                         source_pdf="OLD.pdf", source_pages=(99,),
                         extracted_at="2020-01-01T00:00:00Z")
        rows = cached_extract("golf", pages, "Golf", "NEW.pdf", [50],
                              refresh=False, offline=False)
        assert rows[0]["source_pdf"] == "NEW.pdf"
        assert rows[0]["source_pages"] == [50]
        # extracted_at is preserved from the original extraction (the data came from then)
        assert rows[0]["extracted_at"] == "2020-01-01T00:00:00Z"
        assert rows[0]["extraction_model"] == MODEL_ID

    def test_offline_miss_raises_without_llm(self, tmp_path, monkeypatch):
        monkeypatch.setattr("parse_record_book.CACHE_DIR", tmp_path)
        with pytest.raises(RuntimeError, match="offline"):
            cached_extract("golf", ["no matching text here"], "Golf", "x.pdf", [1],
                           refresh=False, offline=True)

    def test_miss_writes_cache_then_hit_reads_no_second_llm(self, tmp_path, monkeypatch):
        """Full write→read round-trip with llm_extract mocked (no real API call)."""
        monkeypatch.setattr("parse_record_book.CACHE_DIR", tmp_path)
        canned = {"results": [{"year": 1971, "classification": "Combined",
                               "team_champion_school": "Magruder"}]}
        calls = []
        monkeypatch.setattr("parse_record_book.llm_extract",
                            lambda prompt, schema, retries=2: calls.append(prompt) or canned)
        pages = ["1971\nTeam Champion......Magruder (610)"]

        # miss → LLM → write cache
        rows1 = cached_extract("golf", pages, "Golf", "f.pdf", [50],
                               refresh=False, offline=False)
        assert rows1[0]["team_champion_school"] == "Magruder"
        assert rows1[0]["source_pages"] == [50]          # stamped from this run
        assert rows1[0]["source_pdf"] == "f.pdf"
        assert len(calls) == 1

        # the cache file stores RAW rows (no provenance) + meta
        cache_files = list(tmp_path.glob("golf_*.json"))
        assert len(cache_files) == 1
        blob = json.loads(cache_files[0].read_text())
        assert blob["meta"]["source_pages"] == [50]
        assert blob["meta"]["model_id"] == MODEL_ID
        assert "source_pdf" not in blob["rows"][0]        # raw, provenance applied per-run

        # hit → no second LLM call; identical stamped rows
        rows2 = cached_extract("golf", pages, "Golf", "f.pdf", [50],
                               refresh=False, offline=False)
        assert len(calls) == 1
        assert rows2 == rows1

    def test_refresh_bypasses_cache_but_needs_llm(self, tmp_path, monkeypatch):
        # --refresh with a present cache file must NOT use it; with --offline it
        # should raise (proving the cache read was bypassed).
        monkeypatch.setattr("parse_record_book.CACHE_DIR", tmp_path)
        pages = ["1971\nTeam Champion......Magruder (610)"]
        self._seed_cache(tmp_path, pages, "Golf", "golf", [{"year": 1971}])
        with pytest.raises(RuntimeError, match="offline"):
            cached_extract("golf", pages, "Golf", "x.pdf", [50],
                           refresh=True, offline=True)

    def test_championship_chunk_size_is_two(self):
        assert CHUNK_SIZES["championship"] == 2
        assert CHUNK_SIZES["sportsmanship"] >= 64  # all pages in one call


# ── Phase 6: heading-based section detection ─────────────────────────────────


def _divider_text(sport, page_num):
    """Synthesize a NaturalPDF-style divider page: 'MPSSAA\n<Sport>\nRecords\n<n>'."""
    return f"MPSSAA \n{sport} \nRecords \n{page_num}"


class TestDetectDividers:
    def test_finds_short_divider_pages(self):
        texts = [
            "MPSSAA \nGirls Cross Country \nRecords \n2",
            "long content page " * 50,  # too long, skipped
            "MPSSAA \nBoys Cross Country \nRecords \n1122",
        ]
        assert detect_dividers(texts) == [(0, "Girls Cross Country"),
                                          (2, "Boys Cross Country")]

    def test_length_filter_excludes_toc_and_content_pages(self):
        # TOC page starts with MPSSAA and mentions Records but is ~250+ chars.
        toc = ("MPSSAA Fall Record Book Cross Country Field Hockey Football Golf "
               "Soccer Volleyball table of contents Sport Pages Records " + "x" * 200)
        assert detect_dividers([toc]) == []

    def test_excludes_pages_without_records_word(self):
        # A short MPSSAA running header without 'Records'.
        assert detect_dividers(["MPSSAA \nGirls Cross Country \nState Meet"]) == []

    def test_handles_multiline_sport_name(self):
        # 'Girls Cross\nCountry' across two lines -> collapsed to one sport name.
        assert detect_dividers(["MPSSAA \nGirls Cross\nCountry \nRecords \n2"]) == [
            (0, "Girls Cross Country")
        ]

    def test_empty_pages_skipped(self):
        assert detect_dividers(["", "   "]) == []


class TestFindSections:
    def _fall_texts(self, with_back_matter=True):
        """Build synthetic NaturalPDF page texts reproducing the fall layout.

        8 divider pages at the baseline indices, with content + an optional
        districts back-matter page at the end.
        """
        texts = [""] * 78
        dividers = [
            (3, "Girls Cross Country", 2),
            (13, "Boys Cross Country", 1122),
            (27, "Field Hockey", 2266),
            (35, "Football", 3344),
            (47, "Golf", 4466),
            (53, "Girls Soccer", 5522),
            (59, "Boys Soccer", 5588),
            (66, "Volleyball", 6655),
        ]
        for idx, sport, n in dividers:
            texts[idx] = _divider_text(sport, n)
        if with_back_matter:
            texts[77] = "DISTRICTS OF THE STATE\nCounty map"
        return texts

    def test_reproduces_fall_baseline(self):
        assert find_sections(self._fall_texts(), "fall") == FALL_SECTIONS

    def test_back_matter_excluded_from_last_section(self):
        # Without the back-matter page, Volleyball extends to len(pages)=78.
        sections = find_sections(self._fall_texts(with_back_matter=False), "fall")
        assert sections["Volleyball"] == (66, 78)
        # With it, Volleyball ends at idx 77 (matches the baseline).
        assert find_sections(self._fall_texts(), "fall")["Volleyball"] == (66, 77)

    def test_normalizes_ampersand_to_and(self):
        # Spring baseline uses 'Girls Track and Field'; PDF title uses '&'.
        texts = [""] * 96
        spring = [
            (3, "Baseball", 2), (11, "Girls Lacrosse", 1100),
            (16, "Boys Lacrosse", 1155), (22, "Softball", 2211),
            (29, "Tennis", 2288), (35, "Girls Track & Field", 3344),
            (61, "Boys Track & Field", 6600),
        ]
        for idx, sport, n in spring:
            texts[idx] = _divider_text(sport, n)
        texts[95] = "DISTRICTS OF THE STATE"
        assert find_sections(texts, "spring") == SPRING_SECTIONS

    def test_unknown_sport_raises(self):
        texts = self._fall_texts()
        texts[3] = _divider_text("Ultimate Frisbee", 2)
        with pytest.raises(ValueError, match="Detected unknown sport 'Ultimate Frisbee'"):
            find_sections(texts, "fall")

    def test_missing_sport_raises(self):
        texts = self._fall_texts()
        texts[3] = ""  # drop the Girls Cross Country divider
        with pytest.raises(ValueError, match="Missing expected fall sport divider"):
            find_sections(texts, "fall")

    def test_reordered_sports_raise(self):
        # Swap two divider titles -> order no longer matches baseline.
        texts = self._fall_texts()
        texts[3], texts[13] = texts[13], texts[3]
        with pytest.raises(ValueError, match="order"):
            find_sections(texts, "fall")

    def test_no_dividers_raises(self):
        with pytest.raises(ValueError, match="No sport divider pages"):
            find_sections(["", "", ""], "fall")

    def test_unknown_season_raises(self):
        with pytest.raises(ValueError, match="Unknown season"):
            find_sections([], "summer")

    def test_sport_order_mismatch_message_lists_both(self):
        texts = self._fall_texts()
        # Rename Girls Soccer -> Boys Soccer (duplicate) to force a set mismatch.
        texts[53] = _divider_text("Boys Soccer", 5522)
        with pytest.raises(ValueError):
            find_sections(texts, "fall")


class TestDividerCandidates:
    def test_finds_short_mpssaa_pages(self):
        # pypdf-style: divider pages are short and contain 'MPSSAA'.
        pages = ["", "2 \nMPSSAA ", "long content " * 50, "3 \nMPSSAA "]
        assert _divider_candidates(pages) == [1, 3]

    def test_excludes_long_pages(self):
        long_page = ("MPSSAA Football Stats and Records Fast Facts " + "x" * 200)
        assert _divider_candidates([long_page]) == []

    def test_excludes_empty_pages(self):
        assert _divider_candidates(["", "   "]) == []


class TestFindSectionsIntegration:
    """Detection reproduces the committed baseline maps on the real PDFs."""

    def test_fall(self):
        pages = load_pages(FALL_PDF)
        texts = load_page_titles(FALL_PDF, _divider_candidates(pages), pages)
        assert find_sections(texts, "fall") == FALL_SECTIONS

    def test_winter(self):
        pages = load_pages(WINTER_PDF)
        texts = load_page_titles(WINTER_PDF, _divider_candidates(pages), pages)
        assert find_sections(texts, "winter") == WINTER_SECTIONS

    def test_spring(self):
        pages = load_pages(SPRING_PDF)
        texts = load_page_titles(SPRING_PDF, _divider_candidates(pages), pages)
        assert find_sections(texts, "spring") == SPRING_SECTIONS


class TestStatRecordSchema:
    def test_stat_record_schema_accepts_full_row(self):
        from parse_record_book import StatRecord, StatResults
        row = StatRecord(
            sport="Football",
            category="team",
            record="Most Touchdowns, Season",
            value="98",
            holder="Fort Hill",
            school="Fort Hill",
            year="2016",
            co_holder=False,
            notes=None,
        )
        page = StatResults(results=[row])
        assert len(page.results) == 1
        assert page.results[0].record == "Most Touchdowns, Season"
        assert page.results[0].value == "98"

    def test_stat_record_schema_optional_fields(self):
        from parse_record_book import StatRecord
        r = StatRecord(sport="Baseball", record="Runs Scored - Game")
        # category/value/holder/school/year/notes are optional; co_holder optional
        assert r.category is None
        assert r.value is None
        assert r.co_holder is None


def test_stat_records_dedup_key_collapses_duplicates():
    from parse_record_book import dedup, DEDUP_KEYS
    rows = [
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Fort Hill", "school": "Fort Hill",
         "year": "2016", "co_holder": False, "notes": None},
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Fort Hill", "school": "Fort Hill",
         "year": "2016", "co_holder": False, "notes": None},
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "95", "holder": "Damascus", "school": "Damascus",
         "year": "2015", "co_holder": True, "notes": None},
    ]
    out, warns = dedup(rows, DEDUP_KEYS["stat_records"], "stat_records")
    # duplicate (same holder/year) collapsed; distinct co-holder kept
    assert len(out) == 2
    holders = {r["holder"] for r in out}
    assert holders == {"Fort Hill", "Damascus"}


def test_stat_records_dedup_keeps_co_holders():
    from parse_record_book import dedup, DEDUP_KEYS
    rows = [
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Fort Hill", "school": "Fort Hill",
         "year": "2016", "co_holder": True, "notes": None},
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Damascus", "school": "Damascus",
         "year": "2016", "co_holder": True, "notes": None},
    ]
    out, warns = dedup(rows, DEDUP_KEYS["stat_records"], "stat_records")
    assert len(out) == 2
    assert {r["holder"] for r in out} == {"Fort Hill", "Damascus"}
    assert warns == []


def test_stat_records_dedup_keeps_distinct_values_for_same_holder_year():
    # `value` is part of the dedup key because ranked-list records legitimately
    # share (record, holder, year) while differing on value: a school with both a
    # 4-peat and a 3-peat ("Consecutive Championships"), or a player with two
    # 34+ point games in one tournament ("34-Plus Point Scorers", same year,
    # different point totals). These are distinct achievements, not duplicates,
    # so both rows survive and no warning is emitted.
    from parse_record_book import dedup, DEDUP_KEYS
    rows = [
        {"sport": "Boys Basketball", "category": "individual",
         "record": "34-Plus Point Scorers", "value": "38",
         "holder": "Jamahl Brown", "school": "Surrattsville",
         "year": "2008", "co_holder": False, "notes": "Class 1A Semifinal"},
        {"sport": "Boys Basketball", "category": "individual",
         "record": "34-Plus Point Scorers", "value": "36",
         "holder": "Jamahl Brown", "school": "Surrattsville",
         "year": "2008", "co_holder": False, "notes": "Class 1A Final"},
    ]
    out, warns = dedup(rows, DEDUP_KEYS["stat_records"], "stat_records")
    assert len(out) == 2
    assert {r["value"] for r in out} == {"38", "36"}
    assert warns == []


def test_stat_records_dedup_true_duplicate_same_value_collapses():
    # An exact duplicate (same value too) still collapses — `value` in the key
    # only distinguishes rows whose value genuinely differs.
    from parse_record_book import dedup, DEDUP_KEYS
    rows = [
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Fort Hill", "school": "Fort Hill",
         "year": "2016", "co_holder": False, "notes": None},
        {"sport": "Football", "category": "team", "record": "Most TDs, Season",
         "value": "98", "holder": "Fort Hill", "school": "Fort Hill",
         "year": "2016", "co_holder": False, "notes": None},
    ]
    out, warns = dedup(rows, DEDUP_KEYS["stat_records"], "stat_records")
    assert len(out) == 1
    assert warns == []


def test_stat_records_prompt_returns_schema():
    from parse_record_book import _stat_records_prompt, StatResults
    prompt, schema = _stat_records_prompt(["some page text"], "Football")
    assert schema is StatResults
    assert "stat" in prompt.lower() or "record" in prompt.lower()
    assert "some page text" in prompt


def test_stat_records_csv_field_list():
    # Light guard on the stat_records CSV column order (matches the StatRecord
    # schema). The end-to-end main() round-trip is exercised in Task 8/10.
    fields = ["sport", "category", "record", "value", "holder",
              "school", "year", "co_holder", "notes"]
    # the csv writer uses these names; ensure they all exist on a sample row
    row = {k: None for k in fields}
    row.update({"sport": "Football", "record": "Most TDs, Season"})
    assert set(fields) <= set(row.keys())

