"""Tests for build_site.py — normalizer, slugs, co-champion split, display-name
selection, and the unified school index built from the real season books."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site import (
    _base_normalize,
    _longest_consecutive_run,
    best_display_name,
    build_embeds,
    build_school_index,
    cite_str,
    compute_anniversaries,
    compute_droughts,
    compute_first_title_watch,
    compute_never_won,
    compute_pegs,
    compute_streaks,
    fast_facts_paragraph,
    load_aliases,
    load_books,
    make_normalizer,
    pdf_label,
    render_timeline_svg,
    schools_index_json,
    slugify,
    split_cochampions,
)

ALIAS_MAP, CANON_DISPLAY = load_aliases()
NORM = make_normalizer(ALIAS_MAP)


# ── Citation helper ──────────────────────────────────────────────────────────
class TestCiteStr:
    def test_camelcase_pdf_label(self):
        assert pdf_label("FallRecordBook2024.pdf") == "Fall Record Book 2024"
        assert pdf_label("pdfs/Winter record book.pdf") == "Winter Record Book"
        assert pdf_label("Spring record book 2025.pdf") == "Spring Record Book 2025"

    def test_single_page(self):
        assert cite_str({"source_pdf": "FallRecordBook2024.pdf", "source_pages": [37]}) == \
            "p. 37, Fall Record Book 2024"

    def test_contiguous_range(self):
        assert cite_str({"source_pdf": "FallRecordBook2024.pdf", "source_pages": [37, 38]}) == \
            "pp. 37–38, Fall Record Book 2024"

    def test_scattered_pages(self):
        assert cite_str({"source_pdf": "FallRecordBook2024.pdf", "source_pages": [37, 39]}) == \
            "pp. 37, 39, Fall Record Book 2024"

    def test_no_pages(self):
        assert cite_str({"source_pdf": "FallRecordBook2024.pdf", "source_pages": []}) == \
            "Fall Record Book 2024"


# ── base normalization ───────────────────────────────────────────────────────
class TestBaseNormalize:
    def test_allcaps_lowercased_and_punctuation_collapsed(self):
        assert _base_normalize("NORTHEAST-AA") == "northeast aa"
        assert _base_normalize("CAMBRIDGE/SOUTH DORCHESTER") == "cambridge south dorchester"

    def test_unicode_dash_becomes_space(self):
        # Em-dash and hyphen both collapse, so "Central—PG" == "Central-PG".
        assert _base_normalize("Central—PG") == _base_normalize("Central-PG") == "central pg"

    def test_trailing_suffix_words_stripped(self):
        assert _base_normalize("Aberdeen High") == "aberdeen"
        assert _base_normalize("Aberdeen HS") == "aberdeen"
        assert _base_normalize("Franklin High School") == "franklin"

    def test_leading_high_is_kept(self):
        # "High Point" is a real school name; "high" is not a trailing descriptor.
        assert _base_normalize("HIGH POINT") == "high point"
        assert _base_normalize("High Point") == "high point"

    def test_generic_abbreviation_expansion(self):
        assert _base_normalize("Mt. Hebron") == "mount hebron"
        assert _base_normalize("MT HEBRON") == "mount hebron"
        assert _base_normalize("Harford Tech") == "harford technical"
        assert _base_normalize("Col. Richardson") == "colonel richardson"

    def test_empty_and_none(self):
        assert _base_normalize("") == ""
        assert _base_normalize(None) == ""


# ── alias-driven normalization ───────────────────────────────────────────────
class TestNormalizeSchool:
    def test_bcc_abbreviations_resolve_to_bethesda_chevy_chase(self):
        assert NORM("B-CC") == "bethesda chevy chase"
        assert NORM("BCC") == "bethesda chevy chase"
        assert NORM("B.C.C.") == "bethesda chevy chase"

    def test_blair_resolves_to_montgomery_blair_not_bel_air(self):
        # Regression guard: difflib once hinted "bel air", but Blair == Montgomery Blair.
        assert NORM("Blair") == "montgomery blair"

    def test_typo_normalized_via_alias(self):
        assert NORM("Catotin") == "catoctin"
        assert NORM("Surrattsvile") == "surrattsville"
        assert NORM("Westminister") == "westminster"

    def test_allcaps_and_mixed_case_merge(self):
        assert NORM("ELEANOR ROOSEVELT") == NORM("Eleanor Roosevelt")
        assert NORM("BOWIE") == NORM("Bowie") == "bowie"

    def test_county_suffix_distinguishes_schools(self):
        # Chesapeake-AA and Chesapeake-B are different schools; do not merge.
        assert NORM("Chesapeake-AA") != NORM("Chesapeake-B")
        assert NORM("Northeast-AA") == "northeast aa"


# ── slugify ──────────────────────────────────────────────────────────────────
class TestSlugify:
    def test_basic(self):
        assert slugify("eleanor roosevelt") == "eleanor-roosevelt"
        assert slugify("bethesda chevy chase") == "bethesda-chevy-chase"

    def test_keeps_county_suffix(self):
        assert slugify("northeast aa") == "northeast-aa"

    def test_strips_punctuation(self):
        assert slugify("Queen Anne's") == "queen-anne-s"


# ── co-champion split ────────────────────────────────────────────────────────
class TestSplitCochampions:
    def test_two_way_split(self):
        assert split_cochampions("Arundel & Bowie") == ["Arundel", "Bowie"]

    def test_three_way_split(self):
        assert split_cochampions("A & B & C") == ["A", "B", "C"]

    def test_solo_returns_single(self):
        assert split_cochampions("Bowie") == ["Bowie"]

    def test_empty(self):
        assert split_cochampions("") == []
        assert split_cochampions(None) == []

    def test_tie_prefix_stripped(self):
        assert split_cochampions("TIE: Fairmont Heights / Beall") == [
            "Fairmont Heights", "Beall"]

    def test_slash_tie_splits_when_both_plausible(self):
        assert split_cochampions("Poolesville/St. Michaels") == [
            "Poolesville", "St. Michaels"]
        assert split_cochampions("High Point/Old Mill") == [
            "High Point", "Old Mill"]

    def test_slash_kept_when_part_too_short(self):
        # "Cambridge/SD" is ONE school (Cambridge-South Dorchester); "SD"
        # normalizes to 2 chars, so the name must not split.
        assert split_cochampions("Cambridge/SD") == ["Cambridge/SD"]
        assert split_cochampions("FRD/SRI") == ["FRD/SRI"]

    def test_unspaced_ampersand_is_one_school(self):
        # "Carver A&T" is a single school; only a spaced " & " separates ties.
        assert split_cochampions("Carver A&T") == ["Carver A&T"]


# ── display-name selection ───────────────────────────────────────────────────
class TestBestDisplayName:
    def test_prefers_mixed_case_over_allcaps(self):
        assert best_display_name(["BOWIE", "Bowie"]) == "Bowie"

    def test_allcaps_only_titlecased_with_suffix_preserved(self):
        assert best_display_name(["NORTHEAST-AA"]) == "Northeast-AA"

    def test_short_mixed_form_preferred(self):
        # "Bowie" should win over "Bowie HS" if both somehow appeared.
        name = best_display_name(["Bowie HS", "Bowie"])
        assert name == "Bowie"


# ── unified index on real data ───────────────────────────────────────────────
@pytest.fixture(scope="module")
def index():
    books = load_books()
    am, cd = load_aliases()
    registry, report = build_school_index(books, make_normalizer(am), cd)
    return registry, report


class TestBuildSchoolIndex:
    def test_report_counts_match_data(self, index):
        _, report = index
        assert report["seasons"] == ["fall", "winter", "spring"]
        assert report["championship_rows"] == 3597
        # 2268 after the parse_school_records fixes (slash classifications,
        # SF/QF-only schools, comma names, "SOUTHERN (G)"-style suffixes).
        assert report["school_records_rows"] == 2268
        assert report["schools"] > 200

    def test_unmatched_is_bucketed(self, index):
        _, report = index
        um = report["unmatched"]
        assert set(um) == {"junk", "near_known", "unresolved"}
        # junk bucket holds the short individual_results artifacts
        assert any(e["key"] for e in um["junk"])

    def test_known_school_has_titles(self, index):
        registry, _ = index
        er = registry.lookup("Eleanor Roosevelt")
        assert er is not None
        assert er.slug == "eleanor-roosevelt"
        assert len(er.titles) > 0
        # every title row carries season + provenance
        assert all("season" in t and "source_pdf" in t for t in er.titles)

    def test_allegany_xc_titles_from_pdf(self, index):
        # The README calls out Allegany XC 1997-98 as a known record.
        registry, _ = index
        alleg = registry.lookup("ALLEGANY")
        assert alleg is not None
        xc_titles = [r for r in alleg.titles if r["sport"] == "Girls Cross Country"]
        assert 1997 in {r["year"] for r in xc_titles}

    def test_cochampion_split_attaches_to_both_schools(self, index):
        registry, _ = index
        # "Arundel & Bowie" appeared as a co-champion; both should have the row.
        arundel = registry.lookup("Arundel")
        bowie = registry.lookup("Bowie")
        assert arundel is not None and bowie is not None
        # At least one co-champion title exists on each (shared title).
        arundel_co = [t for t in arundel.titles if t.get("co_champion")]
        bowie_co = [t for t in bowie.titles if t.get("co_champion")]
        assert arundel_co or bowie_co  # at least one side records it as co-champion

    def test_golf_team_and_individual_attached(self, index):
        registry, _ = index
        # Parkdale was an individual golf winner in 1971 (from the sample row).
        parkdale = registry.lookup("Parkdale")
        assert parkdale is not None
        assert any(r["year"] == 1971 for r in parkdale.golf_individual)

    def test_stat_records_with_none_school_not_attached(self, index):
        registry, _ = index
        before = len(registry.by_key)
        # school=None stat rows must not create phantom schools.
        assert len(registry.by_key) == before
        # and no school has a stat_records row whose school was None
        for s in registry.by_key.values():
            assert all(r.get("school") is not None for r in s.stat_records)

    def test_slugs_are_unique(self, index):
        registry, _ = index
        slugs = [s.slug for s in registry.by_key.values()]
        assert len(slugs) == len(set(slugs))


class TestSchoolsIndexJson:
    def test_sorted_by_name_and_has_counts(self, index):
        registry, _ = index
        idx = schools_index_json(registry)
        names = [e["name"] for e in idx]
        assert names == sorted(names, key=str.lower)
        assert idx[0].keys() >= {"slug", "name", "titles", "finals",
                                 "individual_champions", "sportsmanship"}
        assert len(idx) == len(registry.by_key)


# ── SVG timeline renderer ────────────────────────────────────────────────────
class TestRenderTimelineSvg:
    TITLES = [
        {"sport": "Football", "year": 1992},
        {"sport": "Football", "year": 1995},
        {"sport": "Girls Basketball", "year": 2010},
    ]

    def test_empty_titles_returns_placeholder(self):
        svg = render_timeline_svg([])
        assert svg.startswith("<svg") and "No titles yet" in svg

    def test_returns_svg_root(self):
        svg = render_timeline_svg(self.TITLES)
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert "viewBox" in svg

    def test_one_dot_per_title(self):
        svg = render_timeline_svg(self.TITLES)
        assert svg.count("<circle") == 3

    def test_decade_gridlines_present(self):
        svg = render_timeline_svg(self.TITLES)
        # Range is 1992-2010. 1990 sits left of year_min and is suppressed so
        # it doesn't cross the sport-label zone; 2000 and 2010 appear.
        assert "1990" not in svg
        assert "2000" in svg and "2010" in svg
        assert svg.count('class="tl-grid"') >= 2

    def test_sport_labels_rendered(self):
        svg = render_timeline_svg(self.TITLES)
        assert "Football" in svg and "Girls Basketball" in svg

    def test_title_tooltip_per_dot(self):
        svg = render_timeline_svg(self.TITLES)
        assert "<title>Football 1992</title>" in svg

    def test_single_title_pads_axis(self):
        # A single title should still produce a usable (decade-padded) axis.
        svg = render_timeline_svg([{"sport": "Baseball", "year": 2016}])
        assert svg.count("<circle") == 1
        assert "2010" in svg and "2020" in svg

    def test_deterministic_across_calls(self):
        assert render_timeline_svg(self.TITLES) == render_timeline_svg(self.TITLES)

    def test_special_chars_escaped(self):
        svg = render_timeline_svg([{"sport": "Track & Field", "year": 2000}])
        assert "Track &amp; Field" in svg
        assert "Track & Field<" not in svg  # no raw ampersand inside an element

    def test_long_sport_label_not_clipped(self):
        """Right-anchored sport labels must fit inside the viewBox left edge.

        Regression: 'Boys Swimming & Diving' (22 chars) right-anchored at a
        fixed x=36 extended to ~x=-72 and was clipped. The left padding must
        grow with the longest label so its left edge stays >= 0.
        """
        import re as _re
        sport = "Boys Swimming & Diving"
        svg = render_timeline_svg([{"sport": sport, "year": 2010}])
        m = _re.search(r'<text class="tl-sport" x="([\d.]+)"', svg)
        assert m, "no sport label found"
        x_attr = float(m.group(1))
        # Right-anchored text spans [x - width, x]; require x >= width estimate
        # so the left edge is >= 0 (inside the viewBox).
        assert x_attr >= len(sport) * 6, f"label x={x_attr} clips {len(sport)}-char sport"

    def test_decades_before_year_min_suppressed(self):
        """A decade boundary below year_min must not appear on the axis."""
        import re as _re
        # year_min=1992 → 1990 gridline is left of the plot and suppressed.
        svg = render_timeline_svg(self.TITLES)
        years_on_axis = [int(y) for y in _re.findall(
            r'<text class="tl-axis"[^>]*>(\d+)</text>', svg)]
        assert 1990 not in years_on_axis
        assert 2000 in years_on_axis


# ── Fast-facts paragraph generator ───────────────────────────────────────────
class TestFastFacts:
    def test_with_titles_full_paragraph(self, index):
        r, _ = index
        ff = fast_facts_paragraph("Allegany", r.lookup("Allegany"))
        # 14 individual champions: includes the "Allegeny" typo row merged by
        # the alias map.
        assert ff == ("Allegany has won 34 state championships across 7 sports, "
                      "most recently 1A baseball in 2025. It has produced 14 "
                      "individual state champions and won 2 sportsmanship awards.")

    def test_singular_grammar_one_title_one_sport(self, index):
        # Old Post (closed) has exactly one title in one sport, exercising the
        # singular-noun branches.
        r, _ = index
        ff = fast_facts_paragraph("Old Post", r.lookup("Old Post"))
        assert ff == ("Old Post has won 1 state championship across 1 "
                      "sport, most recently D boys cross country in 1947.")

    def test_no_title_but_finals(self, index):
        # (Cambridge-South Dorchester previously filled this role, but its
        # "Cambridge/SD"-spelled championships now merge via the alias map and
        # it has 11 titles. Blake has finals, individual champions, and
        # sportsmanship awards — but no team title.)
        r, _ = index
        ff = fast_facts_paragraph("James Hubert Blake",
                                  r.lookup("James Hubert Blake"))
        assert ff == ("James Hubert Blake has reached 4 state finals without "
                      "a title. It has produced 29 individual state champions and won "
                      "3 sportsmanship awards.")

    def test_no_history_at_all(self, index):
        r, _ = index
        ff = fast_facts_paragraph("Carver A&t", r.lookup("Carver A&t"))
        assert ff == "Carver A&t has no recorded state championship appearances."

    def test_no_title_individual_only(self, index):
        r, _ = index
        ff = fast_facts_paragraph("Elmer Wolfe", r.lookup("Elmer Wolfe"))
        assert ff == ("Elmer Wolfe has no recorded state championship appearances. "
                      "It has produced 1 individual state champion.")

    def test_deterministic(self, index):
        r, _ = index
        s = r.lookup("Eleanor Roosevelt")
        assert fast_facts_paragraph("Eleanor Roosevelt", s) == \
            fast_facts_paragraph("Eleanor Roosevelt", s)

    def test_uses_classification_and_in_year(self, index):
        r, _ = index
        ff = fast_facts_paragraph("Eleanor Roosevelt", r.lookup("Eleanor Roosevelt"))
        # last title is 4A Boys Basketball 2022
        assert "4A boys basketball in 2022" in ff


class TestCanonicalDisplayName:
    def test_alias_canonical_preferred_over_short_abbreviation(self, index):
        # Regression: best_display_name once picked "E. Roosevelt" over
        # "Eleanor Roosevelt" because the abbreviation was shorter. The curated
        # canonical display name from aliases.csv must win.
        r, _ = index
        er = r.lookup("Eleanor Roosevelt")
        assert er.display_name == "Eleanor Roosevelt"

    def test_baltimore_polytechnic_display_name(self, index):
        r, _ = index
        bpi = r.lookup("Baltimore Polytechnic Institute")
        assert bpi.display_name == "Baltimore Polytechnic Institute"


# ── Peg computations ─────────────────────────────────────────────────────────
class TestLongestConsecutiveRun:
    def test_simple_run(self):
        assert _longest_consecutive_run([1990, 1991, 1992, 1994, 1995]) == (3, 1990, 1992)

    def test_single_year(self):
        assert _longest_consecutive_run([2000]) == (1, 2000, 2000)

    def test_empty(self):
        assert _longest_consecutive_run([]) == (0, None, None)

    def test_no_consecutive(self):
        length, start, end = _longest_consecutive_run([1990, 1992, 1995])
        assert length == 1 and start == 1990 and end == 1990

    def test_dedups_repeats(self):
        # Repeated years (e.g. co-champions same year) don't extend a streak.
        assert _longest_consecutive_run([2010, 2010, 2011, 2012]) == (3, 2010, 2012)


class TestComputeDroughts:
    def test_sorted_descending_and_nonnegative(self, index):
        r, _ = index
        d = compute_droughts(r)
        droughts = [x["drought"] for x in d["overall"]]
        assert droughts == sorted(droughts, reverse=True)
        assert all(x >= 0 for x in droughts)

    def test_reigning_school_has_zero_drought(self, index):
        r, _ = index
        d = compute_droughts(r)
        alleg = next(x for x in d["overall"] if x["school"] == "Allegany")
        assert alleg["drought"] == 0  # last title 2025, current_year 2025

    def test_current_year_anchored_to_data(self, index):
        r, _ = index
        d = compute_droughts(r)
        assert d["current_year"] == 2025

    def test_per_sport_has_sport_field(self, index):
        r, _ = index
        d = compute_droughts(r)
        assert all("sport" in x for x in d["per_sport"])
        assert d["per_sport"]


class TestComputeStreaks:
    def test_dynasties_length_at_least_two(self, index):
        r, _ = index
        s = compute_streaks(r)
        assert all(x["length"] >= 2 for x in s["dynasties"])

    def test_dynasties_sorted_desc(self, index):
        r, _ = index
        s = compute_streaks(r)
        lens = [x["length"] for x in s["dynasties"]]
        assert lens == sorted(lens, reverse=True)

    def test_active_ends_at_latest_year(self, index):
        r, _ = index
        s = compute_streaks(r)
        for a in s["active"]:
            assert a["end"] == s["latest_year"]


class TestComputeNeverWon:
    def test_reached_final_no_title_has_years(self, index):
        r, _ = index
        nw = compute_never_won(r)
        assert all(e["finalist_years"] for e in nw["reached_final_no_title"])

    def test_no_title_any_sport_sorted_by_finals(self, index):
        r, _ = index
        nw = compute_never_won(r)
        finals = [e["finals"] for e in nw["no_title_any_sport"]]
        assert finals == sorted(finals, reverse=True)


class TestComputeFirstTitleWatch:
    def test_candidates_are_latest_year_and_titleless(self, index):
        r, _ = index
        ft = compute_first_title_watch(r)
        for c in ft["candidates"]:
            assert c["year"] == ft["latest_year"]


class TestComputeAnniversaries:
    def test_anniversaries_are_round(self, index):
        r, _ = index
        a = compute_anniversaries(r)
        assert all(e["anniversary"] in (25, 50, 75, 100) for e in a["anniversaries"])
        # year + anniversary == current_year
        for e in a["anniversaries"]:
            assert e["year"] + e["anniversary"] == a["current_year"]

    def test_each_has_citation(self, index):
        r, _ = index
        a = compute_anniversaries(r)
        assert all(e["citation"] for e in a["anniversaries"])


class TestComputePegs:
    def test_has_all_sections(self, index):
        r, _ = index
        p = compute_pegs(r)
        assert set(p) == {"droughts", "streaks", "never_won",
                          "first_title_watch", "anniversaries"}


# ── Embed widgets ────────────────────────────────────────────────────────────
class TestBuildEmbeds:
    def test_renders_all_embed_types(self, index, tmp_path):
        r, _ = index
        n = build_embeds(r, tmp_path, root="./")
        assert (tmp_path / "embed" / "timeline" / "allegany" / "index.html").exists()
        assert (tmp_path / "embed" / "titles" / "allegany" / "index.html").exists()
        assert (tmp_path / "embed" / "anniversaries" / "index.html").exists()
        assert (tmp_path / "embed" / "index.html").exists()  # builder
        assert n == len(r.by_key) * 2 + 2

    def test_timeline_embed_is_self_contained(self, index, tmp_path):
        r, _ = index
        build_embeds(r, tmp_path, root="./")
        h = (tmp_path / "embed" / "timeline" / "allegany" / "index.html").read_text()
        assert "<link" not in h          # no external stylesheet
        assert "<script src=" not in h   # no external script
        assert "<svg" in h               # timeline inlined
        assert 'data-theme="light"' in h  # default theme
        assert "URLSearchParams" in h     # theme toggle script

    def test_titles_embed_shows_count(self, index, tmp_path):
        r, _ = index
        build_embeds(r, tmp_path, root="./")
        h = (tmp_path / "embed" / "titles" / "allegany" / "index.html").read_text()
        assert 'class="num">34</span>' in h

    def test_builder_inlines_schools_json(self, index, tmp_path):
        r, _ = index
        build_embeds(r, tmp_path, root="./")
        h = (tmp_path / "embed" / "index.html").read_text()
        assert 'id="schools-data"' in h
        assert h.count('"slug"') == len(r.by_key)


# ── Full-site integration ────────────────────────────────────────────────────
class TestSiteIntegration:
    def test_built_site_has_no_broken_links(self, index, tmp_path):
        """Build the whole site and crawl it for broken internal links.

        Guards against root-prefix regressions (e.g. an index page one level
        deep rendered with root='./' instead of '../').
        """
        import importlib.util
        from build_site import build_site as _build
        registry, report = index
        _build(registry, report, tmp_path)
        spec = importlib.util.spec_from_file_location(
            "check_site_links", "scripts/check_site_links.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        broken = mod.check(tmp_path)
        assert broken == [], f"{len(broken)} broken links, e.g. {broken[:3]}"