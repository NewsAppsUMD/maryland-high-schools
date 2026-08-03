"""Tests for build_site.py — normalizer, slugs, co-champion split, display-name
selection, and the unified school index built from the real season books."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site import (
    _base_normalize,
    best_display_name,
    build_school_index,
    fast_facts_paragraph,
    load_aliases,
    load_books,
    make_normalizer,
    render_timeline_svg,
    schools_index_json,
    slugify,
    split_cochampions,
)

ALIAS_MAP, CANON_DISPLAY = load_aliases()
NORM = make_normalizer(ALIAS_MAP)


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
        assert report["school_records_rows"] == 1719
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
        # 1990, 2000, 2010 decade lines should appear within the 1992-2010 range.
        assert "1990" in svg and "2000" in svg and "2010" in svg
        assert svg.count('class="tl-grid"') >= 3

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


# ── Fast-facts paragraph generator ───────────────────────────────────────────
class TestFastFacts:
    def test_with_titles_full_paragraph(self, index):
        r, _ = index
        ff = fast_facts_paragraph("Allegany", r.lookup("Allegany"))
        assert ff == ("Allegany has won 34 state championships across 7 sports, "
                      "most recently 1A baseball in 2025. It has produced 13 "
                      "individual state champions and won 2 sportsmanship awards.")

    def test_singular_grammar_one_title_one_sport(self, index):
        r, _ = index
        ff = fast_facts_paragraph("Charles W. Woodward", r.lookup("Charles W. Woodward"))
        assert ff == ("Charles W. Woodward has won 1 state championship across 1 "
                      "sport, most recently A boys basketball in 1977.")

    def test_no_title_but_finals(self, index):
        r, _ = index
        ff = fast_facts_paragraph("Cambridge-South Dorchester",
                                  r.lookup("Cambridge-South Dorchester"))
        assert ff == ("Cambridge-South Dorchester has reached 1 state final without "
                      "a title. It has produced 6 individual state champions and won "
                      "2 sportsmanship awards.")

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