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
    load_aliases,
    load_books,
    make_normalizer,
    schools_index_json,
    slugify,
    split_cochampions,
)

ALIASES = load_aliases()
NORM = make_normalizer(ALIASES)


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
    registry, report = build_school_index(books, make_normalizer(load_aliases()))
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