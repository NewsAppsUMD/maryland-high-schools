"""Tests for school-name normalization (Fix 7)."""

import json

import pytest

from parse_record_book import (
    canonicalize_rows,
    load_aliases,
    normalize_school,
    slugify_school,
)

ALIASES = {
    "b-cc": "Bethesda-Chevy Chase",
    "e. roosevelt": "Eleanor Roosevelt",
}


class TestNormalizeSchool:
    def test_all_caps_titlecased(self):
        assert normalize_school("ALLEGANY", ALIASES) == "Allegany"

    def test_mixed_case_passthrough(self):
        # Already correctly cased — must not be altered.
        assert normalize_school("McDonough", ALIASES) == "McDonough"

    def test_all_caps_mcdonough(self):
        assert normalize_school("MAURICE J. MCDONOUGH", ALIASES) == "Maurice J. McDonough"

    def test_all_caps_hyphenated(self):
        assert normalize_school("BETHESDA-CHEVY CHASE", ALIASES) == "Bethesda-Chevy Chase"

    def test_alias_case_insensitive(self):
        assert normalize_school("B-CC", ALIASES) == "Bethesda-Chevy Chase"
        assert normalize_school("b-cc", ALIASES) == "Bethesda-Chevy Chase"

    def test_alias_beats_titlecase(self):
        assert normalize_school("E. Roosevelt", ALIASES) == "Eleanor Roosevelt"

    def test_co_champions_each_normalized(self):
        assert normalize_school("B-CC & ALLEGANY", ALIASES) == "Bethesda-Chevy Chase & Allegany"

    def test_whitespace_cleanup(self):
        assert normalize_school("  ALLEGANY   HIGH  ", ALIASES) == "Allegany High"

    def test_empty(self):
        assert normalize_school("", ALIASES) == ""
        assert normalize_school(None, ALIASES) == ""


class TestSlugify:
    def test_basic(self):
        assert slugify_school("Bethesda-Chevy Chase") == "bethesda-chevy-chase"

    def test_ampersand(self):
        assert slugify_school("A & B") == "a-and-b"

    def test_apostrophe_and_period(self):
        assert slugify_school("Queen Anne's") == "queen-annes"
        assert slugify_school("St. Mary's") == "st-marys"


class TestCanonicalizeRows:
    def test_adds_canonical_and_slug(self):
        rows = [{"school": "ALLEGANY"}]
        name_map = {}
        canonicalize_rows(rows, ["school"], ALIASES, name_map)
        assert rows[0]["school"] == "Allegany"
        assert rows[0]["school_slug"] == "allegany"
        assert name_map == {"ALLEGANY": "Allegany"}

    def test_multiple_fields(self):
        rows = [{"champion_school": "B-CC", "finalist_school": "ALLEGANY"}]
        canonicalize_rows(rows, ["champion_school", "finalist_school"], ALIASES, {})
        assert rows[0]["champion_school"] == "Bethesda-Chevy Chase"
        assert rows[0]["champion_school_slug"] == "bethesda-chevy-chase"
        assert rows[0]["finalist_school"] == "Allegany"

    def test_unchanged_name_not_in_map(self):
        rows = [{"school": "Allegany"}]  # already canonical
        name_map = {}
        canonicalize_rows(rows, ["school"], ALIASES, name_map)
        assert name_map == {}

    def test_empty_field_skipped(self):
        rows = [{"school": None}]
        canonicalize_rows(rows, ["school"], ALIASES, {})
        assert "school_slug" not in rows[0]


class TestLoadAliases:
    def test_loads_and_skips_comment(self, tmp_path):
        p = tmp_path / "aliases.json"
        p.write_text(json.dumps({"_comment": "ignore me", "B-CC": "Bethesda-Chevy Chase"}))
        aliases = load_aliases(p)
        assert "_comment" not in aliases
        assert aliases["b-cc"] == "Bethesda-Chevy Chase"

    def test_real_alias_file_loads(self):
        aliases = load_aliases()
        assert aliases["b-cc"] == "Bethesda-Chevy Chase"
