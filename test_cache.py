"""Tests for the per-section extraction cache (Fix 13). No LLM calls."""

import pytest

import parse_record_book as p


@pytest.fixture
def cache_on(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "CACHE_ENABLED", True)
    monkeypatch.setattr(p, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


class TestCachedSection:
    def test_miss_then_hit(self, cache_on):
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return [{"year": 2024, "x": 1}]

        first = p.cached_section("Boys Soccer championship", ["page text"], compute)
        second = p.cached_section("Boys Soccer championship", ["page text"], compute)
        assert first == second == [{"year": 2024, "x": 1}]
        assert calls["n"] == 1  # second call served from cache, no recompute

    def test_input_text_change_invalidates(self, cache_on):
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return [{"n": calls["n"]}]

        p.cached_section("lbl", ["original"], compute)
        p.cached_section("lbl", ["EDITED text"], compute)  # different pages
        assert calls["n"] == 2  # both computed — no false cache hit

    def test_label_change_invalidates(self, cache_on):
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return []

        p.cached_section("Football championship", ["t"], compute)
        p.cached_section("Volleyball championship", ["t"], compute)
        assert calls["n"] == 2

    def test_extraction_version_change_invalidates(self, cache_on, monkeypatch):
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return []

        p.cached_section("lbl", ["t"], compute)
        # Simulate a prompt/logic edit → new fingerprint
        monkeypatch.setattr(p, "_extraction_version", lambda: "v2-different")
        p.cached_section("lbl", ["t"], compute)
        assert calls["n"] == 2

    def test_disabled_never_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(p, "CACHE_ENABLED", False)
        monkeypatch.setattr(p, "CACHE_DIR", tmp_path / "cache")
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return [{"a": 1}]

        p.cached_section("lbl", ["t"], compute)
        p.cached_section("lbl", ["t"], compute)
        assert calls["n"] == 2  # no caching when disabled
        assert not (tmp_path / "cache").exists()

    def test_roundtrip_preserves_rows(self, cache_on):
        rows = [{"sport": "S", "year": 2024, "champion_undefeated": True,
                 "notes": None, "source_pages": "62-63"}]
        p.cached_section("lbl", ["t"], lambda: rows)
        loaded = p.cached_section("lbl", ["t"], lambda: [])  # served from cache
        assert loaded == rows


class TestExtractionVersion:
    def test_stable_and_short(self):
        v1 = p._extraction_version()
        v2 = p._extraction_version()
        assert v1 == v2  # deterministic within a build
        assert len(v1) == 16
