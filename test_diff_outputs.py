"""Tests for diff_outputs.py — semantic diff by natural key vs HEAD."""

import json
import subprocess
from pathlib import Path

import pytest

import diff_outputs as diffmod
from diff_outputs import (
    DEDUP_KEYS,
    PROVENANCE_FIELDS,
    build_markdown,
    diff_season,
    diff_table,
    load_head_book,
    main,
)


def _rows(**overrides):
    """A small championship_results fixture."""
    base = [
        {"sport": "Golf", "year": 2023, "classification": "Combined",
         "champion_school": "Magruder", "score": "300"},
        {"sport": "Golf", "year": 2024, "classification": "Combined",
         "champion_school": "North Hagerstown", "score": "295"},
    ]
    return [{**r, **overrides.get(i, {})} for i, r in enumerate(base)]


KEY = DEDUP_KEYS["championship_results"]


class TestDiffTable:
    def test_identical_is_empty(self):
        rows = _rows()
        d = diff_table(rows, rows, KEY)
        assert d == {"added": [], "removed": [], "changed": []}

    def test_added(self):
        cur = _rows() + [{"sport": "Golf", "year": 2022, "classification": "Combined",
                          "champion_school": "Walter Johnson", "score": "310"}]
        d = diff_table(cur, _rows(), KEY)
        assert len(d["added"]) == 1
        assert d["added"][0]["year"] == 2022
        assert d["removed"] == [] and d["changed"] == []

    def test_removed(self):
        head = _rows()
        cur = [head[0]]  # drop the 2024 row
        d = diff_table(cur, head, KEY)
        assert len(d["removed"]) == 1
        assert d["removed"][0]["year"] == 2024
        assert d["added"] == [] and d["changed"] == []

    def test_changed_payload_flagged(self):
        head = _rows()
        cur = _rows()
        cur[1]["score"] = "290"  # same key, different score
        d = diff_table(cur, head, KEY)
        assert len(d["changed"]) == 1
        old, new = d["changed"][0]
        assert old["score"] == "295" and new["score"] == "290"
        assert d["added"] == [] and d["removed"] == []

    def test_provenance_changes_are_not_changes(self):
        """extraction_model/extracted_at/source_pages changes don't count as changed."""
        head = _rows(extracted_at="2025-01-01")
        cur = _rows(extracted_at="2026-08-01", extraction_model="glm-5.2:cloud",
                    source_pages=[1, 2], source_pdf="new.pdf")
        d = diff_table(cur, head, KEY)
        assert d["changed"] == []  # only provenance differs

    def test_duplicate_keys_keep_first(self):
        """If a natural key repeats within one side, the first row wins."""
        cur = _rows()
        cur.append({"sport": "Golf", "year": 2024, "classification": "Combined",
                    "champion_school": "North Hagerstown", "score": "999"})
        d = diff_table(cur, _rows(), KEY)
        # The duplicate key 2024 already matches HEAD's 2024 row; no add/change.
        assert d["added"] == [] and d["changed"] == []

    def test_school_records_key(self):
        key = DEDUP_KEYS["school_records"]
        assert key == ("sport", "school")
        head = [{"sport": "Golf", "school": "Magruder", "champion_years": [2023]}]
        cur = [{"sport": "Golf", "school": "Magruder", "champion_years": [2023, 2024]}]
        d = diff_table(cur, head, key)
        assert len(d["changed"]) == 1


class TestProvenanceFields:
    def test_provenance_fields_present(self):
        # The parser stamps these on every row; the diff must ignore them.
        assert {"source_pdf", "source_pages", "extracted_at", "extraction_model"} <= set(
            PROVENANCE_FIELDS)


class TestGitIntegration:
    """End-to-end: a real tmp git repo with a committed record_book.json."""

    def _repo(self, tmp_path):
        (tmp_path / "data" / "fall").mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        return tmp_path / "data" / "fall"

    def _commit(self, data_dir, book):
        (data_dir / "record_book.json").write_text(json.dumps(book))
        subprocess.run(["git", "add", "-A"], cwd=data_dir.parent.parent, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "snap"], cwd=data_dir.parent.parent,
                       check=True)

    def test_no_changes_when_identical(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = self._repo(tmp_path)
        book = {"championship_results": _rows()}
        self._commit(data_dir, book)
        result = diff_season(data_dir)
        assert result["head_note"] is None
        for t in diffmod.TABLE_ORDER:
            assert result["tables"][t] == {"added": [], "removed": [], "changed": []}

    def test_added_and_removed_detected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = self._repo(tmp_path)
        head_book = {"championship_results": _rows()}
        self._commit(data_dir, head_book)
        # Working tree: drop 2024, add 2022.
        cur_book = {"championship_results": [
            {"sport": "Golf", "year": 2023, "classification": "Combined",
             "champion_school": "Magruder", "score": "300"},
            {"sport": "Golf", "year": 2022, "classification": "Combined",
             "champion_school": "Walter Johnson", "score": "310"},
        ]}
        (data_dir / "record_book.json").write_text(json.dumps(cur_book))
        result = diff_season(data_dir)
        d = result["tables"]["championship_results"]
        assert len(d["removed"]) == 1 and d["removed"][0]["year"] == 2024
        assert len(d["added"]) == 1 and d["added"][0]["year"] == 2022
        assert d["changed"] == []

    def test_no_head_baseline_marks_everything_added(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data" / "fall"
        data_dir.mkdir(parents=True)
        (data_dir / "record_book.json").write_text(json.dumps(
            {"championship_results": _rows()}))
        # No git repo -> HEAD unavailable.
        result = diff_season(data_dir)
        assert result["head_note"] is not None
        d = result["tables"]["championship_results"]
        assert len(d["added"]) == 2
        assert d["removed"] == [] and d["changed"] == []

    def test_markdown_report_has_table_and_counts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = self._repo(tmp_path)
        self._commit(data_dir, {"championship_results": _rows()})
        (data_dir / "record_book.json").write_text(json.dumps(
            {"championship_results": [_rows()[0]]}))  # remove one row
        result = diff_season(data_dir)
        md = build_markdown({"seasons": [result]})
        assert "championship_results" in md
        assert "| 0 | 1 | 0 |" in md  # 0 added, 1 removed, 0 changed

    def test_main_writes_report_and_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = self._repo(tmp_path)
        self._commit(data_dir, {"championship_results": _rows()})
        rc = main([str(data_dir), "--out", str(tmp_path / "r.md")])
        assert rc == 0
        assert (tmp_path / "r.md").exists()

    def test_main_skips_missing_dir(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = main([str(tmp_path / "data" / "nonexistent")])
        assert rc == 1
        err = capsys.readouterr().err
        assert "no record_book.json" in err


class TestBaselineDir:
    """--baseline-dir compares against an on-disk dir instead of git HEAD."""

    def _book(self, rows):
        return {"championship_results": rows}

    def test_baseline_dir_detects_added(self, tmp_path):
        baseline = tmp_path / "data"
        cur = tmp_path / "build"
        (baseline / "fall").mkdir(parents=True)
        (cur / "fall").mkdir(parents=True)
        head_rows = [{"sport": "Golf", "year": 2023, "classification": "Combined",
                      "champion_school": "Magruder", "score": "300"}]
        cur_rows = head_rows + [{"sport": "Golf", "year": 2024, "classification": "Combined",
                                 "champion_school": "X", "score": "290"}]
        (baseline / "fall" / "record_book.json").write_text(json.dumps(self._book(head_rows)))
        (cur / "fall" / "record_book.json").write_text(json.dumps(self._book(cur_rows)))
        result = diff_season(cur / "fall", baseline_dir=baseline)
        assert len(result["tables"]["championship_results"]["added"]) == 1
        assert result["head_note"] is None

    def test_baseline_dir_missing_is_no_baseline(self, tmp_path):
        cur = tmp_path / "build" / "fall"
        cur.mkdir(parents=True)
        (cur / "record_book.json").write_text(json.dumps(self._book([])))
        result = diff_season(cur, baseline_dir=tmp_path / "data")
        assert result["head_note"] is not None
        # No baseline -> no diff entries.
        assert result["tables"]["championship_results"]["added"] == []

    def test_fail_on_diff_returns_one(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        baseline = tmp_path / "data"
        cur = tmp_path / "build"
        (baseline / "fall").mkdir(parents=True)
        (cur / "fall").mkdir(parents=True)
        (baseline / "fall" / "record_book.json").write_text(json.dumps(self._book([])))
        (cur / "fall" / "record_book.json").write_text(json.dumps(self._book([
            {"sport": "Golf", "year": 2024, "classification": "Combined",
             "champion_school": "X", "score": "290"}])))
        rc = main([str(cur / "fall"), "--baseline-dir", str(baseline),
                   "--out", str(tmp_path / "r.md"), "--fail-on-diff"])
        assert rc == 1

    def test_fail_on_diff_zero_when_identical(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        baseline = tmp_path / "data"
        cur = tmp_path / "build"
        (baseline / "fall").mkdir(parents=True)
        (cur / "fall").mkdir(parents=True)
        rows = [{"sport": "Golf", "year": 2024, "classification": "Combined",
                 "champion_school": "X", "score": "290"}]
        (baseline / "fall" / "record_book.json").write_text(json.dumps(self._book(rows)))
        (cur / "fall" / "record_book.json").write_text(json.dumps(self._book(rows)))
        rc = main([str(cur / "fall"), "--baseline-dir", str(baseline),
                   "--out", str(tmp_path / "r.md"), "--fail-on-diff"])
        assert rc == 0

    def test_provenance_only_diff_does_not_fail(self, tmp_path, monkeypatch):
        """A rebuilt tree differs only in extracted_at — not a real diff."""
        monkeypatch.chdir(tmp_path)
        baseline = tmp_path / "data"
        cur = tmp_path / "build"
        (baseline / "fall").mkdir(parents=True)
        (cur / "fall").mkdir(parents=True)
        rows = [{"sport": "Golf", "year": 2024, "classification": "Combined",
                 "champion_school": "X", "score": "290"}]
        (baseline / "fall" / "record_book.json").write_text(
            json.dumps(self._book([{**r, "extracted_at": "2025-01-01"} for r in rows])))
        (cur / "fall" / "record_book.json").write_text(
            json.dumps(self._book([{**r, "extracted_at": "2026-08-01",
                                    "extraction_model": "glm-5.2:cloud"} for r in rows])))
        rc = main([str(cur / "fall"), "--baseline-dir", str(baseline),
                   "--out", str(tmp_path / "r.md"), "--fail-on-diff"])
        assert rc == 0


class TestLoadHeadBook:
    def test_untracked_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        book, note = load_head_book(tmp_path / "data" / "fall")
        assert book is None
        assert note is not None

    def test_loads_committed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data" / "fall"
        data_dir.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        book = {"championship_results": _rows()}
        (data_dir / "record_book.json").write_text(json.dumps(book))
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=tmp_path, check=True)
        got, note = load_head_book(data_dir)
        assert got is not None and note is None
        assert got["championship_results"][0]["year"] == 2023