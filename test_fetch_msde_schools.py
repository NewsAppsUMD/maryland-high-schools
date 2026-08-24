"""Tests for the MSDE school-list HTML parser (scripts/fetch_msde_schools.py).

Parser-only: no network. Feeds a small HTML fixture mirroring the real MSDE
report-card list markup (``<h2>`` section headings + ``<a>`` school links whose
text ends in a ``(1234)`` code).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from fetch_msde_schools import SchoolListParser, _clean_name

FIXTURE = """
<div class="tab-content">
  <div class="GraphTitle"><h2>Elementary</h2></div>
  <div class="tableData"><ul class="row list-group">
    <li class="col-md-6 list-group-item">
      <a class="link-underline-ACC" href="/Graphs/#/ReportCards/ReportCardSchool/1/E/1/01/2801/0">Beall Elementary                    (2801)</a>
    </li>
  </ul></div>
  <div class="GraphTitle"><h2>High School</h2></div>
  <div class="tableData"><ul class="row list-group">
    <li class="col-md-6 list-group-item">
      <a class="link-underline-ACC" href="/Graphs/#/ReportCards/ReportCardSchool/1/H/1/01/0606/0">Allegany High                       (0606)</a>
    </li>
    <li class="col-md-6 list-group-item">
      <a class="link-underline-ACC" href="/Graphs/#/ReportCards/ReportCardSchool/1/H/1/01/0405/0">Fort Hill High                      (0405)</a>
    </li>
  </ul></div>
  <div class="GraphTitle"><h2>Other</h2></div>
  <div class="tableData"><ul class="row list-group">
    <li class="col-md-6 list-group-item">
      <a class="link-underline-ACC" href="/Graphs/#/ReportCards/ReportCardSchool/1/UC/1/01/0602/0">Allegany County Evening High School  (0602)</a>
    </li>
  </ul></div>
</div>
"""


class TestCleanName:
    def test_strips_trailing_code(self):
        assert _clean_name("Fort Hill High     (0405)") == "Fort Hill High"

    def test_collapses_whitespace(self):
        assert _clean_name("  Allegany   High  ") == "Allegany High"

    def test_no_code(self):
        assert _clean_name("Kent County") == "Kent County"


class TestSchoolListParser:
    def test_high_school_section_only(self):
        p = SchoolListParser({"High School"})
        p.feed(FIXTURE)
        assert p.names == ["Allegany High", "Fort Hill High"]
        # Elementary and Other are excluded.
        assert "Beall Elementary" not in p.names
        assert not any("Evening" in n for n in p.names)

    def test_include_other_section(self):
        p = SchoolListParser({"High School", "Other"})
        p.feed(FIXTURE)
        assert "Allegany County Evening High School" in p.names
        assert "Beall Elementary" not in p.names

    def test_empty_when_no_wanted_section(self):
        p = SchoolListParser({"Middle"})
        p.feed(FIXTURE)
        assert p.names == []
