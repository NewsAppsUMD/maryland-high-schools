#!/usr/bin/env python3
"""Dry-run: enumerate every cache key the current parser would produce for all
three seasons (routing + chunking + prompt-building, NO LLM calls) and report
cache files on disk that no current run would read — i.e. orphans left behind by
prompt/schema/model changes (e.g. the pre-gender-prefix wrestling prompt).

Read-only. Does not call the LLM, does not write the cache, does not touch data.
"""
import os
import sys
from pathlib import Path

# Match the parser's default model id so the keys line up with cached_extract.
os.environ.setdefault("OLLAMA_MODEL", "glm-5.2:cloud")
os.environ.setdefault("MODEL_ID", "glm-5.2:cloud")

import parse_record_book as p

PDFS = {
    "fall": "pdfs/FallRecordBook2024.pdf",
    "winter": "pdfs/Winter record book.pdf",
    "spring": "pdfs/Spring record book 2025.pdf",
}

EXTRACTOR_ROUTES = ("championship", "individual_xc", "individual_results",
                    "golf", "sportsmanship")


def expected_keys():
    """Yield {extractor}_{key[:16]} filenames the current code would read/write."""
    seen = set()
    for season, pdf in PDFS.items():
        if not Path(pdf).exists():
            print(f"  ! skip missing {pdf}", file=sys.stderr)
            continue
        pages = p.load_pages(pdf)
        titles = p.load_page_titles(pdf, p._divider_candidates(pages), pages)
        sections = p.find_sections(titles, season)
        for sport, (start, end) in sections.items():
            routed = p.route_pages(pages, sport, start, end)
            for route in EXTRACTOR_ROUTES:
                pairs = routed.get(route)
                if not pairs:
                    continue
                for chunk in p.chunked(pairs, p.CHUNK_SIZES.get(route, 1)):
                    texts = [t for _, t in chunk]
                    build_prompt, schema, _ = p.EXTRACTORS[route]
                    prompt, _ = build_prompt(texts, sport)
                    key = p._cache_key(p.MODEL_ID, schema, prompt)
                    seen.add(p._cache_path(route, key).name)
    return seen


def main():
    expected = expected_keys()
    on_disk = {f for f in os.listdir(p.CACHE_DIR) if f.endswith(".json")}
    orphans = sorted(on_disk - expected)
    print(f"expected cache files : {len(expected)}")
    print(f"cache files on disk  : {len(on_disk)}")
    print(f"orphans (unreachable): {len(orphans)}")
    for name in orphans:
        print(f"  {name}")
    if orphans:
        print("\nTo remove:")
        for name in orphans:
            print(f"  rm {p.CACHE_DIR}/{name}")


if __name__ == "__main__":
    main()