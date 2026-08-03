# MPSSAA record book pipeline — single-maintainer teaching repo.
# All targets run via `uv run` so the project venv is used automatically.

FALL_PDF   := pdfs/FallRecordBook2024.pdf
WINTER_PDF := pdfs/Winter record book.pdf
SPRING_PDF := pdfs/Spring record book 2025.pdf

SEASONS    := fall winter spring

.PHONY: test routes extract extract-all diff verify rebuild-offline \
        site serve check-links clean clean-build

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	uv run pytest

# ── Routing sanity check (no LLM calls) ───────────────────────────────────────
# Prints page->classifier table for each PDF. Catches a mis-routed section before
# spending money on extraction.
routes:
	uv run parse_record_book.py $(FALL_PDF) --routes
	uv run parse_record_book.py "$(WINTER_PDF)" --routes
	uv run parse_record_book.py "$(SPRING_PDF)" --routes

# ── Extraction (LLM; only changed pages hit the API thanks to the cache) ─────
extract:
	uv run parse_record_book.py $(FALL_PDF)

extract-all:
	uv run parse_record_book.py $(FALL_PDF)
	uv run parse_record_book.py "$(WINTER_PDF)"
	uv run parse_record_book.py "$(SPRING_PDF)"

# ── Human review: added/removed/changed vs last commit ────────────────────────
diff:
	uv run diff_outputs.py

# ── Verification gates (cross-path, continuity, era floors, regression) ────────
# Runs on the committed data/ for each season that has a record_book.json.
verify:
	@for s in $(SEASONS); do \
		[ -f data/$$s/record_book.json ] || continue; \
		echo "── verify $$s ──"; \
		uv run verify_record_book.py data/$$s || exit $$?; \
	done

# ── CI rebuild gate: regenerate data/ from the cache with NO LLM calls ────────
# Rebuilds each season into build/ using --offline (a cache miss errors out,
# proving the committed cache is complete), then semantically diffs the rebuilt
# build/ against the committed data/. Fails if the cache can't reproduce the
# committed output or if any row was lost. Requires `make extract-all` to have
# populated cache/extractions/ first.
rebuild-offline: clean-build
	uv run parse_record_book.py $(FALL_PDF)   --out build/fall   --offline
	uv run parse_record_book.py "$(WINTER_PDF)" --out build/winter --offline
	uv run parse_record_book.py "$(SPRING_PDF)" --out build/spring --offline
	uv run diff_outputs.py build/fall build/winter build/spring \
		--baseline-dir data --fail-on-diff --out build/diff_report.md

# ── Static site (GitHub Pages) ────────────────────────────────────────────────
# Builds site/ from data/*/record_book.json. `make serve` previews it locally.
site:
	uv run build_site.py

serve:
	@echo "Serving site/ at http://localhost:8000/  (Ctrl-C to stop)"
	$(MAKE) site
	uv run python -m http.server -d site 8000

# Internal link check: no 404s in the built site.
check-links: site
	uv run scripts/check_site_links.py site

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean-build:
	rm -rf build

clean: clean-build
	rm -f diff_report.md
	find data -name verification_report.json -delete
	rm -rf site