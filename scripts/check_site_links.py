"""Crawl a built static site and report broken internal links.

Walks every ``*.html`` file under the given site directory, extracts
``href``/``src`` attributes, resolves them relative to each page, and checks
that the target file exists. External links, anchors, and data URIs are
skipped. Exits non-zero if any internal link is broken.

Usage:  uv run scripts/check_site_links.py [SITE_DIR]   (default: site)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

# href="..." and src="..." (single- or double-quoted).
LINK_RE = re.compile(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']')


def _resolve(page: Path, raw: str, site_root: Path) -> Path | None:
    """Resolve a link target to a file under site_root, or None if external."""
    parts = urlsplit(raw)
    if parts.scheme or parts.netloc:
        return None  # external (http://, //host, mailto:, data:, ...)
    path = parts.path
    if not path or path.startswith("#"):
        return None
    # Strip query and fragment already done by urlsplit; resolve relative.
    target = (page.parent / path).resolve()
    # Directory links (trailing slash or empty after dir) -> index.html.
    if path.endswith("/"):
        target = target / "index.html"
    elif target.is_dir():
        target = target / "index.html"
    # Relative path may escape site_root for assets like ../../static; that's fine
    # as long as it resolves under site_root after normalization.
    try:
        target.relative_to(site_root)
    except ValueError:
        return None  # outside the site (shouldn't happen for internal links)
    if not target.suffix:
        target = target / "index.html"
    return target


def check(site_root: Path) -> list[tuple[Path, str, str]]:
    """Return a list of (page, broken_link, reason) tuples."""
    broken: list[tuple[Path, str, str]] = []
    pages = sorted(site_root.rglob("*.html"))
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        for raw in LINK_RE.findall(text):
            if raw.startswith(("http://", "https://", "//", "#", "mailto:", "data:")):
                continue
            target = _resolve(page, raw, site_root)
            if target is None:
                continue
            if not target.exists():
                broken.append((page, raw, str(target.relative_to(site_root))))
    return broken


def main(argv: list[str] | None = None) -> int:
    site_root = (Path(argv[0]) if argv else Path("site")).resolve()
    if not site_root.exists():
        print(f"Error: {site_root} not found. Run `make site` first.", file=sys.stderr)
        return 2
    broken = check(site_root)
    pages = sorted(site_root.rglob("*.html"))
    print(f"Checked {len(pages)} pages, {len(broken)} broken internal link(s).")
    for page, raw, tgt in broken:
        print(f"  {page.relative_to(site_root)}: {raw!r} -> missing {tgt}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))