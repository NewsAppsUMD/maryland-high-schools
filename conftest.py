"""Shared pytest fixtures.

main() mutates the module-level cache globals; snapshot and restore them around
every test so a run of main() (e.g. in the integration test) can't leak
CACHE_ENABLED / CACHE_DIR into unrelated tests.
"""

import pytest

import parse_record_book as p


@pytest.fixture(autouse=True)
def _isolate_cache_globals():
    enabled, cache_dir = p.CACHE_ENABLED, p.CACHE_DIR
    try:
        yield
    finally:
        p.CACHE_ENABLED = enabled
        p.CACHE_DIR = cache_dir
