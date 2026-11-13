"""Optional read-only check of the actual Alfa-Course source catalog."""

import os

import asyncpg
import pytest

from alfa_search import SearchConfig
from alfa_search.semantic.source import CatalogSource
from alfa_search.semantic.text import material_text


async def test_streamed_real_catalog_matches_baseline_keys():
    dsn = os.getenv("CATALOG_DATABASE_URL")
    if not dsn:
        pytest.skip("Set CATALOG_DATABASE_URL to check the existing catalog and BM25F index")
    connection = await asyncpg.connect(dsn)
    try:
        async with connection.transaction(readonly=True, isolation="repeatable_read"):
            actual = set()
            async for batch in CatalogSource().batches(connection, SearchConfig(), 7):
                assert 0 < len(batch) <= 7
                for material in batch:
                    key = (material.type, material.id)
                    assert key not in actual
                    actual.add(key)
                    assert material_text(material)
            baseline = {(r["material_type"], r["material_id"]) for r in
                        await connection.fetch("SELECT material_type,material_id FROM search.materials")}
            assert actual == baseline
            assert actual
    finally:
        await connection.close()
