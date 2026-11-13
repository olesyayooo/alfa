"""Explicit opt-in only: downloads/loads BGE-M3 and tests actual model inference."""

import os
from uuid import UUID, uuid4

import asyncpg
import pytest

from alfa_search import SemanticSearch
from alfa_search.models import Material
from tests.test_semantic_db import FixtureSource


async def test_bge_m3_with_pgvector():
    if os.getenv("RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Explicitly set RUN_REAL_MODEL_TESTS=1 to load the large BGE-M3 model")
    dsn = os.getenv("TEST_VECTOR_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_VECTOR_DATABASE_URL for the real model integration test")
    schema = "semantic_model_test_" + uuid4().hex
    semantic = SemanticSearch(dsn, schema=schema)
    semantic._source = FixtureSource([
        Material(UUID(int=1), "course", "Привлечение клиентов и увеличение продаж",
                 "Маркетинг, реклама, работа с клиентами и продвижение бизнеса"),
        Material(UUID(int=2), "document", "Налоговая декларация по налогу на прибыль",
                 "Форма отчётности, сроки сдачи и уплаты налогов"),
    ])
    try:
        await semantic.build_index()
        rows = await semantic.search("Как найти новых покупателей и продвигать свои товары?", 2)
        assert len(rows) == 2
        assert rows[0]["material_id"] == UUID(int=1)
        assert -1 <= rows[1]["score"] <= rows[0]["score"] <= 1.000001
    finally:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()
