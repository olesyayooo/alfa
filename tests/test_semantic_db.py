"""True pgvector integration with a deterministic model; no model weights needed."""

import os
from dataclasses import replace
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio

from alfa_search import SemanticConfig, SemanticSearch
from alfa_search.models import Material
from alfa_search.semantic.repository import ModelMismatchError
from tests.test_semantic import CountingEncoder

pytestmark = pytest.mark.vector_db
DSN = os.getenv("TEST_VECTOR_DATABASE_URL")


class FixtureSource:
    def __init__(self, materials):
        self.materials = materials
    async def batches(self, conn, policy, batch_size):
        for offset in range(0, len(self.materials), batch_size):
            yield self.materials[offset:offset + batch_size]


@pytest_asyncio.fixture
async def setup_service():
    if not DSN:
        pytest.skip("Set TEST_VECTOR_DATABASE_URL to a PostgreSQL instance with pgvector server files")
    schema = "semantic_test_" + uuid4().hex
    config = SemanticConfig(batch_size=1)
    encoder = CountingEncoder(config)
    service = SemanticSearch(DSN, config, schema=schema, encoder=encoder)
    source = FixtureSource([Material(UUID(int=1), "article", "Маркетинг"),
                            Material(UUID(int=1), "course", "Маркетинг"),
                            Material(UUID(int=2), "article", "Смешанный"),
                            Material(UUID(int=3), "document", "Налоги")])
    service._source = source
    try:
        await service.migrate()
        yield service, source, encoder
    finally:
        connection = await asyncpg.connect(DSN)
        try:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await connection.close()


async def saved_rows(service):
    connection = await asyncpg.connect(DSN)
    try:
        return [dict(r) for r in await connection.fetch(service._repository.sql('''SELECT material_type,
            material_id,embedding::text,content_hash,model_name,model_revision,encoder_signature,
            updated_at,original_tokens,used_tokens,truncated
            FROM search.material_embeddings ORDER BY material_type,material_id'''))]
    finally:
        await connection.close()


async def test_vectors_saved_idempotency_and_no_unnecessary_encoding(setup_service):
    service, source, encoder = setup_service
    await service.build_index()
    before = await saved_rows(service)
    assert len(before) == 4
    assert len(encoder.calls) == 4  # batch size is one
    assert all(len(r["embedding"].strip("[]").split(",")) == 1024 for r in before)
    await service.build_index()
    assert len(encoder.calls) == 4
    assert await saved_rows(service) == before


async def test_exact_cosine_order_composite_keys_and_top_k(setup_service):
    service, source, encoder = setup_service
    await service.build_index()
    rows = await service.search("маркетинг", top_k=3)
    assert [(r["material_type"], r["material_id"]) for r in rows] == [
        ("article", UUID(int=1)), ("course", UUID(int=1)), ("article", UUID(int=2))]
    assert [r["score"] for r in rows] == pytest.approx([1.0, 1.0, 0.6], abs=1e-6)
    assert all(isinstance(r["material_id"], UUID) for r in rows)
    assert rows == await service.search("маркетинг", 3)
    assert len(await service.search("маркетинг", 100)) == 4


async def test_changed_added_and_removed_materials(setup_service):
    service, source, encoder = setup_service
    await service.build_index()
    source.materials = [replace(source.materials[0], title="Налоги"), source.materials[1],
                        Material(UUID(int=4), "podcast", "Маркетинг")]
    await service.update_index()
    assert len(encoder.calls) == 6  # one changed, one added, one reused
    rows = await saved_rows(service)
    assert len(rows) == 3
    assert {r["material_id"] for r in rows} == {UUID(int=1), UUID(int=4)}
    assert (await service.search("налоги", 1))[0]["material_type"] == "article"


async def test_rollback_keeps_old_vectors_after_later_batch_failure(setup_service):
    service, source, encoder = setup_service
    await service.build_index()
    before = await saved_rows(service)
    source.materials = [replace(m, title="Обновлённый " + m.title) for m in source.materials]
    encoder.fail_at = len(encoder.calls) + 2  # one batch has already been written
    with pytest.raises(RuntimeError, match="Injected model failure"):
        await service.update_index()
    assert await saved_rows(service) == before
    assert (await service.search("маркетинг", 1))[0]["score"] == pytest.approx(1.0)


async def test_model_version_mismatch_then_refresh(setup_service):
    service, source, encoder = setup_service
    await service.build_index()
    config = replace(service.config, model_revision="b" * 40)
    other_encoder = CountingEncoder(config)
    other = SemanticSearch(DSN, config, schema=service._repository.schema, encoder=other_encoder)
    other._source = source
    with pytest.raises(ModelMismatchError, match="different model"):
        await other.search("маркетинг")
    assert other_encoder.calls == []
    await other.update_index()
    assert len(other_encoder.calls) == 4
    assert all(r["model_revision"] == "b" * 40 for r in await saved_rows(other))
    assert await other.search("маркетинг")
    with pytest.raises(ModelMismatchError):
        await service.search("маркетинг")


async def test_empty_index_does_not_load_model(setup_service):
    service, source, encoder = setup_service
    assert await service.search("маркетинг") == []  # migrated but never built
    source.materials = []
    await service.build_index()
    assert await service.search("маркетинг") == []
    assert encoder.calls == []


async def test_duplicate_source_keys_roll_back(setup_service):
    service, source, encoder = setup_service
    await service.build_index()
    before = await saved_rows(service)
    source.materials.append(source.materials[0])
    with pytest.raises(asyncpg.UniqueViolationError):
        await service.update_index()
    assert await saved_rows(service) == before
