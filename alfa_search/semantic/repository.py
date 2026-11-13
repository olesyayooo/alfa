"""Persistent vector index; no embeddings are loaded for Python-side ranking."""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import asyncpg

from ..config import SearchConfig
from ..models import SearchResult
from .config import SemanticConfig
from .encoder import Embedding
from .source import CatalogSource
from .text import content_hash, material_text

logger = logging.getLogger(__name__)
Encode = Callable[[list[str]], Awaitable[list[Embedding]]]


class MissingPgvectorError(RuntimeError):
    pass


class SemanticIndexNotReadyError(RuntimeError):
    pass


class ModelMismatchError(RuntimeError):
    pass


def policy_signature(policy: SearchConfig) -> str:
    return json.dumps({"exclude_closed": policy.exclude_closed,
                       "public_articles_only": policy.public_articles_only}, sort_keys=True)


class SemanticRepository:
    def __init__(self, dsn: str, schema: str = "search") -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
            raise ValueError("Invalid search schema")
        self.dsn, self.schema = dsn, schema

    def sql(self, query: str) -> str:
        return query.replace("search.", f'"{self.schema}".')

    async def _check_vector(self, conn: asyncpg.Connection, *, allow_install: bool) -> None:
        namespace = await conn.fetchval('''SELECT n.nspname FROM pg_extension e
            JOIN pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector' ''')
        if namespace:
            if namespace != "public":
                raise MissingPgvectorError("This migration expects pgvector in public; configure it there before indexing")
            return
        available = await conn.fetchval("SELECT default_version FROM pg_available_extensions WHERE name='vector'")
        if not available:
            raise MissingPgvectorError("pgvector server files are missing. See SEMANTIC_SEARCH.md: "
                "use compose.pgvector.yaml with PostgreSQL 17, then run semantic migrate")
        if not allow_install:
            raise MissingPgvectorError("pgvector is available but not enabled. Run python -m alfa_search.semantic migrate")

    async def _migrate(self, conn: asyncpg.Connection) -> None:
        await self._check_vector(conn, allow_install=True)
        migration = Path(__file__).parent.parent.joinpath("migrations/002_semantic.sql").read_text()
        migration = self.sql(migration).replace("SCHEMA IF NOT EXISTS search;", f'SCHEMA IF NOT EXISTS "{self.schema}";')
        try:
            await conn.execute(migration)
        except asyncpg.InsufficientPrivilegeError as exc:
            raise MissingPgvectorError("Migration requires permission to enable vector and create search tables; "
                                       "ask the database owner to run migrations/002_semantic.sql") from exc

    async def _register_vector(self, conn: asyncpg.Connection) -> None:
        from pgvector.asyncpg import register_vector
        await register_vector(conn, schema="public")

    async def migrate(self) -> None:
        conn = await asyncpg.connect(self.dsn)
        try:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", "alfa_semantic:" + self.schema)
                await self._migrate(conn)
        finally:
            await conn.close()

    async def refresh(self, source: CatalogSource, encode: Encode, config: SemanticConfig,
                      policy: SearchConfig) -> None:
        conn = await asyncpg.connect(self.dsn)
        processed = changed_count = truncated_count = 0
        try:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", "alfa_semantic:" + self.schema)
                await self._migrate(conn)
                await self._register_vector(conn)
                await conn.execute('''CREATE TEMP TABLE semantic_seen (
                    material_type text NOT NULL, material_id uuid NOT NULL,
                    PRIMARY KEY(material_type,material_id)) ON COMMIT DROP''')
                async for batch in source.batches(conn, policy, config.batch_size):
                    if not batch:
                        continue
                    keys = [(m.type, m.id) for m in batch]
                    # Duplicate source keys, including across batches, cause rollback.
                    await conn.executemany("INSERT INTO semantic_seen VALUES ($1,$2)", keys)
                    stored = await conn.fetch(self.sql('''SELECT e.material_type,e.material_id,e.content_hash,
                        e.model_name,e.model_revision,e.encoder_signature FROM search.material_embeddings e
                        JOIN unnest($1::text[],$2::uuid[]) AS wanted(kind,id)
                        ON e.material_type=wanted.kind AND e.material_id=wanted.id'''),
                        [k[0] for k in keys], [k[1] for k in keys])
                    existing = {(r["material_type"], r["material_id"]): r for r in stored}
                    changed = []
                    for material in batch:
                        text = material_text(material)
                        digest = content_hash(text)
                        old = existing.get((material.type, material.id))
                        if (old is None or old["content_hash"] != digest or old["model_name"] != config.model_name
                            or old["model_revision"] != config.model_revision or old["encoder_signature"] != config.signature):
                            changed.append((material, text, digest))
                    if changed:
                        embeddings = await encode([text for _, text, _ in changed])
                        records = []
                        for (material, _, digest), embedding in zip(changed, embeddings, strict=True):
                            records.append((material.type, material.id, embedding.vector, digest,
                                config.model_name, config.model_revision, config.signature,
                                embedding.original_tokens, embedding.used_tokens, embedding.truncated))
                            if embedding.truncated:
                                truncated_count += 1
                                logger.warning("Trimmed %s/%s from %d to %d tokens; title/metadata precede description",
                                    material.type, material.id, embedding.original_tokens, embedding.used_tokens)
                        await conn.executemany(self.sql('''INSERT INTO search.material_embeddings
                            (material_type,material_id,embedding,content_hash,model_name,model_revision,
                             encoder_signature,original_tokens,used_tokens,truncated)
                            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                            ON CONFLICT (material_type,material_id) DO UPDATE SET
                            embedding=EXCLUDED.embedding,content_hash=EXCLUDED.content_hash,
                            model_name=EXCLUDED.model_name,model_revision=EXCLUDED.model_revision,
                            encoder_signature=EXCLUDED.encoder_signature,original_tokens=EXCLUDED.original_tokens,
                            used_tokens=EXCLUDED.used_tokens,truncated=EXCLUDED.truncated,updated_at=now()'''), records)
                    processed += len(batch)
                    changed_count += len(changed)
                    logger.info("Semantic indexing: processed=%d encoded=%d reused=%d",
                                processed, changed_count, processed - changed_count)
                deleted = await conn.execute(self.sql('''DELETE FROM search.material_embeddings e
                    WHERE NOT EXISTS (SELECT 1 FROM semantic_seen s
                        WHERE s.material_type=e.material_type AND s.material_id=e.material_id)'''))
                await conn.execute(self.sql('''INSERT INTO search.semantic_index_state
                    (model_name,model_revision,encoder_signature,source_policy,material_count)
                    VALUES ($1,$2,$3,$4,$5) ON CONFLICT (singleton) DO UPDATE SET
                    model_name=EXCLUDED.model_name,model_revision=EXCLUDED.model_revision,
                    encoder_signature=EXCLUDED.encoder_signature,source_policy=EXCLUDED.source_policy,
                    material_count=EXCLUDED.material_count,updated_at=now()'''), config.model_name,
                    config.model_revision, config.signature, policy_signature(policy), processed)
            logger.info("Semantic index committed: materials=%d encoded=%d reused=%d truncated=%d %s",
                        processed, changed_count, processed - changed_count, truncated_count, deleted)
        except Exception:
            logger.exception("Semantic index update failed; all changes rolled back")
            raise
        finally:
            await conn.close()

    async def search(self, text: str, top_k: int, encode: Encode,
                     config: SemanticConfig, policy: SearchConfig) -> list[SearchResult]:
        conn = await asyncpg.connect(self.dsn)
        try:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                await self._check_vector(conn, allow_install=False)
                try:
                    state = await conn.fetchrow(self.sql("SELECT * FROM search.semantic_index_state"))
                except asyncpg.UndefinedTableError as exc:
                    raise SemanticIndexNotReadyError("Run semantic migrate and build before searching") from exc
                if state is None:
                    return []
                if (state["model_name"] != config.model_name or state["model_revision"] != config.model_revision
                    or state["encoder_signature"] != config.signature
                    or state["source_policy"] != policy_signature(policy)):
                    raise ModelMismatchError("Saved semantic index uses a different model, text configuration or "
                                             "source policy; run update_index() with the current settings")
                if not state["material_count"]:
                    return []
                await self._register_vector(conn)
                embedding = (await encode([text]))[0]
                if embedding.truncated:
                    logger.warning("Search query trimmed from %d to %d tokens", embedding.original_tokens, embedding.used_tokens)
                rows = await conn.fetch(self.sql('''SELECT material_id,material_type,
                    1 - (embedding OPERATOR(public.<=>) $1::public.vector) AS score
                    FROM search.material_embeddings
                    WHERE encoder_signature=$2 AND model_name=$3 AND model_revision=$4
                    ORDER BY embedding OPERATOR(public.<=>) $1::public.vector,material_type,material_id LIMIT $5'''),
                    embedding.vector, config.signature, config.model_name, config.model_revision, top_k)
                return [SearchResult(material_id=r["material_id"], material_type=r["material_type"],
                                     score=float(r["score"])) for r in rows]
        finally:
            await conn.close()
