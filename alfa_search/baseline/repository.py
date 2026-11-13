"""Async PostgreSQL persistence and atomic full-index replacement."""

import asyncio
import json
import logging
import re
from pathlib import Path

import asyncpg

from ..config import SearchConfig
from .indexing import prepare_index
from ..models import Material, SearchResult
from .preprocessing import Preprocessor
from ..catalog import PUBLISHED, catalog_sql
from .sql import SEARCH_SQL

logger = logging.getLogger(__name__)


class IndexNotReadyError(RuntimeError):
    """Build the index before searching, or rebuild after preprocessing changes."""


class SearchRepository:
    def __init__(self, dsn: str, schema: str = "search") -> None:
        # Custom schemas allow integration tests to avoid touching the live index.
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
            raise ValueError("Invalid search schema")
        self.dsn = dsn
        self.schema = schema

    def sql(self, query: str) -> str:
        return query.replace("search.", f'"{self.schema}".')

    @staticmethod
    def policy(config: SearchConfig) -> str:
        return json.dumps({"exclude_closed": config.exclude_closed,
                           "public_articles_only": config.public_articles_only}, sort_keys=True)

    async def load_materials(self, conn: asyncpg.Connection, config: SearchConfig) -> list[Material]:
        rows = await conn.fetch(catalog_sql(config.exclude_closed, config.public_articles_only), PUBLISHED)
        return [Material(row["id"], row["type"], row["title"], row["description"],
                         tuple(row["topics"]), tuple(row["tags"]), tuple(row["industries"])) for row in rows]

    async def rebuild(self, preprocessor: Preprocessor, config: SearchConfig) -> int:
        conn = await asyncpg.connect(self.dsn)
        try:
            async with conn.transaction():
                # Serialize writers; DELETE + MVCC lets readers finish on old snapshots.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", "alfa_baseline:" + self.schema)
                migration = Path(__file__).parent.parent.joinpath("migrations/001_initial.sql").read_text()
                migration = self.sql(migration).replace("SCHEMA IF NOT EXISTS search;", f' SCHEMA IF NOT EXISTS "{self.schema}";')
                await conn.execute(migration)
                materials = await self.load_materials(conn, config)
                prepared = await asyncio.to_thread(prepare_index, materials, preprocessor)
                # Entire replacement, including statistics, commits or rolls back together.
                await conn.execute(self.sql("DELETE FROM search.postings; DELETE FROM search.terms; DELETE FROM search.materials; DELETE FROM search.index_stats;"))
                await conn.copy_records_to_table("materials", schema_name=self.schema,
                    columns=["material_type", "material_id", "field_lengths"], records=prepared.materials)
                await conn.copy_records_to_table("terms", schema_name=self.schema,
                    columns=["term", "document_frequency"], records=sorted(prepared.frequencies.items()))
                term_ids = {row["term"]: row["term_id"] for row in
                            await conn.fetch(self.sql("SELECT term_id,term FROM search.terms"))}
                await conn.copy_records_to_table("postings", schema_name=self.schema,
                    columns=["term_id", "material_type", "material_id", "field", "tf"],
                    records=((term_ids[t], kind, id, field, tf) for t, kind, id, field, tf in prepared.postings))
                await conn.execute(self.sql('''INSERT INTO search.index_stats
                    (document_count,avg_lengths,normalization_version,source_policy)
                    VALUES ($1,$2,$3,$4)'''), len(materials), prepared.averages,
                    preprocessor.signature, self.policy(config))
            logger.info("Indexed %d materials in %s", len(materials), self.schema)
            return len(materials)
        except Exception:
            logger.exception("Index rebuild failed; transaction rolled back")
            raise
        finally:
            await conn.close()

    async def search(self, terms: list[str], top_k: int, config: SearchConfig,
                     normalization_version: str) -> list[SearchResult]:
        conn = await asyncpg.connect(self.dsn)
        try:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                try:
                    stats = await conn.fetchrow(self.sql("SELECT normalization_version,source_policy FROM search.index_stats"))
                except (asyncpg.UndefinedTableError, asyncpg.InvalidSchemaNameError) as exc:
                    raise IndexNotReadyError("Run await baseline.build_index() first") from exc
                if not stats:
                    raise IndexNotReadyError("Index statistics missing; run build_index()")
                if stats["normalization_version"] != normalization_version or stats["source_policy"] != self.policy(config):
                    raise IndexNotReadyError("Preprocessing or source policy changed; rebuild index")
                rows = await conn.fetch(self.sql(SEARCH_SQL), terms, list(config.weights),
                                        list(config.b), config.k1, top_k)
                return [SearchResult(material_id=r["material_id"], material_type=r["material_type"],
                                     score=float(r["score"])) for r in rows]
        finally:
            await conn.close()
