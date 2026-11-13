"""Streaming source adapter using exactly the same catalog SQL as BM25F."""

from collections.abc import AsyncIterator

import asyncpg

from ..config import SearchConfig
from ..models import Material
from ..catalog import PUBLISHED, catalog_sql


class CatalogSource:
    async def batches(self, conn: asyncpg.Connection, policy: SearchConfig,
                      batch_size: int) -> AsyncIterator[list[Material]]:
        batch: list[Material] = []
        async for row in conn.cursor(catalog_sql(policy.exclude_closed, policy.public_articles_only),
                                     PUBLISHED, prefetch=batch_size):
            batch.append(Material(row["id"], row["type"], row["title"], row["description"],
                                  tuple(row["topics"]), tuple(row["tags"]), tuple(row["industries"])))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
