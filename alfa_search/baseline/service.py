"""Public async service for future integration with any backend."""

import asyncio
import logging
import os

from ..config import SearchConfig
from ..models import SearchResult
from .preprocessing import Preprocessor
from .repository import SearchRepository

logger = logging.getLogger(__name__)


class BaselineSearch:
    def __init__(self, dsn: str | None = None, config: SearchConfig | None = None,
                 *, schema: str = "search") -> None:
        connection = dsn or os.environ.get("DATABASE_URL")
        if not connection:
            raise ValueError("Set DATABASE_URL or supply dsn")
        self.config = config or SearchConfig()
        self._preprocessor = Preprocessor()
        self._repository = SearchRepository(connection, schema)

    async def build_index(self) -> None:
        """Create search tables and atomically index all eligible source materials."""
        await self._repository.rebuild(self._preprocessor, self.config)

    async def update_index(self) -> None:
        """Atomic full refresh: removes stale materials and recalculates all statistics."""
        await self.build_index()

    async def search(self, query: str, top_k: int = 50) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if not query.strip() or top_k <= 0:
            return []
        terms = sorted(set(await asyncio.to_thread(self._preprocessor.tokens, query)))
        if not terms:
            return []
        try:
            return await self._repository.search(terms, top_k, self.config, self._preprocessor.signature)
        except Exception:
            logger.exception("Baseline search failed")
            raise
