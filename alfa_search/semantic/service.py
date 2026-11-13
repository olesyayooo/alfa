"""Public independent semantic service; no FastAPI or HTTP dependencies."""

import asyncio
import logging
import os

from ..config import SearchConfig
from ..models import SearchResult
from ..text import clean_text
from .config import SemanticConfig
from .encoder import BGEEncoder, Embedding, Encoder, normalized_vector
from .repository import SemanticRepository
from .source import CatalogSource

logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(self, dsn: str | None = None, config: SemanticConfig | None = None,
                 source_policy: SearchConfig | None = None, *, schema: str = "search",
                 encoder: Encoder | None = None) -> None:
        connection = dsn or os.environ.get("DATABASE_URL")
        if not connection:
            raise ValueError("Set DATABASE_URL or supply dsn")
        self.config = config or SemanticConfig()
        self.source_policy = source_policy or SearchConfig()
        self._encoder = encoder or BGEEncoder(self.config)
        if self._encoder.signature != self.config.signature:
            raise ValueError("Encoder signature must match model and text configuration")
        self._repository = SemanticRepository(connection, schema)
        self._source = CatalogSource()
        self._encode_lock = asyncio.Lock()

    async def _encode(self, texts: list[str]) -> list[Embedding]:
        async with self._encode_lock:
            embeddings = await asyncio.to_thread(self._encoder.encode, texts)
        if len(embeddings) != len(texts):
            raise ValueError("Encoder returned an unexpected number of vectors")
        return [Embedding(normalized_vector(e.vector), e.original_tokens, e.used_tokens) for e in embeddings]

    async def migrate(self) -> None:
        """Enable pgvector and create semantic tables without downloading a model."""
        await self._repository.migrate()

    async def build_index(self) -> None:
        """Batch-index new/changed materials; unchanged vectors are reused."""
        await self._repository.refresh(self._source, self._encode, self.config, self.source_policy)

    async def update_index(self) -> None:
        """Update additions/changes, prune unavailable materials and handle model changes."""
        await self.build_index()

    async def search(self, query: str, top_k: int = 50) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if not query.strip() or top_k <= 0:
            return []
        text = clean_text(query)
        if not text:
            return []
        try:
            return await self._repository.search(text, top_k, self._encode, self.config, self.source_policy)
        except Exception:
            logger.exception("Semantic search failed")
            raise
