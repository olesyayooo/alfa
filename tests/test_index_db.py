"""Controlled integration tests; only temporary search schemas are written."""

import asyncio
import math
import os
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import asyncpg

from alfa_search import BaselineSearch, SearchConfig
from alfa_search.models import Material
from alfa_search.baseline.repository import IndexNotReadyError

DSN = os.environ.get("TEST_DATABASE_URL")


def item(number: int, title: str = "", description: str = "", **kwargs) -> Material:
    return Material(UUID(int=number), kwargs.pop("type", "article"), title, description, **kwargs)


@unittest.skipUnless(DSN, "Set TEST_DATABASE_URL to run isolated PostgreSQL tests")
class IndexDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.schema = "search_test_" + uuid4().hex
        self.baseline = BaselineSearch(DSN, schema=self.schema)
        self.source = AsyncMock(return_value=[])
        self.baseline._repository.load_materials = self.source

    async def asyncTearDown(self) -> None:
        conn = await asyncpg.connect(DSN)
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        finally:
            await conn.close()

    async def build(self, materials: list[Material]) -> None:
        self.source.return_value = materials
        await self.baseline.build_index()

    async def snapshot(self) -> dict:
        conn = await asyncpg.connect(DSN)
        try:
            return {table: [dict(r) for r in await conn.fetch(
                f'SELECT * FROM "{self.schema}".{table} ORDER BY 1,2')]
                for table in ("materials", "terms", "postings", "index_stats")}
        finally:
            await conn.close()

    async def test_exact_bm25f(self) -> None:
        await self.build([item(1, "налоги налоги", "налоги отчет", topics=("налог",)),
                          item(2, "отчет", "налог")])
        rows = await self.baseline.search("налоги")
        self.assertEqual(len(rows), 2)
        # Hand calculation: N=2, df=2; avg title=1.5, desc=1.5, topics=.5.
        tf_a = 3 * 2 / (0.25 + 0.75 * 2 / 1.5) + 1 / (0.25 + 0.75 * 2 / 1.5) + 2 / (0.25 + 0.75 / 0.5)
        tf_b = 1 / (0.25 + 0.75 / 1.5)
        idf = math.log(1 + 0.5 / 2.5)
        expected = {UUID(int=1): idf * 2.2 * tf_a / (1.2 + tf_a),
                    UUID(int=2): idf * 2.2 * tf_b / (1.2 + tf_b)}
        for row in rows:
            self.assertAlmostEqual(row["score"], expected[row["material_id"]], places=12)

    async def test_all_fields_and_runtime_weights(self) -> None:
        self.baseline.config = SearchConfig(b=(0,) * 5)
        await self.build([item(1, "налог"), item(2, description="налог"),
                          item(3, topics=("налог",)), item(4, tags=("налог",)),
                          item(5, industries=("налог",))])
        rows = await self.baseline.search("налог")
        self.assertEqual([r["material_id"].int for r in rows], [1, 3, 4, 5, 2])
        self.baseline.config = SearchConfig(weights=(1, 8, 2, 2, 1.5), b=(0,) * 5)
        self.source.reset_mock()
        self.assertEqual((await self.baseline.search("налог"))[0]["material_id"].int, 2)
        self.source.assert_not_awaited()

    async def test_tf_saturation(self) -> None:
        self.baseline.config = SearchConfig(b=(0,) * 5)
        await self.build([item(1, "налог"), item(2, "налог налог налог")])
        rows = await self.baseline.search("налог")
        self.assertEqual(rows[0]["material_id"].int, 2)
        self.assertGreater(rows[0]["score"], rows[1]["score"])
        self.assertLess(rows[0]["score"], 3 * rows[1]["score"])

    async def test_rare_words_score_higher(self) -> None:
        self.baseline.config = SearchConfig(b=(0,) * 5)
        await self.build([item(1, "бизнес нейросеть"), item(2, "бизнес"), item(3, "бизнес")])
        common = (await self.baseline.search("бизнес"))[0]["score"]
        rare = (await self.baseline.search("нейросеть"))[0]["score"]
        self.assertGreater(rare, common)

    async def test_long_description_normalization(self) -> None:
        await self.build([item(1, description="налог"),
                          item(2, description="налог " + "маркетинг " * 100)])
        rows = await self.baseline.search("налог")
        self.assertEqual(rows[0]["material_id"].int, 1)
        self.assertGreater(rows[0]["score"], rows[1]["score"])
        self.baseline.config = SearchConfig(b=(0,) * 5)
        rows = await self.baseline.search("налог")
        self.assertEqual(rows[0]["score"], rows[1]["score"])

    async def test_same_id_types_limits_and_empty(self) -> None:
        await self.build([item(1, "маркетинг"), item(1, "маркетинг", type="course")])
        rows = await self.baseline.search("маркетинг маркетинг")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["material_type"] for r in rows], ["article", "course"])
        self.assertEqual(rows, await self.baseline.search("маркетинг"))
        self.assertEqual(len(await self.baseline.search("маркетинг", 1)), 1)
        for query in ("", "  ", "!!!", "и для", "абракадабра"):
            self.assertEqual(await self.baseline.search(query), [])
        self.assertEqual(await self.baseline.search("маркетинг", 0), [])

    async def test_idempotent_update_and_delete(self) -> None:
        await self.build([item(1, "налог"), item(2, "маркетинг")])
        before = await self.baseline.search("налог маркетинг")
        await self.baseline.build_index()
        self.assertEqual(await self.baseline.search("налог маркетинг"), before)
        self.source.return_value = [item(1, "ИИ")]
        await self.baseline.update_index()
        self.assertEqual(await self.baseline.search("налог маркетинг"), [])
        self.assertEqual((await self.baseline.search("ИИ"))[0]["material_id"].int, 1)
        state = await self.snapshot()
        self.assertEqual(state["index_stats"][0]["document_count"], 1)
        self.assertEqual(state["terms"][0]["term"], "ии")
        self.assertEqual(state["terms"][0]["document_frequency"], 1)
        self.source.return_value = []
        await self.baseline.update_index()
        self.assertEqual(await self.baseline.search("ИИ"), [])
        state = await self.snapshot()
        self.assertEqual(state["index_stats"][0]["document_count"], 0)
        self.assertEqual(state["postings"], [])

    async def test_rollback_after_partial_replacement(self) -> None:
        await self.build([item(1, "налог")])
        before = await self.snapshot()
        # Invalid type fails COPY after old rows have been deleted in the transaction.
        self.source.return_value = [item(2, "ИИ", type="invalid")]
        with self.assertLogs("alfa_search.baseline.repository", level="ERROR"):
            with self.assertRaises(asyncpg.CheckViolationError):
                await self.baseline.update_index()
        self.assertEqual(await self.snapshot(), before)
        self.assertTrue(await self.baseline.search("налог"))

    async def test_index_version_and_policy_checks(self) -> None:
        with self.assertLogs("alfa_search.baseline.service", level="ERROR"):
            with self.assertRaises(IndexNotReadyError):
                await self.baseline.search("налог")
        await self.build([item(1, "налог")])
        self.baseline.config = SearchConfig(exclude_closed=False)
        with self.assertLogs("alfa_search.baseline.service", level="ERROR"):
            with self.assertRaises(IndexNotReadyError):
                await self.baseline.search("налог")

    async def test_readers_see_complete_snapshot_during_refresh(self) -> None:
        await self.build([item(1, "налог")])
        reached, resume = asyncio.Event(), asyncio.Event()
        original = asyncpg.Connection.copy_records_to_table

        async def paused_copy(conn, table, **kwargs):
            if table == "materials" and kwargs.get("schema_name") == self.schema:
                reached.set()
                await resume.wait()
            return await original(conn, table, **kwargs)

        self.source.return_value = [item(2, "ИИ")]
        with patch.object(asyncpg.Connection, "copy_records_to_table", paused_copy):
            task = asyncio.create_task(self.baseline.update_index())
            try:
                await asyncio.wait_for(reached.wait(), 10)
                # Writer has deleted old index rows but has not committed.
                self.assertTrue(await self.baseline.search("налог"))
                self.assertEqual(await self.baseline.search("ИИ"), [])
            finally:
                resume.set()
                await task
        self.assertEqual(await self.baseline.search("налог"), [])
        self.assertTrue(await self.baseline.search("ИИ"))


if __name__ == "__main__":
    unittest.main()
