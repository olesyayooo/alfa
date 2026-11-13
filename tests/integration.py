"""Real catalog smoke check. Run after building the working search index."""

import asyncio
import os

import asyncpg

from alfa_search import BaselineSearch
from alfa_search.catalog import SOURCES


async def main() -> None:
    baseline = BaselineSearch()
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        titles = {}
        expected = set()
        for table, kind, _, has_closed in SOURCES:
            conditions = ['NOT "IsDeleted"', '''"StatusId" IN (SELECT "Id" FROM public."Statuses" WHERE "Title"::text='Опубликовано')''']
            if has_closed:
                conditions.append('NOT "IsClosed"')
            if kind == "article":
                conditions.append('"IsPublic"')
            rows = await conn.fetch(f'SELECT "Id","Title"::text AS title FROM public."{table}" WHERE ' + " AND ".join(conditions))
            expected.update((kind, row["Id"]) for row in rows)
            titles.update({(kind, row["Id"]): row["title"] for row in rows})
        actual = {(r["material_type"], r["material_id"]) for r in
                  await conn.fetch("SELECT material_type,material_id FROM search.materials")}
        assert actual == expected, "Published/closed/public source membership mismatch"
        stats = await conn.fetchrow("SELECT document_count FROM search.index_stats")
        assert stats["document_count"] == len(actual)
        wrong_df = await conn.fetchval('''SELECT count(*) FROM search.terms t
            WHERE t.document_frequency <> (SELECT count(DISTINCT (p.material_type,p.material_id))
                FROM search.postings p WHERE p.term_id=t.term_id)''')
        assert wrong_df == 0, "Document frequencies must count materials, not fields"
        wrong_lengths = await conn.fetchval('''SELECT count(*) FROM search.materials m
            CROSS JOIN generate_series(1,5) f
            WHERE m.field_lengths[f] <> COALESCE((SELECT sum(p.tf)
                FROM search.postings p WHERE p.material_type=m.material_type
                AND p.material_id=m.material_id AND p.field=f),0)''')
        assert wrong_lengths == 0, "Stored field lengths must equal total token frequencies"
        for field in range(1, 6):
            correct_avg = await conn.fetchval(f'''SELECT s.avg_lengths[{field}] =
                (SELECT COALESCE(avg(m.field_lengths[{field}]),0)::double precision
                 FROM search.materials m) FROM search.index_stats s''')
            assert correct_avg, "Stored average lengths are inconsistent"
        print("Indexed materials:", len(actual))
        for query in ("маркетинг", "налоги", "ИИ", "управление бизнесом"):
            rows = await baseline.search(query, top_k=30)
            assert rows and len(rows) <= 30
            keys = [(r["material_type"], r["material_id"]) for r in rows]
            assert len(keys) == len(set(keys))
            assert all(key in expected for key in keys)
            assert all(r["score"] > 0 for r in rows)
            assert rows == await baseline.search(query, top_k=30)
            assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
            print("\n" + query)
            for row in rows[:5]:
                key = (row["material_type"], row["material_id"])
                print(f'{row["score"]:.4f} {key[0]}: {titles[key]}')
        assert await baseline.search("") == []
        assert await baseline.search("абракадабранесуществующая") == []
        print("\nReal catalog checks passed")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
