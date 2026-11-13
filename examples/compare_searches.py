"""Run both real indexes on a fixed set of ten user queries."""

import asyncio
import argparse
import os
from pathlib import Path

import asyncpg

from alfa_search import BaselineSearch, SemanticSearch
from alfa_search.config import SearchConfig
from alfa_search.catalog import PUBLISHED, catalog_sql

QUERIES = [
    "Маркетинг",
    "Налоги ИП",
    "ИИ для бизнеса",
    "Как управлять командой?",
    "Хочу открыть кофейню. Как составить бизнес-план и понять, сколько денег понадобится на запуск?",
    "Продаю товары на маркетплейсах и путаюсь в налогах. Как выбрать налоговый режим и правильно вести учёт?",
    "Клиенты покупают один раз и больше не возвращаются. Как повысить лояльность и увеличить повторные продажи?",
    "Я постоянно занят операционными задачами. Хочу делегировать работу сотрудникам и освободить время для развития бизнеса.",
    "Нужен шаблон трудового договора с сотрудником",
    "Как использовать нейросети, чтобы быстрее создавать рекламные тексты и контент для социальных сетей?",
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["both", "baseline", "semantic"], default="both")
    parser.add_argument("--output", default="search_comparison.md")
    args = parser.parse_args()
    baseline, semantic = BaselineSearch(), SemanticSearch()
    engines = [("BM25F", baseline), ("Semantic Search (BGE-M3)", semantic)]
    if args.engine == "baseline":
        engines = engines[:1]
    elif args.engine == "semantic":
        engines = engines[1:]
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        policy = SearchConfig()
        rows = await conn.fetch(catalog_sql(policy.exclude_closed, policy.public_articles_only), PUBLISHED)
        titles = {(r["type"], r["id"]): " ".join(r["title"].split()) for r in rows}
    finally:
        await conn.close()
    report = ["# Сравнение BM25F и BGE-M3", "", "Реальная выдача top-5 без ручной перестановки результатов. Шкалы score различаются.",
              f"Каталог: {len(titles)} материалов. Модель: {semantic.config.model_name}@{semantic.config.model_revision}. Лимит: {semantic.config.max_tokens} токенов.", ""]
    for number, query in enumerate(QUERIES, 1):
        report.extend([f"## {number}. {query}", ""])
        for label, engine in engines:
            results = await engine.search(query, top_k=5)
            if len(results) != 5:
                raise RuntimeError(f"Expected five results for {label}: {query}")
            report.extend([f"### {label}", ""])
            for rank, result in enumerate(results, 1):
                kind, material_id = result["material_type"], result["material_id"]
                title = titles.get((kind, material_id), "[материал отсутствует в текущем каталоге]")
                report.append(f'{rank}. {title} — {kind}, score={result["score"]:.6f}; id={material_id}')
            report.append("")
            print(f"{number}/10 {label}: {len(results)} results", flush=True)
        Path(args.output).write_text("\n".join(report), encoding="utf-8")
    print("Saved " + args.output, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
