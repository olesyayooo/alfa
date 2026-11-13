"""Export real top-10 retrievals and a blind, unjudged evaluation pool.

No index builds, ranking changes, LLM calls or relevance metrics.
Run from the project root: python -m examples.prepare_evaluation
"""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg

from alfa_search import BaselineSearch, SemanticSearch
from alfa_search.config import FIELDS, SearchConfig
from alfa_search.models import Material
from alfa_search.text import clean_text
from alfa_search.semantic.text import content_hash, material_text
from alfa_search.catalog import PUBLISHED, catalog_sql

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 10
PART_SIZE = 5
ENGINES = ("bm25f", "semantic")
KEY_PATTERN = r"(?:article|course|document|webinar|podcast):[0-9a-f-]{36}"


def read_queries(path: Path) -> list[dict]:
    queries = json.loads(path.read_text(encoding="utf-8"))
    if len(queries) != 40 or [q["query_id"] for q in queries] != [f"q{i:02}" for i in range(1, 41)]:
        raise ValueError("Expected 40 queries with consecutive unique IDs q01..q40")
    if len({q["query"] for q in queries}) != 40:
        raise ValueError("Query texts must be unique")
    if any(not q["query"].strip() or not q["category"].strip() for q in queries):
        raise ValueError("Query text and category must be non-empty")
    return queries


def material_key(kind: str, id: str | UUID) -> str:
    return f"{kind}:{UUID(str(id))}"


async def index_snapshot(conn: asyncpg.Connection) -> dict:
    """Refuse stale/different catalogs rather than reindexing automatically."""
    async with conn.transaction(readonly=True, isolation="repeatable_read"):
        baseline = dict(await conn.fetchrow("SELECT * FROM search.index_stats"))
        semantic = dict(await conn.fetchrow("SELECT * FROM search.semantic_index_state"))
        baseline_keys = {material_key(r["material_type"], r["material_id"]) for r in
                         await conn.fetch("SELECT material_type,material_id FROM search.materials")}
        semantic_keys = {material_key(r["material_type"], r["material_id"]) for r in
                         await conn.fetch("SELECT material_type,material_id FROM search.material_embeddings")}
        policy = SearchConfig()
        source_keys = {material_key(r["type"], r["id"]) for r in await conn.fetch(
            "SELECT type,id FROM (" + catalog_sql(policy.exclude_closed, policy.public_articles_only) + ") catalog",
            PUBLISHED)}
        if baseline_keys != semantic_keys or baseline_keys != source_keys:
            raise RuntimeError("Catalog/index membership differs. Refresh indexes separately before exporting")
        if len(source_keys) != baseline["document_count"] or len(source_keys) != semantic["material_count"]:
            raise RuntimeError("Index material counts are inconsistent")
        if json.loads(baseline["source_policy"]) != json.loads(semantic["source_policy"]):
            raise RuntimeError("Search source policies differ")
        return {"baseline": baseline, "semantic": semantic, "catalog_material_count": len(source_keys)}


def labels(values: list[str]) -> list[str]:
    return sorted({text for value in values for text in [clean_text(value)] if text})


async def fetch_materials(conn: asyncpg.Connection, keys: set[tuple[str, UUID]]) -> dict:
    """Fetch only retrieved materials; retain complete descriptions, including raw HTML."""
    policy = SearchConfig()
    ordered = sorted(keys)
    sql = "SELECT c.* FROM (" + catalog_sql(policy.exclude_closed, policy.public_articles_only) + ''') c
        JOIN unnest($2::text[],$3::uuid[]) wanted(kind,material_id)
        ON c.type=wanted.kind AND c.id=wanted.material_id'''
    async with conn.transaction(readonly=True, isolation="repeatable_read"):
        rows = await conn.fetch(sql, PUBLISHED, [k[0] for k in ordered], [k[1] for k in ordered])
        hashes = {(r["material_type"], r["material_id"]): r["content_hash"] for r in await conn.fetch('''
            SELECT e.material_type,e.material_id,e.content_hash FROM search.material_embeddings e
            JOIN unnest($1::text[],$2::uuid[]) wanted(kind,id)
            ON e.material_type=wanted.kind AND e.material_id=wanted.id''',
            [k[0] for k in ordered], [k[1] for k in ordered])}
    materials = {}
    for row in rows:
        material = Material(row["id"], row["type"], row["title"], row["description"],
                            tuple(row["topics"]), tuple(row["tags"]), tuple(row["industries"]))
        key = (material.type, material.id)
        if content_hash(material_text(material)) != hashes.get(key):
            raise RuntimeError(f"Source text changed since semantic indexing: {material_key(*key)}")
        if key in materials:
            raise RuntimeError(f"Duplicate source material: {key}")
        materials[key] = {
            "material_id": str(material.id), "material_type": material.type,
            "material_key": material_key(*key),
            "title": clean_text(material.title) if material.title is not None else None,
            "description": clean_text(material.description) if material.description is not None else None,
            "description_raw": material.description,
            "topics": labels(row["topics"]), "tags": labels(row["tags"]),
            "industries": labels(row["industries"]),
        }
    if set(materials) != keys:
        raise RuntimeError("Some retrieved materials are absent from the current eligible source catalog")
    return materials


def blind_pool(query: dict) -> list[dict]:
    unique = {}
    for engine in ENGINES:
        for result in query["results"][engine]:
            unique[result["material_key"]] = result
    # Fixed pseudo-random order, independent of rank, score and algorithm iteration.
    return sorted(unique.values(), key=lambda r: hashlib.sha256(
        f'{query["query_id"]}|{r["material_key"]}'.encode()).hexdigest())


def escape_md(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", value)


def description_excerpt(description: str | None, limit: int) -> tuple[str, bool]:
    if description is None or description == "":
        return "[Описание отсутствует в БД.]", False
    if len(description) <= limit:
        return escape_md(description), False
    short = description[:limit]
    last_space = short.rfind(" ")
    if last_space >= limit * 0.8:
        short = short[:last_space]
    return (escape_md(short) + f"…\n\n[Описание обрезано: показано {len(short)} из {len(description)} символов. "
            "Полная версия сохранена в retrieval_results.json.]"), True


def render_blind(queries: list[dict], limit: int, heading: str) -> str:
    lines = [f"# {heading}", "", "Оцени каждый материал относительно запроса по отдельному промпту.",
             "Порядок карточек технический и не отражает релевантность. Идентификатор карточки: тип:UUID.",
             "Отсутствующие поля явно отмечены; длинные описания могут быть обрезаны с пометкой.", ""]
    for query in queries:
        pool = blind_pool(query)
        lines.extend([f'## Запрос `{query["query_id"]}`', "", f'Текст: {escape_md(query["query"])}',
                      f'Категория: {escape_md(query["category"])}',
                      f'Тип запроса: {escape_md(query["query_type"])}',
                      f"Материалов для оценки: {len(pool)}", ""])
        for material in pool:
            description, _ = description_excerpt(material["description"], limit)
            lines.extend([f'### Материал `{material["material_key"]}`', "",
                          f'ID: `{material["material_id"]}`', f'Тип: {material["material_type"]}',
                          f'Название: {escape_md(material["title"]) if material["title"] else "[Не указано в БД.]"}',
                          "", "Описание: " + description, ""])
            for field, title in (("topics", "Темы"), ("tags", "Теги"), ("industries", "Отрасли")):
                lines.append(f'{title}: {escape_md("; ".join(material[field])) if material[field] else "[Не указаны в БД.]"}')
            lines.append("")
    return "\n".join(lines)


def markdown_ids(text: str) -> dict[str, list[str]]:
    found = {}
    query_id = None
    for line in text.splitlines():
        query = re.fullmatch(r"## Запрос `(q\d{2})`", line)
        if query:
            query_id = query[1]
            if query_id in found:
                raise ValueError("Repeated query in blind Markdown")
            found[query_id] = []
        material = re.fullmatch(rf"### Материал `({KEY_PATTERN})`", line)
        if material:
            if query_id is None:
                raise ValueError("Material appears before query")
            found[query_id].append(material[1])
    return found


def validate_export(directory: Path) -> dict:
    queries = read_queries(directory / "queries.json")
    payload = json.loads((directory / "retrieval_results.json").read_text(encoding="utf-8"))
    if [q["query_id"] for q in payload["queries"]] != [q["query_id"] for q in queries]:
        raise ValueError("Queries and retrieval IDs differ")
    expected = {}
    retrieved = 0
    for definition, query in zip(queries, payload["queries"], strict=True):
        if any(definition[field] != query[field] for field in definition):
            raise ValueError("Query text or category mismatch")
        if set(query["results"]) != set(ENGINES):
            raise ValueError("Expected both retrieval algorithms")
        for engine in ENGINES:
            results = query["results"][engine]
            if len(results) > TOP_K or [r["rank"] for r in results] != list(range(1, len(results) + 1)):
                raise ValueError("Expected original ranks, with at most ten results")
            if len({r["material_key"] for r in results}) != len(results):
                raise ValueError("Duplicate material within retrieval")
            scores = [r["score"] for r in results]
            if not all(math.isfinite(s) for s in scores) or scores != sorted(scores, reverse=True):
                raise ValueError("Invalid or reordered scores")
            for result in results:
                if result["material_key"] != material_key(result["material_type"], result["material_id"]):
                    raise ValueError("Composite key mismatch")
                if "relevance" in result:
                    raise ValueError("Export must not contain relevance judgments")
                for field in ("title", "description", "description_raw", "topics", "tags", "industries"):
                    if field not in result:
                        raise ValueError(f"Missing material field: {field}")
                raw = result["description_raw"]
                if result["description"] != (clean_text(raw) if raw is not None else None):
                    raise ValueError("JSON description has been truncated or altered")
            retrieved += len(results)
        pool = blind_pool(query)
        if not 1 <= len(pool) <= 2 * TOP_K:
            raise ValueError("Invalid blind pool size")
        expected[query["query_id"]] = [m["material_key"] for m in pool]
        # Shared materials must carry identical metadata in both retrievals.
        by_key = {}
        for engine in ENGINES:
            for result in query["results"][engine]:
                metadata = {k: v for k, v in result.items() if k not in {"rank", "score"}}
                previous = by_key.setdefault(result["material_key"], metadata)
                if previous != metadata:
                    raise ValueError("Shared material metadata differs")
    main_text = (directory / "deepseek_evaluation.md").read_text(encoding="utf-8")
    limit = payload["metadata"]["markdown_description_limit"]
    expected_main = render_blind(payload["queries"], limit, "Материалы для слепой оценки — все запросы")
    if main_text != expected_main or markdown_ids(main_text) != expected:
        raise ValueError("Full Markdown content or IDs differ from JSON")
    part_files = sorted(directory.glob("deepseek_evaluation_part_*.md"))
    if len(part_files) != 8:
        raise ValueError("Expected eight parts with five queries each")
    combined = {}
    for number, path in enumerate(part_files, 1):
        text = path.read_text(encoding="utf-8")
        part_queries = payload["queries"][(number - 1) * PART_SIZE:number * PART_SIZE]
        if text != render_blind(part_queries, limit, f"Материалы для слепой оценки — часть {number} из 8"):
            raise ValueError("Blind part contains unexpected text")
        mapping = markdown_ids(text)
        if len(mapping) != PART_SIZE or set(combined) & set(mapping):
            raise ValueError("Queries split or duplicated between parts")
        combined.update(mapping)
        if re.search(r"BM25F|BGE-M3|SemanticSearch|\bscore\s*[:=]|\brank\s*[:=]", text):
            raise ValueError("Blind Markdown leaks retrieval information")
    if combined != expected:
        raise ValueError("Blind parts and full Markdown contain different IDs")
    with (directory / "evaluation_template.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["query_id", "material_type", "material_id", "relevance"]:
            raise ValueError("Invalid CSV columns")
        rows = list(reader)
    pairs = [(r["query_id"], material_key(r["material_type"], r["material_id"])) for r in rows]
    expected_pairs = {(query_id, key) for query_id, keys in expected.items() for key in keys}
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs or any(r["relevance"] != "" for r in rows):
        raise ValueError("CSV IDs differ or relevance has been filled")
    short = [{"query_id": q["query_id"], "algorithm": engine, "returned": len(q["results"][engine]), "requested": TOP_K}
             for q in payload["queries"] for engine in ENGINES if len(q["results"][engine]) < TOP_K]
    return {"queries": len(queries), "retrieval_rows": retrieved, "unique_query_material_pairs": len(pairs),
            "distinct_materials": len({key for _, key in pairs}), "blind_parts": len(part_files),
            "min_materials_per_query": min(map(len, expected.values())),
            "max_materials_per_query": max(map(len, expected.values())),
            "truncated_markdown_descriptions": main_text.count("[Описание обрезано:"), "short_retrievals": short}


def write_exports(directory: Path, payload: dict, query_file: Path, prompt_file: Path) -> dict:
    shutil.copyfile(query_file, directory / "queries.json")
    shutil.copyfile(prompt_file, directory / "deepseek_prompt.md")
    (directory / "retrieval_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    queries = payload["queries"]
    limit = payload["metadata"]["markdown_description_limit"]
    (directory / "deepseek_evaluation.md").write_text(
        render_blind(queries, limit, "Материалы для слепой оценки — все запросы"), encoding="utf-8")
    for part in range(8):
        (directory / f"deepseek_evaluation_part_{part + 1:02}.md").write_text(
            render_blind(queries[part * PART_SIZE:(part + 1) * PART_SIZE], limit,
                         f"Материалы для слепой оценки — часть {part + 1} из 8"), encoding="utf-8")
    with (directory / "evaluation_template.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "material_type", "material_id", "relevance"])
        for query in queries:
            for material in blind_pool(query):
                writer.writerow([query["query_id"], material["material_type"], material["material_id"], ""])
    stats = validate_export(directory)
    part_links = "\n".join(
        f'- [Часть {part + 1}: q{part * PART_SIZE + 1:02}–q{(part + 1) * PART_SIZE:02}](deepseek_evaluation_part_{part + 1:02}.md)'
        for part in range(8))
    short = "\n".join(f'- {r["query_id"]}: {r["algorithm"]} вернул {r["returned"]} из запрошенных {r["requested"]}.'
                      for r in stats["short_retrievals"]) or "Все выдачи содержат по десять материалов."
    readme = f'''# Данные для внешней оценки релевантности

Подготовлены 40 запросов, по которым выполнены BM25F и Semantic Search с top_k=10.
Алгоритмы, веса, модель и параметры ранжирования не менялись. Индексы не перестраивались.
Качество не оценивалось, метрики не рассчитывались, внешние LLM не вызывались.

## Что отправлять DeepSeek

Скопируйте [deepseek_prompt.md](deepseek_prompt.md) и приложите **одну** часть
со слепой подборкой материалов. Повторите для остальных частей. Удобнее использовать
отдельный диалог для каждой части и сохранять JSON-ответы отдельно от шаблона.

{part_links}

Не прикладывайте retrieval_results.json: он содержит названия алгоритмов, позиции
и исходные score. Порядок карточек в слепых частях определяется хешем query_id
и составного ID материала, а не поисковым ранжированием.

## Файлы

- queries.json — 40 запросов с ID, тематической категорией и типом формулировки.
- retrieval_results.json — исходные позиции, score и полные данные каждого результата.
- deepseek_evaluation.md — полная слепая подборка всех 40 запросов; для отправки удобнее части.
- deepseek_evaluation_part_01.md … part_08.md — по пять запросов без разделения запроса между файлами.
- deepseek_prompt.md — шкала 0–3 и требования к строгому JSON-массиву ответов.
- evaluation_template.csv — одна строка на уникальную пару запрос–материал; relevance пустое.

ID материалов — UUID-строки. Составной ключ: material_type:material_id.
В JSON description содержит полный очищенный текст без обрезания, description_raw —
полное исходное поле PostgreSQL, включая HTML. NULL не подменяется вымышленным текстом.
В Markdown описания ограничены {limit} символами; каждое сокращение явно отмечено.
Это сокращение только файла для оценки и не меняет входы или параметры поиска.

## Проверка целостности

- Запросов: {stats["queries"]}.
- Строк в исходных выдачах: {stats["retrieval_rows"]}.
- Уникальных пар запрос–материал: {stats["unique_query_material_pairs"]}.
- Различных материалов во всех запросах: {stats["distinct_materials"]}.
- Материалов на запрос после объединения: от {stats["min_materials_per_query"]} до {stats["max_materials_per_query"]}.
- Сокращённых описаний в полной Markdown-подборке: {stats["truncated_markdown_descriptions"]}.

Проверены совпадение ID и содержимого JSON/Markdown, отсутствие дублей, полнота
CSV и слепых частей, последовательность исходных позиций и отсутствие оценок релевантности.

top_k — верхняя граница количества результатов. Короткие выдачи сохранены как есть:

{short}

## Повторный запуск из корня проекта

Задайте DATABASE_URL своим подключением, затем:

```sh
export SEMANTIC_MODEL_CACHE="$PWD/.model-cache"
export SEMANTIC_DEVICE=cpu
export SEMANTIC_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
.venv/bin/python -m examples.prepare_evaluation
```

Выгрузка перезаписывает результаты и пустой CSV-шаблон. Ответы DeepSeek храните отдельно.
Проверить готовые файлы без подключения к БД и без загрузки модели:

```sh
.venv/bin/python -m examples.prepare_evaluation --validate-only
```
'''
    (directory / "README.md").write_text(readme, encoding="utf-8")
    return stats


async def run(output: Path, query_file: Path, limit: int) -> None:
    queries = read_queries(query_file)
    baseline, semantic = BaselineSearch(), SemanticSearch()
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        before = await index_snapshot(conn)
        retrieved_queries = []
        keys = set()
        for query in queries:
            results = {}
            for label, engine in (("bm25f", baseline), ("semantic", semantic)):
                results[label] = await engine.search(query["query"], top_k=TOP_K)
                if len(results[label]) > TOP_K:
                    raise RuntimeError(f'{query["query_id"]}: {label} exceeded top_k')
                keys.update((r["material_type"], r["material_id"]) for r in results[label])
            retrieved_queries.append({**query, "results": results})
            print(f'{query["query_id"]}/q40: BM25F={len(results["bm25f"])} Semantic={len(results["semantic"])}', flush=True)
        materials = await fetch_materials(conn, keys)
        after = await index_snapshot(conn)
        if before != after:
            raise RuntimeError("Indexes changed during retrieval; rerun export on stable indexes")
    finally:
        await conn.close()
    for query in retrieved_queries:
        for engine in ENGINES:
            query["results"][engine] = [
                {"rank": rank, **materials[(r["material_type"], r["material_id"])], "score": r["score"]}
                for rank, r in enumerate(query["results"][engine], 1)]
    metadata = {
        "schema_version": 1, "generated_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "top_k": TOP_K, "material_id_type": "uuid", "catalog_material_count": before["catalog_material_count"],
        "source_policy": json.loads(before["baseline"]["source_policy"]),
        "bm25f": {"weights": dict(zip(FIELDS, baseline.config.weights)),
                  "b": dict(zip(FIELDS, baseline.config.b)), "k1": baseline.config.k1,
                  "normalization_version": before["baseline"]["normalization_version"],
                  "indexed_at": before["baseline"]["indexed_at"].isoformat()},
        "semantic": {"model_name": semantic.config.model_name, "model_revision": semantic.config.model_revision,
                     "encoder_signature": semantic.config.signature, "max_tokens": semantic.config.max_tokens,
                     "indexed_at": before["semantic"]["updated_at"].isoformat()},
        "descriptions": "description: complete plain text; description_raw: complete original DB field, including HTML; NULL preserved",
        "markdown_description_limit": limit, "blind_order": "sha256(query_id|material_type:material_id)",
    }
    payload = {"metadata": metadata, "queries": retrieved_queries}
    output.mkdir(parents=True, exist_ok=True)
    # Publish only after all retrievals, metadata checks and file validation succeed.
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as folder:
        staging = Path(folder)
        stats = write_exports(staging, payload, query_file, ROOT / "evaluation/deepseek_prompt.md")
        for path in staging.iterdir():
            path.replace(output / path.name)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation")
    parser.add_argument("--queries-file", type=Path, default=ROOT / "evaluation/queries.json")
    parser.add_argument("--markdown-description-limit", type=int, default=2000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.markdown_description_limit < 1:
        parser.error("description limit must be positive")
    if args.validate_only:
        print(json.dumps(validate_export(args.output_dir), ensure_ascii=False, indent=2))
    else:
        asyncio.run(run(args.output_dir, args.queries_file, args.markdown_description_limit))


if __name__ == "__main__":
    main()
