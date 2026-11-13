"""Validate supplied judgments and calculate pooled Precision@10 / NDCG@10.

No model inference, index changes or new relevance judgments.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("bm25f", "semantic")
K = 10


def load_judgments(paths: list[Path]) -> list[dict]:
    combined = []
    seen = set()
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].strip()
            if not text.endswith("```"):
                raise ValueError(f"Unclosed JSON fence: {path}")
            text = text[:-3].strip()
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("Judgments must be a JSON array of query objects")
        for query in records:
            if query["query_id"] in seen:
                raise ValueError(f'Duplicate query: {query["query_id"]}')
            seen.add(query["query_id"])
            combined.append(query)
    return sorted(combined, key=lambda q: q["query_id"])


def key(material: dict) -> tuple[str, str]:
    return material["material_type"], str(UUID(material["material_id"]))


def validate_judgments(retrieval: dict, judgments: list[dict]) -> dict[str, dict]:
    expected_queries = {q["query_id"]: q for q in retrieval["queries"]}
    if len(judgments) != len(expected_queries) or {q["query_id"] for q in judgments} != set(expected_queries):
        raise ValueError("Judgments must cover every query exactly once")
    mapped = {}
    for query in judgments:
        qid = query["query_id"]
        labels = {}
        for judgment in query["judgments"]:
            identity = key(judgment)
            if identity in labels:
                raise ValueError(f"Duplicate judgment: {qid} {identity}")
            grade = judgment["relevance"]
            if type(grade) is not int or grade not in range(4):
                raise ValueError(f"Relevance must be an integer in [0,3]: {qid} {identity}")
            labels[identity] = grade
        wanted = set()
        for engine in ENGINES:
            rows = expected_queries[qid]["results"][engine]
            if len(rows) > K or [r["rank"] for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError(f"Invalid retrieval ranks: {qid} {engine}")
            if len({key(row) for row in rows}) != len(rows):
                raise ValueError(f"Duplicate retrieval material: {qid} {engine}")
            wanted.update(key(row) for row in rows)
        if set(labels) != wanted:
            raise ValueError(f"Judgment pool mismatch for {qid}: missing={wanted-set(labels)}, extra={set(labels)-wanted}")
        mapped[qid] = labels
    return mapped


def dcg(grades: list[int], k: int = K) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(grades[:k], 1))


def precision(grades: list[int], threshold: int = 2, k: int = K) -> float:
    return sum(grade >= threshold for grade in grades[:k]) / k


def calculate(retrieval: dict, labels: dict[str, dict]) -> dict:
    per_query = []
    for query in retrieval["queries"]:
        qid = query["query_id"]
        pool = labels[qid]
        ideal = dcg(sorted(pool.values(), reverse=True))
        row = {field: query[field] for field in ("query_id", "query", "category", "query_type")}
        row["pool_size"] = len(pool)
        row["idcg_at_10"] = ideal
        for engine in ENGINES:
            grades = [pool[key(material)] for material in query["results"][engine]]
            score = dcg(grades)
            row[engine] = {
                "returned": len(grades), "grades_in_rank_order": grades,
                "precision_at_10": precision(grades),
                "strict_precision_at_10": precision(grades, threshold=3),
                "dcg_at_10": score, "ndcg_at_10": score / ideal if ideal > 0 else 0.0,
            }
        row["delta_precision_at_10"] = row["semantic"]["precision_at_10"] - row["bm25f"]["precision_at_10"]
        row["delta_ndcg_at_10"] = row["semantic"]["ndcg_at_10"] - row["bm25f"]["ndcg_at_10"]
        per_query.append(row)

    def summary(rows: list[dict]) -> dict:
        result = {"query_count": len(rows)}
        for engine in ENGINES:
            result[engine] = {metric: mean(row[engine][metric] for row in rows)
                              for metric in ("precision_at_10", "strict_precision_at_10", "ndcg_at_10")}
        for metric in ("precision_at_10", "ndcg_at_10"):
            comparisons = Counter("tie" if math.isclose(row["semantic"][metric], row["bm25f"][metric], abs_tol=1e-12)
                                  else "semantic" if row["semantic"][metric] > row["bm25f"][metric] else "bm25f"
                                  for row in rows)
            base, semantic = result["bm25f"][metric], result["semantic"][metric]
            result["comparison_" + metric] = {
                "semantic_wins": comparisons["semantic"], "bm25f_wins": comparisons["bm25f"],
                "ties": comparisons["tie"], "absolute_delta": semantic - base,
                "relative_delta": (semantic - base) / base if base else None,
            }
        return result

    by_type = defaultdict(list)
    by_category = defaultdict(list)
    for row in per_query:
        by_type[row["query_type"]].append(row)
        by_category[row["category"]].append(row)
    return {
        "methodology": {
            "k": K, "precision_relevance_threshold": 2, "strict_precision_relevance_threshold": 3,
            "precision_denominator": K, "missing_positions": "zero relevance",
            "dcg": "sum((2^relevance - 1) / log2(rank + 1)), ranks start at 1",
            "idcg": "shared union of both top-10 lists, sorted by supplied relevance, first 10",
            "zero_idcg_ndcg": 0.0, "aggregation": "macro mean across queries",
            "judge": "DeepSeek responses supplied by the user, not human ground truth",
            "scope": "pooled top-10; no judgments for the rest of the catalog; no full-catalog recall calculated",
        },
        "judged_pairs": sum(len(pool) for pool in labels.values()),
        "grade_distribution": dict(sorted(Counter(grade for pool in labels.values() for grade in pool.values()).items())),
        "summary": summary(per_query),
        "by_query_type": {name: summary(rows) for name, rows in sorted(by_type.items())},
        "by_category": {name: summary(rows) for name, rows in sorted(by_category.items())},
        "per_query": per_query,
    }


def report_text(metrics: dict) -> str:
    summary = metrics["summary"]
    lines = ["# Результаты сравнения поисков", "",
             f'Оценено {summary["query_count"]} запросов и {metrics["judged_pairs"]} уникальных пар запрос–материал.',
             "Оценки 0–3 предоставлены пользователем из ответов DeepSeek; новые оценки не добавлялись.", "",
             "## Средние результаты", "",
             "| Метрика | BM25F | Semantic Search |", "|---|---:|---:|"]
    for metric, name in (("precision_at_10", "Precision@10, оценки 2–3"),
                         ("strict_precision_at_10", "Precision@10, только оценка 3"),
                         ("ndcg_at_10", "NDCG@10")):
        lines.append(f'| {name} | {summary["bm25f"][metric]:.4f} | {summary["semantic"][metric]:.4f} |')
    lines.extend(["", "Precision@10 — число результатов с оценкой не ниже 2, делённое на 10.",
                  "NDCG@10 учитывает все оценки 0–3 и сильнее ценит хорошие результаты вверху выдачи.",
                  "Используется gain = 2^relevance − 1 и discount = log2(rank + 1).",
                  "Идеальная выдача общая для обоих алгоритмов: лучшие 10 материалов из объединённого размеченного пула.",
                  "Если IDCG=0, NDCG=0. Средние считаются по запросам, каждый имеет одинаковый вес.", ""])
    for metric, name in (("precision_at_10", "Precision@10"), ("ndcg_at_10", "NDCG@10")):
        comparison = summary["comparison_" + metric]
        lines.append(f'{name}: Semantic лучше на {comparison["semantic_wins"]} запросах, '
                     f'BM25F — на {comparison["bm25f_wins"]}, равенство — на {comparison["ties"]}. '
                     f'Средняя разница Semantic − BM25F: {comparison["absolute_delta"]:+.4f}.')
    lines.extend(["", "## Выводы", "",
                  f'При пороге релевантности 2–3 среднее число полезных материалов на десять позиций: '
                  f'{summary["bm25f"]["precision_at_10"] * K:.1f} у BM25F и '
                  f'{summary["semantic"]["precision_at_10"] * K:.1f} у Semantic Search.',
                  "На этом наборе семантический поиск лучше и по количеству полезных результатов,",
                  "и по их расположению. Преимущество сохраняется при строгом пороге — только оценка 3.",
                  "Это поддерживает выбор текущего Semantic Search для дальнейшей проверки в продукте,",
                  "но не означает, что его выдача всегда лучше или уже достаточно хороша для пользователей.", "",
                  "Примеры различий по NDCG@10:", ""])
    for qid in ("q02", "q12", "q23", "q32", "q01"):
        row = next(row for row in metrics["per_query"] if row["query_id"] == qid)
        lines.append(f'- {qid} «{row["query"]}»: BM25F {row["bm25f"]["ndcg_at_10"]:.3f}, '
                     f'Semantic {row["semantic"]["ndcg_at_10"]:.3f}.')
    lines.extend(["", "Высокий NDCG не равен высокой доле полезных результатов. Например, для q12",
                  "Semantic имеет NDCG около 0.954, но Precision@10 только 0.4: он хорошо расположил",
                  "немногочисленные полезные материалы из размеченного пула. Не следует читать 0.954 как 95.4% релевантности.",
                  "На коротких и точных темах BM25F иногда лучше: например, «Маркетинг», «НДС»,",
                  "«Персональные данные клиентов». Вывод о превосходстве на каждом типе задач не делается.",
                  "", "## По типу запроса", "",
                  "| Тип | Запросов | P@10 BM25F | P@10 Semantic | NDCG BM25F | NDCG Semantic |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, group in metrics["by_query_type"].items():
        lines.append(f'| {name} | {group["query_count"]} | {group["bm25f"]["precision_at_10"]:.3f} | '
                     f'{group["semantic"]["precision_at_10"]:.3f} | {group["bm25f"]["ndcg_at_10"]:.3f} | '
                     f'{group["semantic"]["ndcg_at_10"]:.3f} |')
    lines.extend(["", "## Все запросы", "",
                  "| ID | Запрос | P@10 BM25F | P@10 Semantic | NDCG BM25F | NDCG Semantic |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in metrics["per_query"]:
        text = row["query"].replace("|", "\\|")
        lines.append(f'| {row["query_id"]} | {text} | {row["bm25f"]["precision_at_10"]:.3f} | '
                     f'{row["semantic"]["precision_at_10"]:.3f} | {row["bm25f"]["ndcg_at_10"]:.3f} | '
                     f'{row["semantic"]["ndcg_at_10"]:.3f} |')
    lines.extend(["", "## Короткая выдача", "",
                  "q14 «Ценообразование»: BM25F вернул один материал с оценкой 3. Precision@10 = 1/10 = 0.1,",
                  "а не 1.0. Девять незаполненных позиций дают нулевой вклад. В NDCG первый материал учитывается",
                  "на своей реальной позиции, далее вклад нулевой. Такое правило одинаково для обоих алгоритмов.",
                  "", "## Ограничения", "",
                  "Это сравнение на 40 фиксированных запросах по оценкам одной LLM, а не реальных пользователей.",
                  "39 описаний в Markdown для оценки были сокращены; полные версии сохранены в исходном JSON.",
                  "Нормализация NDCG использует общий пул найденных материалов, а не весь каталог.",
                  "Результаты характеризуют текущие реализации и настройки, не доказывают превосходство",
                  "алгоритма на любых запросах. Доверительные интервалы и статистическая значимость не рассчитывались.",
                  "Время поиска, стоимость вычислений и пользовательские клики здесь не измерялись.", ""])
    return "\n".join(lines)


def save_outputs(output: Path, judgments: list[dict], metrics: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "judgments.json").write_text(json.dumps(judgments, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics_report.md").write_text(report_text(metrics), encoding="utf-8")
    with (output / "judgments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "material_type", "material_id", "relevance"])
        for query in judgments:
            for item in query["judgments"]:
                writer.writerow([query["query_id"], item["material_type"], item["material_id"], item["relevance"]])
    with (output / "metrics_per_query.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "query", "category", "query_type", "bm25f_returned", "semantic_returned",
                         "bm25f_precision_at_10", "semantic_precision_at_10", "bm25f_ndcg_at_10",
                         "semantic_ndcg_at_10", "delta_precision_at_10", "delta_ndcg_at_10"])
        for row in metrics["per_query"]:
            writer.writerow([row["query_id"], row["query"], row["category"], row["query_type"],
                row["bm25f"]["returned"], row["semantic"]["returned"],
                row["bm25f"]["precision_at_10"], row["semantic"]["precision_at_10"],
                row["bm25f"]["ndcg_at_10"], row["semantic"]["ndcg_at_10"],
                row["delta_precision_at_10"], row["delta_ndcg_at_10"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, nargs="+", default=[ROOT / "evaluation/judgments.json"])
    parser.add_argument("--results", type=Path, default=ROOT / "evaluation/retrieval_results.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation")
    args = parser.parse_args()
    retrieval = json.loads(args.results.read_text(encoding="utf-8"))
    judgments = load_judgments(args.judgments)
    labels = validate_judgments(retrieval, judgments)
    metrics = calculate(retrieval, labels)
    save_outputs(args.output_dir, judgments, metrics)
    print(json.dumps({"judged_pairs": metrics["judged_pairs"], "summary": metrics["summary"],
                      "by_query_type": metrics["by_query_type"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
