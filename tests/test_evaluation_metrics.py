import math

import pytest

from examples.evaluate_searches import calculate, dcg, precision, validate_judgments


def test_dcg_by_hand_and_short_precision():
    assert dcg([3, 2, 1]) == pytest.approx(7 + 3 / math.log2(3) + 1 / 2)
    assert precision([3]) == 0.1
    assert precision([3, 2, 1, 0]) == 0.2
    assert precision([3, 2, 1, 0], threshold=3) == 0.1
    assert dcg([]) == 0


def fixture():
    first = {"material_type": "article", "material_id": "00000000-0000-0000-0000-000000000001"}
    second = {**first, "material_type": "course"}
    retrieval = {"queries": [{"query_id": "q01", "query": "test", "category": "test", "query_type": "short",
                 "results": {"bm25f": [{**first, "rank": 1}],
                             "semantic": [{**second, "rank": 1}, {**first, "rank": 2}]}}]}
    judgments = [{"query_id": "q01", "judgments": [{**first, "relevance": 2}, {**second, "relevance": 3}]}]
    return retrieval, judgments


def test_shared_ideal_and_composite_identity():
    retrieval, judgments = fixture()
    labels = validate_judgments(retrieval, judgments)
    metrics = calculate(retrieval, labels)
    row = metrics["per_query"][0]
    ideal = 7 + 3 / math.log2(3)
    assert row["idcg_at_10"] == pytest.approx(ideal)
    assert row["bm25f"]["ndcg_at_10"] == pytest.approx(3 / ideal)
    assert row["semantic"]["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["judged_pairs"] == 2


def test_reject_missing_duplicate_and_invalid_grade():
    retrieval, judgments = fixture()
    with pytest.raises(ValueError):
        validate_judgments(retrieval, [{"query_id": "q01", "judgments": judgments[0]["judgments"][:1]}])
    with pytest.raises(ValueError):
        validate_judgments(retrieval, [{"query_id": "q01", "judgments": judgments[0]["judgments"] * 2}])
    judgments[0]["judgments"][0]["relevance"] = True
    with pytest.raises(ValueError):
        validate_judgments(retrieval, judgments)


def test_zero_ideal_is_zero_not_nan():
    retrieval, judgments = fixture()
    for item in judgments[0]["judgments"]:
        item["relevance"] = 0
    metrics = calculate(retrieval, validate_judgments(retrieval, judgments))
    assert metrics["summary"]["bm25f"]["ndcg_at_10"] == 0
    assert metrics["summary"]["semantic"]["ndcg_at_10"] == 0
