import math

import pytest

from content.pkt.recommendation.metrics import ranking_metrics


def test_metrics_match_hand_calculation_and_paper_definitions():
    metrics = ranking_metrics([7, 2, 9, 4], {2, 4}, k=3)
    expected_dcg = 1 / math.log2(3)
    expected_idcg = 1 + 1 / math.log2(3)
    assert metrics["hit_rate"] == 1
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"] == pytest.approx(1 / 2)
    assert metrics["ndcg_at_k"] == pytest.approx(expected_dcg / expected_idcg)
    assert metrics["paper_hr"] == pytest.approx(1 / 3)
    assert metrics["paper_ng"] == pytest.approx(expected_dcg / 3)


def test_short_or_empty_lists_and_no_relevant_items():
    short = ranking_metrics([1], {1, 2}, k=5)
    assert short["precision_at_k"] == pytest.approx(1 / 5)
    assert short["paper_hr"] == 1
    assert short["paper_ng"] == 1
    empty = ranking_metrics([], set(), k=5)
    assert empty["hit_rate"] == 0
    assert empty["precision_at_k"] == 0
    assert empty["recall_at_k"] is None
    assert empty["ndcg_at_k"] is None
    assert empty["paper_hr"] is None
    assert empty["paper_ng"] is None


def test_metrics_validate_k_and_duplicate_recommendations():
    with pytest.raises(ValueError, match="positive integer"):
        ranking_metrics([], [], k=0)
    with pytest.raises(ValueError, match="duplicates"):
        ranking_metrics([1, 1], [1])
