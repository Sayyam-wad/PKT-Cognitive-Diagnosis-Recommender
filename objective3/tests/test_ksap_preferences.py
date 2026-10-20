import pytest
from pytest import approx

from content.pkt.recommendation.data import Interaction, LearnerHistory
from content.pkt.recommendation.preferences import difficulty_bucket, difficulty_preference, fit_catalog


def history(learner, rows):
    interactions = tuple(Interaction(learner, i, pid, tuple(concepts), correct, None, "train", 1)
                         for i, (pid, concepts, correct) in enumerate(rows))
    return LearnerHistory(learner, interactions, "train")


def test_difficulty_boundaries_are_paper_aligned():
    assert [difficulty_bucket(x) for x in (0.0, 0.3, 0.30001, 0.69999, 0.7, 1.0)] == [
        "easy", "easy", "moderate", "moderate", "difficult", "difficult"]
    for value in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            difficulty_bucket(value)


def test_catalog_aggregates_retries_and_unions_concepts():
    catalog = fit_catalog([history("a", [(10, [1, 2], 1), (10, [2], 0), (11, [3], 1)])])
    assert catalog == {
        "10": {"problem_id": 10, "concept_ids": [1, 2], "attempts": 2, "correct": 1,
               "difficulty": 0.5, "bucket": "moderate"},
        "11": {"problem_id": 11, "concept_ids": [3], "attempts": 1, "correct": 1,
               "difficulty": 0.0, "bucket": "easy"},
    }


def test_dpm_hand_calculation_unknown_attempts_and_tie_order():
    catalog = {
        "1": {"bucket": "easy"}, "2": {"bucket": "easy"},
        "3": {"bucket": "moderate"}, "4": {"bucket": "difficult"},
    }
    h = history("a", [(1, [1], 1), (2, [1], 0), (3, [2], 1), (999, [2], 1)])
    result = difficulty_preference(h.interactions, catalog)
    assert result["counts"] == {"easy": 2, "moderate": 1, "difficult": 0}
    assert result["correct_counts"] == {"easy": 1, "moderate": 1, "difficult": 0}
    assert result["scores"] == approx({"easy": 7 / 12, "moderate": 2 / 3, "difficult": 0.0})
    assert result["selected_bucket"] == "moderate"
    assert result["unknown_attempts"] == 1
    tie = difficulty_preference(history("b", [(1, [1], 0), (1, [1], 0), (2, [2], 0), (2, [2], 0)]).interactions,
                                {"1": {"bucket": "easy"}, "2": {"bucket": "moderate"}})
    assert tie["scores"]["easy"] == tie["scores"]["moderate"] == 0.25
    assert tie["selected_bucket"] == "moderate"


def test_empty_history_has_finite_zero_scores_and_moderate_fallback():
    result = difficulty_preference((), {})
    assert result == {"scores": {"easy": 0.0, "moderate": 0.0, "difficult": 0.0},
                      "selected_bucket": "moderate", "counts": {"easy": 0, "moderate": 0, "difficult": 0},
                      "correct_counts": {"easy": 0, "moderate": 0, "difficult": 0}, "unknown_attempts": 0}


@pytest.mark.parametrize("weights", [(0.2, 0.2), (-0.1, 1.1), (float("nan"), 0.5)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        difficulty_preference((), {}, *weights)
