from content.pkt.recommendation.ranker import recommend


def setup_data():
    catalog = {
        "10": {"problem_id": 10, "concept_ids": [1, 2], "attempts": 4, "correct": 2, "difficulty": 0.5, "bucket": "moderate"},
        "11": {"problem_id": 11, "concept_ids": [1], "attempts": 8, "correct": 4, "difficulty": 0.5, "bucket": "moderate"},
        "12": {"problem_id": 12, "concept_ids": [2], "attempts": 7, "correct": 3, "difficulty": 0.6, "bucket": "moderate"},
        "13": {"problem_id": 13, "concept_ids": [1], "attempts": 20, "correct": 15, "difficulty": 0.25, "bucket": "easy"},
        "14": {"problem_id": 14, "concept_ids": [1], "attempts": 3, "correct": 1, "difficulty": 0.5, "bucket": "moderate"},
    }
    return catalog


def test_order_per_concept_top_k_dedup_global_cap_and_explanations():
    result = recommend(setup_data(), [2, 1], {"selected_bucket": "moderate"},
                       {10: 0.9, 11: 0.9, 12: 0.2, 14: 0.9}, {11}, {1: 0.9, 2: 0.2},
                       {"edges": [{"source": 1, "target": 2, "kind": "association"}]},
                       top_k_per_concept=1, limit=3)
    rows = result["recommendations"]
    assert [row["problem_id"] for row in rows] == [10, 14]
    assert rows[0]["target_concept"] == 2
    assert rows[0]["unmastered_prerequisites"] == []
    assert rows[1]["target_concept"] == 1
    assert rows[0]["score_source"] == "dmf"
    assert all(row["filter_mode"] == "strict" for row in rows)
    assert result["fallback_used"] is False


def test_attempted_ids_are_excluded_and_ties_use_problem_id():
    result = recommend(setup_data(), [1], {"selected_bucket": "moderate"},
                       {10: 0.5, 11: 0.5, 14: 0.5}, {10}, {1: 0.2}, {"edges": []})
    assert [row["problem_id"] for row in result["recommendations"]] == [11, 14]


def test_fallback_neighbor_bucket_then_popularity_is_labeled():
    catalog = {"13": setup_data()["13"]}
    result = recommend(catalog, [1], {"selected_bucket": "moderate"}, {}, set(), {1: 0.2},
                       {"edges": []}, allow_fallback=True)
    row = result["recommendations"][0]
    assert row["problem_id"] == 13
    assert row["filter_mode"] == "neighbor_bucket_fallback+train_popularity_fallback"
    assert row["score_source"] == "train_popularity"
    assert row["score"] == 20
    assert result["fallback_used"] is True


def test_partial_dmf_scores_use_popularity_for_the_entire_candidate_group():
    result = recommend(setup_data(), [1], {"selected_bucket": "moderate"}, {10: 999}, set(),
                       {1: 0.2}, {"edges": []}, allow_fallback=True)
    rows = result["recommendations"]
    assert [row["problem_id"] for row in rows] == [11, 10, 14]
    assert [row["score"] for row in rows] == [8.0, 4.0, 3.0]
    assert {row["score_source"] for row in rows} == {"train_popularity"}
    assert {row["filter_mode"] for row in rows} == {"train_popularity_fallback"}


def test_strict_missing_scores_and_cold_mastery_return_empty():
    result = recommend(setup_data(), [1], {"selected_bucket": "moderate"}, {}, set(), {1: 0.2}, {"edges": []})
    assert result["recommendations"] == []
    assert result["empty_reason"] == "no eligible problems"
    assert recommend(setup_data(), [], {"selected_bucket": "moderate"}, {}, set(), {1: 0.9}, {"edges": []}) == {
        "recommendations": [], "empty_reason": "no weak concepts", "fallback_used": False}


def test_mastery_and_concept_ids_are_validated():
    import pytest

    with pytest.raises(ValueError, match="mastery values"):
        recommend(setup_data(), [1], {"selected_bucket": "moderate"}, {}, set(),
                  {1: float("nan")}, {"edges": []})
    with pytest.raises(ValueError, match="concept IDs"):
        recommend(setup_data(), [0], {"selected_bucket": "moderate"}, {}, set(),
                  {}, {"edges": []})
