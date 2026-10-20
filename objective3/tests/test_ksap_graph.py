import json

import pytest

from content.pkt.recommendation.data import Interaction, LearnerHistory
from content.pkt.recommendation.knowledge_graph import fit_graph, weak_sequence


def history(learner, rows):
    interactions = tuple(Interaction(learner, i, 100 + i, tuple(tags), correct, None, "train", 1)
                         for i, (tags, correct) in enumerate(rows))
    return LearnerHistory(learner, interactions, "train")


def test_pairwise_rules_include_empty_learner_baskets_in_denominator_and_dag():
    histories = [history("a", [([1, 2], 0)]), history("b", [([1, 2], 0)]), history("c", [([1], 0)]),
                 history("d", [([3], 1)])]
    graph = fit_graph(histories, [0.9, 0.2, 0.5], num_concepts=3, support=0.5, confidence=0.6)
    json.dumps(graph)
    association = next(edge for edge in graph["edges"] if edge["kind"] == "association")
    assert association == {"source": 1, "target": 2, "kind": "association", "support": 0.5,
                           "confidence": 2 / 3, "lift": 4 / 3}
    assert graph["stats"]["learners"] == 4
    assert graph["stats"]["empty_baskets"] == 1
    order = weak_sequence(graph, [0.9, 0.1, 0.4], threshold=0.5)["ordered_concepts"]
    assert order == [1, 3, 2]
    assert set(order) == set(graph["nodes"])


def test_ties_are_recorded_and_auxiliary_edges_do_not_expand_weak_set():
    graph = fit_graph([history("a", [([1, 2], 0)]), history("b", [])], [0.8, 0.8, 0.2], num_concepts=3, support=0.5)
    assert graph["stats"]["mastery_ties"] == [[1, 2]]
    assert graph["stats"]["empty_baskets"] == 1
    assert all(edge["kind"] == "auxiliary" for edge in graph["edges"])
    result = weak_sequence(graph, [0.9, 0.1, 0.9])
    assert result["weak_concepts"] == [2]
    assert result["expanded_concepts"] == [2]


def test_only_qualifying_associations_expand_and_threshold_is_strict():
    graph = {"nodes": [1, 2, 3], "edges": [
        {"source": 1, "target": 2, "kind": "association", "confidence": 0.75, "lift": 2.0},
        {"source": 2, "target": 3, "kind": "association", "confidence": 0.9, "lift": 1.1},
    ]}
    result = weak_sequence(graph, {1: 0.9, 2: 0.5, 3: 0.4}, threshold=0.5)
    assert result == {"weak_concepts": [3], "expanded_concepts": [2, 3], "ordered_concepts": [2, 3]}


def test_expansion_uses_the_fitted_confidence_threshold():
    graph = {"nodes": [1, 2], "thresholds": {"confidence": 0.5}, "edges": [
        {"source": 1, "target": 2, "kind": "association", "confidence": 0.6, "lift": 1.2},
    ]}
    assert weak_sequence(graph, [0.9, 0.4])["expanded_concepts"] == [1, 2]


def test_graph_inputs_and_cycles_are_validated():
    with pytest.raises(ValueError, match="between 0 and 1"):
        fit_graph([], [0.1], num_concepts=1, support=1.1)
    with pytest.raises(ValueError, match="acyclic"):
        weak_sequence({"nodes": [1, 2], "edges": [
            {"source": 1, "target": 2, "kind": "auxiliary"},
            {"source": 2, "target": 1, "kind": "auxiliary"},
        ]}, [0.1, 0.2])
    with pytest.raises(ValueError, match="mastery values"):
        fit_graph([], [1.01], num_concepts=1)
    with pytest.raises(ValueError, match="mastery values"):
        weak_sequence({"nodes": [1], "edges": []}, [-0.01])
