"""Regression tests for the KSAP v2 adaptation, resources and harness."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "content" / "pkt"))
from recommendation import v2  # noqa: E402
from recommendation.data import Interaction, LearnerHistory  # noqa: E402
from recommendation.dmf import train_dmf  # noqa: E402
from recommendation.metrics import ranking_metrics  # noqa: E402
from recommendation.resources import CONCEPT_NAMES, explain, resources_for  # noqa: E402


def _hist(name, rows, split="train"):
    return LearnerHistory(name, tuple(Interaction(name, i, p, tuple(c), y, None, split, 0)
                                      for i, (p, c, y) in enumerate(rows)), split)


def _toy():
    rows_a = [(1, [1], 1), (2, [2], 0), (3, [1, 2], 0), (4, [3], 1), (5, [3, 2], 0)] * 4
    rows_b = [(2, [2], 1), (3, [1, 2], 1), (6, [4], 0), (7, [4, 1], 0), (1, [1], 1)] * 4
    rows_c = [(5, [3, 2], 1), (6, [4], 1), (8, [5], 0), (9, [5, 1], 1), (4, [3], 0)] * 4
    train = [_hist("a", rows_a), _hist("b", rows_b), _hist("c", rows_c)]
    vocab = sorted({e.problem_id for h in train for e in h.interactions})
    dmf = train_dmf(train, vocab, epochs=2, seed=0)
    art = v2.fit(train, [0.5] * 37, dmf_checkpoint=dmf, windows=(5,))
    return train, art


def test_catalog_smoothing_and_quantile_buckets():
    train, _ = _toy()
    raw, _ = v2.build_catalog(train, 0.0, "paper")
    smooth, meta = v2.build_catalog(train, 3.0, "quantile")
    assert 0 < meta["prior_success"] < 1
    for pid, row in raw.items():
        # smoothing pulls every estimate towards the prior
        prior_d = 1 - meta["prior_success"]
        assert abs(smooth[pid]["difficulty"] - prior_d) <= abs(row["difficulty"] - prior_d) + 1e-12
    assert {r["bucket"] for r in smooth.values()} <= {"easy", "moderate", "difficult"}


def test_window_baskets_split_histories():
    train, _ = _toy()
    windows = v2._windows(train, 5)
    assert len(windows) == sum(len(h.interactions) // 5 for h in train)
    assert all(len(w.interactions) == 5 for w in windows)


@pytest.mark.parametrize("cfg", [
    v2.Config(), replace(v2.Config(), graph="window", window=5),
    replace(v2.Config(), weak_mode="relative", weak_k=2), replace(v2.Config(), use_dpm=False),
    replace(v2.Config(), ranker="hybrid"), replace(v2.Config(), graph="none"),
    replace(v2.Config(), dpm_mode="soft", smoothing=3.0, buckets="quantile"),
    replace(v2.Config(), graph="window", window=5, weak_mode="relative", weak_k=2,
            weak_min_candidates=2, top_k_per_concept=1),
])
def test_recommendations_exclude_attempted_and_are_unique(cfg):
    _, art = _toy()
    prefix = _hist("x", [(1, [1], 0), (2, [2], 0)], "valid").interactions
    mastery = [0.2, 0.3] + [0.9] * 35
    out = v2.recommend(art, cfg, prefix, mastery, limit=5)
    ids = [r["problem_id"] for r in out["recommendations"]]
    assert len(ids) == len(set(ids))
    assert not set(ids) & {1, 2}


def test_relative_weak_set_size():
    _, art = _toy()
    prefix = _hist("x", [(1, [1], 0)], "valid").interactions
    mastery = [0.9, 0.1, 0.2, 0.3, 0.4] + [0.95] * 32
    out = v2.recommend(art, replace(v2.Config(), weak_mode="relative", weak_k=2, graph="none"), prefix, mastery)
    assert out["weak_concepts"] == [2, 3]


def test_future_events_cannot_change_recommendations():
    _, art = _toy()
    full = _hist("x", [(1, [1], 0), (2, [2], 1), (6, [4], 0), (8, [5], 1)], "valid").interactions
    altered = full[:2] + tuple(replace(e, correct=1 - e.correct, problem_id=9) for e in full[2:])
    mastery = [0.3] * 37
    for cfg in (v2.Config(), replace(v2.Config(), graph="window", window=5)):
        a = v2.recommend(art, cfg, full[:2], mastery)
        b = v2.recommend(art, cfg, altered[:2], mastery)
        assert a == b


def test_resources_cover_all_concepts_and_explain():
    assert set(CONCEPT_NAMES) == set(range(1, 38))
    for c in CONCEPT_NAMES:
        res = resources_for(c)
        assert res[-1]["type"] == "practice" and res[-1]["url"].startswith("https://codeforces.com/problemset?tags=")
    text = explain({"target_concept": 6, "difficulty_bucket": "easy", "unmastered_prerequisites": [7],
                    "score_source": "dmf"}, {"selected_bucket": "easy"}, [0.5] * 37)
    assert "dp" in text and "implementation" in text


def test_ranking_metrics_hand_computed():
    m = ranking_metrics([5, 1, 9], {1, 2}, k=3)
    assert m["hit_rate"] == 1.0 and m["precision_at_k"] == pytest.approx(1 / 3)
    # one hit at rank 2: DCG = 1/log2(3); ideal for 2 relevant: 1 + 1/log2(3)
    import math
    assert m["ndcg_at_k"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_weak_min_candidates_drops_concepts_without_enough_problems():
    _, art = _toy()
    prefix = _hist("x", [(1, [1], 0)], "valid").interactions  # concept 1 keeps 3 unattempted problems
    mastery = [0.4, 0.3, 0.6, 0.1, 0.05] + [0.95] * 32       # concepts 4 and 5 have only 2 problems each
    for graph in ("none", "window"):
        cfg = replace(v2.Config(), graph=graph, window=5, weak_mode="relative", weak_k=2, weak_min_candidates=3)
        assert v2.recommend(art, cfg, prefix, mastery)["weak_concepts"] == [1, 2]
    thr = replace(v2.Config(), graph="window", window=5, weak_threshold=0.5, weak_min_candidates=3)
    assert v2.recommend(art, thr, prefix, mastery)["weak_concepts"] == [1, 2]
    # option off: the threshold rule alone (unchanged behaviour)
    off = replace(thr, weak_min_candidates=0)
    assert v2.recommend(art, off, prefix, mastery)["weak_concepts"] == [1, 2, 4, 5] + \
        [c for c in range(6, 38) if mastery[c - 1] < 0.5]


def test_explicit_weak_set_must_be_graph_nodes():
    from recommendation.knowledge_graph import weak_sequence
    _, art = _toy()
    graph = art.graphs["window5"]
    assert weak_sequence(graph, [0.9] * 37, weak={3})["weak_concepts"] == [3]
    with pytest.raises(ValueError):
        weak_sequence(graph, [0.9] * 37, weak={99})


def test_explain_distinguishes_prerequisites_from_weak_concepts():
    rec = {"target_concept": 2, "difficulty_bucket": "easy", "score_source": "dmf"}
    mastery = [0.5] * 37
    assert "weak concept" in explain(rec, {"selected_bucket": "easy"}, mastery)
    assert "weak concept" in explain(rec, {"selected_bucket": "easy"}, mastery, weak={2})
    text = explain(rec, {"selected_bucket": "easy"}, mastery, weak={5})
    assert "before your weak concepts" in text and "Practises your weak" not in text


def test_prerequisites_the_learner_knows_are_not_added():
    from recommendation.knowledge_graph import weak_sequence
    graph = {"nodes": [1, 2, 3], "edges": [{"source": 1, "target": 3, "kind": "association", "confidence": 0.9, "lift": 2.0},
                                           {"source": 2, "target": 3, "kind": "association", "confidence": 0.9, "lift": 2.0}]}
    mastery = [0.9, 0.2, 0.1]          # knows concept 1, weak at 2 and 3
    paper = weak_sequence(graph, mastery, threshold=0.15)
    personal = weak_sequence(graph, mastery, threshold=0.15, prereq_below=0.5)
    assert paper["expanded_concepts"] == [1, 2, 3]
    assert personal["expanded_concepts"] == [2, 3]
    assert personal["ordered_concepts"].index(2) < personal["ordered_concepts"].index(3)
