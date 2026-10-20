import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PKT_ROOT = REPO_ROOT / "content" / "pkt"
if str(PKT_ROOT) not in sys.path:
    sys.path.insert(0, str(PKT_ROOT))

from recommendation import pipeline
from recommendation.data import Interaction, LearnerHistory
from recommendation.evaluate import evaluate_recommendations
from recommendation.mastery import digest


def history(learner_id, rows, split="train"):
    interactions = tuple(
        Interaction(learner_id, index, problem_id, tuple(tags), correct,
                    [[1]], split, 1)
        for index, (problem_id, tags, correct) in enumerate(rows)
    )
    return LearnerHistory(learner_id, interactions, split)


def training_histories():
    return [
        history("train-0", [(101, [1], 1), (102, [2], 0)]),
        history("train-1", [(101, [1], 0), (103, [1], 1), (104, [2], 1)]),
        history("train-2", [(102, [2], 1), (103, [1], 0), (104, [2], 0)]),
        history("train-3", [(101, [1], 1), (102, [2], 1), (103, [1], 1), (104, [2], 0)]),
    ]


class FakeHFMastery:
    contexts = []

    def __init__(self, checkpoint, device="cpu", context=200, max_roots=30, attention_window=8):
        self.context = context
        self.sha256 = digest(checkpoint)

    def predict(self, interactions):
        observed = tuple(interactions)
        self.contexts.append(tuple(item.sequence_index for item in observed[-self.context:]))
        return [0.1 if concept in (1, 2) else 0.9 for concept in range(1, 38)]


@pytest.fixture
def fitted_bundle(monkeypatch):
    FakeHFMastery.contexts.clear()
    monkeypatch.setattr(pipeline, "HFMastery", FakeHFMastery)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
            temp = Path(temp_dir)
            checkpoint = temp / "dummy-checkpoint.bin"
            checkpoint.write_bytes(b"local fixture checkpoint")
            bundle = temp / "bundle"
            pipeline.fit_pipeline(training_histories(), checkpoint, bundle, dmf_epochs=1, seed=4)
            system = pipeline.KSAPRecommender(bundle)
            yield bundle, system
    finally:
        torch.set_num_threads(previous_threads)


def test_fit_save_reload_and_attempt_exclusion_use_entire_observed_history(fitted_bundle):
    _, system = fitted_bundle
    observed = history(
        "held-out-long",
        [(101, [1], 1)] + [(102, [2], index % 2) for index in range(1, 250)],
        split="valid1",
    ).interactions
    result = system.recommend(observed, learner_id="held-out-long", limit=4)

    assert system.manifest["dmf_epochs"] == 1
    assert system.dmf.problem_ids == [101, 102, 103, 104]
    assert result["observed_interactions"] == 250
    assert result["model_context_interactions"] == 200
    assert 0 not in FakeHFMastery.contexts[-1]
    assert {item["problem_id"] for item in result["recommendations"]}.isdisjoint({101, 102})
    assert all(item["problem_id"] in {103, 104} for item in result["recommendations"])


@pytest.mark.parametrize("split", ["valid1", "test"])
def test_fit_rejects_validation_and_test_histories(monkeypatch, split):
    monkeypatch.setattr(pipeline, "HFMastery", FakeHFMastery)
    with TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        temp = Path(temp_dir)
        checkpoint = temp / "dummy.bin"
        checkpoint.write_bytes(b"fixture")
        with pytest.raises(ValueError, match="training histories only"):
            pipeline.fit_pipeline([history("held-out", [(101, [1], 1)], split)],
                                  checkpoint, temp / "bundle", dmf_epochs=1)


def test_bundle_checksum_rejects_corrupted_artifact(fitted_bundle):
    bundle, _ = fitted_bundle
    catalog = bundle / "catalog.json"
    catalog.write_text(catalog.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch: catalog.json"):
        pipeline.KSAPRecommender(bundle)


def test_evaluation_uses_prefix_only_and_macro_counts_have_clear_denominators(fitted_bundle):
    _, system = fitted_bundle
    histories = [
        history("valid-no-new-item", [(101, [1], 1), (102, [2], 0),
                                       (101, [1], 0), (102, [2], 1)], "valid1"),
        history("valid-new-items", [(101, [1], 1), (102, [2], 0),
                                     (103, [1], 0), (104, [2], 1)], "valid1"),
    ]
    FakeHFMastery.contexts.clear()
    report = evaluate_recommendations(system, histories, fraction=0.5, k=4)

    assert report["evaluated_learners"] == 2
    assert report["learners_with_future_unseen_items"] == 1
    assert report["learners_with_reachable_future_items"] == 1
    assert report["out_of_catalog_future_items"] == 0
    assert FakeHFMastery.contexts == [(0, 1), (0, 1)]
    for method, summary in report["summary"].items():
        assert summary["all_future_metrics"]["recall_at_k"] == report["snapshots"][1]["methods"][method]["all_future_metrics"]["recall_at_k"]
        assert summary["empty_result_rate"] == pytest.approx(
            sum(row["methods"][method]["returned"] == 0 for row in report["snapshots"]) / 2
        )


def test_evaluation_rejects_training_learner_overlap(fitted_bundle):
    _, system = fitted_bundle
    overlapping = history("train-0", [(101, [1], 1), (102, [2], 0)], "valid1")
    with pytest.raises(ValueError, match="overlap"):
        evaluate_recommendations(system, [overlapping])


def test_future_correctness_and_ast_changes_do_not_change_prefix_recommendations(fitted_bundle):
    _, system = fitted_bundle
    original = history("valid-future", [(101, [1], 1), (102, [2], 0),
                                        (103, [1], 0), (104, [2], 1)], "valid1")
    changed_suffix = LearnerHistory(
        original.learner_id,
        original.interactions[:2] + tuple(
            Interaction(item.learner_id, item.sequence_index, item.problem_id, item.concept_ids,
                        1 - item.correct, [[2]], item.source_split, item.source_record)
            for item in original.interactions[2:]
        ),
        original.source_split,
    )
    base_report = evaluate_recommendations(system, [original], fraction=0.5, k=4)
    changed_report = evaluate_recommendations(system, [changed_suffix], fraction=0.5, k=4)
    base_row, changed_row = base_report["snapshots"][0], changed_report["snapshots"][0]
    assert base_row["recommendation_details"] == changed_row["recommendation_details"]
    assert {name: value["problem_ids"] for name, value in base_row["methods"].items()} == {
        name: value["problem_ids"] for name, value in changed_row["methods"].items()
    }
