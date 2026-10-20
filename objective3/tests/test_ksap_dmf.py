from types import SimpleNamespace
import copy
import tempfile
from pathlib import Path
import unittest

import torch

from content.pkt.recommendation.dmf import _masked_inputs, build_incidence, load_dmf, score_candidates, train_dmf


def history(learner_id, attempts):
    interactions = tuple(
        SimpleNamespace(problem_id=pid, correct=correct)
        for pid, correct in attempts
    )
    return SimpleNamespace(learner_id=learner_id, interactions=interactions)


class DMFTests(unittest.TestCase):
    def fixture(self):
        return [
            history("a", [(10, 0), (11, 1), (10, 1)]),
            history("b", [(11, 1), (12, 0)]),
            history("c", [(10, 1), (12, 1)]),
        ]

    def test_incorrect_attempt_is_incident_and_repeated_outcomes_are_averaged(self):
        matrix, (users, items, labels) = build_incidence(self.fixture(), [10, 11, 12, 13])
        self.assertEqual(matrix[0, 0].item(), 1.0)
        self.assertEqual(matrix[0, 3].item(), 0.0)
        self.assertEqual(float(labels[(users == 0) & (items == 0)].item()), 0.5)

    def test_training_masks_supervised_cell_from_both_tower_inputs(self):
        matrix, (users, items, _) = build_incidence(self.fixture(), [10, 11, 12, 13])
        rows, columns = _masked_inputs(matrix, users, items)
        offsets = torch.arange(len(users))
        self.assertTrue(torch.all(rows[offsets, items] == 0))
        self.assertTrue(torch.all(columns[offsets, users] == 0))
        self.assertTrue(torch.all(matrix[users, items] == 1))

    def test_tiny_fixture_optimizes_and_scores_are_finite_and_reloadable(self):
        checkpoint = train_dmf(self.fixture(), [10, 11, 12, 13], epochs=25, seed=7)
        self.assertLess(checkpoint["history"][-1], checkpoint["history"][0])
        scores = score_candidates(checkpoint, [10, 99])
        self.assertEqual(set(scores), {10, 11, 12, 13})
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in scores.values()))

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "dmf.pt"
            torch.save(checkpoint, path)
            restored = load_dmf(path)
            self.assertEqual(scores, restored.score(restored.problem_ids, [10, 99]))

    def test_zero_history_and_incidence_objective_are_finite(self):
        checkpoint = train_dmf(self.fixture(), [10, 11, 12, 13], epochs=2, seed=3, objective="incidence")
        ranker = load_dmf(checkpoint)
        cold_scores = ranker.score(ranker.problem_ids, [])
        self.assertEqual(checkpoint["objective"], "incidence")
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in cold_scores.values()))

    def test_checkpoint_rejects_nonbinary_or_nonfinite_incidence(self):
        checkpoint = train_dmf(self.fixture(), [10, 11, 12, 13], epochs=1, seed=3)
        for value in (2.0, float("nan")):
            with self.subTest(value=value):
                damaged = copy.deepcopy(checkpoint)
                damaged["incidence"][0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite binary"):
                    load_dmf(damaged)

    def test_checkpoint_rejects_nonfinite_weights(self):
        checkpoint = train_dmf(self.fixture(), [10, 11, 12, 13], epochs=1, seed=3)
        damaged = copy.deepcopy(checkpoint)
        damaged["state_dict"]["user_tower.layers.0.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite tensors"):
            load_dmf(damaged)

