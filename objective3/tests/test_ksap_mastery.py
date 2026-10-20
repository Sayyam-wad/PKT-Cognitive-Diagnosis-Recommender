from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "content" / "pkt"))
from models.astnn.model_longformer import MODEL
from recommendation.data import Interaction
from recommendation.mastery import HFMastery


@pytest.fixture(scope="module")
def provider():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=root) as directory:
        model = MODEL(num_concepts=37, num_problems=7152, hidden_dim=64, hidden_layers=2,
                      concept_embed_dim=64, ast_embed_dim=128, max_tokens=32973,
                      ast_encode_dim=128, ast_pretrained_weight=np.zeros((32973, 128), np.float32),
                      max_len=30, batch_size=1, gpu=-1, attention_window=8)
        checkpoint = Path(directory) / "mastery.pt"
        torch.save(model.state_dict(), checkpoint)
        yield HFMastery(checkpoint, context=2)


def events():
    return tuple(Interaction("learner", i, i + 1, (1,), i % 2, [[1, [2]]], "valid1", 1)
                 for i in range(3))


def test_endpoint_matches_full_prefix_model_and_context_uses_only_latest_events(provider):
    history = events()
    predicted = provider.predict(history)
    assert predicted == provider.predict(history[-2:])
    p_id = torch.tensor([[2, 3]])
    c_id = torch.zeros((1, 2, 75))
    concepts = torch.zeros((1, 2, 38))
    concepts[:, :, 1] = 1
    correctness = torch.tensor([[[0., 1.], [1., 0.]]])
    ast = [[item.ast_roots for item in history[-2:]]]
    with torch.no_grad():
        expected = provider.model.predict_mastery(p_id, c_id, ast, concepts, correctness)[0]
    torch.testing.assert_close(torch.tensor(predicted), expected, rtol=1e-5, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in provider.model.parameters())


def test_cold_start_and_invalid_histories(provider):
    assert provider.predict([]) == [0.5] * 37
    event = events()[0]
    for changed in (replace(event, problem_id=0), replace(event, correct=1.0),
                    replace(event, concept_ids=(38,)), replace(event, ast_roots=[[32973]])):
        with pytest.raises(ValueError):
            provider.predict([changed])
    with pytest.raises(ValueError, match="sequence order"):
        provider.predict(list(reversed(events())))
    with pytest.raises(ValueError, match="one learner"):
        provider.predict([event, replace(events()[1], learner_id="other")])


@pytest.mark.parametrize("context", [True, 1.5, 0])
def test_invalid_context_is_rejected_before_loading(context):
    with pytest.raises(ValueError):
        HFMastery("unused.pt", context=context)


def test_ema_smoothing_averages_per_step_predictions(provider):
    history = events()
    provider.context = 3
    try:
        provider.smoothing = "last"
        last = provider.predict(history)
        provider.smoothing, provider.alpha = "ema", 1.0
        assert provider.predict(history) == pytest.approx(last, abs=1e-6)  # alpha=1 keeps only the newest step
        provider.alpha = 0.1
        smoothed = provider.predict(history)
        assert all(0.0 < v < 1.0 for v in smoothed) and smoothed != pytest.approx(last, abs=1e-9)
    finally:
        provider.smoothing, provider.alpha, provider.context = "last", 0.1, 2


@pytest.mark.parametrize("kwargs", [{"smoothing": "mean"}, {"alpha": 0}, {"alpha": 1.5}])
def test_invalid_smoothing_is_rejected_before_loading(kwargs):
    with pytest.raises(ValueError):
        HFMastery("unused.pt", **kwargs)
