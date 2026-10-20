from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "content" / "pkt"))
from scripts.kt_baselines import DKT, DKVMN, SAKT, NC, next_logits  # noqa: E402


@pytest.mark.parametrize("cls", [DKT, DKVMN, SAKT])
def test_baselines_are_causal_and_shaped(cls):
    torch.manual_seed(0)
    model = cls().eval()
    L = 12
    tags = torch.zeros(1, L, NC)
    for t in range(L):
        tags[0, t, t % NC] = 1
    y = torch.randint(0, 2, (1, L)).float()
    with torch.no_grad():
        base_c, base_n = model.concept_logits_after(tags, y), next_logits(model, tags, y)
        y2 = y.clone(); y2[0, 8:] = 1 - y2[0, 8:]          # change only answers from position 8 on
        alt_c, alt_n = model.concept_logits_after(tags, y2), next_logits(model, tags, y2)
    assert base_c.shape == (1, L, NC) and base_n.shape == (1, L - 1)
    assert torch.allclose(base_c[0, :8], alt_c[0, :8], atol=1e-6)        # states before 8 unaffected
    assert torch.allclose(base_n[0, :8], alt_n[0, :8], atol=1e-6)        # predictions for attempts 1..8 unaffected
    assert not torch.allclose(base_c[0, 8:], alt_c[0, 8:])              # and the change is actually used later
