from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "content" / "pkt"))
from recommendation.calibration import FEATURES, CalibratedDiagnosis, concept_features  # noqa: E402
from recommendation.data import Interaction  # noqa: E402

PARAMS = Path(__file__).resolve().parents[1] / "content" / "pkt" / "results" / "ksap_calibrated" / "calibration.json"


def _events(rows):
    return [Interaction("x", i, p, tuple(c), y, None, "valid", 0) for i, (p, c, y) in enumerate(rows)]


def test_features_use_only_the_prefix():
    rows = [(1, [1], 1), (2, [2], 0), (3, [1, 3], 1), (4, [3], 0)]
    full = _events(rows)
    m = [0.5] * 37
    a = concept_features(full[:2], m, m)
    b = concept_features(_events(rows[:2]), m, m)
    assert a.shape == (37, len(FEATURES)) and np.array_equal(a, b)


def test_shipped_calibration_loads_and_behaves():
    cal = CalibratedDiagnosis(PARAMS)
    m = [0.5] * 37
    assert cal.predict([], m, m) == [0.5] * 37
    base = _events([(1, [1], 1), (2, [2], 0), (3, [3], 1)] * 3)
    before = cal.predict(base, m, m)
    after = cal.predict(base + _events([(9, [5], 1)])[0:1], m, m)
    assert all(0 < v < 1 for v in before)
    # a solved attempt on concept 5 raises concept 5 more than any untouched concept
    gain = np.array(after) - np.array(before)
    assert gain[4] > 0 and gain[4] > np.delete(gain, 4).max()


@pytest.mark.parametrize("bad", [{"features": ["x"]}, {"scale": [0.0] * 8}, {"coef": [1.0] * 7}])
def test_invalid_parameters_are_rejected(bad):
    import json
    params = {**json.loads(PARAMS.read_text()), **bad}
    with pytest.raises((ValueError, KeyError)):
        CalibratedDiagnosis(params)


def test_evidence_adjust_moves_the_practised_concepts_the_right_way():
    from recommendation.calibration import evidence_adjust
    base = _events([(1, [1], 1), (2, [2], 0), (3, [3], 1), (4, [1], 0)] * 3)
    scores = [0.5] * 37
    before = evidence_adjust(scores, base)
    solved = evidence_adjust(scores, base + _events([(9, [5], 1)])[:1])
    failed = evidence_adjust(scores, base + _events([(9, [5], 0)])[:1])
    assert before[4] == pytest.approx(0.5) and before[36] == pytest.approx(0.5)   # never attempted: unchanged
    assert solved[4] > before[4] > failed[4]
    assert evidence_adjust(scores, []) == scores


def test_evidence_adjust_matches_evaluated_formula():
    import math
    from recommendation.calibration import evidence_adjust
    ev = _events([(1, [1, 2], 1), (2, [2], 0), (3, [2], 0), (4, [3], 1)])
    out = evidence_adjust([0.3] * 37, ev, beta=0.5, prior=2.0)
    lg = lambda p: math.log(p / (1 - p))
    ov = (2 + 1.5) / (4 + 3)
    rate2 = (1 + 2 * ov) / (3 + 2)
    assert out[1] == pytest.approx(1 / (1 + math.exp(-(lg(0.3) + 0.5 * (lg(rate2) - lg(ov))))))
