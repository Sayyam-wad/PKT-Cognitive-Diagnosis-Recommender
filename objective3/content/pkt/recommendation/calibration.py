"""Calibrated concept diagnosis (DIAGNOSIS_CALIBRATION.md).

A logistic layer over the frozen ASTNN-Longformer's outputs (last-step and EMA concept scores)
plus the learner's own record on each concept. Fitted on training learners only: target =
solving the next 10 attempts that carry the concept. Output = per-concept probability of
success, comparable across concepts and learners.

Only numpy is needed at run time; the fitted weights live in a small JSON file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

NUM_CONCEPTS = 37
FEATURES = ("model_last", "model_ema", "log_attempts", "concept_rate", "concept_recent", "overall_rate",
            "overall_recent", "log_since_last")


def _logit(p, eps=1e-3):
    p = min(max(float(p), eps), 1 - eps)
    return math.log(p / (1 - p))


def concept_features(prefix, last, ema):
    """NUM_CONCEPTS x len(FEATURES) matrix built from the observed prefix only."""
    prefix = list(prefix)
    n = len(prefix)
    overall = (sum(e.correct for e in prefix) + 1.5) / (n + 3)
    recent = prefix[-20:]
    overall_recent = (sum(e.correct for e in recent) + 1.0) / (len(recent) + 2)
    attempts = [0] * (NUM_CONCEPTS + 1)
    solved = [0] * (NUM_CONCEPTS + 1)
    history = [[] for _ in range(NUM_CONCEPTS + 1)]
    last_seen = [None] * (NUM_CONCEPTS + 1)
    for i, e in enumerate(prefix):
        for c in e.concept_ids:
            attempts[c] += 1
            solved[c] += e.correct
            history[c].append(e.correct)
            last_seen[c] = i
    prior = 3.0
    X = np.empty((NUM_CONCEPTS, len(FEATURES)))
    for c in range(1, NUM_CONCEPTS + 1):
        rate = (solved[c] + prior * overall) / (attempts[c] + prior)
        h5 = history[c][-5:]
        recent_rate = (sum(h5) + prior * overall) / (len(h5) + prior)
        since = n - last_seen[c] if last_seen[c] is not None else n + 50
        X[c - 1] = (_logit(last[c - 1]), _logit(ema[c - 1]), math.log1p(attempts[c]), _logit(rate),
                    _logit(recent_rate), _logit(overall), _logit(overall_recent), math.log1p(since))
    return X


class CalibratedDiagnosis:
    """Standardise -> logistic regression, with weights loaded from JSON."""

    def __init__(self, params):
        if isinstance(params, (str, Path)):
            params = json.loads(Path(params).read_text())
        if tuple(params["features"]) != FEATURES:
            raise ValueError("calibration file was fitted on different features")
        self.mean = np.asarray(params["mean"], float)
        self.scale = np.asarray(params["scale"], float)
        self.coef = np.asarray(params["coef"], float)
        self.intercept = float(params["intercept"])
        if not (self.mean.shape == self.scale.shape == self.coef.shape == (len(FEATURES),)) or \
                not np.isfinite(np.concatenate([self.mean, self.scale, self.coef])).all() or (self.scale <= 0).any():
            raise ValueError("invalid calibration parameters")

    def predict(self, prefix, last, ema):
        """Per-concept success probabilities (list of 37 floats)."""
        prefix = list(prefix)
        if not prefix:
            return [0.5] * NUM_CONCEPTS
        z = (concept_features(prefix, last, ema) - self.mean) / self.scale @ self.coef + self.intercept
        return (1 / (1 + np.exp(-z))).tolist()

    @staticmethod
    def params_from_sklearn(pipeline):
        scaler, lr = pipeline[0], pipeline[-1]
        return {"features": list(FEATURES), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


EVIDENCE_BETA = 0.5   # chosen on validation learners (MASTERY_SMOOTHING.md, "Evidence adjustment")
EVIDENCE_PRIOR = 2.0


def evidence_adjust(scores, prefix, beta=EVIDENCE_BETA, prior=EVIDENCE_PRIOR):
    """Shift each concept's score by the learner's own record on it relative to their overall level.

    logit(out_c) = logit(score_c) + beta * [logit(rate_c) - logit(overall)], where rate_c is the success
    rate on concept c shrunk towards the overall rate with ``prior`` pseudo-attempts. A solved attempt
    raises its concepts and a failed one lowers them; concepts never attempted are unchanged.
    """
    prefix = list(prefix)
    if not prefix:
        return list(scores)
    n = len(prefix)
    overall = (sum(e.correct for e in prefix) + 1.5) / (n + 3)
    attempts = [0] * (NUM_CONCEPTS + 1)
    solved = [0] * (NUM_CONCEPTS + 1)
    for e in prefix:
        for c in e.concept_ids:
            attempts[c] += 1
            solved[c] += e.correct
    out = []
    for c in range(1, NUM_CONCEPTS + 1):
        rate = (solved[c] + prior * overall) / (attempts[c] + prior)
        z = _logit(scores[c - 1]) + beta * (_logit(rate) - _logit(overall))
        out.append(1 / (1 + math.exp(-z)))
    return out
