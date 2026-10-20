"""Observed-history concept scores from the user's ASTNN–Longformer weights."""

from pathlib import Path
import hashlib

import numpy as np
import torch

from models.astnn.model_longformer import MODEL
from recommendation.data import _ast_valid

HF_SHA256 = "b4be633b38edb435e110aba3b72d735eddc6356a63d73a15cceef1d0bbbd96a4"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class HFMastery:
    """Frozen Objective 2 ASTNN-Longformer weights (objective2/results/best_longformer_model.pt); scores are uncalibrated.

    smoothing="last" (default, used for every reported KSAP evaluation) returns the prediction at
    the last observed step. smoothing="ema" returns an exponential moving average (weight
    ``alpha`` on the newest step) of the per-step predictions over the observed window. The last-step vector is dominated by the most
    recent submission: one new attempt moved unrelated concepts by 0.16 on average (up to 0.54).
    The EMA moves them by 0.027 (up to 0.075) with no significant change in next-attempt AUC
    (validation 0.661 vs 0.666; test 0.600 vs 0.621, learner-bootstrap CI of the difference
    [-0.048, 0.006]). It is offered in the demo for live input; KSAP's weak-concept targeting
    drops with it on validation (0.24 -> 0.14), so evaluations keep "last". See MASTERY_SMOOTHING.md.
    """

    def __init__(self, checkpoint, device="cpu", context=200, max_roots=30, attention_window=8,
                 smoothing="last", alpha=0.1):
        if smoothing not in ("ema", "last") or not 0 < alpha <= 1:
            raise ValueError("smoothing must be 'ema' or 'last' and alpha in (0, 1]")
        self.smoothing, self.alpha = smoothing, alpha
        if device not in ("cpu", "cuda") or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in (context, max_roots, attention_window)):
            raise ValueError("device must be cpu/cuda and context/root limits must be positive")
        if attention_window < 2 or attention_window % 2:
            raise ValueError("attention window must be a positive even integer")
        if ((context + attention_window - 1) // attention_window) * attention_window > 504:
            raise ValueError("padded context exceeds this checkpoint's position capacity")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu")
        # The Windows GRU/Longformer combination exits unsuccessfully with cuDNN.
        if device == "cuda":
            torch.backends.cudnn.enabled = False
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (not isinstance(state, dict) or not state or
                any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values())):
            raise ValueError("expected a finite bare model state dictionary")
        embeddings = state.get("encoder.embedding.weight")
        if embeddings is None or tuple(embeddings.shape) != (32973, 128):
            raise ValueError("checkpoint does not match the Codeforces AST vocabulary")
        self.device, self.context, self.max_roots = device, context, max_roots
        self.sha256 = digest(checkpoint)
        self.model = MODEL(
            num_concepts=37, num_problems=7152, hidden_dim=64, hidden_layers=2,
            concept_embed_dim=64, ast_embed_dim=128, max_tokens=32973,
            ast_encode_dim=128, ast_pretrained_weight=np.zeros((32973, 128), np.float32),
            max_len=max_roots, batch_size=1, gpu=0 if device == "cuda" else -1,
            attention_window=attention_window,
        ).to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)

    def predict(self, interactions):
        last, ema = self.predict_both(interactions)
        return last if self.smoothing == "last" else ema

    @torch.no_grad()
    def predict_both(self, interactions):
        """(last-step vector, EMA vector) from one forward pass over the observed window."""
        interactions = list(interactions)
        if not interactions:
            return [0.5] * 37, [0.5] * 37
        if len({item.learner_id for item in interactions}) != 1:
            raise ValueError("history must belong to one learner")
        if any(a.sequence_index >= b.sequence_index for a, b in zip(interactions, interactions[1:])):
            raise ValueError("history must be in increasing sequence order")
        observed = interactions[-self.context:]
        length = len(observed)
        p_id = torch.empty((1, length), dtype=torch.long, device=self.device)
        concepts = torch.zeros((1, length, 38), device=self.device)
        correctness = torch.zeros((1, length, 2), device=self.device)
        ast = [[]]
        for step, item in enumerate(observed):
            if not isinstance(item.problem_id, int) or isinstance(item.problem_id, bool) or not 1 <= item.problem_id <= 7152:
                raise ValueError("problem ID must be in 1..7152")
            if not isinstance(item.correct, int) or isinstance(item.correct, bool) or item.correct not in (0, 1):
                raise ValueError("correctness must be 0 or 1")
            tags = item.concept_ids
            if not tags or any(not isinstance(c, int) or isinstance(c, bool) or not 1 <= c <= 37 for c in tags):
                raise ValueError("concept IDs must be in 1..37")
            _ast_valid(item.ast_roots, f"history step {item.sequence_index}")
            trees = item.ast_roots[:self.max_roots]
            stack = list(trees)
            while stack:
                node = stack.pop()
                if node == [-1]:
                    continue
                if node[0] >= 32973:
                    raise ValueError("AST token exceeds checkpoint vocabulary")
                stack.extend(node[1:])
            p_id[0, step] = item.problem_id
            concepts[0, step, list(tags)] = 1
            correctness[0, step, item.correct] = 1
            ast[0].append(trees)
        code = self.model._encode_ast_batch(ast, 1, length)
        inputs = torch.cat((concepts, code, correctness), dim=-1)
        hidden = self.model.longformer(inputs_embeds=inputs,
                                       attention_mask=torch.ones_like(p_id)).last_hidden_state[0]
        steps = self.model.predict_Linear(hidden).sigmoid()[:, 1:]  # every position sees only observed input
        ema = steps[0]
        for row in steps[1:]:
            ema = self.alpha * row + (1 - self.alpha) * ema
        out = []
        for values in (steps[-1], ema):
            if values.shape != (37,) or not torch.isfinite(values).all():
                raise FloatingPointError("invalid concept mastery output")
            out.append(values.cpu().tolist())
        return tuple(out)
