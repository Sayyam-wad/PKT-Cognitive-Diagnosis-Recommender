"""Knowledge-tracing baselines promised in the report (Section 5.4): DKT, DKVMN and SAKT.

All three read the same interaction stream as the ASTNN model's history input (the attempt's concept
tags and its verdict; no code) and have the same output: one logit per concept. A prediction for the
next attempt is sigmoid(mean logit over its tags), exactly like the ASTNN model's training objective,
and the per-concept vector after the latest attempt is the diagnosis used by KSAP.

Protocol: fit on the training learners, early stopping on validation AUC, 3 seeds; every test
prediction uses only earlier attempts (windows of up to 200 attempts, as in the original loader).

    python content/pkt/scripts/kt_baselines.py --archive data/codeforces.zip --out content/pkt/results/kt_baselines
"""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PKT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT))
from recommendation.data import iter_histories  # noqa: E402
from scripts.v2_cache_mastery import cutoffs_for  # noqa: E402

NC, D, W = 37, 64, 200


def windows(histories):
    out = []
    for h in histories:
        it = h.interactions
        for s in range(0, len(it), W):
            chunk = it[s:s + W]
            if len(chunk) >= 2:
                out.append(chunk)
    return out


def tensors(chunks):
    L = max(len(c) for c in chunks)
    tags = torch.zeros(len(chunks), L, NC + 1)
    y = torch.zeros(len(chunks), L)
    mask = torch.zeros(len(chunks), L, dtype=torch.bool)
    for i, c in enumerate(chunks):
        for t, e in enumerate(c):
            tags[i, t, list(e.concept_ids)] = 1
            y[i, t] = e.correct
            mask[i, t] = True
    return tags[:, :, 1:], y, mask


class Interaction(nn.Module):
    """Embedding of (concept tags, verdict): mean of per-concept embeddings for the observed verdict."""
    def __init__(self):
        super().__init__()
        self.e = nn.Linear(2 * NC, D, bias=False)

    def forward(self, tags, y):
        x = torch.cat([tags * y[..., None], tags * (1 - y[..., None])], -1)
        return self.e(x) / tags.sum(-1, keepdim=True).clamp_min(1)


class DKT(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.rnn, self.out = Interaction(), nn.LSTM(D, D, batch_first=True), nn.Linear(D, NC)

    def states(self, tags, y):          # state after attempt t (sees attempts <= t)
        return self.out(self.rnn(self.inp(tags, y))[0])

    def concept_logits_after(self, tags, y):
        return self.states(tags, y)


class SAKT(nn.Module):
    """Self-attentive KT: the query is the concept to be predicted, keys/values are past interactions."""
    def __init__(self, heads=4):
        super().__init__()
        self.inp, self.q = Interaction(), nn.Embedding(NC, D)
        self.pos = nn.Embedding(W + 1, D)
        self.att = nn.MultiheadAttention(D, heads, batch_first=True, dropout=0.2)
        self.ff = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))

    def logits_for(self, tags, y, query_concepts):
        """query_concepts: (B, Lq, NC) multi-hot; position j attends to interactions < j+1 (i.e. <= j)."""
        B, L, _ = tags.shape
        kv = self.inp(tags, y) + self.pos(torch.arange(L))[None]
        q = query_concepts @ self.q.weight / query_concepts.sum(-1, keepdim=True).clamp_min(1)
        Lq = q.shape[1]
        causal = torch.triu(torch.ones(Lq, L, dtype=torch.bool), diagonal=1)   # query j sees keys 0..j
        h, _ = self.att(q, kv, kv, attn_mask=causal)
        return self.ff(h + q).squeeze(-1)

    def concept_logits_after(self, tags, y):
        """(B, L, NC): logit for every concept as the next attempt after each position."""
        B, L, _ = tags.shape
        eye = torch.eye(NC)
        out = []
        for c in range(NC):
            out.append(self.logits_for(tags, y, eye[c].expand(B, L, NC)))
        return torch.stack(out, -1)


class DKVMN(nn.Module):
    def __init__(self, m=20):
        super().__init__()
        self.key = nn.Parameter(torch.randn(m, D) * 0.1)
        self.v0 = nn.Parameter(torch.randn(m, D) * 0.1)
        self.k_emb, self.inp = nn.Linear(NC, D, bias=False), Interaction()
        self.erase, self.add = nn.Linear(D, D), nn.Linear(D, D)
        self.read = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh(), nn.Linear(D, 1))

    def run(self, tags, y, query=None):
        """Returns per-step memories after update; if query is given, logits for those queries."""
        B, L, _ = tags.shape
        V = self.v0[None].expand(B, -1, -1)
        mems = []
        k = self.k_emb(tags / tags.sum(-1, keepdim=True).clamp_min(1))
        w = torch.softmax(k @ self.key.T, -1)                                 # (B, L, m)
        v = self.inp(tags, y)
        for t in range(L):
            e, a = torch.sigmoid(self.erase(v[:, t])), torch.tanh(self.add(v[:, t]))
            wt = w[:, t, :, None]
            V = V * (1 - wt * e[:, None]) + wt * a[:, None]
            mems.append(V)
        return torch.stack(mems, 1)                                            # (B, L, m, D)

    def readout(self, mem, qtags):
        k = self.k_emb(qtags / qtags.sum(-1, keepdim=True).clamp_min(1))
        w = torch.softmax(k @ self.key.T, -1)
        r = (w[..., None] * mem).sum(-2)
        return self.read(torch.cat([r, k], -1)).squeeze(-1)

    def concept_logits_after(self, tags, y):
        mem = self.run(tags, y)
        eye = torch.eye(NC)
        return torch.stack([self.readout(mem, eye[c].expand(*tags.shape[:2], NC)) for c in range(NC)], -1)


def next_logits(model, tags, y):
    """Logit for attempt t+1 from attempts <= t, for t = 0..L-2 (same readout as the ASTNN model)."""
    nxt = tags[:, 1:]
    if isinstance(model, SAKT):
        return model.logits_for(tags[:, :-1], y[:, :-1], nxt)
    if isinstance(model, DKVMN):
        return model.readout(model.run(tags[:, :-1], y[:, :-1]), nxt)
    s = model.states(tags[:, :-1], y[:, :-1])
    return (s * nxt).sum(-1) / nxt.sum(-1).clamp_min(1)


def evaluate(model, chunks, bs=32):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for i in range(0, len(chunks), bs):
            tags, y, mask = tensors(chunks[i:i + bs])
            lg = next_logits(model, tags, y)
            m = mask[:, 1:]
            ps.append(torch.sigmoid(lg[m]).numpy()); ys.append(y[:, 1:][m].numpy())
    p, t = np.concatenate(ps), np.concatenate(ys)
    return roc_auc_score(t, p), p, t


def train(kind, tr, va, seed, epochs=40, patience=6):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = {"DKT": DKT, "SAKT": SAKT, "DKVMN": DKVMN}[kind]()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); random.shuffle(tr)
        for i in range(0, len(tr), 16):
            tags, y, mask = tensors(tr[i:i + 16])
            lg = next_logits(model, tags, y)
            m = mask[:, 1:]
            loss = nn.functional.binary_cross_entropy_with_logits(lg[m], y[:, 1:][m])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        auc = evaluate(model, va)[0]
        if auc > best + 1e-4:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def diagnosis_cache(model, histories, split):
    """Per-concept probabilities after the last observed attempt, at the v2 evaluation cut-offs."""
    model.eval(); cache = {}
    with torch.no_grad():
        for h in histories:
            for c in cutoffs_for(len(h.interactions)):
                tags, y, _ = tensors([h.interactions[:c][-W:]])
                cache[f"{split}|{h.learner_id}|{c}"] = torch.sigmoid(model.concept_logits_after(tags, y)[0, -1]).tolist()
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["DKT", "DKVMN", "SAKT"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    H = {s: list(iter_histories(args.archive, s, include_asts=False)) for s in ("train", "valid", "test")}
    tr, va, te = windows(H["train"]), windows(H["valid"]), windows(H["test"])
    summary = {}
    for kind in args.models:
        runs = []
        for seed in args.seeds:
            model, vauc, ep = train(kind, list(tr), va, seed)
            tauc = evaluate(model, te)[0]
            runs.append({"seed": seed, "valid_auc": vauc, "test_auc": tauc, "epochs": ep})
            print(f"{kind} seed {seed}: valid {vauc:.4f} test {tauc:.4f} ({ep} epochs)", flush=True)
            if seed == args.seeds[0]:   # diagnosis vectors for KSAP from the first seed
                cache = diagnosis_cache(model, H["valid"], "valid") | diagnosis_cache(model, H["test"], "test")
                (args.out / f"{kind}_mastery_cache.json").write_text(json.dumps(cache))
                full = {h.learner_id: torch.sigmoid(model.concept_logits_after(*tensors([h.interactions[-W:]])[:2])[0, -1]).tolist()
                        for h in H["train"] + H["valid"]}
                (args.out / f"{kind}_full_mastery.json").write_text(json.dumps(full))
        t = [r["test_auc"] for r in runs]
        summary[kind] = {"runs": runs, "test_auc_mean": float(np.mean(t)), "test_auc_sd": float(np.std(t))}
        print(f"== {kind}: test AUC {np.mean(t):.4f} ± {np.std(t):.4f}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
