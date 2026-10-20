"""Attempt-incidence deep matrix factorization for train-catalog ranking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


OBJECTIVES = {"observed_correctness", "incidence"}


def _catalog(problem_ids):
    ids = list(problem_ids)
    if not ids or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in ids):
        raise ValueError("problem_ids must be a nonempty sequence of positive integers")
    if len(ids) != len(set(ids)):
        raise ValueError("problem_ids must be unique")
    return ids


def build_incidence(histories, problem_ids):
    """Build binary attempt rows and mean correctness targets for observed pairs."""
    problem_ids = _catalog(problem_ids)
    item_index = {pid: i for i, pid in enumerate(problem_ids)}
    learner_ids, seen_learners = [], set()
    outcomes = defaultdict(list)
    for history in histories:
        learner_id = history.learner_id
        if learner_id in seen_learners:
            raise ValueError(f"duplicate learner_id {learner_id!r}")
        seen_learners.add(learner_id)
        learner_index = len(learner_ids)
        learner_ids.append(learner_id)
        for interaction in history.interactions:
            if interaction.correct not in (0, 1):
                raise ValueError("interaction correctness must be 0 or 1")
            if interaction.problem_id not in item_index:
                continue
            outcomes[learner_index, item_index[interaction.problem_id]].append(interaction.correct)

    matrix = torch.zeros((len(learner_ids), len(problem_ids)), dtype=torch.float32)
    pairs = sorted(outcomes)
    if pairs:
        users = torch.tensor([u for u, _ in pairs], dtype=torch.long)
        items = torch.tensor([i for _, i in pairs], dtype=torch.long)
        labels = torch.tensor([sum(outcomes[pair]) / len(outcomes[pair]) for pair in pairs], dtype=torch.float32)
        matrix[users, items] = 1.0
    else:
        users = items = torch.empty(0, dtype=torch.long)
        labels = torch.empty(0, dtype=torch.float32)
    return matrix, (users, items, labels)


class _Tower(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, values):
        return self.layers(values)


class DMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim=64):
        super().__init__()
        self.user_tower = _Tower(n_items, hidden_dim)
        self.item_tower = _Tower(n_users, hidden_dim)

    def forward(self, user_rows, item_columns):
        users = self.user_tower(user_rows)
        items = self.item_tower(item_columns)
        # ReLU vectors make cosine nonnegative; clamp endpoints for finite BCE.
        return F.cosine_similarity(users, items, dim=-1, eps=1e-8).clamp(1e-7, 1 - 1e-7)


def _masked_inputs(matrix, users, items):
    user_rows = matrix.index_select(0, users).clone()
    item_columns = matrix.T.index_select(0, items).clone()
    offsets = torch.arange(len(users), device=matrix.device)
    user_rows[offsets, items] = 0.0
    item_columns[offsets, users] = 0.0
    return user_rows, item_columns


def _checkpoint_model(checkpoint, device):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    required = {"state_dict", "problem_ids", "learner_ids", "incidence", "hidden_dim", "history", "objective"}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint is missing required DMF fields")
    if checkpoint["objective"] not in OBJECTIVES:
        raise ValueError("checkpoint has an unsupported objective")
    incidence = torch.as_tensor(checkpoint["incidence"], dtype=torch.float32)
    problem_ids = _catalog(checkpoint["problem_ids"])
    learner_ids = list(checkpoint["learner_ids"])
    if not learner_ids or len(learner_ids) != len(set(learner_ids)):
        raise ValueError("checkpoint learner_ids must be nonempty and unique")
    if incidence.shape != (len(learner_ids), len(problem_ids)):
        raise ValueError("checkpoint incidence shape does not match its vocabularies")
    if not torch.isfinite(incidence).all() or not torch.logical_or(incidence == 0, incidence == 1).all():
        raise ValueError("checkpoint incidence must contain only finite binary values")
    hidden_dim = checkpoint["hidden_dim"]
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
        raise ValueError("checkpoint hidden_dim must be a positive integer")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping) or any(
        not torch.is_tensor(value) or not torch.isfinite(value).all() for value in state_dict.values()
    ):
        raise ValueError("checkpoint state_dict must contain only finite tensors")
    model = DMF(len(learner_ids), len(problem_ids), hidden_dim)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, incidence.to(device), problem_ids


class DMFRanker:
    """Re-encode a held-out attempt row against the fixed training item columns."""

    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model, self.incidence, self.problem_ids = _checkpoint_model(checkpoint, self.device)
        self.item_index = {pid: i for i, pid in enumerate(self.problem_ids)}

    @torch.no_grad()
    def score(self, problem_ids, observed_problem_ids):
        requested = list(problem_ids)
        if any(pid not in self.item_index for pid in requested):
            raise ValueError("problem_ids must belong to the training catalog")
        row = self.incidence.new_zeros((len(self.problem_ids),))
        for pid in observed_problem_ids:
            index = self.item_index.get(pid)
            if index is not None:
                row[index] = 1.0
        item_columns = self.incidence.T
        user_vector = self.model.user_tower(row.unsqueeze(0))
        item_vectors = self.model.item_tower(item_columns)
        scores = F.cosine_similarity(user_vector.expand_as(item_vectors), item_vectors, dim=-1, eps=1e-8)
        scores = scores.clamp(1e-7, 1 - 1e-7)
        return {pid: float(scores[self.item_index[pid]].cpu()) for pid in requested}


def train_dmf(histories, problem_ids, epochs=30, seed=0, device="cpu", objective="observed_correctness"):
    """Train two ReLU towers with sampled unseen pairs and target-cell masking."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    histories = list(histories)
    problem_ids = _catalog(problem_ids)
    matrix, (users, items, labels) = build_incidence(histories, problem_ids)
    n_users, n_items = matrix.shape
    if not n_users or not len(users):
        raise ValueError("training requires at least one learner and one observed catalog pair")
    if objective == "incidence":
        labels = torch.ones_like(labels)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Initialization uses an isolated RNG so caller randomness remains untouched.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DMF(n_users, n_items)
    device = torch.device(device)
    model.to(device).train()
    matrix = matrix.to(device)
    users, items, labels = users.to(device), items.to(device), labels.to(device)

    # Sample two unseen items per observed pair; they are weak negatives, not known dislikes.
    by_user = defaultdict(list)
    for user, item in zip(users.cpu().tolist(), items.cpu().tolist()):
        by_user[user].append(item)
    sample_users, sample_items, sample_labels = users.tolist(), items.tolist(), labels.tolist()
    for user, attempted in sorted(by_user.items()):
        unseen = torch.tensor(sorted(set(range(n_items)) - set(attempted)), dtype=torch.long)
        if not len(unseen):
            continue
        draws = unseen[torch.randint(len(unseen), (2 * len(attempted),), generator=generator)]
        sample_users.extend([user] * len(draws))
        sample_items.extend(draws.tolist())
        sample_labels.extend([0.0] * len(draws))
    sample_users = torch.tensor(sample_users, dtype=torch.long, device=device)
    sample_items = torch.tensor(sample_items, dtype=torch.long, device=device)
    sample_labels = torch.tensor(sample_labels, dtype=torch.float32, device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
    losses = []
    batch_size = 128
    for _ in range(epochs):
        order = torch.randperm(len(sample_users), generator=generator).to(device)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indexes = order[start:start + batch_size]
            batch_users, batch_items = sample_users[indexes], sample_items[indexes]
            # Mask the exact target in both representations for every supervised pair.
            user_rows, item_columns = _masked_inputs(matrix, batch_users, batch_items)
            predictions = model(user_rows, item_columns)
            loss = F.binary_cross_entropy(predictions, sample_labels[indexes])
            if not torch.isfinite(loss):
                raise FloatingPointError("DMF training loss became nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError("DMF gradient became nonfinite")
            optimizer.step()
            total_loss += float(loss.detach()) * len(indexes)
        losses.append(total_loss / len(sample_users))

    model.eval()
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "problem_ids": problem_ids,
        "learner_ids": [history.learner_id for history in histories],
        "incidence": matrix.detach().cpu(),
        "hidden_dim": 64,
        "history": losses,
        "objective": objective,
    }


def load_dmf(checkpoint, device="cpu"):
    """Load a safe-tensor checkpoint mapping or a torch-saved checkpoint path."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return DMFRanker(checkpoint, device=device)


def score_candidates(checkpoint, observed_problem_ids, device="cpu"):
    """Score every train-catalog item for a held-out learner's observed prefix."""
    ranker = load_dmf(checkpoint, device=device)
    return ranker.score(ranker.problem_ids, observed_problem_ids)
