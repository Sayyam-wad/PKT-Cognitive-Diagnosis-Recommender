"""Empirical problem difficulty and explicit DPM preferences."""

from __future__ import annotations

from collections.abc import Mapping
import math


BUCKETS = ("easy", "moderate", "difficult")


def _binary(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")


def difficulty_bucket(difficulty):
    if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)) or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("difficulty must be a finite number between 0 and 1")
    if difficulty <= 0.3:
        return "easy"
    if difficulty < 0.7:
        return "moderate"
    return "difficult"


def fit_catalog(histories):
    """Build a train-history catalog keyed by stringified internal problem ID."""
    stats = {}
    for history in histories:
        for interaction in history.interactions:
            problem_id = interaction.problem_id
            if isinstance(problem_id, bool) or not isinstance(problem_id, int) or problem_id <= 0:
                raise ValueError("problem IDs must be positive integers")
            _binary(interaction.correct, "interaction correctness")
            row = stats.setdefault(problem_id, {"concept_ids": set(), "attempts": 0, "correct": 0})
            for concept in interaction.concept_ids:
                if isinstance(concept, bool) or not isinstance(concept, int) or concept < 1:
                    raise ValueError("concept IDs must be positive integers")
            row["concept_ids"].update(interaction.concept_ids)
            row["attempts"] += 1
            row["correct"] += interaction.correct
    catalog = {}
    for problem_id, row in sorted(stats.items()):
        difficulty = 1 - row["correct"] / row["attempts"]
        catalog[str(problem_id)] = {"problem_id": problem_id, "concept_ids": sorted(row["concept_ids"]),
                                    "attempts": row["attempts"], "correct": row["correct"],
                                    "difficulty": difficulty, "bucket": difficulty_bucket(difficulty)}
    return catalog


def difficulty_preference(history_interactions, catalog, w1=0.5, w2=0.5):
    """Compute explicit bucket scores using only history items present in catalog."""
    for name, value in (("w1", w1), ("w2", w2)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isclose(w1 + w2, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("w1 and w2 must sum to 1")
    if not isinstance(catalog, Mapping):
        raise ValueError("catalog must be a problem-ID mapping")
    counts = {bucket: 0 for bucket in BUCKETS}
    correct_counts = {bucket: 0 for bucket in BUCKETS}
    known_attempts = 0
    unknown_attempts = 0
    for interaction in history_interactions:
        if isinstance(interaction.problem_id, bool) or not isinstance(interaction.problem_id, int) or interaction.problem_id <= 0:
            raise ValueError("problem IDs must be positive integers")
        _binary(interaction.correct, "interaction correctness")
        item = catalog.get(str(interaction.problem_id))
        if item is None:
            unknown_attempts += 1
            continue
        if not isinstance(item, Mapping):
            raise ValueError("catalog entries must be mappings")
        bucket = item.get("bucket")
        if bucket not in counts:
            raise ValueError(f"catalog item has invalid difficulty bucket: {bucket!r}")
        counts[bucket] += 1
        correct_counts[bucket] += interaction.correct
        known_attempts += 1
    scores = {}
    for bucket in BUCKETS:
        rate = counts[bucket] / known_attempts if known_attempts else 0.0
        accuracy = correct_counts[bucket] / counts[bucket] if counts[bucket] else 0.0
        scores[bucket] = w1 * rate + w2 * accuracy
    if known_attempts == 0:
        selected = "moderate"
    else:
        # Explicit order resolves ties in favor of moderate, then easy, then difficult.
        selected = max(("moderate", "easy", "difficult"), key=lambda bucket: (scores[bucket], -("moderate", "easy", "difficult").index(bucket)))
    return {"scores": scores, "selected_bucket": selected, "counts": counts,
            "correct_counts": correct_counts, "unknown_attempts": unknown_attempts}
