"""Deterministic KSAP-style filtering and ranking for problem recommendations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


_BUCKETS = ("easy", "moderate", "difficult")


def _mastery_map(mastery):
    if isinstance(mastery, Mapping):
        values = dict(mastery)
    elif isinstance(mastery, Sequence) and not isinstance(mastery, (str, bytes)):
        values = {index: value for index, value in enumerate(mastery, 1)}
    else:
        raise ValueError("mastery must be a concept-ID mapping or sequence")
    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("mastery values must be finite numbers between 0 and 1")
    return values


def _prerequisites(graph, concept):
    """Return transitive prerequisite ancestors of concept in graph order."""
    edges = graph.get("edges", []) if isinstance(graph, Mapping) else []
    parents = {}
    for edge in edges:
        if isinstance(edge, Mapping) and "source" in edge and "target" in edge:
            parents.setdefault(edge["target"], []).append(edge["source"])
    found, stack = set(), list(parents.get(concept, ()))
    while stack:
        node = stack.pop()
        if node not in found:
            found.add(node)
            stack.extend(parents.get(node, ()))
    return found


def recommend(catalog, ordered_concepts, preference, scores, attempted_problem_ids,
              mastery, graph, top_k_per_concept=3, limit=10, allow_fallback=False):
    """Filter unseen tagged problems by preference, then rank by DMF score.

    Fallbacks are opt-in: neighboring difficulty buckets are considered when the
    preferred bucket is empty. If the selected group has any missing DMF scores,
    popularity ranks the entire group so DMF and attempt counts are never mixed.
    """
    if not isinstance(catalog, Mapping) or not isinstance(preference, Mapping) or not isinstance(scores, Mapping):
        raise ValueError("catalog, preference and scores must be mappings")
    if isinstance(top_k_per_concept, bool) or not isinstance(top_k_per_concept, int) or top_k_per_concept < 1:
        raise ValueError("top_k_per_concept must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(ordered_concepts, Sequence) or isinstance(ordered_concepts, (str, bytes)):
        raise ValueError("ordered_concepts must be a sequence")
    if isinstance(attempted_problem_ids, (str, bytes)):
        raise ValueError("attempted_problem_ids must be an iterable of problem IDs")

    attempted = set(attempted_problem_ids)
    attempted_strings = {str(problem_id) for problem_id in attempted}
    mastery_values = _mastery_map(mastery)
    concepts = list(dict.fromkeys(ordered_concepts))
    if any(isinstance(concept, bool) or not isinstance(concept, int) or concept < 1 for concept in concepts):
        raise ValueError("concept IDs must be positive integers")
    if not concepts:
        return {"recommendations": [], "empty_reason": "no weak concepts", "fallback_used": False}

    preferred = preference.get("selected_bucket")
    if preferred not in _BUCKETS:
        raise ValueError("preference.selected_bucket must be easy, moderate or difficult")
    preferred_index = _BUCKETS.index(preferred)
    output, seen = [], set()
    any_fallback = False

    for concept in concepts:
        candidates = []
        for key, row in catalog.items():
            if not isinstance(row, Mapping):
                continue
            problem_id = row.get("problem_id", key)
            if problem_id in attempted or str(problem_id) in attempted_strings or problem_id in seen:
                continue
            tags = list(row.get("concept_ids", ()))
            if any(isinstance(tag, bool) or not isinstance(tag, int) or tag < 1 for tag in tags):
                raise ValueError("catalog concept IDs must be positive integers")
            if concept not in tags:
                continue
            bucket = row.get("bucket", row.get("difficulty_bucket"))
            if bucket not in _BUCKETS:
                continue
            candidates.append((problem_id, row, bucket))

        selected = [item for item in candidates if item[2] == preferred]
        filter_mode = "strict"
        if not selected and allow_fallback and candidates:
            distances = [abs(_BUCKETS.index(item[2]) - preferred_index) for item in candidates]
            nearest = min(distances)
            selected = [item for item in candidates if abs(_BUCKETS.index(item[2]) - preferred_index) == nearest]
            filter_mode = "neighbor_bucket_fallback"
            any_fallback = True
        if not selected:
            continue

        # Avoid mixing incompatible DMF and popularity scales within a concept.
        def rank_key(item):
            score = scores[item[0]] if item[0] in scores else scores[str(item[0])]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("DMF scores must be finite numbers")
            return (-float(score), item[0])

        has_all_scores = all(item[0] in scores or str(item[0]) in scores for item in selected)
        if has_all_scores:
            selected.sort(key=rank_key)
            score_source = "dmf"
        elif allow_fallback:
            selected.sort(key=lambda item: (-int(item[1].get("attempts", 0)), item[0]))
            score_source = "train_popularity"
            filter_mode = "train_popularity_fallback" if filter_mode == "strict" else filter_mode + "+train_popularity_fallback"
            any_fallback = True
        else:
            # Strict mode only returns candidates the learned ranker can order.
            selected = [item for item in selected if item[0] in scores or str(item[0]) in scores]
            if not selected:
                continue
            selected.sort(key=rank_key)
            score_source = "dmf"

        prerequisites = _prerequisites(graph, concept)
        unmastered = sorted(node for node in prerequisites
                            if mastery_values.get(node, 1.0) < 0.5)
        for problem_id, row, bucket in selected[:top_k_per_concept]:
            if problem_id in seen:
                continue
            seen.add(problem_id)
            tags = list(row.get("concept_ids", ()))
            value = (scores.get(problem_id, scores.get(str(problem_id))) if score_source == "dmf"
                     else int(row.get("attempts", 0)))
            output.append({
                "problem_id": problem_id,
                "concept_ids": tags,
                "target_concept": concept,
                "target_concept_mastery": mastery_values.get(concept),
                "difficulty": row.get("difficulty"),
                "difficulty_training_attempts": row.get("attempts"),
                "difficulty_training_correct": row.get("correct"),
                "difficulty_bucket": bucket,
                "score": float(value),
                "score_source": score_source,
                "filter_mode": filter_mode,
                "other_weak_concepts": sorted(tag for tag in tags if tag != concept and mastery_values.get(tag, 1.0) < 0.5),
                "unmastered_prerequisites": unmastered,
            })
            if len(output) >= limit:
                return {"recommendations": output, "empty_reason": None, "fallback_used": any_fallback}

    return {"recommendations": output, "empty_reason": None if output else "no eligible problems",
            "fallback_used": any_fallback}
