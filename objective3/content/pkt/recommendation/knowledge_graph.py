"""Train-history knowledge graph mining and weak-concept ordering."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _mastery_values(mean_mastery, n):
    if isinstance(mean_mastery, Mapping):
        values = [mean_mastery.get(i) for i in range(1, n + 1)]
    elif isinstance(mean_mastery, Sequence) and not isinstance(mean_mastery, (str, bytes)):
        if len(mean_mastery) != n:
            raise ValueError(f"mean_mastery must contain {n} values")
        values = list(mean_mastery)
    else:
        raise ValueError("mean_mastery must be a sequence or concept-ID mapping")
    if any(value is None for value in values):
        raise ValueError("mean_mastery must contain every concept")
    result = [_finite(value, "mean_mastery value") for value in values]
    if any(not 0 <= value <= 1 for value in result):
        raise ValueError("mastery values must be between 0 and 1")
    return result


def _path_exists(adjacency, start, goal):
    stack, seen = [start], set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(adjacency[node] - seen)
    return False


def _topological(nodes, edges):
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    return order if len(order) == len(nodes) else None


def fit_graph(histories, mean_mastery, num_concepts=37, support=0.7, confidence=0.75):
    """Mine learner-wide incorrect-concept baskets and orient rules as a DAG.

    Each learner contributes one basket, even when it has no incorrect concepts.
    Statistical rules use standard support, confidence and lift. Auxiliary edges
    only order unresolved pairs and carry no fabricated association statistics.
    """
    if isinstance(num_concepts, bool) or not isinstance(num_concepts, int) or num_concepts < 1:
        raise ValueError("num_concepts must be a positive integer")
    support = _finite(support, "support")
    confidence = _finite(confidence, "confidence")
    if not 0 <= support <= 1 or not 0 <= confidence <= 1:
        raise ValueError("support and confidence must be between 0 and 1")
    means = _mastery_values(mean_mastery, num_concepts)

    baskets = []
    for history in histories:
        incorrect = set()
        for interaction in history.interactions:
            if isinstance(interaction.correct, bool) or not isinstance(interaction.correct, int) or interaction.correct not in (0, 1):
                raise ValueError("interaction correctness must be 0 or 1")
            for concept in interaction.concept_ids:
                if isinstance(concept, bool) or not isinstance(concept, int) or not 1 <= concept <= num_concepts:
                    raise ValueError(f"concept IDs must be in 1..{num_concepts}")
                if interaction.correct == 0:
                    incorrect.add(concept)
        baskets.append(incorrect)

    n = len(baskets)
    item_count = [sum(i in basket for basket in baskets) for i in range(num_concepts + 1)]
    pair_count = defaultdict(int)
    for basket in baskets:
        ordered = sorted(basket)
        for index, a in enumerate(ordered):
            for b in ordered[index + 1:]:
                pair_count[a, b] += 1

    nodes = list(range(1, num_concepts + 1))
    edges, rejected, ties = [], [], []
    adjacency = {node: set() for node in nodes}
    for (a, b), count in sorted(pair_count.items()):
        pair_support = count / n if n else 0.0
        if pair_support < support or not n:
            continue
        if means[a - 1] == means[b - 1]:
            ties.append([a, b])
            continue
        source, target = (a, b) if means[a - 1] > means[b - 1] else (b, a)
        source_support = item_count[source] / n
        target_support = item_count[target] / n
        edge_confidence = pair_support / source_support if source_support else 0.0
        lift = edge_confidence / target_support if target_support else 0.0
        if edge_confidence < confidence:
            continue
        edge = {"source": source, "target": target, "kind": "association", "support": pair_support,
                "confidence": edge_confidence, "lift": lift}
        if _path_exists(adjacency, target, source):
            rejected.append(edge)
        else:
            edges.append(edge)
            adjacency[source].add(target)

    for a in nodes:
        for b in nodes:
            if means[a - 1] <= means[b - 1] or _path_exists(adjacency, a, b) or _path_exists(adjacency, b, a):
                continue
            edges.append({"source": a, "target": b, "kind": "auxiliary"})
            adjacency[a].add(b)

    if _topological(nodes, edges) is None:
        raise RuntimeError("internal error: graph construction produced a cycle")
    return {
        "nodes": nodes,
        "edges": edges,
        "rejected_cycles": rejected,
        "thresholds": {"support": support, "confidence": confidence},
        "stats": {"learners": n, "empty_baskets": sum(not basket for basket in baskets),
                  "association_edges": sum(edge["kind"] == "association" for edge in edges),
                  "auxiliary_edges": sum(edge["kind"] == "auxiliary" for edge in edges),
                  "mastery_ties": ties},
    }


def weak_sequence(graph, mastery, threshold=0.5, weak=None, prereq_below=None):
    """Expand weak concepts through qualifying association predecessors, then order them.

    ``weak`` optionally gives the weak set explicitly (otherwise: mastery below ``threshold``).
    ``prereq_below``: if given, a prerequisite is added only when the learner's mastery of it is below
    this value (a prerequisite the learner already knows is not recommended again).
    """
    threshold = _finite(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list) or any(
            isinstance(node, bool) or not isinstance(node, int) or node < 1 for node in nodes):
        raise ValueError("graph must contain positive integer nodes and an edge list")
    if len(nodes) != len(set(nodes)):
        raise ValueError("graph nodes must be unique")
    if isinstance(mastery, Mapping):
        values = {node: _finite(mastery.get(node), "mastery value") for node in nodes}
    elif isinstance(mastery, Sequence) and not isinstance(mastery, (str, bytes)) and len(mastery) == len(nodes):
        values = {node: _finite(value, "mastery value") for node, value in zip(nodes, mastery)}
    else:
        raise ValueError("mastery must map graph nodes or have one value per graph node")
    if any(not 0 <= value <= 1 for value in values.values()):
        raise ValueError("mastery values must be between 0 and 1")
    thresholds = graph.get("thresholds", {})
    confidence_threshold = _finite(thresholds.get("confidence", 0.75), "graph confidence threshold")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("graph confidence threshold must be between 0 and 1")
    adjacency = {node: set() for node in nodes}
    predecessors = defaultdict(list)
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source not in adjacency or target not in adjacency:
            raise ValueError("edge endpoints must exist in graph nodes")
        if target in adjacency[source]:
            continue
        adjacency[source].add(target)
        if edge.get("kind") == "association":
            conf = _finite(edge.get("confidence"), "edge confidence")
            lift = _finite(edge.get("lift"), "edge lift")
            if conf > confidence_threshold and lift > 1:
                predecessors[target].append(source)
    order = _topological(nodes, edges)
    if order is None:
        raise ValueError("graph must be acyclic")
    if weak is None:
        weak = {node for node, value in values.items() if value < threshold}
    else:
        weak = set(weak)
        if not weak <= set(nodes):
            raise ValueError("explicit weak concepts must be graph nodes")
    expanded = set(weak)
    stack = list(weak)
    while stack:
        for source in predecessors[stack.pop()]:
            if prereq_below is not None and values[source] >= prereq_below:
                continue
            if source not in expanded:
                expanded.add(source)
                stack.append(source)
    return {"weak_concepts": sorted(weak), "expanded_concepts": sorted(expanded),
            "ordered_concepts": [node for node in order if node in expanded]}
