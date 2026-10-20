"""Ranking metrics for KSAP recommendations."""

from __future__ import annotations

import math


def ranking_metrics(recommended_ids, relevant_ids, k=10):
    """Return standard top-k metrics and KSAP paper's returned-list metrics.

    The paper calls precision over the returned list HR (Eq. 21), and its NG
    formula (Eq. 22) divides discounted relevant hits by returned-list length.
    Standard HR@k and NDCG@k are included separately for clearer comparisons.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    recommended = list(recommended_ids)
    if len(recommended) != len(set(recommended)):
        raise ValueError("recommended_ids must not contain duplicates")
    relevant = set(relevant_ids)
    ranked = recommended[:k]
    hits = [item in relevant for item in ranked]
    hit_count = sum(hits)
    returned_count = len(ranked)
    ideal_count = min(k, len(relevant))
    dcg = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_count))
    paper_dcg = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
    return {
        "hit_rate": float(hit_count > 0),
        "precision_at_k": hit_count / k,
        "recall_at_k": hit_count / len(relevant) if relevant else None,
        "ndcg_at_k": dcg / idcg if idcg else None,
        "paper_hr": hit_count / returned_count if returned_count else None,
        "paper_ng": paper_dcg / returned_count if returned_count else None,
    }
