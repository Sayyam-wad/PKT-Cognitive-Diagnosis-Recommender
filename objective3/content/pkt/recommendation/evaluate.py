"""Prefix-only validation replay; future attempts are labels, never features."""

import math
from statistics import mean

from recommendation.metrics import ranking_metrics
from recommendation.ranker import recommend


def prefix_at(history, cutoff):
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or not 0 <= cutoff <= len(history.interactions):
        raise ValueError("cutoff must be an observed interaction count within the history")
    return history.interactions[:cutoff]


def evaluate_recommendations(recommender, histories, fraction=0.8, k=10):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    rows, skipped = [], []
    fitted = set(recommender.manifest["training_learners"])
    catalog = recommender.catalog
    for history in histories:
        if history.learner_id in fitted:
            raise ValueError("evaluation learners overlap with fitted learners")
        if len(history.interactions) < 2:
            skipped.append(history.learner_id)
            continue
        cutoff = max(1, min(len(history.interactions) - 1, int(len(history.interactions) * fraction)))
        observed = prefix_at(history, cutoff)
        attempted = {event.problem_id for event in observed}
        output = recommender.recommend(observed, history.learner_id, limit=k)
        values = [output["mastery"][str(i)] for i in range(1, 38)]
        scores = recommender.dmf.score(recommender.dmf.problem_ids, attempted)
        fallback = recommend(catalog, output["ordered_concepts"], output["difficulty_preference"],
                             scores, attempted, values, recommender.graph, limit=k, allow_fallback=True)
        candidates = [int(pid) for pid in catalog if int(pid) not in attempted]
        popularity = sorted(candidates, key=lambda pid: (-catalog[str(pid)]["attempts"], pid))[:k]
        dmf_only = sorted(candidates, key=lambda pid: (-scores[pid], pid))[:k]
        lists = {
            "ksap_strict": [row["problem_id"] for row in output["recommendations"]],
            "ksap_fallback": [row["problem_id"] for row in fallback["recommendations"]],
            "train_popularity": popularity, "dmf_only": dmf_only,
        }
        # Only after producing every list do we inspect the withheld suffix.
        relevant = {event.problem_id for event in history.interactions[cutoff:]} - attempted
        reachable = relevant & set(candidates)
        row = {
            "learner_id": history.learner_id, "cutoff": cutoff,
            "future_unseen_items": sorted(relevant), "reachable_future_items": len(reachable),
            "out_of_catalog_future_items": len(relevant - reachable),
            "recommendation_details": output, "fallback_used": fallback["fallback_used"],
            "methods": {},
        }
        for method, ids in lists.items():
            if len(ids) != len(set(ids)) or set(ids) & attempted:
                raise RuntimeError("recommendation list contains duplicates or observed items")
            row["methods"][method] = {
                "problem_ids": ids, "returned": len(ids),
                "all_future_metrics": ranking_metrics(ids, relevant, k),
                "reachable_future_metrics": ranking_metrics(ids, reachable, k),
            }
        rows.append(row)
    if not rows:
        raise ValueError("no evaluation histories have an observed prefix and future suffix")
    summary = {}
    for method in rows[0]["methods"]:
        metrics = {}
        for scope in ("all_future_metrics", "reachable_future_metrics"):
            fields = rows[0]["methods"][method][scope]
            metrics[scope] = {}
            for key in fields:
                values = [r["methods"][method][scope][key] for r in rows
                          if r["methods"][method][scope][key] is not None]
                metrics[scope][key] = mean(values) if values else None
        summary[method] = {
            **metrics, "empty_result_rate": mean(r["methods"][method]["returned"] == 0 for r in rows),
            "mean_returned": mean(r["methods"][method]["returned"] for r in rows),
        }
    return {
        "protocol": "one prefix at fraction of record order per held-out learner; macro learner means",
        "fraction": fraction, "k": k, "evaluated_learners": len(rows), "skipped_learners": skipped,
        "learners_with_future_unseen_items": sum(bool(r["future_unseen_items"]) for r in rows),
        "learners_with_reachable_future_items": sum(r["reachable_future_items"] > 0 for r in rows),
        "out_of_catalog_future_items": sum(r["out_of_catalog_future_items"] for r in rows),
        "caveat": "Event timestamps are not available (submission order is used). Offline agreement is not learning gain.",
        "summary": summary, "snapshots": rows,
    }
