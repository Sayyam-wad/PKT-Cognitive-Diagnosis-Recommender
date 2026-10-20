from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "content" / "pkt"))
from recommendation import resources  # noqa: E402
from scripts.fetch_cf_practice import build  # noqa: E402


def test_practice_link_rating_band():
    assert resources.practice_link(8) == "https://codeforces.com/problemset?tags=greedy"
    assert resources.practice_link(8, "moderate") == "https://codeforces.com/problemset?tags=greedy%2C1300-1800"
    assert resources.resources_for(6, "difficult")[-1]["title"].endswith("rated 1800-2400")


def test_problem_url_from_code():
    assert resources.problem_url_from_code("1342A") == "https://codeforces.com/problemset/problem/1342/A"
    assert resources.problem_url_from_code("1352B2") == "https://codeforces.com/problemset/problem/1352/B2"


def test_build_and_select(monkeypatch):
    api = {"status": "OK", "result": {
        "problems": [
            {"contestId": 1, "index": "A", "name": "P1", "rating": 900, "tags": ["greedy", "math"]},
            {"contestId": 2, "index": "B", "name": "P2", "rating": 1000, "tags": ["greedy"]},
            {"contestId": 3, "index": "C", "name": "P3", "rating": 1500, "tags": ["greedy"]},
            {"contestId": 4, "index": "D", "name": "NoRating", "tags": ["greedy"]},
            {"contestId": 5, "index": "E", "name": "Other", "rating": 900, "tags": ["not-a-tag"]}],
        "problemStatistics": [
            {"contestId": 1, "index": "A", "solvedCount": 10}, {"contestId": 2, "index": "B", "solvedCount": 50},
            {"contestId": 3, "index": "C", "solvedCount": 99}]}}
    data = build(api)
    assert [r[0] for r in data["problems"]["greedy"]] == ["3C", "2B", "1A"]     # most solved first, unrated dropped
    assert [r[0] for r in data["problems"]["math"]] == ["1A"]
    monkeypatch.setattr(resources, "CF_PRACTICE", {t: [tuple(p) for p in v] for t, v in data["problems"].items()})
    easy = resources.practice_problems(8, "easy")
    assert [p["code"] for p in easy] == ["2B", "1A"]                            # band 800-1300 only
    assert [p["code"] for p in resources.practice_problems(8, "easy", exclude={"2B"})] == ["1A"]
    assert easy[0]["url"] == "https://codeforces.com/problemset/problem/2/B"
    monkeypatch.setattr(resources, "CF_PRACTICE", {})
    assert resources.practice_problems(8, "easy") == []


def test_study_plan():
    from types import SimpleNamespace as E
    mastery = [0.9] * 37
    mastery[2], mastery[5], mastery[7] = 0.6, 0.2, 0.3          # math 0.6, dp 0.2, greedy 0.3
    recs = [{"target_concept": 3}, {"target_concept": 3}, {"target_concept": 6}, {"target_concept": 8}]
    prefix = [E(concept_ids=(6,), correct=0), E(concept_ids=(6, 3), correct=1)]
    graph = {"edges": [{"source": 3, "target": 6, "kind": "association"},
                       {"source": 3, "target": 8, "kind": "auxiliary"}]}
    plan = resources.study_plan(recs, [3, 6, 8, 15], weak={6, 8, 15}, mastery=mastery, prefix=prefix, graph=graph)
    assert [s["concept"] for s in plan["steps"]] == [3, 6, 8]
    math_step, dp_step = plan["steps"][0], plan["steps"][1]
    assert math_step["role"] == "prerequisite" and math_step["leads_to"] == [6] and math_step["ranks"] == [1, 2]
    assert dp_step["role"] == "weak" and (dp_step["attempts"], dp_step["solved"]) == (2, 1)
    assert plan["later"] == [15]
    assert [c for c, _ in plan["weakest"]] == [6, 8, 15]
    assert resources.record_text(0, 0) == "not attempted yet"
    assert resources.study_plan([], [], set(), mastery, []) == {"weakest": [], "steps": [], "later": []}
