from pathlib import Path
import sys
import time

import pytest

PKT = Path(__file__).resolve().parents[1] / "content" / "pkt"
sys.path.insert(0, str(PKT))
from recommendation import live_input  # noqa: E402
from recommendation.data import Interaction  # noqa: E402
from recommendation.resources import problem_label, problem_url  # noqa: E402


def test_oversized_code_is_rejected_before_parsing():
    with pytest.raises(ValueError, match="limit"):
        live_input.code_to_roots("x" * (live_input.MAX_CHARS + 1), {}, 0)


def test_slow_conversion_times_out(monkeypatch):
    import types
    fake = types.ModuleType("cpp_to_astnn")
    fake.source_to_blocks = lambda code, name: time.sleep(5)
    fake.to_ids = lambda b, k, u: b
    monkeypatch.setitem(sys.modules, "cpp_to_astnn", fake)
    t = time.time()
    with pytest.raises(live_input.ConversionTimeout):
        live_input.code_to_roots("int main(){}", {}, 0, timeout=0.2)
    assert time.time() - t < 2


def test_custom_problem_event_needs_tags():
    prev = [Interaction("x", 4, 10, (1,), 1, [[1]], "valid", 0)]
    ev = live_input.make_event("x", prev, live_input.CUSTOM_PROBLEM_ID, [3, 2], True, [[1, [2]]])
    assert ev.sequence_index == 5 and ev.concept_ids == (2, 3) and ev.problem_id == 7152
    with pytest.raises(ValueError, match="concept"):
        live_input.make_event("x", prev, live_input.CUSTOM_PROBLEM_ID, [], True, [[1]])
    with pytest.raises(ValueError):
        live_input.make_event("x", prev, live_input.CUSTOM_PROBLEM_ID, [40], True, [[1]])


def test_problem_links():
    assert problem_label(61) == "Codeforces 1342A"
    assert problem_url(61) == "https://codeforces.com/problemset/problem/1342/A"
    assert problem_url(5599) == "https://codeforces.com/problemset/problem/111/B"
    assert problem_label(314) == "Problem 314" and problem_url(314) is None


def test_code_summary_reports_visible_structure_and_hidden_library_calls(monkeypatch):
    import types
    fake = types.ModuleType("cpp_to_astnn")
    fake._expand_macros = lambda code: code
    monkeypatch.setitem(sys.modules, "cpp_to_astnn", fake)
    f = ["FUNCTION_DECL", ["NAME", ["f"]], ["TYPE_KIND", ["TYPE", ["int"]]],
         ["COMPOUND_STMT", ["RETURN_STMT", ["CALL_EXPR", ["NAME", ["f"]], ["ARGUMENTS", ["INTEGER_LITERAL", ["1"]]]]]]]
    loop = ["FOR_STMT", ["COMPOUND_STMT", ["WHILE_STMT", ["DECL_REF_EXPR", ["x"]],
            ["COMPOUND_STMT", ["BINARY_OPERATOR_%", ["DECL_REF_EXPR", ["a"]], ["INTEGER_LITERAL", ["2"]]],
             ["ARRAY_SUBSCRIPT_EXPR", ["DECL_REF_EXPR", ["v"]], ["INTEGER_LITERAL", ["0"]]]]]]]
    main = ["FUNCTION_DECL", ["NAME", ["main"]], ["TYPE_KIND", ["TYPE", ["int"]]], ["COMPOUND_STMT", loop]]
    code = """#include <map>
    // sort(a) in a comment does not count
    int f(int n){ return f(1); }
    int main(){ string s; for(;;){ while(x){ a % 2; v[0]; } } for (int i = 0; i < s.size(); i++) {} cout << 1; }"""
    out = live_input.code_summary(code, [f, ["NAME", ["f"]], main, loop])   # sub-blocks must not double count
    assert out["functions"] == ["f", "main"] and out["recursive_functions"] == ["f"]
    assert out["loops"] == 2 and out["max_loop_nesting"] == 2 and out["loops_in_source"] == 3
    assert out["modulo"] == 1 and out["array_accesses"] == 1 and out["calls_to_own_functions"] == 1
    assert set(out["library_calls_not_visible"]) == {"string", "cout", "size"}   # not map (#include) nor sort (comment)


def test_next_attempt_probability_matches_training_objective():
    import math
    last = [0.5] * 37
    last[2], last[4] = 0.8, 0.4
    lg = lambda p: math.log(p / (1 - p))
    want = 1 / (1 + math.exp(-(lg(0.8) + lg(0.4)) / 2))
    assert live_input.next_attempt_probability(last, [3, 5]) == pytest.approx(want)
    assert live_input.next_attempt_probability(last, [1]) == pytest.approx(0.5)
