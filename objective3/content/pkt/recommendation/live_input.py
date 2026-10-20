"""Turn a pasted C++ submission into a history event the frozen model can read.

Uses the reverse-engineered converter in ``preprocessing_live/cpp_to_astnn.py``
(libclang 13, no standard headers, macros expanded with pcpp) and the dataset's ``node_w2v_128`` vocabulary.
The conversion reproduces the dataset's trees exactly on real Codeforces submissions,
including 42 held-out records (see OPTION_C_FEASIBILITY.md, section 5).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import zipfile

from recommendation.data import Interaction

_PRE = Path(__file__).resolve().parents[1] / "preprocessing_live"
if str(_PRE) not in sys.path:
    sys.path.insert(0, str(_PRE))


def converter_available():
    """Return (ok, message). libclang 13 and pcpp are optional dependencies."""
    try:
        import clang.cindex as ci  # noqa: F401
        from cpp_to_astnn import source_to_blocks  # noqa: F401
        source_to_blocks("int main(){return 0;}", "probe.cpp")
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def load_vocab(archive):
    """Vocabulary used by the dataset (token -> id) and the unknown-token id.

    Read from preprocessing_live/node_vocab.json (exported from the dataset's node_w2v_128,
    same order); gensim is only needed if that file is missing.
    """
    vocab_json = _PRE / "node_vocab.json"
    if vocab_json.exists():
        keys = json.loads(vocab_json.read_text(encoding="utf-8"))
        return {k: i for i, k in enumerate(keys)}, len(keys)
    from gensim.models import Word2Vec
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as d:
        z.extract("astnn/node_w2v_128", d)
        kv = Word2Vec.load(str(Path(d) / "astnn" / "node_w2v_128")).wv
    return dict(kv.key_to_index), len(kv.index_to_key)


MAX_CHARS = 60_000      # far above any real competitive-programming submission
TIMEOUT_SECONDS = 20.0  # a few pathological files make libclang/pcpp very slow


class ConversionTimeout(RuntimeError):
    pass


def _blocks_with_timeout(code, timeout):
    """Run the converter in a worker thread; give up waiting after ``timeout`` seconds.

    libclang calls release the GIL, so the app stays responsive. A timed-out worker cannot be
    killed and finishes in the background; the caller simply gets an error.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    from cpp_to_astnn import source_to_blocks
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(source_to_blocks, code, "submission.cpp")
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        raise ConversionTimeout(f"conversion took longer than {timeout:.0f} s; try a shorter or simpler version "
                                "of the code (fewer macros/templates)") from None
    finally:
        pool.shutdown(wait=False)


def code_to_roots(code, key_to_index, unk, timeout=TIMEOUT_SECONDS):
    """Convert source text to dataset-format AST roots plus simple conversion statistics."""
    from cpp_to_astnn import to_ids
    if len(code) > MAX_CHARS:
        raise ValueError(f"the code is {len(code):,} characters; the limit is {MAX_CHARS:,}")
    blocks = _blocks_with_timeout(code, timeout)
    roots = [to_ids(b, key_to_index, unk) for b in blocks]
    total = unknown = 0
    stack = list(roots)
    while stack:
        node = stack.pop()
        total += 1
        unknown += node[0] == unk
        stack.extend(node[1:])
    stats = {"blocks": len(roots), "tokens": total, "unknown_tokens": unknown}
    try:
        stats["summary"] = code_summary(code, blocks)
    except Exception:  # noqa: BLE001 - the summary is informational only
        stats["summary"] = None
    return roots, stats


LOOPS = {"FOR_STMT", "WHILE_STMT", "DO_STMT", "CXX_FOR_RANGE_STMT"}
BIT_OPS = {"BINARY_OPERATOR_&", "BINARY_OPERATOR_|", "BINARY_OPERATOR_^", "BINARY_OPERATOR_<<",
           "BINARY_OPERATOR_>>", "COMPOUND_ASSIGNMENT_OPERATOR_&=", "COMPOUND_ASSIGNMENT_OPERATOR_|=",
           "COMPOUND_ASSIGNMENT_OPERATOR_^=", "COMPOUND_ASSIGNMENT_OPERATOR_<<=", "COMPOUND_ASSIGNMENT_OPERATOR_>>="}
LIBRARY_TYPES = ("map", "unordered_map", "set", "unordered_set", "multiset", "vector", "priority_queue", "queue",
                 "stack", "deque", "pair", "string", "bitset", "cin", "cout")
LIBRARY_CALLS = ("sort", "stable_sort", "lower_bound", "upper_bound", "binary_search", "accumulate", "reverse",
                 "next_permutation", "__gcd", "gcd", "memset", "min", "max", "swap", "abs", "scanf", "printf",
                 "getline", "push_back", "pop_back", "insert", "erase", "find", "count", "begin", "end", "size")


def code_summary(code, blocks):
    """What the model actually receives from this code, and what the preprocessing hides from it.

    Counted over FUNCTION_DECL blocks (they hold the full function bodies, so nothing is counted twice).
    The dataset's preprocessing parses code without the C++ standard headers, so calls into the
    standard library (sort, map, set, cin/cout, ...) are not part of the tree the model sees.
    """
    import re
    funcs = [b for b in blocks if b and b[0] == "FUNCTION_DECL"]
    names = [f[1][1][0] for f in funcs if len(f) > 1 and f[1][0] == "NAME" and len(f[1]) > 1]
    counts = {"loops": 0, "conditions": 0, "array_accesses": 0, "bit_operations": 0, "modulo": 0, "calls_to_own_functions": 0}
    max_depth = 0
    recursive = set()

    def walk(node, depth, owner):
        nonlocal max_depth
        tok = node[0]
        if tok in LOOPS:
            counts["loops"] += 1
            depth += 1
            max_depth = max(max_depth, depth)
        elif tok in ("IF_STMT", "CONDITIONAL_OPERATOR"):
            counts["conditions"] += 1
        elif tok == "ARRAY_SUBSCRIPT_EXPR":
            counts["array_accesses"] += 1
        elif tok in BIT_OPS:
            counts["bit_operations"] += 1
        elif tok in ("BINARY_OPERATOR_%", "COMPOUND_ASSIGNMENT_OPERATOR_%="):
            counts["modulo"] += 1
        elif tok == "CALL_EXPR" and len(node) > 1 and node[1][0] == "NAME" and len(node[1]) > 1:
            callee = node[1][1][0]
            if callee in names:
                counts["calls_to_own_functions"] += 1
                if callee == owner:
                    recursive.add(owner)
        for child in node[1:]:
            if isinstance(child, list) and child:
                walk(child, depth, owner)

    for f, name in zip(funcs, names):
        walk(f, 0, name)
    stripped = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', " ", code, flags=re.S)
    stripped = re.sub(r"^\s*#[^\n]*", " ", stripped, flags=re.M)   # #include <map> is not a use
    hidden = [n for n in LIBRARY_TYPES if re.search(r"(?<![\w.])" + re.escape(n) + r"(?!\w)", stripped)]
    hidden += [n for n in LIBRARY_CALLS if re.search(r"(?<![\w])" + re.escape(n) + r"\s*\(", stripped)]
    from cpp_to_astnn import _expand_macros
    expanded = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', " ", _expand_macros(code), flags=re.S)
    expanded = re.sub(r"^\s*#[^\n]*", " ", expanded, flags=re.M)
    loops_in_source = len(re.findall(r"(?<![\w])(?:for|while)\s*\(", expanded))
    return {"functions": names, "max_loop_nesting": max_depth, "recursive_functions": sorted(recursive),
            **counts, "loops_in_source": loops_in_source, "library_calls_not_visible": hidden}


def next_attempt_probability(last, concept_ids):
    """The checkpoint's own forecast that the next attempt (with these tags) succeeds.

    Exactly the training objective: sigmoid of the mean concept logit over the attempt's tags, taken
    at the learner's latest step. It is computed from the history only -- the model never sees the
    code of the attempt it forecasts, so this is an expectation, not a judgement of the code.
    """
    import math
    logits = [math.log(max(min(last[c - 1], 1 - 1e-6), 1e-6) / (1 - max(min(last[c - 1], 1 - 1e-6), 1e-6)))
              for c in concept_ids]
    return 1 / (1 + math.exp(-sum(logits) / len(logits)))


CUSTOM_PROBLEM_ID = 7152  # valid for the model, absent from the KSAP catalogue (diagnosis ignores the ID)


def make_event(learner_id, after, problem_id, concept_ids, correct, roots):
    """A new Interaction placed after the learner's last observed event."""
    if not concept_ids or any(not 1 <= int(c) <= 37 for c in concept_ids):
        raise ValueError("choose at least one concept tag for the problem")
    if not roots:
        raise ValueError("the code produced no statement blocks (does it contain a function definition?)")
    seq = (after[-1].sequence_index + 1) if after else 0
    return Interaction(learner_id, seq, int(problem_id), tuple(sorted(concept_ids)), int(correct), roots, "live", 0)
