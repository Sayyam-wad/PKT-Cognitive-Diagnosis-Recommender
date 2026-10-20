"""Concept names and curated learning material for the 37 Codeforces tags.

Concept names come from ``data/tag_table.csv`` of the source repository
https://github.com/pkt-attn/pkt-attn (ids 1-37). Reading links were curated by
hand (CP-Algorithms / Codeforces); practice links are Codeforces problemset tag
filters. The mapping is editorial, not learned from data.
"""

from urllib.parse import quote

NUM_CONCEPTS = 37

CONCEPT_NAMES = {
    1: "sortings", 2: "dfs and similar", 3: "math", 4: "dsu", 5: "graphs", 6: "dp",
    7: "implementation", 8: "greedy", 9: "binary search", 10: "two pointers",
    11: "brute force", 12: "constructive algorithms", 13: "combinatorics", 14: "bitmasks",
    15: "strings", 16: "data structures", 17: "shortest paths", 18: "number theory",
    19: "expression parsing", 20: "trees", 21: "hashing", 22: "geometry",
    23: "ternary search", 24: "graph matchings", 25: "games", 26: "divide and conquer",
    27: "probabilities", 28: "flows", 29: "string suffix structures", 30: "2-sat",
    31: "chinese remainder theorem", 32: "interactive", 33: "*special problem",
    34: "schedules", 35: "fft", 36: "matrices", 37: "meet-in-the-middle",
}

CPA = "https://cp-algorithms.com/"
ITMO = "https://codeforces.com/edu/course/2"
UG = "https://usaco.guide/"
READING = {
    1: [("Introduction to Sorting (USACO Guide)", UG + "bronze/intro-sorting"),
        ("Custom Comparators and Coordinate Compression (USACO Guide)", UG + "silver/sorting-custom")],
    7: [("Simulation (USACO Guide)", UG + "bronze/simulation")],
    8: [("Introduction to Greedy Algorithms (USACO Guide)", UG + "bronze/intro-greedy"),
        ("Greedy Algorithms with Sorting (USACO Guide)", UG + "silver/greedy-sorting")],
    11: [("Basic Complete Search (USACO Guide)", UG + "bronze/intro-complete")],
    12: [("Ad Hoc Problems (USACO Guide)", UG + "bronze/ad-hoc")],
    27: [("Sums and Expected Value, part 1 (Errichto, Codeforces blog)", "https://codeforces.com/blog/entry/62690")],
    37: [("Meet in the Middle (USACO Guide)", UG + "gold/meet-in-the-middle")],
    2: [("Depth First Search (CP-Algorithms)", CPA + "graph/depth-first-search.html")],
    3: [("Euclidean algorithm for GCD (CP-Algorithms)", CPA + "algebra/euclid-algorithm.html"),
        ("Binary exponentiation (CP-Algorithms)", CPA + "algebra/binary-exp.html")],
    4: [("Disjoint Set Union (CP-Algorithms)", CPA + "data_structures/disjoint_set_union.html")],
    5: [("Breadth-first search (CP-Algorithms)", CPA + "graph/breadth-first-search.html")],
    6: [("Introduction to Dynamic Programming (CP-Algorithms)", CPA + "dynamic_programming/intro-to-dp.html")],
    9: [("Binary Search (CP-Algorithms)", CPA + "num_methods/binary_search.html"),
        ("ITMO Academy pilot course (Codeforces EDU)", ITMO)],
    10: [("ITMO Academy pilot course: two pointers lesson (Codeforces EDU)", ITMO)],
    13: [("Binomial coefficients (CP-Algorithms)", CPA + "combinatorics/binomial-coefficients.html")],
    14: [("Enumerating submasks of a bitmask (CP-Algorithms)", CPA + "algebra/all-submasks.html")],
    15: [("Prefix function / KMP (CP-Algorithms)", CPA + "string/prefix-function.html")],
    16: [("Segment Tree (CP-Algorithms)", CPA + "data_structures/segment_tree.html"),
         ("ITMO Academy pilot course (Codeforces EDU)", ITMO)],
    17: [("Dijkstra's algorithm (CP-Algorithms)", CPA + "graph/dijkstra.html")],
    18: [("Sieve of Eratosthenes (CP-Algorithms)", CPA + "algebra/sieve-of-eratosthenes.html")],
    19: [("Expression parsing (CP-Algorithms)", CPA + "string/expression_parsing.html")],
    20: [("Lowest Common Ancestor (CP-Algorithms)", CPA + "graph/lca.html")],
    21: [("String hashing (CP-Algorithms)", CPA + "string/string-hashing.html")],
    22: [("Basic geometry (CP-Algorithms)", CPA + "geometry/basic-geometry.html")],
    23: [("Ternary Search (CP-Algorithms)", CPA + "num_methods/ternary_search.html")],
    24: [("Kuhn's algorithm, bipartite matching (CP-Algorithms)", CPA + "graph/kuhn_maximum_bipartite_matching.html")],
    25: [("Sprague-Grundy theorem, Nim (CP-Algorithms)", CPA + "game_theory/sprague-grundy-nim.html")],
    26: [("Divide and Conquer DP (CP-Algorithms)", CPA + "dynamic_programming/divide-and-conquer-dp.html")],
    28: [("Edmonds-Karp maximum flow (CP-Algorithms)", CPA + "graph/edmonds_karp.html")],
    29: [("Suffix Array (CP-Algorithms)", CPA + "string/suffix-array.html"),
         ("ITMO Academy pilot course (Codeforces EDU)", ITMO)],
    30: [("2-SAT (CP-Algorithms)", CPA + "graph/2SAT.html")],
    31: [("Chinese Remainder Theorem (CP-Algorithms)", CPA + "algebra/chinese-remainder-theorem.html")],
    32: [("Interactive Problems: Guide for Participants (Codeforces)", "https://codeforces.com/blog/entry/45307")],
    34: [("Scheduling jobs on one machine (CP-Algorithms)", CPA + "schedules/schedule_one_machine.html")],
    35: [("Fast Fourier transform (CP-Algorithms)", CPA + "algebra/fft.html")],
    36: [("Binary exponentiation incl. matrices (CP-Algorithms)", CPA + "algebra/binary-exp.html"),
         ("Gauss method for linear systems (CP-Algorithms)", CPA + "linear_algebra/linear-system-gauss.html")],
}


def concept_name(cid):
    return CONCEPT_NAMES.get(cid, f"concept {cid}")


def _load_problem_codes():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "results" / "problem_codes.json"
    try:
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    except (OSError, ValueError):
        return {}


# internal problem ID -> Codeforces code (e.g. 61 -> "1342A"), recovered through the Codeforces API
PROBLEM_CODES = _load_problem_codes()


def problem_label(pid):
    code = PROBLEM_CODES.get(int(pid))
    return f"Codeforces {code}" if code else f"Problem {pid}"


def problem_url(pid):
    """Codeforces URL for identified problems, else None."""
    code = PROBLEM_CODES.get(int(pid))
    if not code:
        return None
    return problem_url_from_code(code)


# Codeforces rating band shown for each difficulty bucket of the recommended problem (editorial choice).
RATING_BANDS = {"easy": (800, 1300), "moderate": (1300, 1800), "difficult": (1800, 2400)}


def practice_link(cid, bucket=None):
    """Codeforces problemset filtered to the concept's tag (and the bucket's rating band, if given)."""
    tags = CONCEPT_NAMES[cid]
    if bucket in RATING_BANDS:
        lo, hi = RATING_BANDS[bucket]
        tags += f",{lo}-{hi}"
    return "https://codeforces.com/problemset?tags=" + quote(tags)


def _load_cf_practice():
    """Optional list of real Codeforces problems per tag (scripts/fetch_cf_practice.py)."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "results" / "cf_practice.json"
    try:
        data = json.loads(path.read_text())
        return {tag: [tuple(p) for p in probs] for tag, probs in data["problems"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


# tag name -> [(code, name, rating, solved_count), ...], most solved first
CF_PRACTICE = _load_cf_practice()


def practice_problems(cid, bucket=None, exclude=(), n=3):
    """Up to n real Codeforces problems with the concept's tag in the bucket's rating band.

    Most-solved first; ``exclude`` = Codeforces codes the learner already attempted.
    Empty when cf_practice.json has not been fetched.
    """
    lo, hi = RATING_BANDS.get(bucket, (0, 10 ** 4))
    out = []
    for code, name, rating, solved in CF_PRACTICE.get(CONCEPT_NAMES.get(cid, ""), []):
        if lo <= rating <= hi and code not in exclude:
            out.append({"code": code, "name": name, "rating": rating, "solved": solved,
                        "url": problem_url_from_code(code)})
            if len(out) == n:
                break
    return out


def problem_url_from_code(code):
    contest = code[:len(code) - len(code.lstrip("0123456789"))]
    return f"https://codeforces.com/problemset/problem/{contest}/{code[len(contest):]}"


def resources_for(cid, bucket=None):
    """Reading material (may be empty) plus a Codeforces practice filter for the concept."""
    items = [{"type": "read", "title": t, "url": u} for t, u in READING.get(cid, [])]
    band = f" rated {RATING_BANDS[bucket][0]}-{RATING_BANDS[bucket][1]}" if bucket in RATING_BANDS else ""
    items.append({"type": "practice", "title": f"Codeforces problems tagged '{concept_name(cid)}'{band}",
                  "url": practice_link(cid, bucket)})
    return items


def explain(rec, pref, mastery, weak=None):
    """One human-readable sentence explaining a recommendation row from ranker.recommend.

    ``weak``: the diagnosed weak concepts. Targets outside it were added by the knowledge graph as
    prerequisites of weak concepts and are described as such.
    """
    target = rec["target_concept"]
    if weak is not None and target not in weak:
        parts = [f"Builds '{concept_name(target)}' (estimated mastery {mastery[target - 1]:.2f}), which the "
                 "knowledge graph places before your weak concepts"]
    else:
        parts = [f"Practises your weak concept '{concept_name(target)}' (estimated mastery {mastery[target - 1]:.2f})"]
    bucket = rec["difficulty_bucket"]
    if bucket == pref.get("selected_bucket"):
        parts.append(f"{bucket} difficulty, which matches your attempting preference")
    else:
        parts.append(f"{bucket} difficulty")
    if rec.get("unmastered_prerequisites"):
        names = ", ".join(concept_name(c) for c in rec["unmastered_prerequisites"][:3])
        parts.append(f"note: prerequisite concepts still weak ({names})")
    if rec.get("score_source") == "dmf":
        parts.append("learners with similar histories attempted it")
    return "; ".join(parts) + "."


def study_plan(recs, ordered_concepts, weak, mastery, prefix, graph=None, max_items=5):
    """What the learner should work on, in order, derived from one recommendation result.

    Returns {"weakest": [(concept, mastery), ...] (the 3 lowest-scored weak concepts),
             "steps": [{"concept", "mastery", "role", "leads_to", "attempts", "solved", "ranks", "reading"}],
             "later": [weak concepts not covered by this list]}.
    Steps follow the order of the recommendations (prerequisites first); ``leads_to`` names the weak
    concepts a prerequisite comes before in the knowledge graph (association edges only).
    """
    weak = set(weak)
    succ = {}
    for e in (graph or {}).get("edges", []):
        if e.get("kind") == "association":
            succ.setdefault(e["source"], set()).add(e["target"])
    attempts = {c: 0 for c in range(1, NUM_CONCEPTS + 1)}
    solved = dict(attempts)
    for e in prefix:
        for c in e.concept_ids:
            attempts[c] += 1
            solved[c] += e.correct
    steps, index = [], {}
    for rank, r in enumerate(recs, 1):
        c = r["target_concept"]
        if c not in index:
            if len(steps) == max_items:
                continue
            index[c] = len(steps)
            steps.append({"concept": c, "mastery": mastery[c - 1], "role": "weak" if c in weak else "prerequisite",
                          "leads_to": sorted(succ.get(c, set()) & weak, key=lambda k: mastery[k - 1]),
                          "attempts": attempts[c], "solved": solved[c], "ranks": [],
                          "reading": [{"title": t, "url": u} for t, u in READING.get(c, [])]})
        steps[index[c]]["ranks"].append(rank)
    covered = {s["concept"] for s in steps}
    later = [c for c in ordered_concepts if c in weak and c not in covered]
    weakest = sorted(((c, mastery[c - 1]) for c in weak), key=lambda x: x[1])[:3]
    return {"weakest": weakest, "steps": steps, "later": later}


def record_text(attempts, solved):
    if not attempts:
        return "not attempted yet"
    return f"solved {solved} of {attempts} attempt" + ("" if attempts == 1 else "s")
