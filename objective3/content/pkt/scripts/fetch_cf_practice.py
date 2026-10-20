"""Fetch real Codeforces problems per tag for the demo's "Practice on Codeforces" suggestions.

One request to the public Codeforces API (problemset.problems); standard library only.
Writes content/pkt/results/cf_practice.json. The demo works without it (it then shows
rating-filtered Codeforces search links only).

    python content/pkt/scripts/fetch_cf_practice.py
"""
import datetime
import json
from pathlib import Path
import sys
import urllib.request

PKT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKT))
from recommendation.resources import CONCEPT_NAMES  # noqa: E402

URL = "https://codeforces.com/api/problemset.problems"


def build(api):
    if api.get("status") != "OK":
        raise SystemExit(f"Codeforces API error: {api.get('comment')}")
    solved = {(s["contestId"], s["index"]): s["solvedCount"]
              for s in api["result"]["problemStatistics"] if "contestId" in s}
    tags = set(CONCEPT_NAMES.values())
    out = {t: [] for t in tags}
    for p in api["result"]["problems"]:
        if "rating" not in p or "contestId" not in p:
            continue
        code = f"{p['contestId']}{p['index']}"
        for t in p.get("tags", []):
            if t in tags:
                out[t].append([code, p["name"], p["rating"], solved.get((p["contestId"], p["index"]), 0)])
    for t in out:
        out[t].sort(key=lambda r: -r[3])
        out[t] = out[t][:300]          # most-solved 300 per tag is plenty for suggestions
    return {"source": URL, "fetched": datetime.date.today().isoformat(), "problems": out}


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "ksap-demo"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = build(json.load(r))
    path = PKT / "results" / "cf_practice.json"
    path.write_text(json.dumps(data))
    n = sum(len(v) for v in data["problems"].values())
    print(f"saved {n} tag-problem entries for {sum(1 for v in data['problems'].values() if v)} tags -> {path}")


if __name__ == "__main__":
    main()
