"""Find our matched Codeforces submissions in the public open-r1/codeforces-submissions dataset.

Run on a machine with normal internet access, from the objective3_ksap folder:
    python -m pip install "huggingface_hub>=1.5,<2" pyarrow
    python validation/fetch_from_hf.py            # full search
    python validation/fetch_from_hf.py --probe    # look at one file only and print diagnostics

The dataset (~5.6 GB, 36 parquet files) is downloaded ONE file at a time; each file is
searched locally for our submission IDs and deleted afterwards, so at most ~200 MB of disk
is used. Found sources are written to validation/code/<file>. Re-running skips files that
were already searched (progress is kept in validation/hf_progress.json).

IDs are compared as text, so it works whether the dataset stores them as numbers or strings.
Each file also reports how many rows belong to our contests, which tells us whether the
dataset covers them at all.

Optional: set an HF token (free account) to avoid rate limits:
    $env:HF_TOKEN = "hf_..."
"""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
PROGRESS = HERE / "hf_progress.json"
REPO = "open-r1/codeforces-submissions"
VERSION = 2  # bumped when matching changed (v1 compared IDs without normalising the type)


def load_progress():
    if PROGRESS.exists():
        data = json.loads(PROGRESS.read_text())
        if isinstance(data, dict) and data.get("version") == VERSION:
            return data
    return {"version": VERSION, "done": [], "contest_rows": 0, "contest_ids_seen": []}


def find_col(names, *candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def main():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    probe = "--probe" in sys.argv
    pairs = json.loads((HERE / "validation_pairs.json").read_text())
    by_id = {str(p["submission_id"]): p for p in pairs}
    our_contests = {str(p["contest"]) for p in pairs}
    CODE.mkdir(exist_ok=True)
    prog = load_progress()
    token = os.environ.get("HF_TOKEN")
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", token=token)
                   if f.startswith("data/train-") and f.endswith(".parquet"))
    print(f"{len(files)} parquet files; {len(prog['done'])} already searched")
    cache = HERE / "_hf_tmp"
    for i, name in enumerate(files, 1):
        if name in prog["done"] and not probe:
            continue
        for attempt in range(5):
            try:
                path = hf_hub_download(REPO, name, repo_type="dataset", token=token, local_dir=cache)
                break
            except Exception as e:  # noqa: BLE001 - network hiccups / 429
                wait = 30 * (attempt + 1)
                print(f"  download failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"  giving up on {name} for now; re-run later to continue")
            break
        schema = pq.read_schema(path)
        names = schema.names
        id_col = find_col(names, "submission_id", "id")
        src_col = find_col(names, "source", "code")
        contest_col = find_col(names, "contest_id", "contestId", "contest")
        cols = [c for c in (id_col, src_col, contest_col) if c]
        table = pq.read_table(path, columns=cols)
        ids = [str(x) for x in table.column(id_col).to_pylist()]
        if probe or i == 1:
            print(f"  columns: {names}")
            print(f"  id column '{id_col}' type {schema.field(id_col).type}; sample {ids[:3]}")
            nums = [int(x) for x in ids if x.isdigit()]
            if nums:
                print(f"  id range in this file: {min(nums)} .. {max(nums)}  (ours: 54295165 .. 80351270)")
        contest_rows = 0
        if contest_col:
            contests = [str(x) for x in table.column(contest_col).to_pylist()]
            seen = sorted(our_contests.intersection(contests))
            contest_rows = sum(c in our_contests for c in contests)
            prog["contest_rows"] += contest_rows
            prog["contest_ids_seen"] = sorted(set(prog["contest_ids_seen"]) | set(seen))
        hits = [k for k, sid in enumerate(ids) if sid in by_id]
        for k in hits:
            p = by_id[ids[k]]
            src = table.column(src_col)[k].as_py()
            if src:
                (CODE / p["file"]).write_text(src, encoding="utf-8")
                print(f"  FOUND {p['file']} (submission {ids[k]})")
        os.remove(path)
        n_found = sum((CODE / p["file"]).exists() for p in pairs)
        print(f"[{i}/{len(files)}] {name}: {len(hits)} hit(s); rows from our contests {contest_rows}; "
              f"total found {n_found}/{len(pairs)}")
        if probe:
            print(f"  our contests present in this file: {prog['contest_ids_seen'] or 'none'}")
            return
        prog["done"].append(name)
        PROGRESS.write_text(json.dumps(prog))
    n_found = sum((CODE / p["file"]).exists() for p in pairs)
    print(f"\nDone: {n_found}/{len(pairs)} source files in {CODE}")
    print(f"Rows from our {len(our_contests)} contests seen in the dataset: {prog['contest_rows']} "
          f"(contests present: {prog['contest_ids_seen'] or 'none'})")
    if n_found:
        print("Next: python validation/validate_converter.py")


if __name__ == "__main__":
    main()
