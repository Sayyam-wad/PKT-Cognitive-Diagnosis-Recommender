"""Collect every C++ submission (2019-2020) to the 23 identified problems from open-r1/codeforces-submissions.

Run from the objective3_ksap folder on a machine with internet:
    python validation/extract_candidates.py

Only the 8 parquet files that contain our contests are downloaded (~1.2 GB in total, one at a
time, each deleted after use). Output: validation/candidates.parquet (a few MB). The converter
is then run on these sources and the resulting trees are compared with the dataset's records:
an exact tree match is held-out evidence that the converter reproduces the dataset.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "open-r1/codeforces-submissions"
SHARDS = [1, 2, 3, 6, 7, 8, 23, 35]          # files where our contests were seen
ID_MIN, ID_MAX = 54_000_000, 81_000_000      # submission-ID window of the dataset's learners


def main():
    from huggingface_hub import hf_hub_download
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    problems = set(json.loads((HERE / "recovered_problem_ids.json").read_text()).values())
    token = os.environ.get("HF_TOKEN")
    cache = HERE / "_hf_tmp"
    parts = []
    for n in SHARDS:
        name = f"data/train-{n:05d}-of-00036.parquet"
        path = hf_hub_download(REPO, name, repo_type="dataset", token=token, local_dir=cache)
        t = pq.read_table(path, columns=["submission_id", "source", "contestId", "problem_index",
                                         "programmingLanguage", "verdict"])
        key = pc.binary_join_element_wise(pc.cast(t["contestId"], pa.string()),
                                          pc.cast(t["problem_index"], pa.string()), "")
        sid = pc.cast(t["submission_id"], pa.int64())
        mask = pc.and_(pc.and_(pc.is_in(key, value_set=pa.array(sorted(problems))),
                               pc.match_substring(t["programmingLanguage"], "C++")),
                       pc.and_(pc.greater_equal(sid, ID_MIN), pc.less_equal(sid, ID_MAX)))
        sel = t.filter(mask)
        parts.append(sel)
        os.remove(path)
        print(f"{name}: {sel.num_rows} candidate submissions")
    out = pa.concat_tables(parts)
    pq.write_table(out, HERE / "candidates.parquet", compression="zstd")
    print(f"\nWrote {out.num_rows} candidates to {HERE / 'candidates.parquet'} "
          f"({(HERE / 'candidates.parquet').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
