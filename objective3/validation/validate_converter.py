"""Exact-parity check: pasted Codeforces source vs the dataset's ASTNN trees.

Usage (from the objective3_ksap folder, after pasting code into validation/code/):
    python validation/validate_converter.py

Needs: libclang==13.0.0 and pcpp (vocabulary is read from node_vocab.json). For every pair in validation_pairs.json with a
pasted file, converts the code and compares it with the dataset record:
  * exact      - identical block list (every token and tree shape)
  * roots      - identical sequence of root kinds
  * token F1   - overlap of the token multisets (structure + identifiers)
and prints the first differing block so converter rules can be fixed.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "content" / "pkt"))
sys.path.insert(0, str(ROOT / "content" / "pkt" / "preprocessing_live"))
from cpp_to_astnn import source_to_blocks, pretty  # noqa: E402
from recommendation.data import iter_histories  # noqa: E402

ARCHIVE = ROOT / "data" / "codeforces.zip"
CODE_DIR = Path(__file__).resolve().parent / "code"


def load_vocab():
    vocab_json = ROOT / "content" / "pkt" / "preprocessing_live" / "node_vocab.json"
    if vocab_json.exists():
        return json.loads(vocab_json.read_text(encoding="utf-8")) + ["<UNK>"]
    from gensim.models import Word2Vec
    with zipfile.ZipFile(ARCHIVE) as z, tempfile.TemporaryDirectory() as d:
        z.extract("astnn/node_w2v_128", d)
        kv = Word2Vec.load(str(Path(d) / "astnn" / "node_w2v_128")).wv
    return kv.index_to_key + ["<UNK>"]


def decode(node, keys):
    if node == [-1]:
        return None
    return [keys[node[0]]] + [c for c in (decode(k, keys) for k in node[1:]) if c is not None]


def flat(trees):
    out = []
    stack = list(trees)
    while stack:
        n = stack.pop()
        out.append(n[0])
        stack.extend(n[1:])
    return Counter(out)


def map_unk(node, vocab):
    return [node[0] if node[0] in vocab else "<UNK>"] + [map_unk(c, vocab) for c in node[1:]]


def main():
    keys = load_vocab()
    vocab = set(keys)
    pairs = json.loads((Path(__file__).resolve().parent / "validation_pairs.json").read_text())
    histories = {}
    for split in {p["split"] for p in pairs}:
        for h in iter_histories(ARCHIVE, split):
            histories[h.learner_id] = h
    results = []
    for p in pairs:
        f = CODE_DIR / p["file"]
        if not f.exists():
            continue
        record = histories[p["learner"]].interactions[p["record_index"]]
        want = [decode(r, keys) for r in record.ast_roots if r != [-1]]
        got = [map_unk(b, vocab) for b in source_to_blocks(f.read_text(encoding="utf-8", errors="replace"), f.name)]
        exact = got == want
        roots = [b[0] for b in got] == [b[0] for b in want]
        a, b = flat(got), flat(want)
        inter = sum((a & b).values())
        f1 = 2 * inter / (sum(a.values()) + sum(b.values())) if (a or b) else 1.0
        results.append((p["file"], exact, roots, f1))
        print(f"{p['file']:22s} exact={exact!s:5s} roots={roots!s:5s} tokenF1={f1:.3f}  blocks got/want={len(got)}/{len(want)}")
        if not exact:
            for i, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    print(f"   first differing block #{i}:\n   GOT:  " + " ".join(pretty(g))[:300] +
                          "\n   WANT: " + " ".join(pretty(w))[:300])
                    break
    if results:
        n = len(results)
        print(f"\n{n} files | exact {sum(r[1] for r in results)}/{n} | same root sequence {sum(r[2] for r in results)}/{n} | "
              f"mean token F1 {sum(r[3] for r in results) / n:.3f}")
    else:
        print(f"No pasted files found in {CODE_DIR}")


if __name__ == "__main__":
    main()
