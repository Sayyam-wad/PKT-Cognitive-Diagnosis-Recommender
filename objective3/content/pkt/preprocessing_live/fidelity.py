# Exploration script (16 Oct 2026); paths below point to the cloud workspace used for the feasibility study.
"""Compare converter output on real C++ (PDKT sample) with the Codeforces ASTNN dataset."""
import os
NODE_W2V = os.environ.get("NODE_W2V", "node_w2v_128")  # path to the dataset node_w2v_128
import collections, json, sys, time
sys.path.insert(0, "/home/claude/work/ksap/content/pkt/preprocessing_live")
from cpp_to_astnn import source_to_blocks
from gensim.models import Word2Vec

kv = Word2Vec.load(NODE_W2V).wv
keys = kv.index_to_key + ['<UNK>']
vocab = kv.key_to_index
is_struct = lambda s: (s.isupper() and len(s) > 1 and not s.isdigit()) or s == 'End'


def stats_from_trees(sub_trees):
    s = collections.Counter(); root_kinds = collections.Counter(); struct_tok = collections.Counter()
    nroots, ntok, oov, oov_struct = [], 0, 0, collections.Counter()
    for trees in sub_trees:
        nroots.append(len(trees))
        for t in trees:
            root_kinds[t[0]] += 1
            stack = [t]
            while stack:
                n = stack.pop()
                tok = n[0]
                ntok += 1
                if is_struct(tok):
                    struct_tok[tok] += 1
                if tok not in vocab:
                    oov += 1
                    if is_struct(tok):
                        oov_struct[tok] += 1
                stack.extend(n[1:])
    return dict(subs=len(sub_trees), mean_roots=sum(nroots) / len(nroots), tokens=ntok, oov_rate=oov / ntok,
                root_kinds=root_kinds, struct_tok=struct_tok, oov_struct=oov_struct)


def dataset_trees(path, limit=3000):
    lines = open(path).read().split('\n')
    out = []
    def dec(n):
        if n == [-1]: return None
        return [keys[n[0]]] + [c for c in (dec(k) for k in n[1:]) if c is not None]
    for i in range(0, len(lines) - 7, 8):
        for a in lines[i + 4].split(';'):
            out.append([dec(r) for r in json.loads(a) if r != [-1]])
            if len(out) >= limit: return out
    return out


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('/mnt/user-data/uploads/pkt_backup/objective3_ksap/pdkt_cpp_sample.jsonl')]
    t = time.time(); conv, fails = [], 0
    for r in rows:
        try:
            b = source_to_blocks(r['code'])
            conv.append(b)
        except Exception as e:
            fails += 1
    print(f'converted {len(conv)}/{len(rows)} in {time.time()-t:.0f}s, failures {fails}, empty {sum(not c for c in conv)}')
    conv = [c for c in conv if c]
    A = stats_from_trees(conv)
    B = stats_from_trees(dataset_trees('/home/claude/optc/astnn/astnn_test'))
    print(f"mean roots/sub: converted {A['mean_roots']:.1f}  dataset {B['mean_roots']:.1f}")
    print(f"token OOV rate: converted {A['oov_rate']:.4f}  dataset(UNK) {B['oov_rate']:.4f}")
    print('structural tokens we emit that are NOT in the vocabulary:', A['oov_struct'].most_common(15))
    tot_a, tot_b = sum(A['struct_tok'].values()), sum(B['struct_tok'].values())
    print('structural token share (converted vs dataset), top 25 by dataset:')
    for k, v in B['struct_tok'].most_common(25):
        print(f"   {k:36s} {A['struct_tok'][k]/tot_a:.3f}  {v/tot_b:.3f}")
    only_a = [(k, v) for k, v in A['struct_tok'].most_common() if B['struct_tok'][k] == 0][:12]
    print('kinds emitted but never seen in dataset trees:', only_a)
    ra, rb = sum(A['root_kinds'].values()), sum(B['root_kinds'].values())
    print('root kind share top 12 (converted vs dataset):')
    for k, v in B['root_kinds'].most_common(12):
        print(f"   {k:30s} {A['root_kinds'][k]/ra:.3f}  {v/rb:.3f}")
