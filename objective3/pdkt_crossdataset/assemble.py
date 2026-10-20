"""Step 3: assemble the converted submissions into one 8-line ASTNN file (WORK/pdkt_astnn_test), learners with >= 3 submissions."""
import csv, json, glob, collections, sys
from mapping import PDKT_TO_CF, FALLBACK
from paths import RAW, WORK
csv.field_size_limit(10**9)
pt = collections.defaultdict(set)
for r in csv.DictReader(open(RAW / 'problem_tags.csv')): pt[int(r['problem_id'])].add(int(r['problemtag_id']))
rows = [r for r in csv.DictReader(open(RAW / 'user_sequence.csv', encoding='utf-8', errors='replace')) if r['language'] == 'C++']
conv = {}
for f in glob.glob(str(WORK / 'parts' / 'conv_*.json')):
    conv.update({int(k): v for k, v in json.load(open(f)).items()})
unk = len(json.load(open(WORK / 'vocab.json')))
seqs = collections.OrderedDict(); dropped = fallback = tot = oov = 0
for i, r in enumerate(rows):
    roots = conv.get(i) or []
    if not roots: dropped += 1; continue
    st = list(roots)
    while st:
        n = st.pop(); tot += 1; oov += n[0] == unk; st.extend(n[1:])
    pid = int(r['problem_id'])
    tags = sorted({c for tg in pt.get(pid, ()) for c in PDKT_TO_CF.get(tg, [])})
    if not tags: tags = FALLBACK; fallback += 1
    seqs.setdefault(r['user_id'], []).append((pid, tags, roots, int(r['result'])))
n_users = n_events = 0; pos = 0
with open(WORK / 'pdkt_astnn_test', 'w') as f:
    for u, ev in seqs.items():
        if len(ev) < 3: continue
        n_users += 1; n_events += len(ev); pos += sum(e[3] for e in ev)
        f.write(f"{u},{len(ev)}\n" + ";".join(str(e[0]) for e in ev) + "\n" + ";".join(json.dumps(e[1]) for e in ev) + "\n\n"
                + ";".join(json.dumps(e[2], separators=(',', ':')) for e in ev) + "\n\n" + ";".join(str(e[3]) for e in ev) + "\n\n")
print(f"C++ submissions {len(rows)} | no code tree (dropped) {dropped} | topic fallback to 'implementation' {fallback}")
print(f"test file: {n_users} users, {n_events} submissions, accepted {pos/n_events:.1%}, token OOV {oov/tot:.1%}")
