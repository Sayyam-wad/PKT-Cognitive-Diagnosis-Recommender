"""Step 4: split the PDKT test file (built by assemble.py) by learner into pdkt.zip: 40% train / 20% valid / 40% test, seed 0."""
import random, zipfile
from paths import HERE, WORK
recs = open(WORK / 'pdkt_astnn_test').read().split('\n')
records = [recs[i:i + 8] for i in range(0, len(recs) - 7, 8)]
random.seed(0); order = list(range(len(records))); random.shuffle(order)
n = len(order); cut1, cut2 = int(n * 0.4), int(n * 0.6)
parts = {'astnn_train1': order[:cut1], 'astnn_valid1': order[cut1:cut2], 'astnn_test': order[cut2:]}
with zipfile.ZipFile(HERE / 'pdkt.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for name, idx in parts.items():
        z.writestr('astnn/' + name, '\n'.join('\n'.join(records[i]) for i in sorted(idx)) + '\n')
