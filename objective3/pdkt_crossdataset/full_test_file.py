"""Rebuild the full PDKT file (all 584 learners, 8-line ASTNN format) from the included pdkt.zip,
for the zero-shot Objective 2 evaluation: writes PDKT_WORK/pdkt_astnn_test.

    python full_test_file.py
    python evaluate_original.py work/pdkt_astnn_test ../../objective2/data/codeforces/astnn/node_w2v_128 ../../objective2/results/best_longformer_model.pt
"""
import zipfile
from paths import HERE, WORK
with zipfile.ZipFile(HERE / 'pdkt.zip') as z, open(WORK / 'pdkt_astnn_test', 'w') as out:
    for name in ('astnn/astnn_train1', 'astnn/astnn_valid1', 'astnn/astnn_test'):
        text = z.read(name).decode()
        out.write(text if text.endswith('\n') else text + '\n')
print('written', WORK / 'pdkt_astnn_test')
