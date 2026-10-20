"""Step 2: crash-tolerant conversion of every C++ submission (libclang can crash or hang on a few files)."""
import subprocess, os, json, sys
from paths import HERE, WORK
P = WORK / 'parts'; P.mkdir(exist_ok=True)
N = len(json.load(open(WORK / 'cpp_codes.json'))); CH = 500; env = dict(os.environ); crashed = []
def run(a, b, timeout):
    try:
        return subprocess.run([sys.executable, str(HERE / 'conv_worker.py'), str(a), str(b)], env=env,
                              timeout=timeout, capture_output=True).returncode == 0
    except subprocess.TimeoutExpired:
        return False
for a in range(0, N, CH):
    b = a + CH
    if (P / f'conv_{a}_{b}.json').exists(): continue
    singles = [(P / f'conv_{i}_{i + 1}.json').exists() for i in range(a, min(b, N))]
    if sum(singles) >= len(singles) - 5:                 # finished file by file earlier
        crashed += [i for i, ok in zip(range(a, min(b, N)), singles) if not ok]; continue
    if not run(a, b, 120):
        for i in range(a, min(b, N)):            # redo one by one; skip files that crash or hang
            if not run(i, i + 1, 60): crashed.append(i)
    print(f'{min(b, N)}/{N} done; crashed/hung so far {len(crashed)}', flush=True)
json.dump(crashed, open(P / 'crashed.json', 'w'))
print('FINISHED', len(crashed), 'files skipped')
