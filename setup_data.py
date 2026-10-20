"""One-time setup: download the Codeforces dataset and put every input where the code expects it.

    python setup_data.py                 # downloads data/codeforces.zip from the original PKT repository
    python setup_data.py --zip PATH      # or use a copy you already have

It verifies SHA-256 checksums and then:
  * objective3/data/codeforces.zip                         (read directly by Objective 3)
  * objective2/data/codeforces/astnn/{astnn_train1, astnn_valid1, astnn_test, node_w2v_128}
                                                            (the inputs of the Objective 2 training code)
  * objective3/content/pkt/results/ksap_hf_v1/mastery.pt   (copy of objective2/results/best_longformer_model.pt,
                                                            the trained ASTNN-Longformer used for diagnosis)
Standard library only.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = "https://github.com/pkt-attn/pkt-attn/raw/main/data/codeforces.zip"
REPO = "https://github.com/pkt-attn/pkt-attn.git"
ZIP_SHA = "cd317f221a65c2cb50e1ea03683960b300690145e5c5ebbb40e3b113bf5cd70d"
CKPT_SHA = "b4be633b38edb435e110aba3b72d735eddc6356a63d73a15cceef1d0bbbd96a4"
MEMBERS = ("astnn/astnn_train1", "astnn/astnn_valid1", "astnn/astnn_test", "astnn/node_w2v_128")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="existing copy of codeforces.zip (skips the download)")
    args = ap.parse_args()
    target = ROOT / "objective3" / "data" / "codeforces.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.zip:
        if args.zip.resolve() != target.resolve():
            shutil.copyfile(args.zip, target)
    elif not target.exists() or sha256(target) != ZIP_SHA:
        print(f"downloading {URL} (85 MB) ...", flush=True)
        try:
            urllib.request.urlretrieve(URL, target)
        except Exception as exc:                      # e.g. a proxy that blocks raw downloads
            print(f"direct download failed ({exc}); trying git clone of {REPO} ...", flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["git", "clone", "--depth", "1", REPO, tmp], check=True)
                shutil.copyfile(Path(tmp) / "data" / "codeforces.zip", target)
    if sha256(target) != ZIP_SHA:
        sys.exit(f"checksum mismatch for {target}; expected the PKT Codeforces dataset (sha256 {ZIP_SHA[:12]}...)")
    print("dataset OK:", target)

    out = ROOT / "objective2" / "data" / "codeforces" / "astnn"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target) as z:
        for m in MEMBERS:
            dest = out / Path(m).name
            if not dest.exists() or dest.stat().st_size != z.getinfo(m).file_size:
                with z.open(m) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print("Objective 2 inputs OK:", out)

    ckpt = ROOT / "objective2" / "results" / "best_longformer_model.pt"
    if sha256(ckpt) != CKPT_SHA:
        print("note: objective2/results/best_longformer_model.pt differs from the reported checkpoint "
              "(expected after retraining); Objective 3 will use your retrained model.")
    bundle = ROOT / "objective3" / "content" / "pkt" / "results" / "ksap_hf_v1" / "mastery.pt"
    shutil.copyfile(ckpt, bundle)
    print("Objective 3 diagnosis checkpoint OK:", bundle)


if __name__ == "__main__":
    main()
