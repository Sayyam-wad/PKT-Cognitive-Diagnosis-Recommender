"""Shared paths for the PDKT pipeline. Set PDKT_RAW to the folder holding the raw PDKT export
(user_sequence.csv, problem_tags.csv); intermediate files go to PDKT_WORK (default: ./work)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBJ3 = HERE.parent                                   # objective3/
REPO = OBJ3.parent
PKT = OBJ3 / "content" / "pkt"
OBJ2 = REPO / "objective2"
RAW = Path(os.environ.get("PDKT_RAW", HERE / "raw"))
WORK = Path(os.environ.get("PDKT_WORK", HERE / "work"))
W2V = OBJ2 / "data" / "codeforces" / "astnn" / "node_w2v_128"
WORK.mkdir(parents=True, exist_ok=True)
