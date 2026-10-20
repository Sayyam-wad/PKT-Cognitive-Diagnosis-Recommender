"""Validate and summarize the supplied Codeforces ASTNN archive."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content.pkt.recommendation.data import inspect_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Path to codeforces.zip")
    parser.add_argument("--output", help="Optional path for the JSON summary")
    args = parser.parse_args()
    summary = inspect_archive(args.archive)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
