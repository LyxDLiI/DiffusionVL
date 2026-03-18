from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from utils_metrics import simulate_early_commit, summarize_records


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MME trajectory JSONL outputs.")
    parser.add_argument("--trajectory-jsonl", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--early-thresholds", default="0.6,0.7,0.8,0.9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_jsonl(Path(args.trajectory_jsonl))
    summary = summarize_records(records)
    thresholds = [float(item) for item in args.early_thresholds.split(",") if item.strip()]
    summary["early_commit"] = simulate_early_commit(records, thresholds)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
