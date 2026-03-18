from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from utils_metrics import (
    compute_official_mme_scores,
    enrich_records,
    simulate_early_commit,
    summarize_final_wrong,
    summarize_flip_patterns,
    summarize_pairs,
    summarize_records,
)


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _infer_predictions_path(trajectory_jsonl: Path) -> Optional[Path]:
    run_dir = trajectory_jsonl.parent.parent if trajectory_jsonl.parent.name == "trajectory" else trajectory_jsonl.parent
    candidates = sorted(run_dir.glob("*_samples_mme.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MME trajectory outputs with optional lmms_eval sample logs.")
    parser.add_argument("--trajectory-jsonl", required=True)
    parser.add_argument("--predictions-jsonl", default=None, help="Path to lmms_eval *_samples_mme.jsonl. If omitted, the script will try to infer it from the run directory.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--early-thresholds", default="0.6,0.7,0.8,0.9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_path = Path(args.trajectory_jsonl)
    trajectory_records = _load_jsonl(trajectory_path)

    predictions_path = Path(args.predictions_jsonl) if args.predictions_jsonl else _infer_predictions_path(trajectory_path)
    prediction_records = _load_jsonl(predictions_path) if predictions_path and predictions_path.exists() else None

    records = enrich_records(trajectory_records, prediction_records)
    thresholds = [float(item) for item in args.early_thresholds.split(",") if item.strip()]

    sample_summary = summarize_records(records)
    pair_summary = summarize_pairs(records)
    final_official = compute_official_mme_scores(records, answer_mode="final")
    ever_official = compute_official_mme_scores(records, answer_mode="ever")
    vote_official = compute_official_mme_scores(records, answer_mode="vote")

    summary = {
        "trajectory_jsonl": str(trajectory_path),
        "predictions_jsonl": str(predictions_path) if predictions_path else None,
        **sample_summary,
        "pair_summary": pair_summary,
        "official_mme": {
            "final": final_official,
            "ever": ever_official,
            "vote": vote_official,
            "ever_final_gap": ever_official.get("total_score", 0.0) - final_official.get("total_score", 0.0),
        },
        "flip_patterns": summarize_flip_patterns(records),
        "final_wrong_analysis": summarize_final_wrong(records),
        "early_commit": simulate_early_commit(records, thresholds),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
