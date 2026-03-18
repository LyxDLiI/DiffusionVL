from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional

from utils_answers import majority_answer, normalize_answer_mme


@dataclass
class SampleMetrics:
    sample_id: str
    category: str
    target: str
    final_answer: str
    ever_correct: bool
    final_correct: bool
    flip_count: int
    earliest_correct_step: Optional[int]
    stable_convergence_step: Optional[int]
    total_steps: int


def _flip_count(step_answers: List[str]) -> int:
    count = 0
    prev = None
    for answer in step_answers:
        if prev is not None and answer != prev:
            count += 1
        prev = answer
    return count


def _earliest_correct_step(step_answers: List[str], target: str) -> Optional[int]:
    for idx, answer in enumerate(step_answers, start=1):
        if answer == target:
            return idx
    return None


def _stable_convergence_step(step_answers: List[str], final_answer: str) -> Optional[int]:
    if not step_answers:
        return None
    earliest = len(step_answers)
    for idx in range(len(step_answers) - 1, -1, -1):
        if step_answers[idx] == final_answer:
            earliest = idx + 1
        else:
            break
    return earliest


def sample_metrics(record: Dict) -> SampleMetrics:
    target = normalize_answer_mme(record.get("target") or record.get("meta", {}).get("target"))
    step_answers = [normalize_answer_mme(step.get("decoded_answer")) for step in record.get("steps", [])]
    final_answer = normalize_answer_mme(record.get("final_output"))
    if final_answer == "unknown" and step_answers:
        final_answer = step_answers[-1]
    return SampleMetrics(
        sample_id=str(record.get("sample_id", "")),
        category=record.get("category") or record.get("meta", {}).get("category", "unknown"),
        target=target,
        final_answer=final_answer,
        ever_correct=target in step_answers,
        final_correct=final_answer == target,
        flip_count=_flip_count(step_answers),
        earliest_correct_step=_earliest_correct_step(step_answers, target),
        stable_convergence_step=_stable_convergence_step(step_answers, final_answer),
        total_steps=len(step_answers),
    )


def summarize_records(records: Iterable[Dict]) -> Dict:
    records = list(records)
    metrics = [sample_metrics(record) for record in records]
    if not metrics:
        return {
            "num_samples": 0,
            "final_pass_accuracy": 0.0,
            "ever_pass_accuracy": 0.0,
            "ever_final_gap": 0.0,
        }

    final_acc = mean(1.0 if item.final_correct else 0.0 for item in metrics)
    ever_acc = mean(1.0 if item.ever_correct else 0.0 for item in metrics)
    voting_acc = mean(
        1.0
        if majority_answer(step.get("decoded_answer") for step in record.get("steps", [])) == item.target
        else 0.0
        for record, item in zip(records, metrics)
    )
    nonempty_flips = [item.flip_count for item in metrics]
    earliest = [item.earliest_correct_step for item in metrics if item.earliest_correct_step is not None]
    stable = [item.stable_convergence_step for item in metrics if item.stable_convergence_step is not None]

    per_category: Dict[str, Dict] = {}
    categories = sorted({item.category for item in metrics})
    for category in categories:
        subset = [item for item in metrics if item.category == category]
        per_category[category] = {
            "count": len(subset),
            "final_pass_accuracy": mean(1.0 if item.final_correct else 0.0 for item in subset),
            "ever_pass_accuracy": mean(1.0 if item.ever_correct else 0.0 for item in subset),
            "avg_flip_count": mean(item.flip_count for item in subset) if subset else 0.0,
        }

    return {
        "num_samples": len(metrics),
        "final_pass_accuracy": final_acc,
        "ever_pass_accuracy": ever_acc,
        "ever_final_gap": ever_acc - final_acc,
        "trajectory_vote_accuracy": voting_acc,
        "avg_flip_count": mean(nonempty_flips) if nonempty_flips else 0.0,
        "avg_earliest_correct_step": mean(earliest) if earliest else None,
        "avg_stable_convergence_step": mean(stable) if stable else None,
        "per_category": per_category,
    }


def simulate_early_commit(records: Iterable[Dict], thresholds: Iterable[float]) -> List[Dict]:
    outputs = []
    records = list(records)
    for threshold in thresholds:
        triggered = 0
        saved_steps = []
        correct = 0
        total = 0
        for record in records:
            total += 1
            target = normalize_answer_mme(record.get("target") or record.get("meta", {}).get("target"))
            steps = record.get("steps", [])
            final = normalize_answer_mme(record.get("final_output"))
            chosen = final
            chosen_step = len(steps)
            for idx, step in enumerate(steps, start=1):
                gap = step.get("gap")
                answer = normalize_answer_mme(step.get("decoded_answer"))
                if gap is not None and answer != "unknown" and gap >= threshold:
                    chosen = answer
                    chosen_step = idx
                    triggered += 1
                    saved_steps.append(max(len(steps) - idx, 0))
                    break
            if chosen == target:
                correct += 1
        outputs.append(
            {
                "threshold": threshold,
                "trigger_rate": triggered / total if total else 0.0,
                "avg_saved_steps": mean(saved_steps) if saved_steps else 0.0,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return outputs
