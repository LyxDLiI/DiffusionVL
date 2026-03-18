from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from utils_answers import majority_answer, normalize_answer_mme


@dataclass
class SampleMetrics:
    sample_id: str
    doc_id: Optional[int]
    question_id: str
    category: str
    target: str
    final_answer: str
    vote_answer: str
    ever_correct: bool
    final_correct: bool
    vote_correct: bool
    flip_count: int
    earliest_correct_step: Optional[int]
    stable_convergence_step: Optional[int]
    total_steps: int


@dataclass
class PairMetrics:
    question_id: str
    category: str
    final_pair_correct: bool
    ever_pair_correct: bool
    vote_pair_correct: bool


def _step_answers(record: Dict) -> List[str]:
    return [normalize_answer_mme(step.get("decoded_answer")) for step in record.get("steps", [])]


def _flip_count(step_answers: List[str]) -> int:
    count = 0
    prev = None
    for answer in step_answers:
        if prev is not None and answer != prev:
            count += 1
        prev = answer
    return count


def _flip_breakdown(step_answers: List[str]) -> Counter:
    counter: Counter = Counter()
    prev = None
    for answer in step_answers:
        if prev is not None and answer != prev:
            counter[f"{prev}->{answer}"] += 1
        prev = answer
    return counter


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


def _get_target(record: Dict) -> str:
    return normalize_answer_mme(record.get("target") or record.get("meta", {}).get("target"))


def sample_metrics(record: Dict) -> SampleMetrics:
    target = _get_target(record)
    step_answers = _step_answers(record)
    final_answer = normalize_answer_mme(record.get("final_output") or record.get("prediction", {}).get("filtered_resps", [None])[0])
    if final_answer == "unknown" and step_answers:
        final_answer = step_answers[-1]
    vote_answer = majority_answer(step.get("decoded_answer") for step in record.get("steps", []))
    return SampleMetrics(
        sample_id=str(record.get("sample_id", "")),
        doc_id=record.get("doc_id"),
        question_id=str(record.get("question_id") or record.get("prediction", {}).get("doc", {}).get("question_id", "")),
        category=record.get("category") or record.get("meta", {}).get("category") or record.get("prediction", {}).get("doc", {}).get("category", "unknown"),
        target=target,
        final_answer=final_answer,
        vote_answer=vote_answer,
        ever_correct=target in step_answers,
        final_correct=final_answer == target,
        vote_correct=vote_answer == target,
        flip_count=_flip_count(step_answers),
        earliest_correct_step=_earliest_correct_step(step_answers, target),
        stable_convergence_step=_stable_convergence_step(step_answers, final_answer),
        total_steps=len(step_answers),
    )


def enrich_records(trajectory_records: Iterable[Dict], prediction_records: Optional[Iterable[Dict]] = None) -> List[Dict]:
    trajectory_records = [dict(record) for record in trajectory_records]
    if not prediction_records:
        return trajectory_records

    by_doc_id = {int(pred["doc_id"]): pred for pred in prediction_records if "doc_id" in pred}
    for record in trajectory_records:
        doc_id = record.get("doc_id")
        if doc_id is None:
            continue
        pred = by_doc_id.get(int(doc_id))
        if pred is None:
            continue
        doc = pred.get("doc", {})
        record["prediction"] = pred
        record.setdefault("question_id", doc.get("question_id"))
        record.setdefault("category", doc.get("category"))
        record.setdefault("target", pred.get("target") or doc.get("answer"))
        record.setdefault("prompt", pred.get("input"))
    return trajectory_records


def _category_subset(metrics: List[SampleMetrics], category: str) -> Dict:
    subset = [item for item in metrics if item.category == category]
    return {
        "count": len(subset),
        "final_pass_accuracy": mean(1.0 if item.final_correct else 0.0 for item in subset) if subset else 0.0,
        "ever_pass_accuracy": mean(1.0 if item.ever_correct else 0.0 for item in subset) if subset else 0.0,
        "vote_accuracy": mean(1.0 if item.vote_correct else 0.0 for item in subset) if subset else 0.0,
        "avg_flip_count": mean(item.flip_count for item in subset) if subset else 0.0,
    }


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
    voting_acc = mean(1.0 if item.vote_correct else 0.0 for item in metrics)
    earliest = [item.earliest_correct_step for item in metrics if item.earliest_correct_step is not None]
    stable = [item.stable_convergence_step for item in metrics if item.stable_convergence_step is not None]

    per_category: Dict[str, Dict] = {}
    for category in sorted({item.category for item in metrics}):
        per_category[category] = _category_subset(metrics, category)

    return {
        "num_samples": len(metrics),
        "final_pass_accuracy": final_acc,
        "ever_pass_accuracy": ever_acc,
        "ever_final_gap": ever_acc - final_acc,
        "trajectory_vote_accuracy": voting_acc,
        "avg_flip_count": mean(item.flip_count for item in metrics),
        "avg_earliest_correct_step": mean(earliest) if earliest else None,
        "avg_stable_convergence_step": mean(stable) if stable else None,
        "per_category": per_category,
    }


def summarize_flip_patterns(records: Iterable[Dict]) -> Dict:
    records = list(records)
    transition_counter: Counter = Counter()
    semantic_flip_samples = 0
    unknown_flip_samples = 0
    for record in records:
        step_answers = _step_answers(record)
        breakdown = _flip_breakdown(step_answers)
        transition_counter.update(breakdown)
        if breakdown.get("yes->no", 0) or breakdown.get("no->yes", 0):
            semantic_flip_samples += 1
        if any("unknown" in key for key in breakdown):
            unknown_flip_samples += 1
    total = len(records)
    return {
        "transition_counts": dict(sorted(transition_counter.items())),
        "semantic_flip_sample_rate": semantic_flip_samples / total if total else 0.0,
        "unknown_involved_flip_sample_rate": unknown_flip_samples / total if total else 0.0,
    }


def summarize_final_wrong(records: Iterable[Dict]) -> Dict:
    records = list(records)
    metrics = [sample_metrics(record) for record in records]
    final_wrong = [item for item in metrics if not item.final_correct]
    if not final_wrong:
        return {"count": 0, "ever_correct_rate": 0.0, "vote_correct_rate": 0.0}
    return {
        "count": len(final_wrong),
        "ever_correct_rate": mean(1.0 if item.ever_correct else 0.0 for item in final_wrong),
        "vote_correct_rate": mean(1.0 if item.vote_correct else 0.0 for item in final_wrong),
        "avg_flip_count": mean(item.flip_count for item in final_wrong),
    }


def _group_pairs(metrics: List[SampleMetrics]) -> Dict[Tuple[str, str], List[SampleMetrics]]:
    grouped: Dict[Tuple[str, str], List[SampleMetrics]] = defaultdict(list)
    for item in metrics:
        if item.question_id:
            grouped[(item.category, item.question_id)].append(item)
    return grouped


def summarize_pairs(records: Iterable[Dict]) -> Dict:
    metrics = [sample_metrics(record) for record in records]
    grouped = _group_pairs(metrics)
    valid_pairs: List[PairMetrics] = []
    incomplete = 0
    for (category, question_id), items in grouped.items():
        if len(items) != 2:
            incomplete += 1
            continue
        valid_pairs.append(
            PairMetrics(
                question_id=question_id,
                category=category,
                final_pair_correct=all(item.final_correct for item in items),
                ever_pair_correct=all(item.ever_correct for item in items),
                vote_pair_correct=all(item.vote_correct for item in items),
            )
        )

    if not valid_pairs:
        return {"num_pairs": 0, "incomplete_pairs": incomplete}

    per_category: Dict[str, Dict] = {}
    for category in sorted({pair.category for pair in valid_pairs}):
        subset = [pair for pair in valid_pairs if pair.category == category]
        per_category[category] = {
            "count": len(subset),
            "final_pair_accuracy": mean(1.0 if pair.final_pair_correct else 0.0 for pair in subset),
            "ever_pair_accuracy": mean(1.0 if pair.ever_pair_correct else 0.0 for pair in subset),
            "vote_pair_accuracy": mean(1.0 if pair.vote_pair_correct else 0.0 for pair in subset),
        }

    return {
        "num_pairs": len(valid_pairs),
        "incomplete_pairs": incomplete,
        "final_pair_accuracy": mean(1.0 if pair.final_pair_correct else 0.0 for pair in valid_pairs),
        "ever_pair_accuracy": mean(1.0 if pair.ever_pair_correct else 0.0 for pair in valid_pairs),
        "ever_final_pair_gap": mean(1.0 if pair.ever_pair_correct else 0.0 for pair in valid_pairs)
        - mean(1.0 if pair.final_pair_correct else 0.0 for pair in valid_pairs),
        "vote_pair_accuracy": mean(1.0 if pair.vote_pair_correct else 0.0 for pair in valid_pairs),
        "per_category": per_category,
    }


def compute_official_mme_scores(records: Iterable[Dict], answer_mode: str = "final") -> Dict:
    metrics = [sample_metrics(record) for record in records]
    grouped = _group_pairs(metrics)
    category2scores: Dict[str, Dict[str, List[int]]] = defaultdict(dict)

    def get_correct(item: SampleMetrics) -> bool:
        if answer_mode == "final":
            return item.final_correct
        if answer_mode == "ever":
            return item.ever_correct
        if answer_mode == "vote":
            return item.vote_correct
        raise ValueError(f"Unknown answer_mode: {answer_mode}")

    incomplete_pairs = 0
    for (category, question_id), items in grouped.items():
        if len(items) != 2:
            incomplete_pairs += 1
            continue
        category2scores[category][question_id] = [1 if get_correct(item) else 0 for item in items]

    per_category = {}
    total_score = 0.0
    for category, question_map in sorted(category2scores.items()):
        total_category_score = 0.0
        for scores in question_map.values():
            acc = sum(scores) / len(scores) * 100.0
            acc_plus = 100.0 if sum(scores) == 2 else 0.0
            total_category_score += acc_plus + acc
        category_score = total_category_score / len(question_map) if question_map else 0.0
        per_category[category] = category_score
        total_score += category_score

    return {
        "answer_mode": answer_mode,
        "num_pairs": sum(len(v) for v in category2scores.values()),
        "incomplete_pairs": incomplete_pairs,
        "total_score": total_score,
        "per_category": per_category,
    }


def simulate_early_commit(records: Iterable[Dict], thresholds: Iterable[float]) -> List[Dict]:
    outputs = []
    records = list(records)
    for threshold in thresholds:
        triggered = 0
        saved_steps = []
        chosen_records = []
        for record in records:
            target = _get_target(record)
            steps = record.get("steps", [])
            final = normalize_answer_mme(record.get("final_output"))
            chosen = final
            for idx, step in enumerate(steps, start=1):
                gap = step.get("gap")
                answer = normalize_answer_mme(step.get("decoded_answer"))
                if gap is not None and answer != "unknown" and gap >= threshold:
                    chosen = answer
                    triggered += 1
                    saved_steps.append(max(len(steps) - idx, 0))
                    break
            cloned = dict(record)
            cloned["final_output"] = chosen
            cloned["target"] = target
            chosen_records.append(cloned)
        sample_summary = summarize_records(chosen_records)
        pair_summary = summarize_pairs(chosen_records)
        official_summary = compute_official_mme_scores(chosen_records, answer_mode="final")
        outputs.append(
            {
                "threshold": threshold,
                "trigger_rate": triggered / len(records) if records else 0.0,
                "avg_saved_steps": mean(saved_steps) if saved_steps else 0.0,
                "sample_accuracy": sample_summary.get("final_pass_accuracy", 0.0),
                "pair_accuracy": pair_summary.get("final_pair_accuracy", 0.0),
                "official_total_score": official_summary.get("total_score", 0.0),
            }
        )
    return outputs
