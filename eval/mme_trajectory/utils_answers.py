import re
from collections import Counter
from typing import Iterable, Optional


YES_PATTERNS = [
    re.compile(r"\byes\b"),
    re.compile(r"\by\b"),
    re.compile(r"\btrue\b"),
    re.compile(r"\bcorrect\b"),
    re.compile(r"\b1\b"),
    re.compile(r"\ba\b"),
]
NO_PATTERNS = [
    re.compile(r"\bno\b"),
    re.compile(r"\bn\b"),
    re.compile(r"\bfalse\b"),
    re.compile(r"\bincorrect\b"),
    re.compile(r"\b0\b"),
    re.compile(r"\bb\b"),
]


def normalize_answer_mme(text: Optional[str]) -> str:
    if text is None:
        return "unknown"

    cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
    cleaned = cleaned.replace(".", " ").replace(":", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "unknown"

    prefix = cleaned[:16]
    if any(pattern.search(prefix) for pattern in YES_PATTERNS):
        return "yes"
    if any(pattern.search(prefix) for pattern in NO_PATTERNS):
        return "no"
    return "unknown"


def majority_answer(answers: Iterable[str]) -> str:
    normalized = [normalize_answer_mme(ans) for ans in answers]
    normalized = [ans for ans in normalized if ans != "unknown"]
    if not normalized:
        return "unknown"
    counts = Counter(normalized)
    best = max(counts.values())
    winners = sorted([key for key, value in counts.items() if value == best])
    return winners[0]
