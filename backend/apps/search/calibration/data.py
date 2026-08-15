from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CaseType = Literal["حقوقی", "کیفری"]


@dataclass
class AnnotatedSample:
    ruling_id: str
    case_type: CaseType
    query: str
    gold_articles: list[str]
    gold_routing: dict
    gold_checklist: list[dict]


def load_case_annotations(path: str | Path) -> list[AnnotatedSample]:
    """Loads the 100 case-grounded annotation samples (query field required)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        if "query" not in item or not item["query"]:
            raise ValueError(
                f"ruling_id={item.get('ruling_id')} has no query field — "
                "cannot run search pipeline without it."
            )
        samples.append(AnnotatedSample(
            ruling_id=item["ruling_id"],
            case_type=item["case_type"],
            query=item["query"],
            gold_articles=item["gold_articles"],
            gold_routing=item["gold_routing"],
            gold_checklist=item["gold_checklist"],
        ))
    return samples


def stratified_test_split(
    samples: list[AnnotatedSample],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[AnnotatedSample], list[AnnotatedSample]]:
    """
    Splits into train_val / test, preserving the case_type ratio in both.
    With 60 civil / 40 criminal, this yields ~12 civil + 8 criminal in test
    (20 total), and ~48 civil + 32 criminal in train_val (80 total).
    """
    import random
    rng = random.Random(seed)

    by_type: dict[str, list[AnnotatedSample]] = {}
    for s in samples:
        by_type.setdefault(s.case_type, []).append(s)

    train_val, test = [], []
    for case_type, group in by_type.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_test = round(len(shuffled) * test_fraction)
        test.extend(shuffled[:n_test])
        train_val.extend(shuffled[n_test:])

    return train_val, test


def stratified_k_folds(
    samples: list[AnnotatedSample],
    k: int = 5,
    seed: int = 42,
) -> list[tuple[list[AnnotatedSample], list[AnnotatedSample]]]:
    """
    Returns k (train, val) pairs, each preserving the case_type ratio.
    Uses round-robin assignment within each case_type group so class
    balance is maintained across all folds, not just approximately.
    """
    import random
    rng = random.Random(seed)

    by_type: dict[str, list[AnnotatedSample]] = {}
    for s in samples:
        by_type.setdefault(s.case_type, []).append(s)

    fold_buckets: list[list[AnnotatedSample]] = [[] for _ in range(k)]
    for case_type, group in by_type.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        for i, sample in enumerate(shuffled):
            fold_buckets[i % k].append(sample)

    folds = []
    for i in range(k):
        val = fold_buckets[i]
        train = [s for j, bucket in enumerate(fold_buckets) if j != i for s in bucket]
        folds.append((train, val))
    return folds