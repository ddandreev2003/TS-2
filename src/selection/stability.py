"""Stability metrics for feature selection across CV folds."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def dice(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    denom = len(a) + len(b)
    if denom == 0:
        return 0.0
    return 2 * len(a & b) / denom


def kuncheva(a: set[str], b: set[str], p: int) -> float:
    """Kuncheva index for feature subset stability."""
    if p == 0:
        return 1.0
    k = len(a)
    r = len(a & b)
    if k == 0:
        return 1.0
    expected = k * k / p
    denom = k - expected
    if denom == 0:
        return 1.0
    return (r - expected) / denom


def stability(sets: Iterable[set[str]], p: int | None = None) -> dict[str, float]:
    """Compute mean pairwise Jaccard, Dice, and Kuncheva across fold sets."""
    sets_list = list(sets)
    if len(sets_list) < 2:
        return {"jaccard": 1.0, "dice": 1.0, "kuncheva": 1.0}

    if p is None:
        p = max(len(s) for s in sets_list)

    j_scores, d_scores, k_scores = [], [], []
    for a, b in combinations(sets_list, 2):
        j_scores.append(jaccard(a, b))
        d_scores.append(dice(a, b))
        k_scores.append(kuncheva(a, b, p))

    return {
        "jaccard": float(sum(j_scores) / len(j_scores)),
        "dice": float(sum(d_scores) / len(d_scores)),
        "kuncheva": float(sum(k_scores) / len(k_scores)),
    }


def combined_loss(kuncheva_score: float, mae_norm: float, gamma: float) -> float:
    """Lower is better: balance stability and predictive performance."""
    return gamma * (1.0 - kuncheva_score) + (1.0 - gamma) * mae_norm
