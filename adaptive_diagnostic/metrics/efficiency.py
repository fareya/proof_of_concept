"""Diagnostic-efficiency metrics (the RQ1 headline).

The efficiency curve plots held-out item-correctness AUC against the number of probes
administered. From it we read the two headline numbers: *items-to-target* (how many
probes to reach a target AUC) and the *area under the efficiency curve*.

AUC is implemented in numpy so the synthetic loop and smoke test stay dependency-light;
for real runs, ``sklearn.metrics.roc_auc_score`` is equivalent and faster on large sets.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (handles ties).

    Returns 0.5 (chance) when only one class is present, so an undefined slice of the
    efficiency curve degrades gracefully rather than raising.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average ranks within tied scores.
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def area_under_efficiency_curve(aucs: Sequence[float]) -> float:
    """Mean AUC across probe counts — a single scalar summary of the whole curve."""
    aucs = np.asarray(aucs, dtype=float)
    return float(np.mean(aucs)) if aucs.size else float("nan")


def items_to_target(aucs: Sequence[float], target: float) -> Optional[int]:
    """Number of probes (1-indexed) at which the curve first reaches ``target`` AUC.

    Returns ``None`` if the target is never reached within the budget.
    """
    for i, auc in enumerate(aucs):
        if auc >= target:
            return i + 1
    return None
