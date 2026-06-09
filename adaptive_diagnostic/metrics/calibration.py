"""Belief calibration.

A diagnostic that is confidently wrong is worse than one that is uncertain, so we track
the Expected Calibration Error (ECE) of the estimator's predicted correctness
probabilities against realized outcomes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> float:
    """Equal-width binned ECE.

    Args:
        probs: predicted P(correct) for each held-out item.
        outcomes: realized correctness (0/1).
        n_bins: number of equal-width confidence bins on [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probs.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(outcomes[mask]))
        ece += (np.sum(mask) / probs.size) * abs(conf - acc)
    return float(ece)
