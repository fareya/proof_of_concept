"""Non-adaptive / heuristic baselines for RQ1.

These anchor the central claim: the information-gain selector must reach a target
held-out prediction accuracy in *fewer* items than each of these.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schemas import Item
from ..estimator.base import Estimator
from .base import Selector


class RandomSelector(Selector):
    """Uniformly random item from the remaining pool."""

    def select(
        self, candidates: Sequence[Item], estimator: Estimator, asked: set[int]
    ) -> Item:
        available = self._available(candidates, asked)
        return available[int(self.rng.integers(len(available)))]


class FixedDifficultySelector(Selector):
    """Difficulty-ascending sweep — a simple CAT-flavoured non-adaptive policy.

    Ignores the belief entirely and walks items from easiest to hardest, giving a
    fixed, content-balanced ordering to compare adaptivity against.
    """

    def select(
        self, candidates: Sequence[Item], estimator: Estimator, asked: set[int]
    ) -> Item:
        available = self._available(candidates, asked)
        return min(available, key=lambda it: it.difficulty)


class MaxCoverageSelector(Selector):
    """Greedy maximum coverage over knowledge components.

    Prefers KCs that have been probed least so far, spreading the budget across the
    skill space rather than concentrating where the model is uncertain.
    """

    def __init__(self, n_kcs: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_kcs = n_kcs
        self._kc_counts: dict[int, int] = {}

    def reset_counts(self) -> None:
        self._kc_counts = {}

    def select(
        self, candidates: Sequence[Item], estimator: Estimator, asked: set[int]
    ) -> Item:
        available = self._available(candidates, asked)
        # Recompute coverage from `asked` so the selector is stateless across students.
        counts: dict[int, int] = {}
        by_id = {it.item_id: it for it in candidates}
        for item_id in asked:
            kc = by_id[item_id].kc
            counts[kc] = counts.get(kc, 0) + 1
        least = min(available, key=lambda it: (counts.get(it.kc, 0), it.item_id))
        # Among items in the least-covered KC, pick at random for fairness.
        target_kc = least.kc
        pool = [it for it in available if it.kc == target_kc]
        return pool[int(self.rng.integers(len(pool)))]
