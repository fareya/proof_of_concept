"""Selector interface.

A selector picks the next item to administer from a candidate pool, given the current
estimator belief and what has already been asked. This is the locus of RQ1: the
information-gain selector vs. the non-adaptive baselines.
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence

import numpy as np

from ..data.schemas import Item
from ..estimator.base import Estimator


class Selector(abc.ABC):
    """Choose the next probe."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()

    @abc.abstractmethod
    def select(
        self,
        candidates: Sequence[Item],
        estimator: Estimator,
        asked: set[int],
    ) -> Item:
        """Return the next item to administer (must not already be in ``asked``)."""

    @staticmethod
    def _available(candidates: Sequence[Item], asked: set[int]) -> list[Item]:
        return [it for it in candidates if it.item_id not in asked]
