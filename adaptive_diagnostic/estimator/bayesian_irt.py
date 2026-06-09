"""Grid-Bayes IRT estimator (numpy, fully runnable).

Belief over each KC's latent ability ``theta_kc`` is a categorical distribution over a
shared discretized grid. Updates are exact Bayesian conditioning under the 2PL model:

    P(correct | theta) = sigmoid(a * (theta - b)).

A per-KC Dirichlet-multinomial head tracks the misconception class a student tends to
choose when wrong, supervised by observed distractor labels (RQ2). This estimator is a
deliberately simple, transparent baseline; ``estimator/dkt.py`` is the deep-model swap.
"""

from __future__ import annotations

import numpy as np

from ..data.schemas import Interaction, Item
from .base import Estimator


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


class BayesianIRTEstimator(Estimator):
    def __init__(
        self,
        n_kcs: int,
        grid_size: int = 41,
        grid_min: float = -4.0,
        grid_max: float = 4.0,
        dirichlet_prior: float = 1.0,
    ) -> None:
        self.n_kcs = n_kcs
        self.grid = np.linspace(grid_min, grid_max, grid_size)
        self.dirichlet_prior = float(dirichlet_prior)
        # Standard-normal prior over the grid for each KC.
        self._prior = np.exp(-0.5 * self.grid**2)
        self._prior /= self._prior.sum()
        # Misconception counts populated lazily per (kc, class).
        self._mis_counts: dict[int, dict[int, float]] = {}
        self.reset()

    # --------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        self.belief = np.tile(self._prior, (self.n_kcs, 1))  # (n_kcs, grid_size)
        self._mis_counts = {}

    # --------------------------------------------------------------- updates
    def _likelihood_correct(self, item: Item) -> np.ndarray:
        """P(correct | theta) evaluated across the grid for this item's KC."""
        return _sigmoid(item.discrimination * (self.grid - item.difficulty))

    def update(self, item: Item, interaction: Interaction) -> None:
        like_correct = self._likelihood_correct(item)
        like = like_correct if interaction.correct else (1.0 - like_correct)
        posterior = self.belief[item.kc] * like
        total = posterior.sum()
        if total > 0:
            self.belief[item.kc] = posterior / total

        if not interaction.correct and interaction.misconception is not None:
            counts = self._mis_counts.setdefault(item.kc, {})
            counts[interaction.misconception] = counts.get(interaction.misconception, 0.0) + 1.0

    # ------------------------------------------------------------- prediction
    def predict_correct(self, item: Item) -> float:
        like_correct = self._likelihood_correct(item)
        return float(np.dot(self.belief[item.kc], like_correct))

    def predict_misconception(self, item: Item) -> np.ndarray:
        classes = item.misconception_classes
        if not classes:
            return np.array([])
        counts = self._mis_counts.get(item.kc, {})
        weights = np.array(
            [self.dirichlet_prior + counts.get(c, 0.0) for c in classes], dtype=float
        )
        return weights / weights.sum()

    # ------------------------------------------------ selector-facing summaries
    def kc_belief_entropy(self, kc: int) -> float:
        return _entropy(self.belief[kc])

    def misconception_entropy(self, item: Item) -> float:
        p = self.predict_misconception(item)
        return _entropy(p) if p.size else 0.0
