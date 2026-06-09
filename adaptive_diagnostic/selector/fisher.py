"""Fisher-information (classical CAT) selector.

Anchors the method against the psychometrics literature (Lord, 1980): select the item
that maximizes Fisher information about the latent ability at the current point
estimate. For the 2PL model the item information at ability ``theta`` is

    I(theta) = a^2 * p * (1 - p),     p = sigmoid(a * (theta - b)).

We use the posterior-mean ability as the operating point, mirroring how a CAT system
plugs in its running ability estimate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schemas import Item
from ..estimator.base import Estimator
from ..estimator.bayesian_irt import BayesianIRTEstimator, _sigmoid
from .base import Selector


class FisherInformationSelector(Selector):
    def select(
        self, candidates: Sequence[Item], estimator: Estimator, asked: set[int]
    ) -> Item:
        available = self._available(candidates, asked)
        infos = np.array([self._fisher_info(it, estimator) for it in available])
        best = np.flatnonzero(infos >= infos.max() - 1e-12)
        return available[int(self.rng.choice(best))]

    def _fisher_info(self, item: Item, estimator: Estimator) -> float:
        theta = self._theta_hat(item.kc, estimator)
        p = float(_sigmoid(item.discrimination * (theta - item.difficulty)))
        return float(item.discrimination**2 * p * (1.0 - p))

    @staticmethod
    def _theta_hat(kc: int, estimator: Estimator) -> float:
        if isinstance(estimator, BayesianIRTEstimator):
            return float(np.dot(estimator.belief[kc], estimator.grid))
        return 0.0  # operating point unknown for opaque estimators; use the prior mean
