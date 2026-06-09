"""Expected-information-gain probe selector (the RQ1 method).

For each candidate item we estimate the expected reduction in uncertainty about the
target if that item were administered, then take the argmax. This is the
Bayesian-active-learning / BALD criterion (Houlsby et al., 2011) specialized to the
diagnostic loop, approximated by marginalizing over the predicted response.

Two targets are supported:
  * ``target="kc"``           -> reduce entropy of the latent ability belief for the
                                 item's KC (RQ1, proficiency-level diagnosis).
  * ``target="misconception"``-> reduce entropy of the misconception-class belief (RQ2).

For a binary-response item the expected posterior entropy is

    E[H(posterior)] = p * H(posterior | correct) + (1 - p) * H(posterior | incorrect),

where ``p`` is the current posterior-predictive probability of a correct answer. The
gain is the current entropy minus that expectation; it is always non-negative.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schemas import Item
from ..estimator.base import Estimator
from ..estimator.bayesian_irt import BayesianIRTEstimator, _entropy, _sigmoid
from .base import Selector


class InfoGainSelector(Selector):
    def __init__(self, target: str = "kc", **kwargs) -> None:
        super().__init__(**kwargs)
        if target not in ("kc", "misconception"):
            raise ValueError(f"target must be 'kc' or 'misconception', got {target!r}")
        self.target = target

    def select(
        self, candidates: Sequence[Item], estimator: Estimator, asked: set[int]
    ) -> Item:
        available = self._available(candidates, asked)
        if not available:
            raise ValueError("No candidate items remain to select from.")
        gains = np.array([self._expected_gain(it, estimator) for it in available])
        # Break ties at random for fair comparison against the baselines.
        best = np.flatnonzero(gains >= gains.max() - 1e-12)
        return available[int(self.rng.choice(best))]

    # ------------------------------------------------------------------ gains
    def _expected_gain(self, item: Item, estimator: Estimator) -> float:
        if self.target == "misconception":
            return self._misconception_gain(item, estimator)
        return self._kc_gain(item, estimator)

    def _kc_gain(self, item: Item, estimator: Estimator) -> float:
        """Expected entropy reduction of the KC ability belief.

        Exact for :class:`BayesianIRTEstimator` (we can form both hypothetical
        posteriors over the grid); for other estimators we fall back to the predictive
        Bernoulli entropy, which still rewards items the model is most uncertain about.
        """
        if isinstance(estimator, BayesianIRTEstimator):
            belief = estimator.belief[item.kc]
            like_c = _sigmoid(item.discrimination * (estimator.grid - item.difficulty))
            p_correct = float(np.dot(belief, like_c))
            prior_h = _entropy(belief)

            post_c = belief * like_c
            post_i = belief * (1.0 - like_c)
            post_c = post_c / post_c.sum() if post_c.sum() > 0 else belief
            post_i = post_i / post_i.sum() if post_i.sum() > 0 else belief
            expected_h = p_correct * _entropy(post_c) + (1 - p_correct) * _entropy(post_i)
            return prior_h - expected_h

        # Estimator-agnostic fallback: predictive uncertainty (max at p = 0.5).
        p = estimator.predict_correct(item)
        p = min(max(p, 1e-9), 1 - 1e-9)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))

    def _misconception_gain(self, item: Item, estimator: Estimator) -> float:
        """Reward incorrect-likely items that sharpen the misconception belief (RQ2).

        Information about *which* misconception only arrives on an incorrect answer, so
        we weight the misconception-belief entropy by P(incorrect).
        """
        classes = item.misconception_classes
        if not classes:
            return 0.0
        p_incorrect = 1.0 - estimator.predict_correct(item)
        p_mis = estimator.predict_misconception(item)
        return float(p_incorrect * _entropy(p_mis))
