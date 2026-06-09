"""Deep knowledge-tracing estimator (STUB).

Target for Gate G1 (W3-5): a torch DKT/AKT/simpleKT model from pyKT, wrapped in the
:class:`~adaptive_diagnostic.estimator.base.Estimator` interface so it drops into the
existing selector + experiment harness unchanged. For RQ2, add a categorical
*misconception head* (per item, a distribution over distractor/misconception classes)
supervised by Eedi distractor labels.

Implementation notes for whoever picks this up:
  * Keep a running encoded history (the RNN/attention hidden state) as the "belief".
  * ``predict_correct`` is the model's next-step correctness probability for the item.
  * ``kc_belief_entropy`` has no exact analogue in a point-estimate DKT; approximate
    uncertainty with MC-dropout or a small deep ensemble so the info-gain selector
    still has a posterior to reduce (this mirrors BALD's epistemic-uncertainty use).
"""

from __future__ import annotations

import numpy as np

from ..data.schemas import Interaction, Item
from .base import Estimator


class DKTEstimator(Estimator):
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - stub
        raise NotImplementedError(
            "DKTEstimator is a stub for Gate G1. Wire a pyKT DKT/AKT/simpleKT model "
            "here behind the Estimator interface, with MC-dropout or an ensemble for "
            "the epistemic uncertainty the info-gain selector consumes. Until then, "
            "use BayesianIRTEstimator."
        )

    def reset(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def update(self, item: Item, interaction: Interaction) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict_correct(self, item: Item) -> float:  # pragma: no cover - stub
        raise NotImplementedError

    def predict_misconception(self, item: Item) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def kc_belief_entropy(self, kc: int) -> float:  # pragma: no cover - stub
        raise NotImplementedError
