"""Estimator interface.

An estimator maintains a *belief* over a single student's latent state and exposes
the two predictive quantities the selector needs:

  * ``predict_correct(item)``       -> P(correct) under the current belief, and
  * ``predict_misconception(item)`` -> P(misconception class | incorrect), for RQ2.

The selector treats the estimator as a black box, so swapping the numpy grid-Bayes
estimator for a torch DKT/AKT model (``estimator/dkt.py``) requires no selector change.
"""

from __future__ import annotations

import abc

import numpy as np

from ..data.schemas import Interaction, Item


class Estimator(abc.ABC):
    """Belief over one student's knowledge/misconception state."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Return the belief to its prior (start of a new student's session)."""

    @abc.abstractmethod
    def update(self, item: Item, interaction: Interaction) -> None:
        """Condition the belief on an observed response."""

    @abc.abstractmethod
    def predict_correct(self, item: Item) -> float:
        """Posterior predictive P(student answers ``item`` correctly)."""

    @abc.abstractmethod
    def predict_misconception(self, item: Item) -> np.ndarray:
        """Posterior P(misconception class | incorrect) over ``item.misconception_classes``.

        Returns a probability vector aligned with ``item.misconception_classes``;
        an empty array when the item carries no misconception labels.
        """

    # --- quantities used by the information-gain selector --------------------

    @abc.abstractmethod
    def kc_belief_entropy(self, kc: int) -> float:
        """Entropy of the current belief over the latent state of one KC (nats)."""
