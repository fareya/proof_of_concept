"""Estimator agent (STUB, RQ3).

Parses the student's free-text reply back into a structured response — the chosen option
and, when wrong, the implied distractor/misconception class — then hands it to the
numeric estimator's ``update``. Use constrained decoding or a small dedicated parser so
the structured signal driving the belief stays reliable.
"""

from __future__ import annotations

from ..data.schemas import Interaction, Item


class EstimatorAgent:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - stub
        raise NotImplementedError(
            "EstimatorAgent is a Gate-G3 stub. Parse free-text reply -> Interaction "
            "(chosen option + implied misconception) via constrained decoding or a "
            "small parser, then call the numeric Estimator.update."
        )

    def parse(self, item: Item, reply_text: str) -> Interaction:  # pragma: no cover
        raise NotImplementedError
