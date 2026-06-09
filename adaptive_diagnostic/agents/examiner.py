"""Examiner agent (STUB, RQ3).

Renders the selected structured probe as a natural-language question, optionally in the
context of the dialogue so far, so the loop reads as a tutoring conversation rather than
a bare item sequence. Inference-only in v1 (no LLM training required).
"""

from __future__ import annotations

from ..data.schemas import Item


class ExaminerAgent:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - stub
        raise NotImplementedError(
            "ExaminerAgent is a Gate-G3 stub. Render Item -> natural-language question "
            "(LLM, inference only). Keep the structured Item as the source of truth so "
            "the estimator still updates on a parsed option, not free text."
        )

    def render(self, item: Item, history: list | None = None) -> str:  # pragma: no cover
        raise NotImplementedError
