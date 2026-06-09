"""Dialogue orchestrator (STUB, RQ3).

Wires examiner agent + estimator agent + simulated student into the same
select -> ask -> answer -> update loop as the structured experiment, but in natural
language. Orchestrate with LangGraph or AutoGen. The key RQ3 question is whether the NL
wrapper *preserves* the efficiency/accuracy advantage of the structured selector — so
this should reuse the exact selector/estimator objects, only swapping how the probe is
posed and how the reply is parsed.
"""

from __future__ import annotations


class DialogueOrchestrator:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - stub
        raise NotImplementedError(
            "DialogueOrchestrator is a Gate-G3 stub. Reuse the structured selector + "
            "estimator unchanged; wrap probe rendering (ExaminerAgent) and reply parsing "
            "(EstimatorAgent) around them, orchestrated with LangGraph/AutoGen. See "
            "experiment/run_adaptivity.py:run_session for the loop to mirror."
        )
