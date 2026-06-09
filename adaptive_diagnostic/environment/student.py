"""Simulated-student environments.

``SimulatedStudent`` wraps a synthetic ground-truth profile and answers items via the
IRT response model — this is the trusted, reproducible environment used for the RQ1/RQ2
experiments. ``LLMSimulatedStudent`` is a stub for the RQ3 closed loop: an LLM
conditioned on a held-out real response pattern, whose turn-level behaviour must be
*validated against QATD-2k* (Scarlatos et al., 2026) before it is trusted as an env.
"""

from __future__ import annotations

from ..data.schemas import Interaction, Item, StudentProfile
from ..data.synthetic import SyntheticWorld


class SimulatedStudent:
    """A synthetic student that answers items from a known latent profile."""

    def __init__(self, world: SyntheticWorld, profile: StudentProfile) -> None:
        self.world = world
        self.profile = profile

    def answer(self, item: Item) -> Interaction:
        return self.world.respond(self.profile, item)

    def true_p_correct(self, item: Item) -> float:
        """Ground-truth P(correct) — used only for diagnostics, never by the loop."""
        return self.world.p_correct(self.profile, item)


class LLMSimulatedStudent:
    """LLM-backed simulated student (STUB, RQ3).

    Intended design:
      * Condition an LLM on a held-out real student's response history / compact profile
        (cf. history-conditioned simulators, arXiv:2605.30051).
      * Answer the examiner agent's natural-language question; the estimator agent
        parses the reply back into a structured (option, distractor) response.
      * Before use as an environment, validate turn-level fidelity against QATD-2k with
        the metrics of Scarlatos et al. (2026).
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - stub
        raise NotImplementedError(
            "LLMSimulatedStudent is a Gate-G3 stub. Implement an LLM conditioned on a "
            "held-out real response pattern and validate it against QATD-2k before "
            "trusting it as an environment. Use SimulatedStudent for RQ1/RQ2."
        )
