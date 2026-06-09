"""Core data structures shared across the estimator, selector, and environment.

These are deliberately minimal and framework-agnostic (plain dataclasses, no torch)
so they can describe both synthetic items and real items loaded from pyKT/Eedi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class Item:
    """A single multiple-choice diagnostic item.

    The IRT parameters (``difficulty``, ``discrimination``) drive the synthetic
    response model and the Fisher-information baseline. ``misconception_by_option``
    maps each *incorrect* option index to a named misconception class, mirroring the
    Eedi "distractor-as-misconception" annotation used for RQ2.
    """

    item_id: int
    kc: int  # knowledge component this item primarily probes
    difficulty: float  # IRT b
    discrimination: float = 1.0  # IRT a
    n_options: int = 4
    correct_option: int = 0
    # option index -> misconception class id (only for incorrect options)
    misconception_by_option: dict[int, int] = field(default_factory=dict)

    @property
    def misconception_classes(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.misconception_by_option.values())))


@dataclass(frozen=True)
class Interaction:
    """One observed student response to an item."""

    item_id: int
    chosen_option: int
    correct: bool
    # Misconception class implied by the chosen distractor, if any (RQ2 supervision).
    misconception: Optional[int] = None


@dataclass
class StudentProfile:
    """Latent ground-truth profile for a synthetic student.

    Real students have no observable latent profile; this exists so the synthetic
    environment can (a) generate responses and (b) provide held-out items for the
    falsifiable predictive metric. ``ability[k]`` is the per-KC IRT theta;
    ``misconception_pref[k]`` is the distractor family this student gravitates to
    when they answer KC ``k`` incorrectly.
    """

    student_id: int
    ability: Sequence[float]  # per-KC latent ability (theta), length = n_kcs
    misconception_pref: dict[int, int] = field(default_factory=dict)  # kc -> class id
