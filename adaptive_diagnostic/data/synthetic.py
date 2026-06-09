"""Synthetic item bank and student population with known ground truth.

The generator is an IRT (2-parameter logistic) world augmented with misconception
labels on distractors. It exists so the whole adaptivity loop is runnable and the
predictive metric is *grounded* (we know the true held-out responses) before any real
data is wired in. Replace with ``data/loaders.py`` outputs when real sequences exist.
"""

from __future__ import annotations

import numpy as np

from .schemas import Item, Interaction, StudentProfile


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class SyntheticWorld:
    """A reproducible population of items + students with latent ground truth."""

    def __init__(
        self,
        n_students: int,
        n_items: int,
        n_kcs: int,
        n_misconceptions_per_kc: int = 3,
        n_options: int = 4,
        ability_std: float = 1.5,
        difficulty_std: float = 0.6,
        seed: int = 0,
    ) -> None:
        # ability_std > difficulty_std makes per-KC ability (which must be *learned* by
        # probing) the dominant driver of held-out correctness, rather than item
        # difficulty (which the prior already knows). This is what lets the efficiency
        # curve rise with probes and gives adaptive selection room to win on RQ1.
        self.rng = np.random.default_rng(seed)
        self.n_kcs = n_kcs
        self.n_misconceptions_per_kc = n_misconceptions_per_kc
        self.ability_std = ability_std
        self.difficulty_std = difficulty_std
        self.items = self._make_items(n_items, n_kcs, n_misconceptions_per_kc, n_options)
        self.students = self._make_students(n_students, n_kcs)
        self._item_by_id = {it.item_id: it for it in self.items}

    # ------------------------------------------------------------------ items
    def _make_items(
        self, n_items: int, n_kcs: int, n_mis: int, n_options: int
    ) -> list[Item]:
        items: list[Item] = []
        for i in range(n_items):
            kc = int(self.rng.integers(n_kcs))
            difficulty = float(self.rng.normal(0.0, self.difficulty_std))
            discrimination = float(np.clip(self.rng.normal(1.0, 0.25), 0.3, 2.5))
            correct = int(self.rng.integers(n_options))
            # Map each incorrect option to one of this KC's misconception classes.
            misconception_by_option: dict[int, int] = {}
            for opt in range(n_options):
                if opt == correct:
                    continue
                cls = kc * n_mis + int(self.rng.integers(n_mis))
                misconception_by_option[opt] = cls
            items.append(
                Item(
                    item_id=i,
                    kc=kc,
                    difficulty=difficulty,
                    discrimination=discrimination,
                    n_options=n_options,
                    correct_option=correct,
                    misconception_by_option=misconception_by_option,
                )
            )
        return items

    # --------------------------------------------------------------- students
    def _make_students(self, n_students: int, n_kcs: int) -> list[StudentProfile]:
        students: list[StudentProfile] = []
        for s in range(n_students):
            ability = self.rng.normal(0.0, self.ability_std, size=n_kcs).tolist()
            # Each student has a preferred misconception class per KC.
            pref = {
                kc: kc * self.n_misconceptions_per_kc
                + int(self.rng.integers(self.n_misconceptions_per_kc))
                for kc in range(n_kcs)
            }
            students.append(
                StudentProfile(student_id=s, ability=ability, misconception_pref=pref)
            )
        return students

    # ------------------------------------------------------------- responses
    def item(self, item_id: int) -> Item:
        return self._item_by_id[item_id]

    def p_correct(self, student: StudentProfile, item: Item) -> float:
        theta = student.ability[item.kc]
        return float(sigmoid(item.discrimination * (theta - item.difficulty)))

    def respond(self, student: StudentProfile, item: Item) -> Interaction:
        """Sample a response: correct via 2PL; if incorrect, pick a distractor
        biased toward the student's preferred misconception class for this KC."""
        p = self.p_correct(student, item)
        if self.rng.random() < p:
            return Interaction(
                item_id=item.item_id,
                chosen_option=item.correct_option,
                correct=True,
                misconception=None,
            )
        chosen = self._sample_distractor(student, item)
        return Interaction(
            item_id=item.item_id,
            chosen_option=chosen,
            correct=False,
            misconception=item.misconception_by_option.get(chosen),
        )

    def _sample_distractor(self, student: StudentProfile, item: Item) -> int:
        distractors = [o for o in range(item.n_options) if o != item.correct_option]
        pref_class = student.misconception_pref.get(item.kc)
        weights = np.array(
            [
                3.0 if item.misconception_by_option.get(o) == pref_class else 1.0
                for o in distractors
            ],
            dtype=float,
        )
        weights /= weights.sum()
        return int(self.rng.choice(distractors, p=weights))
