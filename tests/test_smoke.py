"""End-to-end smoke test for the adaptivity loop.

Tiny and fast: asserts the full estimator -> selector -> simulated-student -> metrics
loop runs and produces a sensible efficiency curve, and sanity-checks the individual
runnable components. Heavier stubs (DKT, NL agents, real loaders) are not exercised.
"""

from __future__ import annotations

import numpy as np

from adaptive_diagnostic.config import ExperimentConfig
from adaptive_diagnostic.data.synthetic import SyntheticWorld
from adaptive_diagnostic.environment.student import SimulatedStudent
from adaptive_diagnostic.estimator.bayesian_irt import BayesianIRTEstimator
from adaptive_diagnostic.experiment.run_adaptivity import run_experiment, run_selector
from adaptive_diagnostic.metrics.efficiency import items_to_target, roc_auc
from adaptive_diagnostic.selector import build_selector


def _tiny_config(**overrides) -> ExperimentConfig:
    base = dict(
        n_students=20,
        n_items=30,
        n_kcs=3,
        budget=10,
        holdout_fraction=0.4,
        seeds=(0,),
        selectors=("info_gain", "random", "fisher"),
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_world_and_student_respond():
    world = SyntheticWorld(n_students=5, n_items=10, n_kcs=2, seed=0)
    student = SimulatedStudent(world, world.students[0])
    item = world.items[0]
    interaction = student.answer(item)
    assert interaction.item_id == item.item_id
    assert 0 <= interaction.chosen_option < item.n_options
    assert interaction.correct == (interaction.chosen_option == item.correct_option)


def test_estimator_belief_updates_and_is_normalized():
    world = SyntheticWorld(n_students=5, n_items=10, n_kcs=2, seed=1)
    est = BayesianIRTEstimator(n_kcs=2, grid_size=21)
    item = world.items[0]
    before = est.kc_belief_entropy(item.kc)
    # Feed several correct answers; entropy should not increase and belief stays valid.
    student = SimulatedStudent(world, world.students[0])
    for it in world.items[:5]:
        est.update(it, student.answer(it))
    for kc in range(2):
        np.testing.assert_allclose(est.belief[kc].sum(), 1.0, atol=1e-9)
    after = est.kc_belief_entropy(item.kc)
    assert after <= before + 1e-6  # observing evidence should not add uncertainty


def test_info_gain_select_is_non_repeating_and_valid():
    world = SyntheticWorld(n_students=5, n_items=12, n_kcs=2, seed=2)
    est = BayesianIRTEstimator(n_kcs=2, grid_size=21)
    selector = build_selector("info_gain", rng=np.random.default_rng(0))
    asked: set[int] = set()
    for _ in range(5):
        item = selector.select(world.items, est, asked)
        assert item.item_id not in asked
        asked.add(item.item_id)
        est.update(item, SimulatedStudent(world, world.students[0]).answer(item))
    assert len(asked) == 5


def test_roc_auc_matches_known_values():
    # Perfect separation -> 1.0; reversed -> 0.0; single class -> 0.5.
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([1, 1, 1], [0.3, 0.6, 0.9]) == 0.5


def test_end_to_end_efficiency_curve_runs():
    config = _tiny_config()
    result = run_selector(config, "info_gain", seed=0)
    curve = result.efficiency_curve
    assert len(curve) == min(config.budget, int(config.n_items * (1 - config.holdout_fraction)))
    assert all(0.0 <= a <= 1.0 for a in curve)
    # A learning estimator should end up better than chance on held-out prediction.
    assert result.final_auc > 0.55
    assert items_to_target(curve, 0.5) == 1  # chance is reached immediately


def test_full_experiment_over_all_selectors():
    config = _tiny_config()
    results = run_experiment(config)
    assert set(results) == set(config.selectors)
    for name, runs in results.items():
        assert len(runs) == len(config.seeds)
        for r in runs:
            assert np.isfinite(r.final_auc)
