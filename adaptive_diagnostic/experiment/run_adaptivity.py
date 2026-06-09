"""RQ1 adaptivity experiment (end-to-end, synthetic).

For each (seed, selector) we run the diagnostic loop over a held-out population of
students and measure the diagnostic-efficiency curve: after each administered probe, we
freeze the belief and predict every student's *held-out* item responses, scoring AUC.
The information-gain selector should reach a target AUC in fewer probes than the
baselines (random / fixed-difficulty / max-coverage / Fisher).

This is the harness the proposal's Gate G1 ("adaptive reaches the fixed policy's
full-length AUC with >=25% fewer items") will be evaluated on once real estimators and
datasets are wired in. Run:

    python -m adaptive_diagnostic.experiment.run_adaptivity --students 60 --items 80
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import ExperimentConfig
from ..data.schemas import Item
from ..data.synthetic import SyntheticWorld
from ..environment.student import SimulatedStudent
from ..estimator.bayesian_irt import BayesianIRTEstimator
from ..metrics.calibration import expected_calibration_error
from ..metrics.efficiency import (
    area_under_efficiency_curve,
    items_to_target,
    roc_auc,
)
from ..selector import build_selector


@dataclass
class SelectorResult:
    name: str
    efficiency_curve: list[float]  # mean held-out AUC after each probe count
    final_auc: float
    auec: float  # area under the efficiency curve
    ece: float
    items_to_90pct: int | None  # probes to reach 90% of the full-length AUC


def _split_items(
    items: Sequence[Item], holdout_fraction: float, rng: np.random.Generator
) -> tuple[list[Item], list[Item]]:
    """Split the item bank into a probe pool and a held-out evaluation pool."""
    idx = rng.permutation(len(items))
    n_holdout = max(1, int(round(holdout_fraction * len(items))))
    holdout = [items[i] for i in idx[:n_holdout]]
    probe_pool = [items[i] for i in idx[n_holdout:]]
    return probe_pool, holdout


def run_session(
    student: SimulatedStudent,
    probe_pool: Sequence[Item],
    holdout: Sequence[Item],
    estimator: BayesianIRTEstimator,
    selector,
    budget: int,
) -> tuple[list[float], list[tuple[float, int]]]:
    """Run one student's diagnostic session.

    Returns:
        per_probe_auc_inputs: for each probe step, the list of (pred, true) over the
            held-out items is reduced upstream; here we return the AUC building blocks.
        calibration_pairs: (predicted P(correct), realized correctness) on held-out
            items at the *final* belief, for ECE.
    """
    estimator.reset()
    asked: set[int] = set()
    # True held-out outcomes are fixed for this student (sample once for a fair target).
    holdout_truth = [(it, student.answer(it).correct) for it in holdout]

    per_probe_aucs: list[tuple[list[float], list[int]]] = []
    n_steps = min(budget, len(probe_pool))
    for _ in range(n_steps):
        item = selector.select(probe_pool, estimator, asked)
        interaction = student.answer(item)
        estimator.update(item, interaction)
        asked.add(item.item_id)

        preds = [estimator.predict_correct(it) for it, _ in holdout_truth]
        truth = [int(c) for _, c in holdout_truth]
        per_probe_aucs.append((preds, truth))

    # Final-belief calibration pairs.
    calib = [
        (estimator.predict_correct(it), int(c)) for it, c in holdout_truth
    ]
    # Encode the per-probe (preds, truth) as AUC-ready tuples for aggregation.
    return per_probe_aucs, calib


def run_selector(
    config: ExperimentConfig, selector_name: str, seed: int
) -> SelectorResult:
    rng = np.random.default_rng(seed)
    world = SyntheticWorld(
        n_students=config.n_students,
        n_items=config.n_items,
        n_kcs=config.n_kcs,
        n_misconceptions_per_kc=config.n_misconceptions_per_kc,
        n_options=config.n_options,
        ability_std=config.ability_std,
        difficulty_std=config.difficulty_std,
        seed=seed,
    )
    probe_pool, holdout = _split_items(world.items, config.holdout_fraction, rng)
    budget = min(config.budget, len(probe_pool))

    selector = build_selector(
        selector_name, rng=np.random.default_rng(seed + 1), **_selector_kwargs(selector_name, config)
    )

    # Accumulate held-out (pred, truth) across students at each probe step.
    step_preds: list[list[float]] = [[] for _ in range(budget)]
    step_truth: list[list[int]] = [[] for _ in range(budget)]
    calib_probs: list[float] = []
    calib_outcomes: list[int] = []

    for profile in world.students:
        estimator = BayesianIRTEstimator(
            n_kcs=config.n_kcs,
            grid_size=config.grid_size,
            grid_min=config.grid_min,
            grid_max=config.grid_max,
            dirichlet_prior=config.dirichlet_prior,
        )
        student = SimulatedStudent(world, profile)
        per_probe, calib = run_session(
            student, probe_pool, holdout, estimator, selector, budget
        )
        for step, (preds, truth) in enumerate(per_probe):
            step_preds[step].extend(preds)
            step_truth[step].extend(truth)
        for p, o in calib:
            calib_probs.append(p)
            calib_outcomes.append(o)

    efficiency_curve = [
        roc_auc(step_truth[s], step_preds[s]) for s in range(budget)
    ]
    final_auc = efficiency_curve[-1] if efficiency_curve else float("nan")
    target = 0.9 * final_auc
    return SelectorResult(
        name=selector_name,
        efficiency_curve=efficiency_curve,
        final_auc=final_auc,
        auec=area_under_efficiency_curve(efficiency_curve),
        ece=expected_calibration_error(calib_probs, calib_outcomes),
        items_to_90pct=items_to_target(efficiency_curve, target),
    )


def _selector_kwargs(name: str, config: ExperimentConfig) -> dict:
    if name == "max_coverage":
        return {"n_kcs": config.n_kcs}
    if name == "info_gain":
        return {"target": "kc"}
    return {}


def run_experiment(config: ExperimentConfig) -> dict[str, list[SelectorResult]]:
    """Run every selector across every seed. Returns {selector_name: [per-seed result]}."""
    results: dict[str, list[SelectorResult]] = {name: [] for name in config.selectors}
    for seed in config.seeds:
        for name in config.selectors:
            results[name].append(run_selector(config, name, seed))
    return results


# --------------------------------------------------------------------- reporting
# Non-adaptive policies whose full-length AUC defines the shared Gate-G1 target.
NON_ADAPTIVE_BASELINES = ("random", "fixed_difficulty", "max_coverage")


def _mean_curve(runs: list[SelectorResult]) -> np.ndarray:
    return np.mean(np.array([r.efficiency_curve for r in runs]), axis=0)


def _format_report(results: dict[str, list[SelectorResult]], config: ExperimentConfig) -> str:
    mean_curves = {name: _mean_curve(runs) for name, runs in results.items()}

    # Shared target = the strongest non-adaptive baseline's full-length AUC. Reaching
    # *this* in fewer probes is the RQ1 claim (a conservative, comparable bar).
    present_baselines = [b for b in NON_ADAPTIVE_BASELINES if b in mean_curves]
    ref_target = (
        max(float(mean_curves[b][-1]) for b in present_baselines)
        if present_baselines
        else max(float(c[-1]) for c in mean_curves.values())
    )
    # Probes the reference baseline itself needs (denominator for the % reduction).
    ref_items = None
    for b in present_baselines:
        hit = items_to_target(mean_curves[b], ref_target)
        if hit is not None:
            ref_items = hit if ref_items is None else max(ref_items, hit)

    lines = ["Adaptive Diagnostic Dialogue — RQ1 efficiency (synthetic)"]
    lines.append(
        f"students={config.n_students} items={config.n_items} kcs={config.n_kcs} "
        f"budget={config.budget} holdout={config.holdout_fraction} seeds={list(config.seeds)}"
    )
    lines.append(f"shared target AUC = {ref_target:.3f} (best non-adaptive baseline, full budget)")
    lines.append("")
    header = f"{'selector':<18}{'final AUC':>12}{'AUEC':>10}{'items→target':>14}{'ECE':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, runs in results.items():
        final = np.array([r.final_auc for r in runs])
        auec = np.array([r.auec for r in runs])
        ece = np.array([r.ece for r in runs])
        hit = items_to_target(mean_curves[name], ref_target)
        itt_str = str(hit) if hit is not None else ">budget"
        lines.append(
            f"{name:<18}{final.mean():>9.3f}±{final.std():<2.2f}"
            f"{auec.mean():>10.3f}{itt_str:>14}{ece.mean():>9.3f}"
        )

    lines.append("")
    ig_items = items_to_target(mean_curves["info_gain"], ref_target) if "info_gain" in mean_curves else None
    if ig_items is not None and ref_items:
        reduction = 100.0 * (ref_items - ig_items) / ref_items
        direction = "fewer" if reduction >= 0 else "MORE"
        lines.append(
            f"RQ1: info_gain reaches the shared target in {ig_items} probes vs "
            f"{ref_items} for the best non-adaptive baseline "
            f"({abs(reduction):.0f}% {direction} items; Gate G1 wants ≥25% fewer)."
        )
    else:
        lines.append(
            "RQ1: at least one policy did not reach the shared target within budget — "
            "raise --budget or adjust the synthetic world."
        )
    lines.append(
        "(Synthetic sanity check only; the real test is this harness on pyKT/Eedi with a "
        "DKT/AKT estimator. Numbers vary with seeds and world parameters.)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RQ1 adaptivity experiment (synthetic).")
    parser.add_argument("--students", type=int, default=60)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--kcs", type=int, default=6)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--holdout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)

    config = ExperimentConfig(
        n_students=args.students,
        n_items=args.items,
        n_kcs=args.kcs,
        budget=args.budget,
        holdout_fraction=args.holdout,
        seeds=tuple(args.seed),
    )
    results = run_experiment(config)
    print(_format_report(results, config))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
