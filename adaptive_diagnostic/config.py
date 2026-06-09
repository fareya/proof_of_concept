"""Experiment configuration.

A single dataclass keeps the synthetic experiment reproducible and self-documenting.
``from_yaml`` is optional sugar for when you start running real-dataset sweeps; it is
guarded so the core loop never hard-depends on PyYAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    # --- synthetic world ---
    n_students: int = 60
    n_items: int = 80
    n_kcs: int = 6
    n_misconceptions_per_kc: int = 3
    n_options: int = 4
    ability_std: float = 1.5  # per-KC latent ability spread
    difficulty_std: float = 0.6  # item difficulty spread (< ability_std by design)

    # --- adaptivity loop ---
    budget: int = 20  # number of probes administered per student
    holdout_fraction: float = 0.4  # items reserved for the predictive metric

    # --- estimator (grid-Bayes IRT) ---
    grid_size: int = 41  # discretization of the latent ability axis
    grid_min: float = -4.0
    grid_max: float = 4.0
    dirichlet_prior: float = 1.0  # misconception-head smoothing

    # --- protocol ---
    seeds: tuple[int, ...] = (0, 1, 2)
    selectors: tuple[str, ...] = (
        "info_gain",
        "random",
        "fixed_difficulty",
        "max_coverage",
        "fisher",
    )

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        try:
            import yaml  # optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only with PyYAML absent
            raise ImportError(
                "PyYAML is required for ExperimentConfig.from_yaml; "
                "install with `pip install adaptive-diagnostic[kt]`."
            ) from exc
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        extra = {k: v for k, v in raw.items() if k not in cls.__dataclass_fields__}
        if extra:
            known["extra"] = extra
        return cls(**known)
