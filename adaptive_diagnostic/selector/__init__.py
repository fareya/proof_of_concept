"""Probe selectors: choose the next item to administer."""

from .base import Selector
from .baselines import FixedDifficultySelector, MaxCoverageSelector, RandomSelector
from .fisher import FisherInformationSelector
from .info_gain import InfoGainSelector

REGISTRY: dict[str, type[Selector]] = {
    "info_gain": InfoGainSelector,
    "random": RandomSelector,
    "fixed_difficulty": FixedDifficultySelector,
    "max_coverage": MaxCoverageSelector,
    "fisher": FisherInformationSelector,
}


def build_selector(name: str, **kwargs) -> Selector:
    if name not in REGISTRY:
        raise KeyError(f"Unknown selector {name!r}; options: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "Selector",
    "InfoGainSelector",
    "RandomSelector",
    "FixedDifficultySelector",
    "MaxCoverageSelector",
    "FisherInformationSelector",
    "REGISTRY",
    "build_selector",
]
