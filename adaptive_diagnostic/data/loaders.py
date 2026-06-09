"""Real-dataset loaders (STUBS, Gate G0 / W1-2).

Each loader should return the same ``Item`` / interaction-sequence structures the
synthetic world produces, so the estimator, selector, and experiment harness need no
changes when moving from synthetic to real data.

Datasets (see proposal 3.1):
  * pyKT / ASSISTments / Statics2011 -> per-student (item, KC, correct) sequences (RQ1).
  * Eedi "Mining Misconceptions"     -> MCQ items with distractor->misconception maps (RQ2).
  * QATD-2k (+ optionally MathDial)   -> real tutoring dialogues for face validity (RQ3).

LICENSING: QATD-2k is CC-BY-NC-SA and UK-only — flag this for any bias discussion, and
do not commit raw data (see .gitignore).
"""

from __future__ import annotations

from typing import Iterator

from .schemas import Interaction, Item


def load_pykt_sequences(dataset: str, data_dir: str) -> Iterator[list[Interaction]]:
    """Yield per-student interaction sequences from a pyKT-format dataset."""
    raise NotImplementedError(
        "G0 stub. Parse pyKT-preprocessed CSVs (e.g. ASSISTments2009, Statics2011) into "
        "per-student lists of Interaction, plus an Item bank carrying KC ids. Keep the "
        "Item/Interaction schema so the rest of the pipeline is unchanged."
    )


def load_eedi_items(data_dir: str) -> list[Item]:
    """Load Eedi MCQ items with distractor->misconception annotations (RQ2 target)."""
    raise NotImplementedError(
        "G0 stub. Build Item objects with misconception_by_option populated from the "
        "Eedi distractor->misconception mapping; these supervise the misconception head."
    )


def load_qatd_dialogues(data_dir: str):
    """Load QATD-2k tutoring dialogues for face-validity checks / simulator calibration."""
    raise NotImplementedError(
        "G3 stub. Load QATD-2k turns for (a) face-validity surface stats vs generated "
        "dialogues and (b) validating the LLM simulated student. Respect CC-BY-NC-SA, "
        "UK-only licensing; do not commit raw data."
    )
