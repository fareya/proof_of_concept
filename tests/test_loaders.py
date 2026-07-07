"""Tests for the real-dataset loaders (Gate G0).

These parse tiny in-format fixtures (synthetic data in the real on-disk layouts) and
check that the output drops cleanly into the schema + estimator the rest of the pipeline
uses.
"""

from __future__ import annotations

import os

import pytest

from adaptive_diagnostic.data import loaders
from adaptive_diagnostic.data.schemas import Interaction, Item
from adaptive_diagnostic.estimator.bayesian_irt import BayesianIRTEstimator

FIXTURES = os.path.join(os.path.dirname(loaders.__file__), "fixtures")


def _path(name: str) -> str:
    return os.path.join(FIXTURES, name)


# --------------------------------------------------------------------- long CSV
def test_load_long_csv_sequences_and_ordering():
    log = loaders.load_long_csv(_path("long_sample.csv"))
    assert set(log.sequences) == {1, 2}
    assert len(log.sequences[1]) == 3
    assert len(log.sequences[2]) == 3
    # Student 1's responses are ordered by timestamp: correct, wrong, wrong.
    assert [ix.correct for ix in log.sequences[1]] == [True, False, False]
    assert all(isinstance(ix, Interaction) for ix in log.sequences[1])


def test_load_long_csv_builds_misconception_map():
    log = loaders.load_long_csv(_path("long_sample.csv"))
    by_id = {it.item_id: it for it in log.items}
    # Item 11 was answered wrong as option C(=2) with misconception 5.
    assert by_id[11].misconception_by_option == {2: 5}
    # Item 12 was answered wrong as option B(=1) with misconception 7.
    assert by_id[12].misconception_by_option == {1: 7}
    # Item 10 was answered both correctly and (option B) wrong with misconception 4.
    assert by_id[10].misconception_by_option == {1: 4}


def test_long_csv_requires_core_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("user_id,item_id,correct\n1,2,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        loaders.load_long_csv(str(bad))


# ---------------------------------------------------------------- KT triplet txt
def test_load_kt_text():
    log = loaders.load_kt_text(_path("kt_sample.txt"))
    assert set(log.sequences) == {0, 1}
    assert len(log.sequences[0]) == 3
    assert len(log.sequences[1]) == 4
    # In the triplet format the id doubles as the KC.
    for it in log.items:
        assert it.kc == it.item_id
    # Distinct ids across both students: {1,2,3,5}.
    assert {it.item_id for it in log.items} == {1, 2, 3, 5}


def test_kt_text_length_mismatch_raises(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("3\n1,2\n1,0,1\n", encoding="utf-8")  # 2 ids but declares 3
    with pytest.raises(ValueError, match="length mismatch"):
        loaders.load_kt_text(str(bad))


# --------------------------------------------------------------- pyKT dispatch
def test_load_pykt_sequences_dispatches_by_extension():
    assert loaders.load_pykt_sequences("kt_sample", FIXTURES).sequences  # .txt
    assert loaders.load_pykt_sequences("long_sample", FIXTURES).sequences  # .csv


def test_load_pykt_sequences_missing_file():
    with pytest.raises(FileNotFoundError):
        loaders.load_pykt_sequences("does_not_exist", FIXTURES)


# ----------------------------------------------------------------------- Eedi
def test_load_eedi_items():
    items = loaders.load_eedi_items(_path("eedi_sample.csv"))
    by_id = {it.item_id: it for it in items}
    assert set(by_id) == {100, 101}

    q100 = by_id[100]
    assert q100.correct_option == 1  # CorrectAnswer=2, 1-indexed -> index 1 (B)
    assert q100.kc == 33  # ConstructId
    assert q100.misconception_by_option == {2: 41, 3: 55}  # C and D distractors

    q101 = by_id[101]
    assert q101.correct_option == 0  # CorrectAnswer=1 -> index 0 (A)
    assert q101.misconception_by_option == {1: 12, 2: 13, 3: 14}


# ------------------------------------------------------------- integration
def test_loaded_log_drives_the_estimator():
    """A loaded sequence must update the grid-Bayes estimator without surprises."""
    log = loaders.load_long_csv(_path("long_sample.csv"))
    n_kcs = max(it.kc for it in log.items) + 1
    est = BayesianIRTEstimator(n_kcs=n_kcs, grid_size=21)
    item_by_id = {it.item_id: it for it in log.items}

    for interaction in log.sequences[1]:
        item = item_by_id[interaction.item_id]
        p = est.predict_correct(item)
        assert 0.0 <= p <= 1.0
        est.update(item, interaction)

    # Belief stays a valid distribution after replaying real interactions.
    import numpy as np

    for kc in range(n_kcs):
        np.testing.assert_allclose(est.belief[kc].sum(), 1.0, atol=1e-9)
