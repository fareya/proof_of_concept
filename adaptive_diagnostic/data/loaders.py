"""Real-dataset loaders for the diagnostic loop (Gate G0 / W1-2).

Each loader returns the same ``Item`` / ``Interaction`` structures the synthetic world
produces, so the estimator, selector, and experiment harness need no changes when
moving from synthetic to real data.

Implemented (W1):
  * :func:`load_long_csv`       -- generic long-format logs (one row per response)
  * :func:`load_kt_text`        -- the classic DKT/pyKT triplet text format
  * :func:`load_pykt_sequences` -- convenience dispatch over a pyKT dataset directory
  * :func:`load_eedi_items`     -- Eedi MCQ items with distractor->misconception labels (RQ2)

Still stubbed (later gates):
  * :func:`load_qatd_dialogues` -- G3 face validity / simulated-student calibration

Notes:
  * Real KT logs carry only correctness, not IRT item parameters, so loaded items get
    placeholder ``difficulty=0.0`` / ``discrimination=1.0``. Calibrate these (fit IRT, or
    swap in the DKT estimator) before reading the grid-Bayes estimator's predictions as
    proficiency; the parsers here are about getting the data into the schema.
  * LICENSING: real datasets are licensed (QATD-2k is CC-BY-NC-SA, UK-only). Do not
    commit raw data; see ``.gitignore``. The tiny files under ``data/fixtures/`` are
    *synthetic* samples in the real on-disk formats, used only to exercise the parsers.
"""

from __future__ import annotations

import csv
import os
from typing import NamedTuple, Optional, Sequence

from .schemas import Interaction, Item


class InteractionLog(NamedTuple):
    """Parsed real-data log: an item bank plus per-student interaction sequences."""

    items: list[Item]
    sequences: dict[int, list[Interaction]]  # student_id -> chronological interactions


# Accepted spellings for a correctness flag.
_TRUE = {"1", "1.0", "true", "t", "yes", "y", "correct"}
_FALSE = {"0", "0.0", "false", "f", "no", "n", "incorrect", ""}


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"Cannot parse correctness flag from {value!r}")


def _option_to_index(value: str, one_indexed: bool) -> Optional[int]:
    """Map an option label to a 0-based index.

    Accepts single letters ('A'->0, 'B'->1, ...) or integers (respecting
    ``one_indexed``). Returns ``None`` for blanks/unparseable values.
    """
    v = value.strip()
    if v == "":
        return None
    if len(v) == 1 and v.isalpha():
        return ord(v.upper()) - ord("A")
    if v.lstrip("-").isdigit():
        idx = int(v)
        return idx - 1 if one_indexed else idx
    return None


def _require_columns(
    fieldnames: Optional[Sequence[str]], required: Sequence[str], path: str
) -> None:
    have = set(fieldnames or [])
    missing = [c for c in required if c not in have]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}; found {list(fieldnames or [])}"
        )


def _present(row: dict, col: Optional[str]) -> bool:
    return bool(col) and row.get(col, "") not in ("", None)


# --------------------------------------------------------------------- long CSV
def load_long_csv(
    path: str,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    kc_col: str = "kc",
    correct_col: str = "correct",
    option_col: Optional[str] = "chosen_option",
    misconception_col: Optional[str] = "misconception",
    difficulty_col: Optional[str] = "difficulty",
    discrimination_col: Optional[str] = "discrimination",
    correct_option_col: Optional[str] = "correct_option",
    timestamp_col: Optional[str] = "timestamp",
    n_options: int = 4,
    one_indexed_options: bool = False,
) -> InteractionLog:
    """Load a long-format interaction log (one row per student response).

    Only ``user_col``, ``item_col``, ``kc_col`` and ``correct_col`` are required; the
    rest are used when present (e.g. Eedi-style ``chosen_option`` + ``misconception``
    populate each item's distractor->misconception map for RQ2). Rows are ordered by
    ``timestamp_col`` when every row in a sequence has one, else by file order.
    """
    items: dict[int, dict] = {}
    rows_by_user: dict[int, list[tuple]] = {}

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, [user_col, item_col, kc_col, correct_col], path)
        for row in reader:
            uid = int(float(row[user_col]))
            iid = int(float(row[item_col]))
            kc = int(float(row[kc_col]))
            correct = _parse_bool(row[correct_col])
            chosen = (
                _option_to_index(row[option_col], one_indexed_options)
                if _present(row, option_col)
                else None
            )
            misc = int(float(row[misconception_col])) if _present(row, misconception_col) else None

            entry = items.setdefault(
                iid,
                {
                    "kc": kc,
                    "difficulty": 0.0,
                    "discrimination": 1.0,
                    "n_options": n_options,
                    "correct_option": 0,
                    "misconception_by_option": {},
                },
            )
            if _present(row, difficulty_col):
                entry["difficulty"] = float(row[difficulty_col])
            if _present(row, discrimination_col):
                entry["discrimination"] = float(row[discrimination_col])
            if _present(row, correct_option_col):
                co = _option_to_index(row[correct_option_col], one_indexed_options)
                if co is not None:
                    entry["correct_option"] = co
            if (not correct) and chosen is not None and misc is not None:
                entry["misconception_by_option"][chosen] = misc

            ts = float(row[timestamp_col]) if _present(row, timestamp_col) else None
            rows_by_user.setdefault(uid, []).append((ts, iid, chosen, correct, misc))

    item_objs = [
        Item(
            item_id=iid,
            kc=e["kc"],
            difficulty=e["difficulty"],
            discrimination=e["discrimination"],
            n_options=e["n_options"],
            correct_option=e["correct_option"],
            misconception_by_option=dict(e["misconception_by_option"]),
        )
        for iid, e in sorted(items.items())
    ]
    item_by_id = {it.item_id: it for it in item_objs}

    sequences: dict[int, list[Interaction]] = {}
    for uid, rows in rows_by_user.items():
        if all(r[0] is not None for r in rows):
            rows = sorted(rows, key=lambda r: r[0])
        seq: list[Interaction] = []
        for _ts, iid, chosen, correct, misc in rows:
            it = item_by_id[iid]
            chosen_opt = chosen if chosen is not None else (it.correct_option if correct else -1)
            seq.append(
                Interaction(
                    item_id=iid, chosen_option=chosen_opt, correct=correct, misconception=misc
                )
            )
        sequences[uid] = seq

    return InteractionLog(items=item_objs, sequences=sequences)


# ---------------------------------------------------------------- KT triplet txt
def load_kt_text(path: str, n_options: int = 4) -> InteractionLog:
    """Load the classic DKT/pyKT triplet text format.

    Each student is three non-empty lines: the sequence length ``N``; ``N``
    comma-separated skill/question ids; and ``N`` comma-separated 0/1 responses. In this
    format the question id doubles as the knowledge component.
    """
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() != ""]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"{path}: expected groups of 3 lines (N / ids / responses), "
            f"got {len(lines)} non-empty lines"
        )

    items: dict[int, int] = {}  # item_id -> kc (identical here)
    sequences: dict[int, list[Interaction]] = {}
    for uid, i in enumerate(range(0, len(lines), 3)):
        n = int(lines[i])
        ids = [int(x) for x in lines[i + 1].split(",") if x.strip() != ""]
        resp = [_parse_bool(x) for x in lines[i + 2].split(",") if x.strip() != ""]
        if not (len(ids) == len(resp) == n):
            raise ValueError(
                f"{path}: sequence {uid} length mismatch "
                f"(declared {n}, ids {len(ids)}, responses {len(resp)})"
            )
        seq = []
        for iid, correct in zip(ids, resp):
            items.setdefault(iid, iid)
            seq.append(
                Interaction(
                    item_id=iid,
                    chosen_option=0 if correct else -1,  # options unknown in KT logs
                    correct=correct,
                    misconception=None,
                )
            )
        sequences[uid] = seq

    item_objs = [
        Item(item_id=iid, kc=kc, difficulty=0.0, discrimination=1.0, n_options=n_options)
        for iid, kc in sorted(items.items())
    ]
    return InteractionLog(items=item_objs, sequences=sequences)


def load_pykt_sequences(dataset: str, data_dir: str) -> InteractionLog:
    """Dispatch over a pyKT dataset directory.

    Loads ``{dataset}.csv`` as a long-format log, else ``{dataset}.txt`` as the triplet
    format. pyKT's preprocessed exports are CSVs; map their columns to the names
    :func:`load_long_csv` expects (or pass the ``*_col`` overrides directly).
    """
    csv_path = os.path.join(data_dir, f"{dataset}.csv")
    txt_path = os.path.join(data_dir, f"{dataset}.txt")
    if os.path.exists(csv_path):
        return load_long_csv(csv_path)
    if os.path.exists(txt_path):
        return load_kt_text(txt_path)
    raise FileNotFoundError(
        f"No {dataset}.csv or {dataset}.txt under {data_dir!r}. Convert the pyKT "
        "preprocessed export to a long CSV with columns user_id,item_id,kc,correct,..."
    )


# ----------------------------------------------------------------------- Eedi
def load_eedi_items(
    path: str,
    *,
    question_col: str = "QuestionId",
    correct_col: str = "CorrectAnswer",
    construct_col: str = "ConstructId",
    subject_col: str = "SubjectId",
    option_letters: Sequence[str] = ("A", "B", "C", "D"),
    misconception_col_fmt: str = "Misconception{}Id",
    one_indexed_correct: bool = True,
) -> list[Item]:
    """Load Eedi MCQ items with distractor->misconception annotations (RQ2 target).

    One row per question. ``CorrectAnswer`` may be a letter or a (1-indexed) integer.
    Each option's ``Misconception{Letter}Id`` column, when present and not the correct
    option, populates ``Item.misconception_by_option``. The knowledge component is the
    construct id when available, else the subject id, else 0.
    """
    items: list[Item] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames, [question_col, correct_col], path)
        fields = set(reader.fieldnames or [])
        for row in reader:
            qid = int(float(row[question_col]))
            correct_idx = _option_to_index(row[correct_col], one_indexed_correct)
            correct_idx = 0 if correct_idx is None else correct_idx
            if construct_col in fields and _present(row, construct_col):
                kc = int(float(row[construct_col]))
            elif subject_col in fields and _present(row, subject_col):
                kc = int(float(row[subject_col]))
            else:
                kc = 0

            misconception_by_option: dict[int, int] = {}
            for opt_idx, letter in enumerate(option_letters):
                if opt_idx == correct_idx:
                    continue
                col = misconception_col_fmt.format(letter)
                if not _present(row, col):
                    continue
                try:
                    misconception_by_option[opt_idx] = int(float(row[col]))
                except ValueError:
                    continue

            items.append(
                Item(
                    item_id=qid,
                    kc=kc,
                    difficulty=0.0,
                    discrimination=1.0,
                    n_options=len(option_letters),
                    correct_option=correct_idx,
                    misconception_by_option=misconception_by_option,
                )
            )
    return items


# ------------------------------------------------------------------- QATD (stub)
def load_qatd_dialogues(data_dir: str):
    """Load QATD-2k tutoring dialogues for face-validity / simulator calibration (STUB)."""
    raise NotImplementedError(
        "G3 stub. Load QATD-2k turns for (a) face-validity surface stats vs generated "
        "dialogues and (b) validating the LLM simulated student. Respect CC-BY-NC-SA, "
        "UK-only licensing; do not commit raw data."
    )
