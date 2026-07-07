#!/usr/bin/env python3
"""Standalone SFT-data builder: prompt Claude to generate simulated-student
training data, using every trace-generation method from both local projects.

This file is SELF-CONTAINED so it can be copied to another machine on its own.
It merges (and documents) the methods from two codebases:

  sim_misconception/scripts/distill_traces.py   verifier-filtered distillation
  sim_misconception/src/simstudent/baselines/prompts.py   prompt conditions
  sim_misconception/src/simstudent/sft/build_sft_data.py  MAP corpus + mixing
  misconception/reasoning_generator.py          extra trace scaffolds
                                                (first-error, debate, MALT)

════════════════════════════════════════════════════════════════════════════
 1. WHAT THIS PRODUCES
════════════════════════════════════════════════════════════════════════════
TRL-style chat JSONL, one example per line:

  {"messages": [{"role": "user", "content": <canonical student prompt>},
                {"role": "assistant", "content": "<explanation>...</explanation>
                                                  <answer>B</answer>"}],
   "source": ..., "method": ..., "QuestionId": ..., "MisconceptionId": ...}

DESIGN DECISION (from distill_traces.py): the *user* turn is ALWAYS the
canonical zero-shot-ss prompt, no matter which teacher scaffold generated the
trace. Teacher scaffolds (backward-reconstruction, debate, MALT, ...) are
construction details; the fine-tuned student must sit behind the exact same
interface that the prompted baselines are evaluated on. The assistant turn is
canonicalized (re-wrapped in <explanation>/<answer> tags) so the student model
learns exactly the output contract the eval parser expects.

════════════════════════════════════════════════════════════════════════════
 2. METHODS (--methods flag; ablation axis #1)
════════════════════════════════════════════════════════════════════════════
From sim_misconception (first-class teacher conditions):
  zero-shot-ss   Student-simulating: first-person student who unknowingly
                 holds misconception M. The canonical interface.
  zero-shot-br   Backward-reconstruction: third person; reconstruct the flawed
                 solution path implied by M, written in the student's voice.
                 Rationale: separating "plan the error" from "voice the error"
                 can yield more misconception-faithful traces.
  persona        Epistemic-state profile: competence level + style constraints
                 + the misconception. Tests whether richer persona framing
                 improves stylistic realism.
  few-shot-icl   Worked-mistake ICL: k answer-only examples of the SAME
                 misconception on other train questions. NOTE: the pilot
                 showed answer-only shots UNDERPERFORM zero-shot; kept as a
                 control.
  few-shot-icl-traces  Same, but each shot includes the student's reasoning
                 trace (needs --trace-bank, a JSONL previously produced by
                 this script with zero-shot-ss). Tests whether worked examples
                 with traces close the few-shot gap (examples > descriptions).

From the misconception repo (reasoning_generator.py scaffolds, adapted: the
original repo used them as free-form trace generators for retrieval; here they
are given the same <explanation>/<answer> output contract so their output can
be filtered and canonicalized identically):
  first-error    Identify the earliest divergence point from the correct
                 approach, then voice the resulting flawed solution as the
                 student. Rationale: concentrates the error signal at one
                 decision, avoiding "uniformly sloppy" traces.
  debate         Two-agent debate (proposer/critic) refines the flawed pathway
                 internally; only the final student-voiced trace is emitted.
  malt           MALT-style draft→verify→refine pipeline inside one prompt;
                 only the refined student-voiced trace is emitted.

════════════════════════════════════════════════════════════════════════════
 3. VERIFIER FILTERS (ablation axis #2, --no-filter to disable)
════════════════════════════════════════════════════════════════════════════
Every generation must survive three filters, in a FIXED order so rejection
rates decompose cleanly (a trace failing "answer" is never also counted
against "break" or "cycle"):

  answer   parsed <answer> letter equals the target distractor. One comparison
           rejects both failure modes: drifting to the correct answer
           (leakage) and drifting to a different wrong answer.
  break    character-break regexes (meta-commentary / self-correction /
           reveals-correct). Cheap and deterministic.
  cycle    cycle-consistency: a frozen TF-IDF retriever over misconception
           names must recover the conditioning misconception from
           (question, distractor, trace) within top-k (default 25).
           The retriever is deliberately weak: it may reject good traces
           (recall ceiling) but cannot inject signal a trace doesn't carry,
           so downstream SFT gains stay attributable to trace quality.
           Requires scikit-learn; auto-skipped (with a warning) if missing.

Rejected pairs are resampled up to --attempts fresh variants (the variant id
salts the cache key, so each retry is a genuinely new sample at temperature
0.7). If all attempts fail, the pair is DROPPED: corpus coverage is
negotiable, filter guarantees are not.

Per-filter rejection counts are written to <out>/stats_<method>.json — these
are RESULTS (they belong in the paper's data-construction table), not
plumbing.

════════════════════════════════════════════════════════════════════════════
 4. NON-CLAUDE CORPORA + MIXING (ablation axis #3)
════════════════════════════════════════════════════════════════════════════
  map        Real student explanations from the MAP dataset
             (Category == False_Misconception, non-null explanation).
             Signal = stylistic/epistemic realism of human text, NOT answer
             format (MAP has no option letters; <answer> carries answer text,
             and conditioning is the coarse MAP tag, not an Eedi description).
  mix        Interleave two JSONL corpora at ratio N:1 (e.g. distilled:map at
             1:1 and 3:1). Interleaving (not concatenation) keeps both sources
             evenly spread across the epoch regardless of the trainer's
             shuffle buffer. Both sources are truncated so the RATIO holds
             exactly across the whole file — the ablation comparison is
             ratio-vs-ratio, not tokens-vs-tokens.

WITHDRAWN: the former MalruleLib-based corpora must NOT be rebuilt or used —
its license prohibits use in a pipeline that evaluates closed-source LLMs.

════════════════════════════════════════════════════════════════════════════
 5. DATA YOU NEED ON THE TARGET MACHINE
════════════════════════════════════════════════════════════════════════════
(a) Eedi "Mining Misconceptions in Mathematics" (Kaggle) — REQUIRED for
    generation. Pass its folder as --eedi-dir. Files used:
      train.csv                  ~1,869 questions; per-option misconception
                                 ids (Misconception{A..D}Id), AnswerXText,
                                 CorrectAnswer, ConstructName
      misconception_mapping.csv  2,587 rows: MisconceptionId, MisconceptionName
(b) MAP "Charting Student Math Misunderstandings" (Kaggle) — only for the
    `map` subcommand. Pass --map-csv pointing at its train.csv. Columns used:
      QuestionId, QuestionText, MC_Answer, StudentExplanation, Category,
      Misconception
(c) OPTIONAL but recommended for comparability: the frozen split file
    sim_misconception/data/splits/eedi_pairs.csv (and map_questions.csv).
    Pass via --splits-csv; rows with split == "train" are kept. Without it,
    this script falls back to a seeded 80/10/10 split BY QUESTION — fine for
    a fresh project, but NOT identical to the frozen sim_misconception splits,
    so don't compare numbers across the two without it.
(d) An Anthropic API key in $CLAUDE_KEY or $ANTHROPIC_API_KEY, and
    `pip install anthropic pandas numpy scikit-learn`.

════════════════════════════════════════════════════════════════════════════
 6. USAGE
════════════════════════════════════════════════════════════════════════════
  # Pilot (40 pairs, no corpus written beyond the pilot file):
  python make_sft_data.py generate --eedi-dir <dir> --limit 40 \
      --methods zero-shot-ss

  # Full distillation with several teacher scaffolds:
  python make_sft_data.py generate --eedi-dir <dir> \
      --methods zero-shot-ss zero-shot-br first-error debate malt

  # Real-student corpus:
  python make_sft_data.py map --map-csv <map>/train.csv

  # Mixtures:
  python make_sft_data.py mix --a out/sft_zero-shot-ss.jsonl \
      --b out/sft_map_only.jsonl --ratio 1
  python make_sft_data.py mix --a out/sft_zero-shot-ss.jsonl \
      --b out/sft_map_only.jsonl --ratio 3

  # Cost-free smoke test of prompts/plumbing (no API calls):
  python make_sft_data.py generate --eedi-dir <dir> --limit 5 --dry-run

Everything is cached under --cache-dir (JSONL keyed by a hash of
model/method/variant/temperature/prompt), so re-runs are free and
interrupted runs resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260706  # matches sim_misconception frozen seed
OPTIONS = ("A", "B", "C", "D")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt library
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_CONTRACT = (
    "Respond with your reasoning as the student would write it, inside "
    "<explanation>...</explanation> tags (2-4 sentences, informal, first person), "
    "then your chosen option inside <answer>...</answer> tags containing only the "
    "letter A, B, C, or D."
)

STAY_IN_CHARACTER = (
    "Stay fully in character: never mention the misconception, never correct "
    "yourself, never reveal or hint at the right answer. Your working genuinely "
    "feels correct to you."
)


def format_question(row) -> str:
    lines = [f"Question: {row['QuestionText']}"] + [
        f"{o}. {row['options_text'][o]}" for o in OPTIONS
    ]
    return "\n".join(lines)


def zero_shot_ss(row, **_) -> str:
    """Canonical student interface — also the user turn of EVERY SFT example."""
    return (
        "You are a real middle-school maths student. You hold this specific "
        f"misunderstanding, without being aware of it: {row['MisconceptionName']}.\n"
        f"{STAY_IN_CHARACTER}\n\n{format_question(row)}\n\n{OUTPUT_CONTRACT}"
    )


def zero_shot_br(row, **_) -> str:
    return (
        "A middle-school maths student holds this misconception: "
        f"{row['MisconceptionName']}.\n"
        "Reconstruct, step by step, the solution path this student would actually "
        "take on the question below — apply the misconception faithfully wherever "
        "it is relevant, and follow the flawed path to whichever option it leads to. "
        "Do not evaluate or correct the path.\n\n"
        f"{format_question(row)}\n\nWrite the reconstruction in the student's own "
        f"voice. {OUTPUT_CONTRACT}"
    )


def persona(row, competence: str = "a solid B-grade student who is confident in class", **_) -> str:
    return (
        "Adopt this student profile completely.\n"
        f"- Overall maths competence: {competence}.\n"
        f"- Known misunderstanding (the student is NOT aware of it): "
        f"{row['MisconceptionName']}.\n"
        "- Style: writes short, informal explanations; never uses teacher language.\n"
        f"{STAY_IN_CHARACTER}\n\n{format_question(row)}\n\n{OUTPUT_CONTRACT}"
    )


def few_shot_icl(row, examples: pd.DataFrame = None, **_) -> str:
    """Answer-only shots of the same misconception on other train questions.
    Pilot result: UNDERPERFORMS zero-shot — kept as a control condition."""
    blocks = []
    for i, (_, ex) in enumerate(examples.iterrows(), 1):
        blocks.append(
            f"Example {i}:\n{format_question(ex)}\n"
            f"The student answered: {ex['option']}. {ex['options_text'][ex['option']]}"
        )
    shots = "\n\n".join(blocks)
    return (
        "Here are answers one particular maths student gave on past questions. "
        "The student makes the same kind of mistake every time.\n\n"
        f"{shots}\n\nNow answer the next question exactly as this same student "
        f"would, making the same kind of mistake if it applies. {STAY_IN_CHARACTER}\n\n"
        f"{format_question(row)}\n\n{OUTPUT_CONTRACT}"
    )


def few_shot_icl_traces(row, examples: pd.DataFrame = None, **_) -> str:
    """Worked-mistake shots WITH reasoning traces (needs a `trace` column,
    sourced from a previous zero-shot-ss run via --trace-bank)."""
    blocks = []
    for i, (_, ex) in enumerate(examples.iterrows(), 1):
        blocks.append(
            f"Example {i}:\n{format_question(ex)}\n"
            f"The student wrote: \"{ex['trace']}\"\n"
            f"The student answered: {ex['option']}. {ex['options_text'][ex['option']]}"
        )
    shots = "\n\n".join(blocks)
    return (
        "Here is how one particular maths student worked through past questions. "
        "The student makes the same kind of mistake every time.\n\n"
        f"{shots}\n\nNow answer the next question exactly as this same student "
        f"would, reasoning the same way and making the same kind of mistake if it "
        f"applies. {STAY_IN_CHARACTER}\n\n{format_question(row)}\n\n{OUTPUT_CONTRACT}"
    )


# Scaffolds adapted from misconception/reasoning_generator.py. The originals
# produced free-form third-person traces for retrieval experiments; here each
# is instructed to end with the student-voiced trace under the shared output
# contract, so one parser + one filter stack serves every method.

def first_error(row, **_) -> str:
    return (
        "A middle-school maths student holds this misconception, without being "
        f"aware of it: {row['MisconceptionName']}.\n"
        "First, silently identify the SINGLE earliest point where this student's "
        "reasoning would diverge from the correct approach on the question below. "
        "Then write the student's full (flawed) working as they would, where that "
        "one early mistake naturally leads them to a wrong option. Do not mention "
        "the divergence analysis in the output.\n\n"
        f"{format_question(row)}\n\n{STAY_IN_CHARACTER}\n\n{OUTPUT_CONTRACT}"
    )


def debate(row, **_) -> str:
    return (
        "You will silently simulate a short debate between two agents to design a "
        "realistic wrong solution, then output only the final result.\n"
        f"The target student unknowingly holds this misconception: {row['MisconceptionName']}.\n"
        "Agent A proposes a plausible student-like flawed pathway through the "
        "question below; Agent B challenges it (is it faithful to the misconception? "
        "does a real student talk like this?) and refines it. Iterate briefly.\n"
        "Then output ONLY the refined trace, written first-person in the student's "
        f"voice. {STAY_IN_CHARACTER}\n\n{format_question(row)}\n\n{OUTPUT_CONTRACT}"
    )


def malt(row, **_) -> str:
    return (
        "Use a three-step internal pipeline — draft, verify, refine — and output "
        "only the final result.\n"
        f"The student unknowingly holds this misconception: {row['MisconceptionName']}.\n"
        "Draft: write the student's flawed working on the question below. "
        "Verify: check the draft is misconception-faithful, lands on a wrong "
        "option, and sounds like a real student. Refine: fix any issues.\n"
        "Output ONLY the refined version, first-person in the student's voice. "
        f"{STAY_IN_CHARACTER}\n\n{format_question(row)}\n\n{OUTPUT_CONTRACT}"
    )


METHODS = {
    "zero-shot-ss": zero_shot_ss,
    "zero-shot-br": zero_shot_br,
    "persona": persona,
    "few-shot-icl": few_shot_icl,           # needs examples
    "few-shot-icl-traces": few_shot_icl_traces,  # needs examples w/ trace col
    "first-error": first_error,
    "debate": debate,
    "malt": malt,
}
FEW_SHOT_METHODS = {"few-shot-icl", "few-shot-icl-traces"}

# ─────────────────────────────────────────────────────────────────────────────
# Parsing + verifier filters
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.I)
_EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.I | re.S)

CHARACTER_BREAK_PATTERNS = {
    "meta_commentary": re.compile(
        r"\bas an ai\b|\blanguage model\b|\bi am simulating\b|\bthe student (would|might|is)\b|"
        r"\bthis (misconception|error) (is|means)\b|\broleplay\b",
        re.I,
    ),
    "self_correction": re.compile(
        r"\bwait,? (no|that)\b|\bactually,? (no|the right|the correct)\b|\bon second thought\b|"
        r"\bi (was|am) wrong\b|\blet me reconsider\b|\bcorrection:\b|\bi apologize\b",
        re.I,
    ),
    "reveals_correct": re.compile(
        r"\bthe (correct|right|actual) answer is\b|\bcorrectly,? (it|this|the)\b|"
        r"\bshould (actually|really) be\b|\bin reality\b",
        re.I,
    ),
}


def parse_response(text: str) -> dict:
    ans = _ANSWER_RE.search(text)
    expl = _EXPL_RE.search(text)
    if not ans:  # fallback: bare final letter
        ans = re.search(r"\b([ABCD])\b[^ABCD]*$", text.strip())
    return {
        "predicted_option": ans.group(1).upper() if ans else None,
        "trace": expl.group(1).strip() if expl else text.strip(),
    }


def character_break_flags(trace: str) -> dict:
    return {n: bool(p.search(trace)) for n, p in CHARACTER_BREAK_PATTERNS.items()}


class TfidfMisconceptionRetriever:
    """Name-only TF-IDF retriever over the 2,587-entry taxonomy (deliberately
    weak — see module docstring §3). None if sklearn is unavailable."""

    def __init__(self, mapping: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: deferred
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.ids = np.array(mapping["MisconceptionId"].astype(int).tolist())
        self.matrix = self.vec.fit_transform(mapping["MisconceptionName"].tolist())

    def rank(self, query: str, k: int = 25) -> list[int]:
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        return self.ids[sims.argsort()[::-1][:k]].tolist()


def check(parsed: dict, row, retriever, cycle_k: int) -> str | None:
    """Name of the first failed filter, or None if accepted. Order is fixed so
    rejection rates decompose (answer -> break -> cycle)."""
    if parsed["predicted_option"] != row["option"]:
        return "answer"
    if any(character_break_flags(parsed["trace"]).values()):
        return "break"
    if retriever is not None:
        query = f"{row['QuestionText']} {row['distractor_text']} {parsed['trace']}"
        if row["MisconceptionId"] not in retriever.rank(query, k=cycle_k):
            return "cycle"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_eedi_pairs(eedi_dir: Path) -> pd.DataFrame:
    """Long format: one row per labeled (question, distractor, misconception)."""
    df = pd.read_csv(eedi_dir / "train.csv")
    mapping = pd.read_csv(eedi_dir / "misconception_mapping.csv").set_index(
        "MisconceptionId")["MisconceptionName"]
    rows = []
    for _, r in df.iterrows():
        opts = {o: r[f"Answer{o}Text"] for o in OPTIONS}
        for opt in OPTIONS:
            mid = r[f"Misconception{opt}Id"]
            if pd.isna(mid) or opt == r["CorrectAnswer"]:
                continue
            rows.append({
                "QuestionId": int(r["QuestionId"]),
                "QuestionText": r["QuestionText"],
                "CorrectAnswer": r["CorrectAnswer"],
                "option": opt,
                "distractor_text": opts[opt],
                "options_text": opts,
                "MisconceptionId": int(mid),
                "MisconceptionName": mapping.get(int(mid), ""),
            })
    return pd.DataFrame(rows)


def restrict_to_train(pairs: pd.DataFrame, splits_csv: Path | None) -> pd.DataFrame:
    """Keep train-split rows. Frozen split file if given (comparable with
    sim_misconception); otherwise a seeded 80/10/10 fallback BY QUESTION."""
    if splits_csv is not None:
        split = pd.read_csv(splits_csv)
        train_qids = set(split.loc[split["split"] == "train", "QuestionId"])
        kept = pairs[pairs["QuestionId"].isin(train_qids)]
        print(f"splits file: kept {len(kept)}/{len(pairs)} pairs (train questions)")
        return kept
    qids = np.array(sorted(pairs["QuestionId"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(qids)
    n = len(qids)
    train_qids = set(qids[: int(0.8 * n)])
    print(f"WARNING: no --splits-csv; seeded fallback split "
          f"({len(train_qids)}/{n} questions -> train). NOT the frozen "
          f"sim_misconception split — don't compare numbers across the two.")
    return pairs[pairs["QuestionId"].isin(train_qids)]


# ─────────────────────────────────────────────────────────────────────────────
# Cached Claude client
# ─────────────────────────────────────────────────────────────────────────────

class CachedLLM:
    """JSONL-file cache in front of the Anthropic API. The cache key hashes
    (model, method, variant, temperature, prompt), so retries with a new
    `variant` are genuinely fresh samples and interrupted runs resume free."""

    def __init__(self, model: str, method: str, cache_dir: Path,
                 temperature: float = 0.7, max_tokens: int = 1024,
                 dry_run: bool = False, delay: float = 0.1):
        self.model, self.method = model, method
        self.temperature, self.max_tokens = temperature, max_tokens
        self.dry_run, self.delay = dry_run, delay
        self.path = cache_dir / model.replace("/", "_") / f"{method}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        self.cache[e["key"]] = e["completion"]
        self.client = None
        if not dry_run:
            key = os.getenv("CLAUDE_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                sys.exit("Set CLAUDE_KEY or ANTHROPIC_API_KEY (or use --dry-run).")
            from anthropic import Anthropic
            self.client = Anthropic(api_key=key)

    def _key(self, prompt: str, variant: int) -> str:
        blob = f"{self.model}|{self.method}|{variant}|{self.temperature}|{prompt}"
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, prompt: str, variant: int = 0) -> str | None:
        key = self._key(prompt, variant)
        if key in self.cache:
            return self.cache[key]
        if self.dry_run:
            return None  # uncached in dry-run: caller counts it and moves on
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        self.cache[key] = text
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "variant": variant,
                                "prompt": prompt, "completion": text}) + "\n")
        if self.delay:
            time.sleep(self.delay)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot example selection
# ─────────────────────────────────────────────────────────────────────────────

def load_trace_bank(path: Path) -> dict[tuple[int, int], str]:
    """(QuestionId, MisconceptionId) -> trace, from a JSONL previously written
    by this script (assistant turn is parsed back out)."""
    bank = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            parsed = parse_response(ex["messages"][1]["content"])
            bank[(ex["QuestionId"], ex["MisconceptionId"])] = parsed["trace"]
    return bank


def few_shot_examples(row, pairs: pd.DataFrame, k: int, rng: np.random.Generator,
                      trace_bank: dict | None) -> pd.DataFrame | None:
    """k train pairs sharing MisconceptionId, from OTHER questions. For the
    traces variant, only pairs whose trace exists in the bank qualify.
    Returns None if fewer than k shots exist (pair is skipped, counted)."""
    cand = pairs[(pairs["MisconceptionId"] == row["MisconceptionId"])
                 & (pairs["QuestionId"] != row["QuestionId"])]
    if trace_bank is not None:
        cand = cand[cand.apply(
            lambda r: (r["QuestionId"], r["MisconceptionId"]) in trace_bank, axis=1)]
        if len(cand) < k:
            return None
        cand = cand.copy()
        cand["trace"] = cand.apply(
            lambda r: trace_bank[(r["QuestionId"], r["MisconceptionId"])], axis=1)
    if len(cand) < k:
        return None
    return cand.iloc[rng.choice(len(cand), size=k, replace=False)]


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: generate (Claude distillation with verifier filtering)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate(args) -> None:
    eedi_dir = Path(args.eedi_dir)
    pairs = load_eedi_pairs(eedi_dir)
    pairs = restrict_to_train(pairs, Path(args.splits_csv) if args.splits_csv else None)
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"pairs to distill: {len(pairs)}  methods: {args.methods}")

    retriever = None
    if not args.no_filter:
        try:
            mapping = pd.read_csv(eedi_dir / "misconception_mapping.csv").dropna()
            retriever = TfidfMisconceptionRetriever(mapping)
        except ImportError:
            print("WARNING: scikit-learn missing — cycle filter SKIPPED. "
                  "Rejection stats will not be comparable to filtered runs.")

    trace_bank = load_trace_bank(Path(args.trace_bank)) if args.trace_bank else None
    if "few-shot-icl-traces" in args.methods and trace_bank is None:
        sys.exit("few-shot-icl-traces needs --trace-bank (a JSONL from a "
                 "previous zero-shot-ss run of this script).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for method in args.methods:
        llm = CachedLLM(args.model, method, Path(args.cache_dir),
                        temperature=args.temperature, dry_run=args.dry_run,
                        delay=args.delay)
        examples, stats = [], Counter()
        for i, (_, row) in enumerate(pairs.iterrows()):
            shots = None
            if method in FEW_SHOT_METHODS:
                shots = few_shot_examples(
                    row, pairs, args.shots, rng,
                    trace_bank if method == "few-shot-icl-traces" else None)
                if shots is None:
                    stats["no_shots"] += 1
                    continue
            prompt = METHODS[method](row, examples=shots)

            accepted = None
            for variant in range(args.attempts):
                completion = llm.complete(prompt, variant=variant)
                if completion is None:  # dry run, uncached
                    stats["uncached"] += 1
                    break
                stats["generations"] += 1
                parsed = parse_response(completion)
                if args.no_filter:
                    accepted = (parsed, variant)
                    break
                failed = check(parsed, row, retriever, args.cycle_k)
                if failed is None:
                    accepted = (parsed, variant)
                    break
                stats[f"reject_{failed}"] += 1
            if accepted is None:
                stats["pairs_exhausted"] += 1
                continue
            parsed, variant = accepted
            stats["accepted"] += 1
            examples.append({
                # User turn is ALWAYS canonical zero-shot-ss (see docstring §1);
                # assistant turn is canonicalized to the output contract.
                "messages": [
                    {"role": "user", "content": zero_shot_ss(row)},
                    {"role": "assistant",
                     "content": f"<explanation>{parsed['trace']}</explanation>\n"
                                f"<answer>{parsed['predicted_option']}</answer>"},
                ],
                "source": "claude_distilled",
                "method": method,
                "teacher_model": args.model,
                "variant": variant,
                "filtered": not args.no_filter,
                "QuestionId": int(row["QuestionId"]),
                "MisconceptionId": int(row["MisconceptionId"]),
            })
            if (i + 1) % 25 == 0:
                print(f"  [{method}] {i+1}/{len(pairs)}  accepted={stats['accepted']}")

        suffix = f"_pilot{args.limit}" if args.limit else ""
        suffix += "_nofilter" if args.no_filter else ""
        corpus_path = out_dir / f"sft_{method}{suffix}.jsonl"
        with open(corpus_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        report = {"model": args.model, "method": method, "n_pairs": len(pairs),
                  "attempts": args.attempts, "cycle_k": args.cycle_k,
                  "filtered": not args.no_filter, "n_examples": len(examples),
                  "stats": dict(stats)}
        if stats.get("generations"):
            report["acceptance_rate"] = stats.get("accepted", 0) / stats["generations"]
        stats_path = out_dir / f"stats_{method}{suffix}.json"
        stats_path.write_text(json.dumps(report, indent=2))
        print(f"{method}: {len(examples)} examples -> {corpus_path}")
        print(json.dumps(report["stats"], indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: map (real student explanations)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_map(args) -> None:
    m = pd.read_csv(args.map_csv)
    if args.splits_csv:
        split = pd.read_csv(args.splits_csv)
        train_qids = set(split.loc[split["split"] == "train", "QuestionId"])
    else:
        qids = np.array(sorted(m["QuestionId"].unique()))
        rng = np.random.default_rng(SEED)
        rng.shuffle(qids)
        train_qids = set(qids[: int(0.8 * len(qids))])
        print("WARNING: seeded fallback split for MAP (no --splits-csv).")
    m = m[m["QuestionId"].isin(train_qids)
          & (m["Category"] == "False_Misconception")
          & m["StudentExplanation"].notna()]

    out = []
    for _, r in m.iterrows():
        # Mirrors zero-shot-ss with two MAP-specific differences: conditioning
        # is the coarse MAP tag (not an Eedi description) and <answer> carries
        # answer TEXT (MAP has no option letters). The human explanation's
        # style/epistemic realism is the training signal, not answer format.
        user = (
            "You are a real middle-school maths student. You tend to make this "
            f"kind of mistake, without being aware of it: {r['Misconception']}.\n"
            f"{STAY_IN_CHARACTER}\n\n"
            f"Question: {r['QuestionText']}\n\n"
            "Respond with your reasoning inside <explanation>...</explanation> "
            "tags, then your final answer inside <answer>...</answer> tags."
        )
        assistant = (f"<explanation>{r['StudentExplanation']}</explanation>\n"
                     f"<answer>{r['MC_Answer']}</answer>")
        out.append({"messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": assistant}],
                    "source": "map", "map_tag": r["Misconception"]})

    rng = np.random.default_rng(SEED)
    rng.shuffle(out)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sft_map_only.jsonl"
    with open(path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")
    print(f"map_only: {len(out)} examples -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: mix (ratio-exact interleaving)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_mix(args) -> None:
    def read(p):
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]

    a, b = read(args.a), read(args.b)
    rng = np.random.default_rng(SEED)
    rng.shuffle(a)
    rng.shuffle(b)
    r = args.ratio
    # Pattern = (a * r, b) repeated. Strided slices a[j::r] partition `a`, so
    # no example is duplicated; truncation to the limiting source keeps the
    # RATIO exact across the whole file (see docstring §4).
    n = min(len(a) // r, len(b))
    if n == 0:
        sys.exit(f"not enough examples to mix at {r}:1 (|a|={len(a)}, |b|={len(b)})")
    groups = zip(*[a[j::r][:n] for j in range(r)], b[:n])
    mixed = [x for grp in groups for x in grp]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"sft_mixed_{r}to1.jsonl"
    path = out_dir / name
    with open(path, "w") as f:
        for ex in mixed:
            f.write(json.dumps(ex) + "\n")
    print(f"mixed {r}:1 ({Path(args.a).name}:{Path(args.b).name}): "
          f"{len(mixed)} examples -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Claude distillation with verifier filters")
    g.add_argument("--eedi-dir", required=True,
                   help="Folder with Eedi train.csv + misconception_mapping.csv")
    g.add_argument("--methods", nargs="+", default=["zero-shot-ss"],
                   choices=sorted(METHODS))
    g.add_argument("--model", default="claude-sonnet-4-6")
    g.add_argument("--splits-csv", default=None,
                   help="Frozen eedi_pairs split CSV (QuestionId,split); "
                        "omit for seeded fallback split")
    g.add_argument("--limit", type=int, default=None, help="Pilot: first N pairs")
    g.add_argument("--attempts", type=int, default=2,
                   help="Rejection-sampling retries per (pair, method)")
    g.add_argument("--cycle-k", type=int, default=25)
    g.add_argument("--no-filter", action="store_true",
                   help="ABLATION: skip all three verifier filters")
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--shots", type=int, default=3, help="k for few-shot methods")
    g.add_argument("--trace-bank", default=None,
                   help="JSONL from a previous run (for few-shot-icl-traces)")
    g.add_argument("--out-dir", default="sft_out")
    g.add_argument("--cache-dir", default="sft_cache")
    g.add_argument("--delay", type=float, default=0.1)
    g.add_argument("--dry-run", action="store_true",
                   help="No API calls; exercises plumbing + counts cache hits")
    g.set_defaults(fn=cmd_generate)

    m = sub.add_parser("map", help="Real-student corpus from MAP")
    m.add_argument("--map-csv", required=True, help="MAP train.csv")
    m.add_argument("--splits-csv", default=None,
                   help="Frozen map_questions split CSV")
    m.add_argument("--out-dir", default="sft_out")
    m.set_defaults(fn=cmd_map)

    x = sub.add_parser("mix", help="Interleave two corpora at exact ratio N:1")
    x.add_argument("--a", required=True, help="Majority corpus JSONL")
    x.add_argument("--b", required=True, help="Minority corpus JSONL")
    x.add_argument("--ratio", type=int, default=1, help="N in N:1 (a:b)")
    x.add_argument("--name", default=None, help="Output filename override")
    x.add_argument("--out-dir", default="sft_out")
    x.set_defaults(fn=cmd_mix)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
