# Adaptive Diagnostic Dialogue

**Conversation-as-Assessment for Misconception-Level Knowledge Estimation** (Plan A).

A diagnostic loop that actively selects the next probe by *expected information gain*
(à la CAT/BALD), targets *misconception-level* state rather than scalar proficiency,
and is anchored to a falsifiable predictive metric: a diagnostic either predicts a
student's held-out responses better, or in fewer turns, than a non-adaptive baseline.

This repository is **starter scaffolding**. The core adaptivity loop (estimator →
selector → simulated student → metrics) runs end-to-end on *synthetic* data today, so
you can iterate on the experiment harness before wiring in real datasets and models.
Heavier pieces — a torch DKT/AKT estimator, the natural-language examiner agents, and
the real dataset loaders (pyKT/ASSISTments, Eedi, QATD-2k) — are present as clearly
marked stubs with stable interfaces.

## What runs today

```bash
pip install -e .
# end-to-end RQ1 adaptivity experiment on synthetic data:
python -m adaptive_diagnostic.experiment.run_adaptivity --students 60 --items 80 --seed 0
# smoke test (tiny, fast, asserts the loop produces an efficiency curve):
pytest -q
```

The experiment compares the **information-gain selector** against the baselines
(random, fixed difficulty-ascending, max-coverage over KCs, Fisher-information CAT)
and reports the diagnostic-efficiency curve — held-out item-correctness AUC as a
function of the number of probes administered — plus belief calibration (ECE).

## Layout

```
adaptive_diagnostic/
  config.py              # ExperimentConfig dataclass (+ optional YAML load)
  data/
    schemas.py           # Item, Interaction, StudentProfile dataclasses
    synthetic.py         # IRT-style synthetic item bank + students (ground truth)
    loaders.py           # STUBS: pyKT/ASSISTments, Eedi, QATD-2k loaders
  estimator/
    base.py              # Estimator ABC: belief / update / predict / misconception head
    bayesian_irt.py      # RUNNABLE numpy estimator (grid-Bayes per KC + Dirichlet head)
    dkt.py               # STUB: torch DKT/AKT estimator (pyKT-compatible interface)
  selector/
    base.py              # Selector ABC
    info_gain.py         # RUNNABLE expected-information-gain selector (BALD-style)
    baselines.py         # random / fixed-difficulty / max-coverage selectors
    fisher.py            # Fisher-information CAT selector
  environment/
    student.py           # SimulatedStudent (synthetic ground-truth env) + LLM stub
  agents/
    examiner.py          # STUB: NL examiner agent (renders probe as a question)
    estimator_agent.py   # STUB: parses NL reply -> structured response/distractor
    orchestrator.py      # STUB: LangGraph/AutoGen wiring; plain loop provided
  metrics/
    efficiency.py        # efficiency curve, items-to-target, area-under-curve, AUC
    calibration.py       # expected calibration error (ECE)
  experiment/
    run_adaptivity.py    # RQ1 driver: compares selectors, emits efficiency curves
tests/
  test_smoke.py          # end-to-end synthetic smoke test
```

## Research questions (see proposal)

- **RQ1 (adaptivity):** does info-driven probe selection reach a target held-out
  prediction accuracy in *fewer items* than random / fixed-difficulty / coverage?
- **RQ2 (misconception resolution):** can the loop identify a student's *specific*
  misconception (distractor family) better than base-rate, and in how many turns?
- **RQ3 (dialogue layer):** does wrapping the selector/estimator in NL examiner +
  estimator agents preserve the advantage, and are dialogues face-valid vs QATD-2k?
- **RQ4 (downstream value, stretch):** does misconception-level feedback beat
  proficiency-only feedback for a simulated student's subsequent correctness?

## Next steps (maps to the gated timeline)

- **W1–2 (G0):** replace `data/loaders.py` stubs with pyKT/ASSISTments + Eedi loaders.
- **W3–5 (G1):** swap in `estimator/dkt.py`; run `run_adaptivity` on real sequences;
  confirm the adaptive policy hits the fixed policy's full-length AUC with ≥25% fewer
  items, stable across ≥3 seeds.
- **W6–7 (G2):** turn on the misconception head against Eedi distractor labels.
- **W8–10 (G3):** implement `agents/*` for the NL layer and validate the simulated
  student against QATD-2k before trusting it as an environment.
