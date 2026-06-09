"""Adaptive Diagnostic Dialogue (Plan A).

Conversation-as-assessment for misconception-level knowledge estimation. The package
is organized around a single closed loop:

    estimator (belief over knowledge state)
        -> selector (pick the most informative next probe)
        -> environment (simulated student answers)
        -> estimator.update(...)  ->  repeat

Everything in this loop runs on synthetic data with numpy alone; heavier components
(torch estimators, NL agents, real dataset loaders) are stubbed with stable interfaces.
"""

__version__ = "0.1.0"
