"""
Runner — the framework that drives all algos uniformly.

Public modules:
    runner.io            — TIFF loading / dataset discovery
    runner.csv_db        — multi-CSV "database" of results
    runner.runtime_log   — GPU/CPU sampling + stage timer
    runner.eval_runner   — metric computation
    runner.plots         — paper-style figures
    runner.preprocessing — pluggable normalization + temporal-target
    runner.core          — `run_one(...)` the top-level entry point

Anything in the runner package is algo-agnostic; algorithm code lives
under `algos/` and is reached only through the registry.
"""
