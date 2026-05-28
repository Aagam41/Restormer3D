"""
configs — preset configs per (algo, target hardware) combination.

Each config is just a Python module that defines a top-level CONFIG dict.
The CONFIG dict must contain at minimum:
    algo: <name in algos/__init__.py REGISTRY>
plus whatever kwargs the algo's train_self_supervised expects.

Optional top-level fields:
    name:         short label for tables/plots
    description:  one-line summary
    paper_frame:  default frame number for the 1x4 paper grid (default 750)

To use a config:
    python scripts/run_one.py --config configs/restormer3d_finetune.py --noisy <path>

You can also build a config dict in code without a file:
    from runner.core import run_one
    run_one("restormer3d", {"dim": 32, ...}, ...)
"""

import importlib.util
from pathlib import Path


def load_config(path) -> dict:
    """Load a .py config file and return its CONFIG dict."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CONFIG"):
        raise ValueError(
            f"Config {path} does not define a top-level CONFIG dict."
        )
    cfg = dict(mod.CONFIG)
    # Annotate with the source path for traceability
    cfg.setdefault("__config_path", str(path))
    cfg.setdefault("__config_name", path.stem)
    return cfg
