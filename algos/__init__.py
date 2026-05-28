"""
Algorithm registry. This is the Restormer3D-only build of the framework.

To switch which algo runs, change the `algo` field of your config — no
code changes required.

To add a NEW algorithm:
  1. Drop a new module into algos/ that exposes the uniform API:
       compute_norm_params, normalize, denormalize
       train_self_supervised(stack, device, config) -> (model, cfg)
       denoise_stack(model, stack, config, device)  -> np.ndarray
       save_checkpoint(model, config, path)
       load_checkpoint(path, device=None) -> (model, cfg)
  2. Add an entry to REGISTRY below.

The runner ONLY interacts with algos through this registry + the
uniform API. It never imports algo modules directly.
"""

import importlib
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AlgoSpec:
    name: str                       # short identifier used everywhere
    module: str                     # importable module path
    family: str                     # grouping label (for tables/plots)
    description: str                # one-line summary
    paper: str = ""                 # optional reference
    tags: list = field(default_factory=list)


REGISTRY: Dict[str, AlgoSpec] = {

    "restormer3d": AlgoSpec(
        name="restormer3d",
        module="algos.restormer3d",
        family="Restormer",
        description="3D adaptation of Restormer (MDTA + GDFN) for denoising.",
        paper="Zamir et al., CVPR 2022 (arXiv 2111.09881)",
        tags=["transformer", "mdta", "gdfn"],
    ),
}


def get_algo(name: str):
    """Import and return the algo module for `name`."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: "
            f"{sorted(REGISTRY.keys())}"
        )
    spec = REGISTRY[name]
    return importlib.import_module(spec.module)


def get_spec(name: str) -> AlgoSpec:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: "
            f"{sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]


def list_algos():
    """Return the registry as a list of dicts (for CSV writing)."""
    return [
        {
            "name": s.name,
            "module": s.module,
            "family": s.family,
            "description": s.description,
            "paper": s.paper,
            "tags": ";".join(s.tags),
        }
        for s in REGISTRY.values()
    ]
