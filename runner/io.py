"""
I/O helpers. Loading and saving TIFF stacks, locating noisy/clean pairs
in a directory tree.

The dataset layout we assume:

    <dataset_root>/
        noisy/
            F1.tif
            F2.tif
            ...
        clean/         (optional — present for eval)
            F1.tif
            F2.tif
            ...

Other layouts are supported via explicit --noisy-dir / --clean-dir flags
on the scripts.
"""

from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


def _load_tif(path) -> np.ndarray:
    """Try tifffile first, fall back to SimpleITK."""
    path = str(path)
    try:
        import tifffile
        return tifffile.imread(path)
    except ImportError:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _save_tif(arr: np.ndarray, path, compression: str = "zlib"):
    """Save with tifffile if available, fall back to SimpleITK."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile
        tifffile.imwrite(str(path), arr, compression=compression)
    except ImportError:
        import SimpleITK as sitk
        sitk.WriteImage(
            sitk.GetImageFromArray(arr), str(path), useCompression=True,
        )


def load_stack(path) -> np.ndarray:
    """Load a noisy/clean stack. Returns the raw dtype (e.g. int16)."""
    arr = _load_tif(path)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D stack [F, H, W] at {path}, got shape {arr.shape}"
        )
    return arr


def save_stack(arr: np.ndarray, path):
    """Save a denoised stack to `path`."""
    _save_tif(arr, path)


def stack_info(name: str, arr: np.ndarray, path) -> dict:
    """Build a dict describing this stack — written to stacks.csv."""
    return {
        "stack_name": name,
        "path": str(path),
        "frames":  int(arr.shape[0]),
        "height":  int(arr.shape[1]),
        "width":   int(arr.shape[2]),
        "dtype":   str(arr.dtype),
        "min":     float(arr.min()),
        "max":     float(arr.max()),
        "mean":    float(arr.mean()),
        "std":     float(arr.std()),
        "n_bytes": int(arr.nbytes),
    }


# ── Dataset discovery ─────────────────────────────────────────

def find_stacks(noisy_dir, clean_dir: Optional[Path] = None,
                 pattern: str = "*.tif") -> List[Tuple[str, Path, Optional[Path]]]:
    """
    Return a list of (stack_name, noisy_path, clean_path|None).

    Stack name = file stem (e.g. F1.tif -> "F1").
    If `clean_dir` is provided, match clean files by filename.
    """
    noisy_dir = Path(noisy_dir)
    noisy_files = sorted(noisy_dir.glob(pattern))
    pairs = []
    for nf in noisy_files:
        name = nf.stem
        cf = None
        if clean_dir is not None:
            candidate = Path(clean_dir) / nf.name
            if candidate.exists():
                cf = candidate
        pairs.append((name, nf, cf))
    return pairs


# ── Output paths (group-scoped) ───────────────────────────────
#
# Every run lives under a "group" — a single short tag that identifies
# one benchmark invocation (or one run_one invocation, if used
# standalone). The layout is:
#
#   <results_dir>/<group_id>/runs.csv               ← per-group CSVs
#   <results_dir>/<group_id>/metrics.csv
#   <results_dir>/<group_id>/config.csv
#   <results_dir>/<group_id>/timing.csv
#   <results_dir>/<group_id>/gpu_log.csv
#   <results_dir>/<group_id>/stacks.csv
#   <results_dir>/<group_id>/algos.csv
#   <results_dir>/<group_id>/group_manifest.json    ← human-readable
#                                                     config snapshot
#   <results_dir>/<group_id>/outputs/<run_id>/<stack>.tif
#   <results_dir>/<group_id>/checkpoints/<run_id>/
#   <results_dir>/<group_id>/errors/<run_id>.log
#   <figures_dir>/<group_id>/group_manifest.json    ← mirror
#   <figures_dir>/<group_id>/<run_id>/<stack>_frame0750.png
#   <figures_dir>/<group_id>/_leaderboard/<metric>.png

import uuid as _uuid
from datetime import datetime as _datetime, timezone as _timezone


def make_group_id() -> str:
    """Fresh group_id used to scope one benchmark invocation."""
    ts = _datetime.now(_timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha = _uuid.uuid4().hex[:6]
    return f"g_{ts}_{sha}"


def group_results_dir(results_dir, group_id: str) -> Path:
    """All result CSVs + outputs for one group."""
    out = Path(results_dir) / group_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def group_figures_dir(figures_dir, group_id: str) -> Path:
    """All paper figures for one group."""
    out = Path(figures_dir) / group_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_output_dir(results_dir, group_id: str, run_id: str) -> Path:
    """Where this run's per-stack denoised TIFFs go (group-scoped)."""
    out = Path(results_dir) / group_id / "outputs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_figure_dir(figures_dir, group_id: str, run_id: str) -> Path:
    """Where this run's paper-style comparison figures go (group-scoped)."""
    out = Path(figures_dir) / group_id / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_checkpoint_dir(results_dir, group_id: str, run_id: str) -> Path:
    out = Path(results_dir) / group_id / "checkpoints" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Group manifest (human-readable per-algo config snapshot) ─────

def write_group_manifest(results_dir, group_id: str,
                          manifest: dict,
                          figures_dir=None):
    """
    Write `<results_dir>/<group_id>/group_manifest.json` and optionally
    mirror to `<figures_dir>/<group_id>/group_manifest.json`.

    `manifest` should contain at minimum:
        group_id, started_at, host, gpu_name, configs (dict[algo -> cfg])
    """
    import json
    grp_path = Path(results_dir) / group_id / "group_manifest.json"
    grp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(grp_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str, sort_keys=True)

    if figures_dir is not None:
        fig_path = Path(figures_dir) / group_id / "group_manifest.json"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fig_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str, sort_keys=True)
    return grp_path
