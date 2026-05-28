#!/usr/bin/env python3
"""
run_one.py — run a single (algo, stack) job.

Usage:
    python scripts/run_one.py \\
        --config  configs/restormer3d_finetune.py \\
        --noisy   /path/to/noisy/F1.tif \\
        --clean   /path/to/clean/F1.tif        # optional, for eval
        --results-dir benchmark_results        # default: ./benchmark_results
        --figures-dir paper_figures            # default: ./paper_figures
        --frame   750                          # paper-figure frame

Outputs:
    benchmark_results/runs.csv             — master table, one row per run
    benchmark_results/metrics.csv          — all metrics this run
    benchmark_results/config.csv           — flattened config
    benchmark_results/timing.csv           — per-stage durations
    benchmark_results/gpu_log.csv          — GPU samples during the run
    benchmark_results/outputs/<run_id>/<stack>.tif
    benchmark_results/checkpoints/<run_id>/<stack>.pth
    paper_figures/<run_id>/<stack>_frame0750.png
"""

import argparse
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# Make `algos` / `runner` / `configs` importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.core import run_one
from runner import io as io_, runtime_log
from configs import load_config


def _collect_env() -> dict:
    return runtime_log.env_info()


def _write_or_update_manifest(*, results_dir, figures_dir, group_id,
                               algo, config_name, config_path,
                               full_config, env_info):
    """
    Maintain a `group_manifest.json` describing every algo that ran
    under this group. Idempotent: re-running adds new algo entries; the
    same algo+config_name in the same group is treated as an update,
    not a duplicate.
    """
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    manifest_path = results_dir / group_id / "group_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "group_id":   group_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "host":       env_info.get("host", ""),
            "python":     env_info.get("python", ""),
            "torch":      env_info.get("torch", ""),
            "gpu_name":   env_info.get("gpu_name", ""),
            "configs":    {},
        }

    # The KEY is `<algo>__<config_name>` so the same algo with different
    # configs gets separate entries.
    key = f"{algo}__{config_name}"
    manifest["configs"][key] = {
        "algo":        algo,
        "config_name": config_name,
        "config_path": config_path,
        "logged_at":   datetime.now(timezone.utc).isoformat(),
        "full_config": _serializable(full_config),
    }
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["total_configs"] = len(manifest["configs"])

    io_.write_group_manifest(
        results_dir=results_dir, group_id=group_id,
        manifest=manifest, figures_dir=figures_dir,
    )


def _serializable(d):
    """Best-effort JSON-friendly version of a config dict."""
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = _serializable(v)
        else:
            out[k] = str(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True,
                    help="Path to a configs/*.py file")
    p.add_argument("--noisy",      required=True,
                    help="Path to the noisy input .tif")
    p.add_argument("--clean",      default=None,
                    help="Optional path to the clean ground-truth .tif")
    p.add_argument("--results-dir", default=str(ROOT / "benchmark_results"),
                    help="Where CSVs, outputs, and checkpoints are written")
    p.add_argument("--figures-dir", default=str(ROOT / "paper_figures"),
                    help="Where paper figures are written")
    p.add_argument("--group-id", default=None,
                    help="Group ID for output scoping. If omitted, a fresh "
                         "one is generated. Pass an existing group_id to "
                         "add this run to a prior benchmark group.")
    p.add_argument("--frame",      type=int, default=None,
                    help="Frame index for the 1x4 paper figure (default: "
                         "from config or 750)")
    p.add_argument("--no-checkpoint", action="store_true",
                    help="Don't save the trained model checkpoint")
    p.add_argument("--no-figures",   action="store_true",
                    help="Don't generate paper figures")
    p.add_argument("--quiet", action="store_true",
                    help="Less verbose output")
    args = p.parse_args()

    cfg = load_config(args.config)
    algo = cfg.pop("algo")     # mandatory
    # Pull out framework-level keys
    name = cfg.pop("name", None)
    desc = cfg.pop("description", None)
    paper_frame = args.frame or cfg.pop("paper_frame", 750)
    config_name = Path(args.config).stem    # e.g. "restormer3d_finetune"

    print(f"\nConfig: {Path(args.config).name}")
    if name:
        print(f"  name: {name}")
    if desc:
        print(f"  desc: {desc}")
    print(f"  algo: {algo}")
    print(f"  paper_frame: {paper_frame}")

    summary = run_one(
        algo=algo,
        config=cfg,
        noisy_path=args.noisy,
        clean_path=args.clean,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        group_id=args.group_id,
        config_name=config_name,
        paper_frame=paper_frame,
        save_checkpoint=not args.no_checkpoint,
        save_figures=not args.no_figures,
        verbose=not args.quiet,
    )

    # Write a per-run manifest under the group folder so the group
    # alone is enough to know what configs ran. If multiple run_one
    # invocations share a group, each appends an entry.
    _write_or_update_manifest(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        group_id=summary["group_id"],
        algo=algo, config_name=config_name,
        config_path=str(Path(args.config).resolve()),
        full_config={**cfg, "algo": algo, "paper_frame": paper_frame,
                      "name": name, "description": desc},
        env_info=_collect_env(),
    )

    print(f"\n────  Run complete  ────")
    print(f"  status   : {summary['status']}")
    print(f"  run_id   : {summary['run_id']}")
    print(f"  group_id : {summary['group_id']}")
    print(f"  config   : {config_name}")
    print(f"  output   : {summary.get('output_path', '(none)')}")
    print(f"  figure   : {summary.get('figure_path', '(none)')}")
    if summary.get("metrics"):
        m = summary["metrics"]
        print(f"  stSNR    : {m.get('stSNR', float('nan')):.4f}")
        print(f"  stPSNR   : {m.get('stPSNR', float('nan')):.4f}")
        print(f"  stSI_PSNR: {m.get('stSI_PSNR', float('nan')):.4f}")
    print(f"  total_sec: {summary.get('total_sec', 0):.1f}")

    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
