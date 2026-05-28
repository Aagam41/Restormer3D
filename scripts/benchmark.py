#!/usr/bin/env python3
"""
benchmark.py — run every (algo, stack) pair that hasn't been run yet.

Discovers noisy/clean pairs under a dataset directory and runs the
selected algos on each. Already-completed (algo, stack) pairs (per
runs.csv) are SKIPPED so the script is safe to rerun.

Usage:
    python scripts/benchmark.py \\
        --noisy-dir /path/to/noisy \\
        --clean-dir /path/to/clean \\
        --algos restormer3d
        --configs configs/restormer3d_eval_only.py configs/restormer3d_finetune.py
        --results-dir benchmark_results
        --figures-dir paper_figures

If --configs is not given, the script uses ONE default config per algo
named configs/<algo>_default.py if it exists, else configs/<algo>_t4.py.

Outputs are exactly the same as scripts/run_one.py — one consolidated
benchmark_results/ tree with CSVs and per-run TIFF/PNG outputs.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import algos as _algos_pkg
from configs import load_config
from runner.core import run_one
from runner.io import find_stacks
from runner.csv_db import (
    completed_pairs, write_algo_registry,
)


def _serializable(d):
    """Best-effort JSON-friendly version of a config dict."""
    if d is None:
        return None
    if isinstance(d, (str, int, float, bool)):
        return d
    if isinstance(d, (list, tuple)):
        return [_serializable(x) for x in d]
    if isinstance(d, dict):
        return {k: _serializable(v) for k, v in d.items()}
    return str(d)


def _resolve_config_for_algo(algo: str, config_paths) -> Path:
    """Resolve the config file for an algo.

    If `config_paths` contains a config whose 'algo' key matches, use it.
    Otherwise fall back to configs/<algo>_default.py or configs/<algo>_t4.py.

    NOTE: this returns at most ONE path. Use `_resolve_configs_for_algos`
    below for multi-config support (passing several configs per algo).
    """
    for cp in config_paths or []:
        cp = Path(cp)
        try:
            cfg = load_config(cp)
            if cfg.get("algo") == algo:
                return cp
        except Exception:
            continue
    candidates = [
        ROOT / "configs" / f"{algo}_default.py",
        ROOT / "configs" / f"{algo}_t4.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No config found for algo '{algo}'. "
        f"Pass --configs explicitly or create configs/{algo}_default.py."
    )


def _resolve_configs_for_algos(algo_names, config_paths):
    """Resolve potentially MULTIPLE configs per algo.

    Returns a list of (algo, config_path) tuples — one per (algo, config)
    pair to run. So passing `--algos dvt_unet3d --configs A.py B.py`
    where both target dvt_unet3d returns two entries for dvt_unet3d,
    each with its own config — both will run as separate jobs with
    distinct config_name in the CSVs.

    Behaviour:
      * If --configs is given: every config whose `algo` key is in
        `algo_names` produces one (algo, cfg) entry. Algos that are in
        `algo_names` but have NO matching config fall back to the
        default lookup.
      * If --configs is not given: every algo gets its default config.
    """
    pairs = []
    matched_algos = set()
    if config_paths:
        for cp in config_paths:
            cp = Path(cp)
            try:
                cfg = load_config(cp)
            except Exception as e:
                print(f"  [skip] {cp.name}: {e}")
                continue
            a = cfg.get("algo")
            if a is None:
                print(f"  [skip] {cp.name}: config has no 'algo' key")
                continue
            if a not in algo_names:
                # User passed this config but didn't list its algo.
                # Be helpful: include it anyway, since they probably meant
                # to. Add the algo to the active set.
                print(f"  [note] {cp.name}: targets algo '{a}' which "
                      f"isn't in --algos — adding it.")
            pairs.append((a, cp))
            matched_algos.add(a)

    # For any algo that's in --algos but didn't get a matching --config,
    # fall back to the default lookup.
    for a in algo_names:
        if a in matched_algos:
            continue
        try:
            pairs.append((a, _resolve_config_for_algo(a, None)))
        except FileNotFoundError as e:
            print(f"  [skip] {a}: {e}")
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy-dir",  required=True,
                    help="Directory with noisy stacks (*.tif)")
    p.add_argument("--clean-dir",  default=None,
                    help="Directory with clean GT stacks (*.tif). "
                         "If absent, metrics are not computed.")
    p.add_argument("--algos",      nargs="+", default=["all"],
                    help="Algo names from algos/__init__.py REGISTRY, "
                         "or 'all'. If --configs is given, configs may "
                         "ADD to this list (the algo of each passed "
                         "config is included even if not in --algos).")
    p.add_argument("--configs",    nargs="*", default=None,
                    help="Explicit config files. EACH config produces a "
                         "separate job — pass multiple configs for the "
                         "SAME algo (e.g. *_best.py *_normalised.py) to "
                         "run different variants side by side. Configs "
                         "are matched to algos via their 'algo' key.")
    p.add_argument("--results-dir", default=str(ROOT / "benchmark_results"))
    p.add_argument("--figures-dir", default=str(ROOT / "paper_figures"))
    p.add_argument("--group-id", default=None,
                    help="Group ID for output scoping. If omitted, a "
                         "fresh one is generated AND we skip (algo, stack) "
                         "pairs that succeeded in any prior group. If "
                         "given, we run only the (algo, stack) pairs that "
                         "have NOT yet succeeded WITHIN that group — "
                         "useful for re-running a specific group without "
                         "interference from other groups (see also "
                         "--rerun).")
    p.add_argument("--rerun", action="store_true",
                    help="Run every (algo, stack) pair regardless of "
                         "prior completions. Use with --group-id to "
                         "force a complete redo of that group.")
    p.add_argument("--frame", type=int, default=None,
                    help="Override paper_frame from config.")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--no-figures",    action="store_true")
    p.add_argument("--pretrained", type=Path, default=None,
                    help="Optional path to a pretrained checkpoint (.pth). "
                         "When given, every (algo, stack) job loads these "
                         "weights as initialization before training. "
                         "Combined with a config that has 0 training iters, "
                         "this yields a pure 'load + infer' eval. Combined "
                         "with a short-schedule config, this yields a "
                         "fine-tune pass. Algos without init_state_dict "
                         "support (currently only dvt_unet3d and "
                         "restormer3d support it) print a warning and "
                         "train from scratch.")
    p.add_argument("--dry-run", action="store_true",
                    help="List jobs that would run; don't execute.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Group ID ──────────────────────────────────────────────
    from runner import io as io_
    group_id = args.group_id or io_.make_group_id()
    user_supplied_group = args.group_id is not None
    (results_dir / group_id).mkdir(parents=True, exist_ok=True)
    (figures_dir / group_id).mkdir(parents=True, exist_ok=True)
    print(f"\nGroup ID for this benchmark: {group_id}"
          + (" (user-supplied)" if user_supplied_group else ""))

    # Resolve algos
    if "all" in args.algos:
        algo_names = sorted(_algos_pkg.REGISTRY.keys())
    else:
        algo_names = list(args.algos)
        for a in algo_names:
            if a not in _algos_pkg.REGISTRY:
                sys.exit(f"Unknown algo: {a}. "
                         f"Known: {sorted(_algos_pkg.REGISTRY.keys())}")

    # Write the registry snapshot to algos.csv (group-scoped)
    write_algo_registry(results_dir, _algos_pkg.list_algos(),
                         group_id=group_id)

    # Resolve configs per algo (NEW: supports multiple configs per algo).
    # Returns a list of (algo, config_path) tuples — duplicates of an
    # algo are allowed and become separate jobs with distinct config_name.
    cfg_pairs = _resolve_configs_for_algos(algo_names, args.configs)
    if not cfg_pairs:
        sys.exit("No algos resolved — nothing to run.")

    # Discover stacks
    pairs = find_stacks(args.noisy_dir, args.clean_dir)
    if not pairs:
        sys.exit(f"No .tif files found under {args.noisy_dir}.")

    # ── Decide which "completed" set to consult ───────────────
    # By default (fresh group, no --rerun): skip pairs done in any group.
    # With --group-id (user-supplied) and no --rerun: skip pairs already
    # done in THIS group only.
    # With --rerun: skip nothing.
    if args.rerun:
        done = set()
        print(f"  --rerun: ignoring all prior completions.")
    elif user_supplied_group:
        # Scoped to this group: only skip what's done IN this group
        done = set()
        from runner.csv_db import read_runs as _read_runs
        for row in _read_runs(results_dir, group_id=group_id):
            if row.get("status", "").lower() == "success":
                # NEW: completion key is (algo, stack, config_name) so
                # different configs of the same algo don't collide.
                done.add((row["algo"], row["stack_name"],
                          row.get("config_name", "")))
        print(f"  --group-id given: only skipping completions IN this "
              f"group ({len(done)} found).")
    else:
        done = completed_pairs(results_dir)
        print(f"  default: skipping pairs completed in any group "
              f"({len(done)} found).")

    # ── Build the job list ─────────────────────────────────────
    jobs = []
    for stack_name, noisy_path, clean_path in pairs:
        for algo, cfg_path in cfg_pairs:
            config_name = Path(cfg_path).stem
            # In user-supplied-group mode, completion is per
            # (algo, stack, config_name) so multiple variants of the
            # same algo can coexist. In default mode, completion is per
            # (algo, stack) for backward compat with the cross-group
            # skip behaviour.
            if user_supplied_group and not args.rerun:
                if (algo, stack_name, config_name) in done:
                    continue
            elif not args.rerun:
                if (algo, stack_name) in done:
                    continue
            jobs.append((algo, stack_name, noisy_path, clean_path,
                         cfg_path))

    print(f"\nBenchmark plan:")
    print(f"  noisy dir : {args.noisy_dir}")
    print(f"  clean dir : {args.clean_dir or '(none)'}")
    print(f"  jobs      : {len(jobs)} new")
    if args.pretrained is not None:
        if not args.pretrained.exists():
            sys.exit(f"--pretrained file not found: {args.pretrained}")
        print(f"  pretrained: {args.pretrained}  "
              f"(loaded into init_state_dict for supported algos)")
    print(f"  configs   :")
    seen = set()
    for a, cp in cfg_pairs:
        key = (a, cp.name)
        if key in seen:
            continue
        seen.add(key)
        print(f"    {a:25s}  ← {cp.name}")
    print(f"  stacks    : {len(pairs)} ({[n for n,_,_ in pairs]})")
    print(f"  results   : {results_dir / group_id}")
    print(f"  figures   : {figures_dir / group_id}")

    if args.dry_run or len(jobs) == 0:
        for i, (algo, stack, noisy, clean, cfg_path) in enumerate(jobs, 1):
            print(f"  [{i:3d}] {algo:25s}  on  {stack:15s}  "
                  f"({cfg_path.name})")
        return 0

    # ── Build the group manifest BEFORE running any jobs ──────
    # We snapshot every algo's resolved config up front so the manifest
    # is useful even if a later job crashes. After all jobs complete we
    # update completed_at + summary fields.
    from runner import runtime_log as _rtl
    env_info = _rtl.env_info()
    manifest = {
        "group_id":      group_id,
        "started_at":    datetime.now(timezone.utc).isoformat(),
        "host":          env_info.get("host", ""),
        "python":        env_info.get("python", ""),
        "torch":         env_info.get("torch", ""),
        "gpu_name":      env_info.get("gpu_name", ""),
        "noisy_dir":     str(args.noisy_dir),
        "clean_dir":     str(args.clean_dir) if args.clean_dir else "",
        "stacks":        [name for name, _, _ in pairs],
        "configs":       {},
        "total_jobs":    len(jobs),
    }
    for algo, cfg_path in cfg_pairs:
        try:
            cfg_snapshot = load_config(cfg_path)
        except Exception as e:
            cfg_snapshot = {"_load_error": str(e)}
        cn = Path(cfg_path).stem
        manifest["configs"][f"{algo}__{cn}"] = {
            "algo":        algo,
            "config_name": cn,
            "config_path": str(Path(cfg_path).resolve()),
            "full_config": _serializable(cfg_snapshot),
        }
    io_.write_group_manifest(
        results_dir=results_dir, group_id=group_id,
        manifest=manifest, figures_dir=figures_dir,
    )

    # Execute
    successes = 0; failures = 0
    for i, (algo, stack, noisy, clean, cfg_path) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {algo}  on  {stack}  (group={group_id})")
        cfg = load_config(cfg_path)
        cfg.pop("algo")
        paper_frame = args.frame or cfg.pop("paper_frame", 750)
        cfg.pop("name", None); cfg.pop("description", None)
        config_name = Path(cfg_path).stem
        try:
            summary = run_one(
                algo=algo, config=cfg,
                noisy_path=noisy, clean_path=clean,
                results_dir=results_dir, figures_dir=figures_dir,
                group_id=group_id, config_name=config_name,
                paper_frame=paper_frame,
                save_checkpoint=not args.no_checkpoint,
                save_figures=not args.no_figures,
                verbose=True,
                pretrained_path=args.pretrained,
            )
            if summary.get("status") == "success":
                successes += 1
            else:
                failures += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return 130
        except Exception as e:
            print(f"  Job failed: {type(e).__name__}: {e}")
            # run_one already logs errors; continue to next job
            continue

    # ── Update manifest with completion summary ────────────────
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["successes"]    = successes
    manifest["failures"]     = failures
    io_.write_group_manifest(
        results_dir=results_dir, group_id=group_id,
        manifest=manifest, figures_dir=figures_dir,
    )

    print(f"\n{'='*70}\nBenchmark complete: {len(jobs)} job(s) executed "
          f"({successes} success, {failures} failed).")
    print(f"Group: {group_id}")
    print(f"Manifest: {results_dir / group_id / 'group_manifest.json'}")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
