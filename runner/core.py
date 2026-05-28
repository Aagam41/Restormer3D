"""
runner.core — the top-level `run_one(...)` function.

Given an algo name, a config dict, a noisy stack and an optional clean
stack, it:
    1. Generates a unique run_id.
    2. Starts a GPU sampler thread.
    3. Times stages (training, inference, eval) separately.
    4. Calls the algo's train_self_supervised + denoise_stack.
    5. Saves the denoised TIFF, the checkpoint, and a paper figure.
    6. Computes metrics if a clean stack is given.
    7. Writes one row each to runs.csv / config.csv / timing.csv /
       metrics.csv / stacks.csv (and continuous samples to gpu_log.csv).
    8. Returns a summary dict so the caller can print or chain.

Failures are caught and logged with status="error"; a stack trace is
saved to <results_dir>/errors/<run_id>.log.
"""

import os
import sys
import time
import uuid
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

import algos
from . import io as io_, csv_db, runtime_log, eval_runner, plots
from . import preprocessing as prep


def _make_run_id(algo: str, stack_name: str) -> str:
    """Stable but unique id: <algo>__<stack>__<utc-yyyymmdd-hhmmss>__<sha8>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha = uuid.uuid4().hex[:8]
    return f"{algo}__{stack_name}__{ts}__{sha}"


def run_one(
    algo: str,
    config: Dict[str, Any],
    noisy_path,
    clean_path: Optional[Path] = None,
    *,
    results_dir,
    figures_dir,
    group_id: Optional[str] = None,
    config_name: Optional[str] = None,
    paper_frame: int = 750,
    save_checkpoint: bool = True,
    save_figures: bool = True,
    gpu_sample_interval: float = 2.0,
    verbose: bool = True,
    pretrained_path: Optional[Path] = None,
):
    """
    Run a single (algo, noisy_path) job. Returns a summary dict.

    Args:
        group_id:    Group this run belongs to. Used to scope output
                     folders and CSV tables. If None, a fresh group_id
                     is generated. When multiple runs share a group_id
                     (e.g. all algos in one benchmark sweep), their
                     results live under the same `<results_dir>/<group_id>/`
                     subtree and share a single `group_manifest.json`.
        config_name: Human-readable config label (typically the stem of
                     the config .py file, e.g. "dvt_unet3d_t4"). Stored
                     in runs.csv so you can later filter results by
                     config name when collating data across groups.
        pretrained_path: Optional path to a pretrained checkpoint. If
                     given AND the algo's train_self_supervised accepts
                     an `init_state_dict` parameter, the model is
                     initialized from this checkpoint before training.
                     Combined with a config that has 0 training iters
                     (warmup_iters=0, n2v_iters=0), this gives a pure
                     "load + infer" pass. Combined with a short-schedule
                     config, this gives a fine-tune pass. Algos without
                     init_state_dict support log a warning and proceed
                     with from-scratch training.

    Side effects (all paths group-scoped):
        - One row added to <results_dir>/<group_id>/runs.csv
        - Rows added to config.csv, timing.csv, metrics.csv (if clean),
          stacks.csv, gpu_log.csv
        - Denoised TIFF saved under <results_dir>/<group_id>/outputs/<run_id>/
        - Checkpoint saved under <results_dir>/<group_id>/checkpoints/<run_id>/
        - Paper figure under <figures_dir>/<group_id>/<run_id>/
    """
    import torch

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve or generate the group_id
    if group_id is None:
        group_id = io_.make_group_id()
    if config_name is None:
        config_name = config.get("__config_name", config.get("name", ""))

    # Make sure the group folders exist
    io_.group_results_dir(results_dir, group_id)
    io_.group_figures_dir(figures_dir, group_id)

    noisy_path = Path(noisy_path)
    stack_name = noisy_path.stem
    run_id = _make_run_id(algo, stack_name)

    if verbose:
        print(f"\n{'='*70}\n RUN  algo={algo}  stack={stack_name}\n"
              f"      run_id={run_id}\n"
              f"      group_id={group_id}  config={config_name}\n"
              f"{'='*70}")

    # ── Static env info written eagerly so we can debug if something
    # blows up below. ──────────────────────────────────────────
    env = runtime_log.env_info()

    # GPU sampler in the background (group-scoped output)
    sampler = runtime_log.GPUSampler(
        results_dir=results_dir / group_id, run_id=run_id,
        interval_sec=gpu_sample_interval,
    )
    sampler.start()

    timer = runtime_log.StageTimer()
    status = "success"
    error_msg = ""
    summary: Dict[str, Any] = {"run_id": run_id, "algo": algo,
                                "stack_name": stack_name}

    denoised = None
    metrics: Dict[str, float] = {}

    try:
        # ── Load noisy ────────────────────────────────────────
        with timer.stage("load_noisy"):
            noisy = io_.load_stack(noisy_path)
        info_n = io_.stack_info(stack_name, noisy, noisy_path)
        info_n["role"] = "noisy"
        info_n["run_id"] = run_id
        info_n["group_id"] = group_id
        csv_db.write_stack_info(results_dir, info_n, group_id=group_id)
        if verbose:
            print(f"   Noisy: shape={noisy.shape} dtype={noisy.dtype} "
                  f"range=[{noisy.min()}, {noisy.max()}]")

        # ── Load clean if provided ────────────────────────────
        clean = None
        if clean_path is not None and Path(clean_path).exists():
            with timer.stage("load_clean"):
                clean = io_.load_stack(clean_path)
            info_c = io_.stack_info(f"{stack_name}_clean", clean, clean_path)
            info_c["role"] = "clean"
            info_c["run_id"] = run_id
            info_c["group_id"] = group_id
            csv_db.write_stack_info(results_dir, info_c, group_id=group_id)
            if verbose:
                print(f"   Clean: shape={clean.shape} dtype={clean.dtype} "
                      f"range=[{clean.min()}, {clean.max()}]")

        # ── Resolve algo module ───────────────────────────────
        mod = algos.get_algo(algo)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Pluggable preprocessing (Option A — strict refactor) ──
        # `normalization` and `temporal_target` are optional string keys.
        # Each algo declares DEFAULT_NORMALIZATION and
        # DEFAULT_TEMPORAL_TARGET at module level, and reads these
        # config keys itself to choose the strategy at runtime.
        # We just record what was requested + what the algo defaults are
        # so the run is traceable from the CSV log.
        eff_config = dict(config)
        algo_default_norm = getattr(mod, "DEFAULT_NORMALIZATION", None)
        algo_default_temp = getattr(mod, "DEFAULT_TEMPORAL_TARGET", None)
        eff_config.setdefault("__algo_default_normalization",
                                algo_default_norm)
        eff_config.setdefault("__algo_default_temporal_target",
                                algo_default_temp)
        # Validate strategy names early so we fail fast with a clear msg
        norm_name = eff_config.get("normalization", algo_default_norm)
        temp_name = eff_config.get("temporal_target", algo_default_temp)
        if norm_name is not None:
            try:
                ns = prep.resolve_normalization(norm_name)
                eff_config["__resolved_normalization"] = ns.name
                if verbose:
                    print(f"   Normalization: {ns.name}"
                          + (" (algo default)" if norm_name == algo_default_norm
                             else " (config override)"))
            except KeyError as e:
                raise ValueError(f"Bad normalization '{norm_name}': {e}")
        if temp_name is not None:
            try:
                ts = prep.resolve_temporal_target(temp_name)
                eff_config["__resolved_temporal_target"] = ts.name
                if verbose:
                    print(f"   Temporal target: {ts.name} "
                          f"(returns {ts.returns})"
                          + (" (algo default)" if temp_name == algo_default_temp
                             else " (config override)"))
            except KeyError as e:
                raise ValueError(f"Bad temporal_target '{temp_name}': {e}")

        # ── Load pretrained init (optional) ───────────────────
        # Build train_kwargs dict; only add init_state_dict if the algo
        # accepts it (currently dvt_unet3d, restormer3d).
        train_kwargs = dict(
            stack=noisy, device=device, config=eff_config, verbose=verbose,
        )
        pretrained_state = None
        if pretrained_path is not None:
            import inspect
            sig = inspect.signature(mod.train_self_supervised)
            if "init_state_dict" not in sig.parameters:
                print(f"   WARNING: algo '{algo}' does not support "
                      f"init_state_dict; --pretrained ignored, "
                      f"training from scratch.")
            else:
                try:
                    if verbose:
                        print(f"   Loading pretrained init from "
                              f"{pretrained_path}…")
                    pre_model, _ = mod.load_checkpoint(
                        str(pretrained_path), device=device,
                    )
                    pretrained_state = pre_model.state_dict()
                    del pre_model
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    train_kwargs["init_state_dict"] = pretrained_state
                    # Record in config snapshot so it lands in config.csv
                    eff_config["__pretrained_path"] = str(pretrained_path)
                except Exception as e:
                    print(f"   WARNING: failed to load pretrained from "
                          f"{pretrained_path}: {e}. Falling back to "
                          f"from-scratch training.")

        # ── Training ──────────────────────────────────────────
        if verbose:
            print(f"   Training on {device}…")
        with timer.stage("train"):
            model, returned_cfg = mod.train_self_supervised(**train_kwargs)

        # ── Inference ─────────────────────────────────────────
        if verbose:
            print(f"   Inference…")
        with timer.stage("inference"):
            denoised = mod.denoise_stack(
                model=model, stack=noisy.astype(np.float32),
                config=returned_cfg, device=device, verbose=verbose,
            )

        # Convert to a sensible dtype for output (match input)
        if np.issubdtype(noisy.dtype, np.integer):
            info_ = np.iinfo(noisy.dtype)
            denoised_save = np.clip(denoised, info_.min, info_.max)
            denoised_save = np.round(denoised_save).astype(noisy.dtype)
        else:
            denoised_save = denoised.astype(np.float32)

        # ── Save outputs ──────────────────────────────────────
        out_dir = io_.run_output_dir(results_dir, group_id, run_id)
        out_path = out_dir / f"{stack_name}.tif"
        with timer.stage("save_output"):
            io_.save_stack(denoised_save, out_path)
        summary["output_path"] = str(out_path)

        if save_checkpoint:
            ckpt_dir = io_.run_checkpoint_dir(results_dir, group_id, run_id)
            ckpt_path = ckpt_dir / f"{stack_name}.pth"
            try:
                mod.save_checkpoint(model, returned_cfg, str(ckpt_path))
                summary["checkpoint_path"] = str(ckpt_path)
            except Exception as e:
                if verbose:
                    print(f"   (checkpoint save failed: {e})")

        # ── Evaluate if clean given ───────────────────────────
        if clean is not None:
            if verbose:
                print(f"   Evaluating…")
            with timer.stage("evaluate"):
                metrics = eval_runner.evaluate_pair(denoised, clean)
            csv_db.write_metrics(results_dir, run_id, metrics,
                                  group_id=group_id)
            if verbose:
                key_metrics = ("stSNR", "stPSNR", "stSI_PSNR",
                               "sSNR", "tSNR")
                print("   Metrics:")
                for k in key_metrics:
                    if k in metrics:
                        print(f"     {k:12s} = {metrics[k]:.4f}")

        # ── Paper figure ──────────────────────────────────────
        if save_figures:
            fig_dir = io_.run_figure_dir(figures_dir, group_id, run_id)
            fig_path = fig_dir / f"{stack_name}_frame{paper_frame:04d}.png"
            metric_str = ""
            if metrics:
                metric_str = (
                    f"stSNR={metrics.get('stSNR', float('nan')):.2f}  "
                    f"stPSNR={metrics.get('stPSNR', float('nan')):.2f}  "
                    f"stSI_PSNR={metrics.get('stSI_PSNR', float('nan')):.2f}"
                )
            try:
                with timer.stage("paper_figure"):
                    plots.comparison_grid(
                        noisy_stack=noisy,
                        clean_stack=clean,
                        denoised_stack=denoised,
                        frame=paper_frame,
                        save_path=fig_path,
                        title=f"{algo}  /  {stack_name}",
                        metric_str=metric_str,
                    )
                summary["figure_path"] = str(fig_path)
            except Exception as e:
                if verbose:
                    print(f"   (paper figure failed: {e})")

    except Exception as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        err_dir = results_dir / group_id / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        with open(err_dir / f"{run_id}.log", "w") as f:
            f.write(traceback.format_exc())
        if verbose:
            print(f"   ERROR: {error_msg}")
            traceback.print_exc()
    finally:
        sampler.stop()

    # ── Per-stage timings ─────────────────────────────────────
    csv_db.write_timing(results_dir, run_id, timer.timings,
                         group_id=group_id)

    # ── GPU/CPU summary (gpu_log lives inside the group folder) ──
    gpu_summary = runtime_log.summarize_gpu_log(
        results_dir / group_id, run_id,
    )

    # ── Config (flattened) ────────────────────────────────────
    # Start with the user-supplied config, then overlay the resolved
    # values from eff_config (which has __resolved_normalization etc.)
    # and from returned_cfg (the algo may have updated knobs). We DO
    # log __-prefixed metadata since that's how the resolved strategies
    # surface to downstream analysis; we only skip single-underscore
    # entries which are reserved for transient runtime state.
    cfg_for_log = {}
    for source in (config, locals().get("eff_config", {}),
                    locals().get("returned_cfg", {}) or {}):
        for k, v in source.items():
            # Skip private callable / object-valued entries that would
            # not serialize cleanly to a CSV row.
            if k.startswith("_") and not k.startswith("__"):
                continue
            if callable(v) or k == "norm_params":
                # norm_params is a dict; record it specially so it stays
                # readable in the CSV.
                if k == "norm_params" and isinstance(v, dict):
                    for nk, nv in v.items():
                        cfg_for_log[f"norm_params.{nk}"] = nv
                continue
            cfg_for_log[k] = v
    cfg_for_log["__resolved_device"] = str(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg_for_log["__group_id"] = group_id
    cfg_for_log["__config_name"] = config_name
    csv_db.write_config(results_dir, run_id, cfg_for_log,
                         group_id=group_id)

    # ── Master row in runs.csv ────────────────────────────────
    run_row = {
        "run_id":     run_id,
        "group_id":   group_id,           # group scope (NEW)
        "config_name": config_name,        # human config label (NEW)
        "started_at": datetime.now(timezone.utc).isoformat(),
        "algo":       algo,
        "stack_name": stack_name,
        "status":     status,
        "error":      error_msg,
        "noisy_path": str(noisy_path),
        "clean_path": str(clean_path) if clean_path else "",
        "output_path": summary.get("output_path", ""),
        "checkpoint_path": summary.get("checkpoint_path", ""),
        "figure_path": summary.get("figure_path", ""),
        # primary metrics in the master row for convenience
        "stSNR":      metrics.get("stSNR", ""),
        "stPSNR":     metrics.get("stPSNR", ""),
        "stSI_PSNR":  metrics.get("stSI_PSNR", ""),
        "sSNR":       metrics.get("sSNR", ""),
        "tSNR":       metrics.get("tSNR", ""),
        # stage timings inline
        "train_sec":      timer.timings.get("train", ""),
        "inference_sec":  timer.timings.get("inference", ""),
        "evaluate_sec":   timer.timings.get("evaluate", ""),
        "total_sec":      sum(timer.timings.values()),
        # gpu summary
        **gpu_summary,
        # static env
        "host":           env.get("host", ""),
        "python":         env.get("python", ""),
        "torch":          env.get("torch", ""),
        "gpu_name":       env.get("gpu_name", ""),
        "gpu_total_mib":  env.get("gpu_total_mem_mib", ""),
    }
    csv_db.write_run(results_dir, run_row, group_id=group_id)

    summary.update(run_row)
    summary["metrics"] = metrics
    summary["timings"] = timer.timings
    summary["status"] = status
    return summary
