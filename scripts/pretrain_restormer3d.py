#!/usr/bin/env python3
"""
Restormer3D pretraining across multiple training stacks with validation-based
model selection.

USAGE
─────

    python scripts/pretrain_restormer3d.py \\
        --train-dir   /path/to/train/noisy \\
        --val-noisy-dir /path/to/val/noisy \\
        --val-clean-dir /path/to/val/clean \\
        --output-dir  ./pretrained_restormer3d \\
        --config      configs/restormer3d_pretrain.py \\
        --total-iters 50000 \\
        --checkpoint-interval 500

WHAT IT DOES
────────────

  1. Loads every *.tif from --train-dir as a noisy training stack (NO ground truth needed).
  2. Loads paired (noisy, clean) stacks from --val-noisy-dir / --val-clean-dir.
  3. Trains Restormer3D using its existing 2-stage N2V flow, but with multi-stack
     sampling: each iteration picks a random training stack and a random patch.
  4. Every --checkpoint-interval iters:
       - Runs denoise_stack() on each val stack (forward-only, no gradients).
       - Computes stSNR vs clean using the CIDC25 evaluation metric.
       - If mean val stSNR is best so far, saves --output-dir/restormer3d_best.pth.
       - Always overwrites --output-dir/restormer3d_last.pth.
  5. Appends one row to --output-dir/pretraining_log.csv per checkpoint
     (so you can `tail -f` to monitor).
  6. Saves --output-dir/pretraining_config.json with full run provenance.

VAL ISOLATION GUARANTEE
───────────────────────
Val stacks are read in `forward()` mode only. No val patch ever enters the
training batch. No gradient flows from val. Val is used solely to decide
WHICH checkpoint to save as "best".

This means val IS used to choose iteration count (which checkpoint is best).
That's the cleanest possible use of val — but the val signal won't generalize
to entirely different test distributions. The test set must come from the
same imaging conditions as train + val for this to work.

DEFAULTS
────────
By default, refuses to overwrite an existing restormer3d_best.pth in
--output-dir (so you don't lose a good run by accident). Pass
--force-overwrite to bypass.

CONFIG REQUIREMENTS
───────────────────
The config file at --config (default: configs/restormer3d_pretrain.py) must
define a CONFIG dict with at least: dim, num_blocks, num_refinement_blocks,
heads, ffn_expansion_factor, bias_free, patch_d, patch_hw, batch_size, lr,
mask_ratio, mask_radius, normalization, temporal_target. warmup_iters and
n2v_iters in the config are OVERRIDDEN by --total-iters split: 1/25 of total
iters go to warmup (or use --warmup-iters to override).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algos.restormer3d import (
    Restormer3D,
    denoise_stack,
    save_checkpoint,
    n2v_mask_3d,
    DEFAULT_NORMALIZATION,
    DEFAULT_TEMPORAL_TARGET,
)
from runner import preprocessing as _prep
from runner._eval_metrics import compute_all_metrics


# ══════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════

def _load_config_file(path: Path) -> dict:
    """Exec a config .py file and return its CONFIG dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    ns = {"__file__": str(path), "__name__": "__main__"}
    with open(path) as f:
        code = compile(f.read(), str(path), "exec")
    exec(code, ns)
    if "CONFIG" not in ns:
        raise RuntimeError(f"{path.name} must define top-level CONFIG dict.")
    return dict(ns["CONFIG"])


def _list_tifs(directory: Path):
    """Return sorted list of *.tif / *.tiff under `directory`."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    paths = sorted(
        list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    )
    if not paths:
        raise RuntimeError(f"No *.tif files under {directory}")
    return paths


def _load_stack(path: Path) -> np.ndarray:
    """Load a TIFF as int/float32 numpy array, shape [F, H, W]."""
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(
            f"{path.name} has shape {arr.shape}; expected 3D [F, H, W]"
        )
    return arr


def _stem_match(noisy_paths, clean_paths):
    """Pair noisy and clean files by filename stem.

    Returns list of (stem, noisy_path, clean_path). Warns on unmatched.
    """
    clean_by_stem = {p.stem: p for p in clean_paths}
    pairs = []
    for np_path in noisy_paths:
        stem = np_path.stem
        if stem not in clean_by_stem:
            warnings.warn(
                f"Val noisy {np_path.name} has no matching clean file "
                f"(stem '{stem}' not found in clean dir). Skipping."
            )
            continue
        pairs.append((stem, np_path, clean_by_stem[stem]))
    return pairs


# ══════════════════════════════════════════════════════════════
# 3D augmentation (same primitive used in restormer3d.py)
# ══════════════════════════════════════════════════════════════

def _augment_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


# ══════════════════════════════════════════════════════════════
# Pretraining loop
# ══════════════════════════════════════════════════════════════

def run_pretraining(args):
    print("=" * 70)
    print("RESTORMER3D PRETRAINING")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "restormer3d_best.pth"
    last_ckpt_path = output_dir / "restormer3d_last.pth"
    log_csv_path = output_dir / "pretraining_log.csv"
    cfg_json_path = output_dir / "pretraining_config.json"

    # Overwrite guard
    if best_ckpt_path.exists() and not args.force_overwrite:
        print(f"\nERROR: {best_ckpt_path} already exists.")
        print(f"Use --force-overwrite to bypass, or pick a different "
              f"--output-dir.")
        sys.exit(1)

    # ── Load config ────────────────────────────────────────
    cfg = _load_config_file(Path(args.config))
    # Strip framework-only keys
    for k in ("algo", "name", "description", "paper_frame"):
        cfg.pop(k, None)

    # Override schedule from CLI
    if args.warmup_iters is not None:
        cfg["warmup_iters"] = args.warmup_iters
        n2v_iters = args.total_iters - args.warmup_iters
    else:
        # Default 4% warmup, 96% N2V
        cfg["warmup_iters"] = max(args.total_iters // 25, 200)
        n2v_iters = args.total_iters - cfg["warmup_iters"]
    cfg["n2v_iters"] = max(n2v_iters, 10)

    print(f"\n[config] from {args.config}")
    for k in ("dim", "num_blocks", "num_refinement_blocks", "heads",
              "ffn_expansion_factor", "bias_free",
              "patch_d", "patch_hw", "batch_size", "lr",
              "mask_ratio", "mask_radius", "warmup_iters", "n2v_iters",
              "normalization", "temporal_target"):
        print(f"  {k:22s} = {cfg.get(k)}")

    # ── Device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        print(f"  WARNING: no CUDA detected — pretraining on CPU "
              f"will be impractically slow.")

    # ── Load training stacks ────────────────────────────────
    print(f"\n[train data] {args.train_dir}")
    train_paths = _list_tifs(Path(args.train_dir))
    print(f"  Found {len(train_paths)} training stack(s)")

    # Load all stacks into a list. For large stacks, we keep them on CPU
    # as float32 numpy and ship patches to GPU per iter (cheaper than
    # filling 32GB VRAM with raw data).
    train_stacks_raw = []
    train_stacks_norm = []
    train_norm_params = []
    train_temporal_med = []
    train_temporal_med_returns = []
    norm_strategy = _prep.resolve_normalization(
        cfg.get("normalization", DEFAULT_NORMALIZATION)
    )
    tt_strategy = _prep.resolve_temporal_target(
        cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    )

    for i, p in enumerate(train_paths):
        stack = _load_stack(p)
        norm_params = norm_strategy.compute_params(stack)
        stack_norm = norm_strategy.forward(stack, norm_params).astype(np.float32)
        # Temporal median target (2D)
        if tt_strategy.returns != "2d":
            _tt3d = tt_strategy.compute(stack_norm)
            tmed = np.median(_tt3d, axis=0).astype(np.float32)
        else:
            tmed = tt_strategy.compute(stack_norm).astype(np.float32)
        train_stacks_raw.append(stack)
        train_stacks_norm.append(stack_norm)
        train_norm_params.append(norm_params)
        train_temporal_med.append(tmed)
        if i < 3 or i == len(train_paths) - 1:
            print(f"  [{i+1}/{len(train_paths)}] {p.name}: "
                  f"shape={stack.shape}, "
                  f"range=[{stack.min()}, {stack.max()}], "
                  f"norm.shift={norm_params['shift']:.2f}, "
                  f"norm.scale={norm_params['scale']:.2f}")
        elif i == 3:
            print(f"  ... ({len(train_paths)-4} more)")

    # ── Load val pairs ──────────────────────────────────────
    print(f"\n[val data] noisy={args.val_noisy_dir}  clean={args.val_clean_dir}")
    val_noisy_paths = _list_tifs(Path(args.val_noisy_dir))
    val_clean_paths = _list_tifs(Path(args.val_clean_dir))
    val_pairs = _stem_match(val_noisy_paths, val_clean_paths)
    if args.val_stacks_subset:
        val_pairs = val_pairs[: args.val_stacks_subset]
        print(f"  (Limited to first {args.val_stacks_subset} val stack(s) "
              f"via --val-stacks-subset)")
    print(f"  Using {len(val_pairs)} val pair(s)")
    if not val_pairs:
        sys.exit("No paired val data found.")

    val_stacks_raw = []
    val_stacks_clean = []
    val_norm_params = []
    for i, (stem, np_path, cp_path) in enumerate(val_pairs):
        noisy_s = _load_stack(np_path)
        clean_s = _load_stack(cp_path)
        if noisy_s.shape != clean_s.shape:
            sys.exit(f"Val pair {stem}: noisy {noisy_s.shape} != "
                     f"clean {clean_s.shape}")
        vnp = norm_strategy.compute_params(noisy_s)
        val_stacks_raw.append(noisy_s)
        val_stacks_clean.append(clean_s)
        val_norm_params.append(vnp)
        if i < 3:
            print(f"  [{i+1}/{len(val_pairs)}] {stem}: shape={noisy_s.shape}")

    # ── Build model ─────────────────────────────────────────
    print(f"\n[model] Restormer3D")
    model = Restormer3D(
        in_channels=1, out_channels=1,
        dim=cfg["dim"],
        num_blocks=tuple(cfg["num_blocks"]),
        num_refinement_blocks=cfg["num_refinement_blocks"],
        heads=tuple(cfg["heads"]),
        ffn_expansion_factor=cfg["ffn_expansion_factor"],
        bias_free=cfg["bias_free"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # ── Init optimizer + scheduler ─────────────────────────
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]

    # ── Init log + provenance ──────────────────────────────
    log_csv_path.write_text(
        "iter,phase,running_train_loss,val_stSNR_mean,val_stSNR_per_stack,"
        "val_stPSNR_mean,val_stSI_PSNR_mean,is_best,elapsed_sec\n"
    )
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "config": _serializable(cfg),
        "train_stacks": [str(p) for p in train_paths],
        "val_pairs": [(stem, str(np_p), str(cp_p))
                       for stem, np_p, cp_p in val_pairs],
        "n_params": int(n_params),
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else None),
    }
    with open(cfg_json_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    # ── Patch sampler (multi-stack) ────────────────────────
    n_train = len(train_stacks_norm)

    def random_train_patch(rng_np):
        """Pick a random training stack, return (noisy_patch, tmed_crop)
        as cpu float tensors. The caller moves them to device."""
        idx = rng_np.integers(0, n_train)
        stack_norm = train_stacks_norm[idx]
        tmed = train_temporal_med[idx]
        F_, H, W = stack_norm.shape
        t0_ = rng_np.integers(0, max(F_ - pd, 1))
        y0 = rng_np.integers(0, max(H - phw, 1))
        x0 = rng_np.integers(0, max(W - phw, 1))
        d = min(pd, F_); h = min(phw, H); w = min(phw, W)
        # NumPy slice → torch
        sub_np = stack_norm[t0_:t0_+d, y0:y0+h, x0:x0+w]
        tmed_np = tmed[y0:y0+h, x0:x0+w]
        return (torch.from_numpy(sub_np.copy()),
                torch.from_numpy(tmed_np.copy()),
                idx)

    # ── Evaluation function (val-only, no gradients) ──────
    @torch.no_grad()
    def eval_on_val(model):
        """Forward each val stack through the model, return mean stSNR
        (and per-stack list) plus a few other metrics."""
        model.eval()
        per_stack = []
        for i, (stem, np_path, cp_path) in enumerate(val_pairs):
            noisy = val_stacks_raw[i]
            clean = val_stacks_clean[i]
            np_for_infer = val_norm_params[i]
            # denoise_stack expects config to have norm_params + a
            # resolved normalization name (matching what was used at
            # training time).
            infer_cfg = {
                **cfg,
                "norm_params": np_for_infer,
                "__resolved_normalization": norm_strategy.name,
                "__resolved_temporal_target": tt_strategy.name,
            }
            try:
                denoised = denoise_stack(
                    model=model, stack=noisy.astype(np.float32),
                    config=infer_cfg, device=device, verbose=False,
                )
                metrics = compute_all_metrics(denoised, clean)
                per_stack.append({
                    "stem": stem,
                    "stSNR": float(metrics.get("stSNR", float("nan"))),
                    "stPSNR": float(metrics.get("stPSNR", float("nan"))),
                    "stSI_PSNR": float(metrics.get("stSI_PSNR", float("nan"))),
                })
            except Exception as e:
                print(f"  [eval WARNING] {stem}: {e}")
                per_stack.append({
                    "stem": stem, "stSNR": float("nan"),
                    "stPSNR": float("nan"), "stSI_PSNR": float("nan"),
                })
        model.train()
        # Mean over stacks, ignoring NaN
        def _mean(key):
            vals = [d[key] for d in per_stack
                     if d[key] == d[key]]   # exclude NaN
            return float(np.mean(vals)) if vals else float("nan")
        return per_stack, _mean("stSNR"), _mean("stPSNR"), _mean("stSI_PSNR")

    # ── Training loop ──────────────────────────────────────
    rng_np = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    best_val_stSNR = -float("inf")
    t_start = time.time()
    running_loss = 0.0
    running_count = 0

    def _log_row(it, phase, val_per_stack, val_mean, val_psnr,
                  val_sipsnr, is_best):
        rl = running_loss / max(running_count, 1)
        with open(log_csv_path, "a") as f:
            w = csv.writer(f)
            per_stack_compact = ";".join(
                f"{d['stem']}:{d['stSNR']:.4f}" for d in val_per_stack
            )
            w.writerow([
                it, phase, f"{rl:.6f}",
                (f"{val_mean:.4f}" if val_mean == val_mean else ""),
                per_stack_compact,
                (f"{val_psnr:.4f}" if val_psnr == val_psnr else ""),
                (f"{val_sipsnr:.4f}" if val_sipsnr == val_sipsnr else ""),
                "1" if is_best else "0",
                f"{time.time()-t_start:.1f}",
            ])

    # ─── Stage 0: temporal-median warmup ──
    warmup_iters = cfg["warmup_iters"]
    n2v_iters = cfg["n2v_iters"]
    total_iters = warmup_iters + n2v_iters

    print(f"\n[training] total={total_iters}  "
          f"warmup={warmup_iters}  n2v={n2v_iters}  "
          f"checkpoint every {args.checkpoint_interval} iters")
    print(f"[training] sampling uniformly across {n_train} training stack(s)")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                              weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, total_iters, eta_min=cfg["lr"] * 0.05,
    )

    model.train()

    def _do_checkpoint(it, phase):
        nonlocal best_val_stSNR, running_loss, running_count
        per_stack, mn, pn, sin = eval_on_val(model)
        is_best = mn == mn and mn > best_val_stSNR
        if is_best:
            best_val_stSNR = mn
            save_checkpoint(
                model,
                {**cfg, "pretrained_iter": it, "pretrained_val_stSNR": mn,
                 "__resolved_normalization": norm_strategy.name,
                 "__resolved_temporal_target": tt_strategy.name},
                str(best_ckpt_path),
            )
        # Always save last
        save_checkpoint(
            model,
            {**cfg, "pretrained_iter": it,
             "__resolved_normalization": norm_strategy.name,
             "__resolved_temporal_target": tt_strategy.name},
            str(last_ckpt_path),
        )
        _log_row(it, phase, per_stack, mn, pn, sin, is_best)
        per_stack_str = ", ".join(
            f"{d['stem']}={d['stSNR']:.3f}" for d in per_stack
        )
        marker = "  *** NEW BEST ***" if is_best else ""
        elapsed = time.time() - t_start
        print(
            f"\n[iter {it:>6}/{total_iters}] phase={phase}  "
            f"train_loss={running_loss/max(running_count,1):.6f}  "
            f"val_stSNR={mn:.4f}  best={best_val_stSNR:.4f}{marker}\n"
            f"  per-stack: {per_stack_str}\n"
            f"  elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)"
        )
        running_loss = 0.0
        running_count = 0

    # === Warmup stage ===
    for it in range(1, warmup_iters + 1):
        patches_inp, patches_tgt = [], []
        for _ in range(bs):
            sub, tmed_crop, _ = random_train_patch(rng_np)
            aug = int(rng_np.integers(0, 8))
            sub = _augment_3d(sub, aug)
            tmed_crop_b = _augment_3d(
                tmed_crop.unsqueeze(0).expand(sub.shape[0], -1, -1), aug,
            )
            patches_inp.append(sub.unsqueeze(0))
            patches_tgt.append(tmed_crop_b.unsqueeze(0))
        inp = torch.stack(patches_inp, dim=0).to(device)
        tgt = torch.stack(patches_tgt, dim=0).to(device)

        opt.zero_grad()
        pred = model(inp)
        loss = F.l1_loss(pred, tgt)
        if not torch.isfinite(loss):
            warnings.warn(f"non-finite loss at iter {it}, skipping")
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        running_loss += loss.item()
        running_count += 1

        if (it % args.checkpoint_interval == 0) or it == warmup_iters:
            _do_checkpoint(it, phase="warmup")

    # === N2V stage ===
    for it in range(1, n2v_iters + 1):
        all_orig = []
        patches = []
        for _ in range(bs):
            sub, _, _ = random_train_patch(rng_np)
            aug = int(rng_np.integers(0, 8))
            sub = _augment_3d(sub, aug)
            masked, (mz, my, mx), orig = n2v_mask_3d(
                sub.to(device),
                mask_ratio=cfg["mask_ratio"],
                radius=cfg["mask_radius"],
            )
            patches.append(masked.unsqueeze(0))
            all_orig.append((mz, my, mx, orig))
        inp = torch.stack(patches, dim=0).to(device)

        opt.zero_grad()
        pred = model(inp)
        loss = torch.tensor(0.0, device=device)
        for b, (mz, my, mx, orig) in enumerate(all_orig):
            loss = loss + F.l1_loss(pred[b, 0, mz, my, mx], orig)
        loss = loss / bs

        if not torch.isfinite(loss):
            warnings.warn(f"non-finite loss at n2v_iter {it}, skipping")
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        running_loss += loss.item()
        running_count += 1

        global_it = warmup_iters + it
        if (global_it % args.checkpoint_interval == 0) or it == n2v_iters:
            _do_checkpoint(global_it, phase="n2v")

    # ── Final report ───────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PRETRAINING COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  best val stSNR: {best_val_stSNR:.4f}")
    print(f"  best ckpt:      {best_ckpt_path}")
    print(f"  last ckpt:      {last_ckpt_path}")
    print(f"  log:            {log_csv_path}")
    print(f"  provenance:     {cfg_json_path}")
    print("=" * 70)


def _serializable(x):
    """Recursively convert numpy/torch types to native Python so JSON
    serialization works."""
    if isinstance(x, dict):
        return {k: _serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_serializable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train-dir", required=True, type=Path,
                    help="Directory containing noisy training stacks (*.tif).")
    p.add_argument("--val-noisy-dir", required=True, type=Path,
                    help="Directory containing noisy validation stacks.")
    p.add_argument("--val-clean-dir", required=True, type=Path,
                    help="Directory containing clean (GT) validation stacks. "
                         "Paired with --val-noisy-dir by filename stem.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Where to write checkpoints + log + provenance.")
    p.add_argument("--config", type=Path,
                    default=ROOT / "configs" / "restormer3d_pretrain.py",
                    help="Restormer3D config file. "
                         "Default: configs/restormer3d_pretrain.py")
    p.add_argument("--total-iters", type=int, default=50_000,
                    help="Total training iterations "
                         "(warmup + n2v stages combined). Default: 50000")
    p.add_argument("--warmup-iters", type=int, default=None,
                    help="Override warmup iters specifically. "
                         "Default: total_iters // 25.")
    p.add_argument("--checkpoint-interval", type=int, default=500,
                    help="Evaluate val + save best/last every N iters. "
                         "Default: 500.")
    p.add_argument("--val-stacks-subset", type=int, default=None,
                    help="If set, only use first N val stacks "
                         "(speeds up checkpointing for long runs). "
                         "Default: use all paired val stacks.")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for patch sampling.")
    p.add_argument("--force-overwrite", action="store_true",
                    help="Allow overwriting an existing "
                         "<output-dir>/restormer3d_best.pth. By default, "
                         "the script refuses to clobber it.")
    return p.parse_args()


def main():
    args = parse_args()
    run_pretraining(args)


if __name__ == "__main__":
    main()
