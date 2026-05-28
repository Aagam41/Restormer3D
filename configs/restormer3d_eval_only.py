"""
restormer3d_eval_only — config for evaluating a pretrained Restormer3D
checkpoint WITHOUT any per-stack training.

Use with `--pretrained <ckpt.pth>` to load the checkpoint and run inference
directly. Both `warmup_iters` and `n2v_iters` are 0, so the training loop
is a no-op and the loaded weights are used as-is for inference.

Architecture MUST match the pretrained checkpoint's config (and the
fine-tune config) — see restormer3d_pretrain.py.

Expected timing on T4:
  Train:       0 s (skipped entirely)
  Inference:  ~30-60 s/stack (Restormer3D inference cost depends on
                              MDTA attention + sliding-window patches)
  Total:      ~30-60 s/stack

This is mode A in the pretraining workflow: zero adaptation to the test
stack's specific noise, maximum speed. Use it to benchmark the pretrained
model in isolation.

Compare against restormer3d_finetune.py to see how much the 800-iter
fine-tune helps on YOUR data.
"""

CONFIG = {
    "algo":         "restormer3d",
    "name":         "restormer3d_eval_only",
    "description":  "Restormer3D eval only (load pretrained, skip training, infer).",
    "paper_frame":  750,

    # ── Backbone — MUST match restormer3d_pretrain.py ────────────
    "dim":                   32,
    "num_blocks":            (2, 2, 2, 3),
    "num_refinement_blocks": 2,
    "heads":                 (1, 2, 4, 8),
    "ffn_expansion_factor":  2.0,
    "bias_free":             True,

    # ── Patch sampling (used at inference only) ──────────────────
    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   2,

    # ── Schedule: ZERO TRAINING ──────────────────────────────────
    "warmup_iters": 0,
    "n2v_iters":    0,
    "lr":           1e-4,

    # ── Noise2Void mask (unused when n2v_iters=0) ───────────────
    "mask_ratio":   0.015,
    "mask_radius":  1,

    # ── Preprocessing — MUST match restormer3d_pretrain.py ───────
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
