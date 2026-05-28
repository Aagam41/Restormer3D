"""
restormer3d_finetune — config for per-stack fine-tuning at submission time
on T4.

Loads pretrained weights from /opt/ml/model/restormer3d.pth (saved by
scripts/pretrain_restormer3d.py), then runs a short fine-tune on the
input stack before inference.

Architecture MUST match restormer3d_pretrain.py exactly (same backbone,
preprocessing) so the pretrained state_dict loads cleanly.

Expected timing on T4 at this config:
  Stage 0 (warmup):    0 iters (skipped — pretrained model already knows
                                  the data distribution)
  Stage 1 (n2v):       800 iters
  Inference:           ~30-50 s
  Total per stack:     ~6-7 min (rough — Restormer3D's MDTA attention is
                                 the bottleneck; profile on your hardware)
  7 stacks:            ~45-50 min target

If pretrained init is missing, the inference.py wrapper falls back to
training from scratch, which is slow — so always check that
/opt/ml/model/restormer3d.pth is present in your submission.
"""

CONFIG = {
    "algo":         "restormer3d",
    "name":         "restormer3d_finetune",
    "description":  "Restormer3D fine-tune (load pretrained, short adapt, infer).",
    "paper_frame":  750,

    # ── Backbone — MUST match restormer3d_pretrain.py ────────────
    "dim":                   32,
    "num_blocks":            (2, 2, 2, 3),
    "num_refinement_blocks": 2,
    "heads":                 (1, 2, 4, 8),
    "ffn_expansion_factor":  2.0,
    "bias_free":             True,

    # ── Patch sampling ────────────────────────────────────────────
    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   2,            # T4 16GB; keep batch small

    # ── Schedule (short fine-tune from pretrained init) ──────────
    # No warmup — pretrained model already understands the data distribution
    # so warmup against temporal median is redundant.
    "warmup_iters": 0,
    "n2v_iters":    800,
    # Lower LR than scratch — standard 5-10× reduction for fine-tuning
    "lr":           1e-4,

    # ── Noise2Void mask — same as pretrain ───────────────────────
    "mask_ratio":   0.015,
    "mask_radius":  1,

    # ── Preprocessing — MUST match restormer3d_pretrain.py ───────
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
