"""
restormer3d_pretrain — config for offline pretraining of Restormer3D
across multiple training stacks. Run on RTX 5090 (or any 24GB+ GPU).

This config is for `scripts/pretrain_restormer3d.py`. The architecture is
identical to restormer3d_default.py (the proven config) so the pretrained
checkpoint can be loaded into fine-tuning later without architecture
mismatches.

Differences from restormer3d_default.py:
  - batch_size 2 → 4   (5090 has 32GB VRAM)
  - warmup_iters/n2v_iters are placeholders; scripts/pretrain_restormer3d.py
    overrides them based on --total-iters and --warmup-iters CLI flags

Everything else (dim=32, num_blocks=(2,2,2,3), num_refinement_blocks=2,
heads=(1,2,4,8), ffn_expansion_factor=2.0, bias_free=True, patch_d=32,
patch_hw=64, lr, mask params, normalization, temporal target) matches
restormer3d_default exactly.
"""

CONFIG = {
    "algo":         "restormer3d",
    "name":         "restormer3d_pretrain",
    "description":  "Restormer3D pretraining config (multi-stack on RTX 5090).",
    "paper_frame":  750,

    # ── Backbone — MUST match the fine-tune config ────────────────
    "dim":                   32,
    "num_blocks":            (2, 2, 2, 3),
    "num_refinement_blocks": 2,
    "heads":                 (1, 2, 4, 8),
    "ffn_expansion_factor":  2.0,
    "bias_free":             True,

    # ── Patch sampling ────────────────────────────────────────────
    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   4,            # 5090 fits batch=4 at this patch size

    # ── Schedule — overridden by scripts/pretrain_restormer3d.py CLI
    # These defaults are used only if someone calls train_self_supervised
    # directly with this config (which won't happen via the pretrain script).
    "warmup_iters": 2000,
    "n2v_iters":    48000,
    "lr":           3e-4,

    # ── Noise2Void mask ──────────────────────────────────────────
    "mask_ratio":   0.015,
    "mask_radius":  1,

    # ── Preprocessing — MUST match the fine-tune config ──────────
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
