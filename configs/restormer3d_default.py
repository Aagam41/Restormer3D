"""restormer3d — 3D Restormer (MDTA + GDFN)."""

CONFIG = {
    "algo":         "restormer3d",
    "name":         "restormer3d_t4",
    "description":  "3D Restormer for denoising, fp32.",
    "paper_frame":  750,

    "dim":                    32,
    "num_blocks":             (2, 2, 2, 3),
    "num_refinement_blocks":  2,
    "heads":                  (1, 2, 4, 8),
    "ffn_expansion_factor":   2.0,
    "bias_free":              True,

    "patch_d":       32,
    "patch_hw":      64,
    "batch_size":    2,

    "warmup_iters":  200,
    "n2v_iters":     3000,
    "lr":            3e-4,
    "mask_ratio":    0.015,
    "mask_radius":   1,
}
