"""
Paper-style figure generation.

Two things are produced:

  1. Per-stack 1x4 comparison grid (PNG):
        | noisy | ground truth | denoised | residual |
     for a chosen frame (default 750). Used in the paper to illustrate
     qualitative results for each (algo, stack) run.

  2. Leaderboard summary plot (PNG):
        bar chart of stSNR/stPSNR across algos, with one bar per stack.

Outputs live in <figures_root>/<run_id>/.
"""

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")     # no display needed
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
# Per-frame 1x4 comparison
# ══════════════════════════════════════════════════════════════

def _percentile_range(arr, lo: float = 1.0, hi: float = 99.0):
    """Robust display range, ignoring extreme outliers."""
    vmin = float(np.percentile(arr, lo))
    vmax = float(np.percentile(arr, hi))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def comparison_grid(
    noisy_stack: np.ndarray,
    clean_stack: Optional[np.ndarray],
    denoised_stack: np.ndarray,
    frame: int,
    save_path,
    title: str = "",
    metric_str: str = "",
) -> Path:
    """
    Render a 1x4 (or 1x3 if no clean) comparison grid for one frame and
    save as PNG.

    The display range for noisy / clean / denoised panels is shared (so
    differences are visible at a glance). The residual panel uses a
    diverging colormap centered at zero with its own symmetric range.

    Returns the saved path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    F_ = noisy_stack.shape[0]
    f = max(0, min(F_ - 1, frame))

    noisy_f    = noisy_stack[f].astype(np.float32)
    denoised_f = denoised_stack[f].astype(np.float32)
    clean_f    = (clean_stack[f].astype(np.float32)
                   if clean_stack is not None else None)

    # Common display range: based on the clean stack if available,
    # else the noisy.
    ref = clean_f if clean_f is not None else noisy_f
    vmin, vmax = _percentile_range(ref, 1.0, 99.0)

    # Residual = noisy - denoised. The expected content is "what was
    # removed" — should look like pure noise if denoising worked.
    residual_f = noisy_f - denoised_f
    rmax = max(abs(np.percentile(residual_f, 1)),
               abs(np.percentile(residual_f, 99)))
    if rmax < 1e-6:
        rmax = 1.0

    n_cols = 4 if clean_f is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.2))

    panels = [("Noisy input", noisy_f, "gray", (vmin, vmax))]
    if clean_f is not None:
        panels.append(("Ground truth", clean_f, "gray", (vmin, vmax)))
    panels.append(("Denoised", denoised_f, "gray", (vmin, vmax)))
    panels.append(("Residual (noisy − denoised)", residual_f, "RdBu_r",
                   (-rmax, rmax)))

    for ax, (panel_title, img, cmap, (lo, hi)) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    suptitle = title
    if metric_str:
        suptitle = f"{title}\n{metric_str}" if title else metric_str
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    fig.text(
        0.5, 0.02, f"frame {f}", ha="center", fontsize=9,
        color="dimgray",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.94 if suptitle else 1.0))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════
# Side-by-side TIME-COURSE comparison for one pixel
# ══════════════════════════════════════════════════════════════

def time_course_plot(
    noisy_stack: np.ndarray,
    clean_stack: Optional[np.ndarray],
    denoised_stack: np.ndarray,
    pixel_yx: tuple,
    save_path,
    title: str = "",
) -> Path:
    """
    Plot the time course of one pixel across noisy / clean / denoised.

    Useful for showing transient preservation in the paper. `pixel_yx`
    is (y, x).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    y, x = pixel_yx
    n = noisy_stack[:, y, x].astype(np.float32)
    d = denoised_stack[:, y, x].astype(np.float32)
    fig, ax = plt.subplots(figsize=(9, 3.3))
    if clean_stack is not None:
        c = clean_stack[:, y, x].astype(np.float32)
        ax.plot(c, color="black", linewidth=1.0, label="Ground truth")
    ax.plot(n, color="lightcoral", alpha=0.6, linewidth=0.6, label="Noisy")
    ax.plot(d, color="C0", linewidth=1.0, label="Denoised")
    ax.set_xlabel("frame"); ax.set_ylabel("intensity")
    ax.set_title(title or f"Time course at pixel (y={y}, x={x})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════
# Leaderboard bar chart across algos
# ══════════════════════════════════════════════════════════════

def leaderboard_bar(
    rows: List[Dict],
    metric: str = "stSNR",
    save_path: Optional[Path] = None,
    title: str = "",
):
    """
    Grouped bar chart: one group per stack, one bar per algorithm, height
    is the chosen metric.

    `rows` is a list of dicts each containing: algo, stack_name, <metric>.
    Typically built from a join of runs.csv + metrics.csv.
    """
    from collections import defaultdict
    by_algo = defaultdict(dict)        # algo -> {stack -> value}
    stacks = set()
    for r in rows:
        if r.get("metric") != metric:
            continue
        algo = r["algo"]; stack = r["stack_name"]
        try:
            val = float(r["value"])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(val):
            continue
        by_algo[algo][stack] = val
        stacks.add(stack)
    stacks = sorted(stacks)
    algos = sorted(by_algo.keys())
    if not algos or not stacks:
        return None

    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(stacks) + 2), 4.5))
    width = 0.8 / max(len(algos), 1)
    xpos = np.arange(len(stacks))
    for i, algo in enumerate(algos):
        ys = [by_algo[algo].get(s, np.nan) for s in stacks]
        ax.bar(xpos + i * width - 0.4 + width / 2, ys, width,
               label=algo)
    ax.set_xticks(xpos)
    ax.set_xticklabels(stacks)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by algo / stack")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path
