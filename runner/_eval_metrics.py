"""
Evaluation script for AI4Life-CIDC25 Calcium Imaging Denoising Challenge.

Computes all metrics defined at:
  https://ai4life-cidc25.grand-challenge.org/cidc25-evaluation-metrics/

Metrics:
  - stSNR  (main ranking metric = 0.5 * sSNR + 0.5 * tSNR)
  - stPSNR (data_range = p97 - p3 of ground truth stack)
  - stSI_PSNR (scale-invariant PSNR from TasNet / careamics)

Usage:
    python eval.py --denoised /path/to/denoised.tif --clean /path/to/clean.tif
    python eval.py --denoised /path/to/denoised.tif --clean /path/to/clean.tif -o results.csv

    # Multiple file pairs (validation set):
    python eval.py \\
        --denoised den1.tif den2.tif \\
        --clean    gt1.tif  gt2.tif  \\
        -o results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
    def load_stack(path):
        return tifffile.imread(str(path)).astype(np.float64)
except ImportError:
    import SimpleITK
    def load_stack(path):
        img = SimpleITK.ReadImage(str(path))
        return SimpleITK.GetArrayFromImage(img).astype(np.float64)

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


def _finite_mean(vals) -> float:
    """Mean of finite values only (filters out inf and nan)."""
    arr = np.array(vals, dtype=np.float64)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(arr[mask]))


def _finite_std(vals) -> float:
    """Std of finite values only."""
    arr = np.array(vals, dtype=np.float64)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return float("nan")
    return float(np.std(arr[mask]))


# ══════════════════════════════════════════════════════════════
#  CORE METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════

def snr(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in dB.

    SNR = 10 * log10( ||gt||² / ||gt - pred||² )

    Args:
        gt:   Ground truth signal (any shape, flattened internally).
        pred: Denoised signal (same shape as gt).
    Returns:
        SNR in dB.
    """
    gt_f = gt.ravel().astype(np.float64)
    pred_f = pred.ravel().astype(np.float64)
    signal_power = np.sum(gt_f ** 2)
    noise_power = np.sum((gt_f - pred_f) ** 2)
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return -float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    """
    Peak Signal-to-Noise Ratio using scikit-image.

    Uses data_range = p97 - p3 of the full ground truth stack
    (as specified by the challenge).
    """
    return float(skimage_psnr(gt.astype(np.float64),
                               pred.astype(np.float64),
                               data_range=data_range))


def _zero_mean(x: np.ndarray) -> np.ndarray:
    """Subtract mean (for SI-PSNR)."""
    return x - np.mean(x)


def _si_fix(gt_zm: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Scale-invariant projection (TasNet / careamics).

    Projects pred onto gt direction, then returns the zero-meaned
    and normalized version for PSNR computation.

    s_target = <pred_zm, gt_zm> / ||gt_zm||² * gt_zm
    Returns: s_target (zero-meaned, since gt_zm is already zero-meaned)
    """
    pred_zm = _zero_mean(pred)
    # Optimal scale factor
    alpha = np.dot(pred_zm.ravel(), gt_zm.ravel()) / (np.dot(gt_zm.ravel(), gt_zm.ravel()) + 1e-12)
    return alpha * gt_zm


def scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Scale-Invariant PSNR (from careamics, based on TasNet SI-SNR).

    Normalizes both signals to zero-mean, finds optimal scale,
    then computes PSNR with range_parameter = (max-min)/std of gt.
    """
    gt_f = gt.astype(np.float64)
    pred_f = pred.astype(np.float64)

    gt_std = np.std(gt_f)
    if gt_std < 1e-12:
        # Constant signal — SI-PSNR is undefined
        return np.nan

    # Range parameter for PSNR
    range_param = (np.max(gt_f) - np.min(gt_f)) / gt_std

    # Zero-mean and normalize gt
    gt_norm = _zero_mean(gt_f) / gt_std

    # Compute the target projection and residual in normalized space
    target_proj = _si_fix(gt_norm, pred_f)

    return float(skimage_psnr(
        _zero_mean(gt_norm),
        target_proj,
        data_range=range_param,
    ))


# ══════════════════════════════════════════════════════════════
#  SPATIAL / TEMPORAL / SPATIO-TEMPORAL WRAPPERS
# ══════════════════════════════════════════════════════════════

def compute_all_metrics(denoised: np.ndarray, clean: np.ndarray) -> dict:
    """
    Compute all challenge metrics for a single file pair.

    Args:
        denoised: Denoised stack [F, H, W] (float64).
        clean:    Ground truth stack [F, H, W] (float64).

    Returns:
        Dict with all metric names as keys.
    """
    assert denoised.shape == clean.shape, \
        f"Shape mismatch: denoised {denoised.shape} vs clean {clean.shape}"

    F, H, W = clean.shape

    # Data range for PSNR: p97 - p3 of full ground truth stack
    data_range = float(np.percentile(clean, 97) - np.percentile(clean, 3))

    # ── Spatial metrics (computed per frame, averaged over frames) ──
    s_snr_vals = []
    s_psnr_vals = []
    s_si_psnr_vals = []

    for t in range(F):
        gt_frame = clean[t]
        pred_frame = denoised[t]

        s_snr_vals.append(snr(gt_frame, pred_frame))
        s_psnr_vals.append(psnr(gt_frame, pred_frame, data_range))
        s_si_psnr_vals.append(scale_invariant_psnr(gt_frame, pred_frame))

    sSNR = _finite_mean(s_snr_vals)
    sPSNR = _finite_mean(s_psnr_vals)
    sSI_PSNR = _finite_mean(s_si_psnr_vals)
    sSNR_std = _finite_std(s_snr_vals)
    sPSNR_std = _finite_std(s_psnr_vals)
    sSI_PSNR_std = _finite_std(s_si_psnr_vals)

    # ── Temporal metrics (computed per pixel (i,j), averaged over spatial grid) ──
    # For each spatial location, the temporal signal is clean[:, i, j]
    # To make this tractable for 490×490, we vectorize.
    t_snr_vals = []
    t_psnr_vals = []
    t_si_psnr_vals = []

    for i in range(H):
        for j in range(W):
            gt_signal = clean[:, i, j]       # temporal signal at pixel (i,j)
            pred_signal = denoised[:, i, j]

            t_snr_vals.append(snr(gt_signal, pred_signal))
            t_psnr_vals.append(psnr(gt_signal, pred_signal, data_range))
            t_si_psnr_vals.append(scale_invariant_psnr(gt_signal, pred_signal))

    tSNR = _finite_mean(t_snr_vals)
    tPSNR = _finite_mean(t_psnr_vals)
    tSI_PSNR = _finite_mean(t_si_psnr_vals)
    tSNR_std = _finite_std(t_snr_vals)
    tPSNR_std = _finite_std(t_psnr_vals)
    tSI_PSNR_std = _finite_std(t_si_psnr_vals)

    # ── Spatio-temporal (convex combination: 0.5 * spatial + 0.5 * temporal) ──
    stSNR = 0.5 * sSNR + 0.5 * tSNR
    stPSNR = 0.5 * sPSNR + 0.5 * tPSNR
    stSI_PSNR = 0.5 * sSI_PSNR + 0.5 * tSI_PSNR

    return {
        # Main ranking metric
        "stSNR": stSNR,
        # Spatial
        "sSNR": sSNR,
        "sSNR_std": sSNR_std,
        # Temporal
        "tSNR": tSNR,
        "tSNR_std": tSNR_std,
        # PSNR variants
        "stPSNR": stPSNR,
        "sPSNR": sPSNR,
        "sPSNR_std": sPSNR_std,
        "tPSNR": tPSNR,
        "tPSNR_std": tPSNR_std,
        # SI-PSNR variants
        "stSI_PSNR": stSI_PSNR,
        "sSI_PSNR": sSI_PSNR,
        "sSI_PSNR_std": sSI_PSNR_std,
        "tSI_PSNR": tSI_PSNR,
        "tSI_PSNR_std": tSI_PSNR_std,
        # Data range used
        "data_range": data_range,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CIDC25 Evaluation: compute stSNR, stPSNR, stSI_PSNR"
    )
    parser.add_argument(
        "--denoised", nargs="+", required=True,
        help="Path(s) to denoised TIFF file(s)"
    )
    parser.add_argument(
        "--clean", nargs="+", required=True,
        help="Path(s) to clean ground truth TIFF file(s)"
    )
    parser.add_argument(
        "-o", "--output", default="eval_results.csv",
        help="Output CSV path (default: eval_results.csv)"
    )
    args = parser.parse_args()

    if len(args.denoised) != len(args.clean):
        print(f"ERROR: Got {len(args.denoised)} denoised files but "
              f"{len(args.clean)} clean files. Must be equal.")
        sys.exit(1)

    all_results = []

    for den_path, gt_path in zip(args.denoised, args.clean):
        print(f"\n{'='*60}")
        print(f"Denoised: {den_path}")
        print(f"Clean:    {gt_path}")
        print(f"{'='*60}")

        denoised = load_stack(den_path)
        clean = load_stack(gt_path)

        print(f"  Denoised: shape={denoised.shape}, range=[{denoised.min():.2f}, {denoised.max():.2f}]")
        print(f"  Clean:    shape={clean.shape}, range=[{clean.min():.2f}, {clean.max():.2f}]")

        print("  Computing metrics (this may take a few minutes for large stacks)...")
        metrics = compute_all_metrics(denoised, clean)

        metrics["file_denoised"] = Path(den_path).name
        metrics["file_clean"] = Path(gt_path).name
        all_results.append(metrics)

        # Print key metrics
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  stSNR     = {metrics['stSNR']:>10.4f} dB  (RANKING) │")
        print(f"  │  sSNR      = {metrics['sSNR']:>10.4f} dB             │")
        print(f"  │  tSNR      = {metrics['tSNR']:>10.4f} dB             │")
        print(f"  ├─────────────────────────────────────────┤")
        print(f"  │  stPSNR    = {metrics['stPSNR']:>10.4f} dB             │")
        print(f"  │  sPSNR     = {metrics['sPSNR']:>10.4f} dB             │")
        print(f"  │  tPSNR     = {metrics['tPSNR']:>10.4f} dB             │")
        print(f"  ├─────────────────────────────────────────┤")
        print(f"  │  stSI_PSNR = {metrics['stSI_PSNR']:>10.4f} dB             │")
        print(f"  │  sSI_PSNR  = {metrics['sSI_PSNR']:>10.4f} dB             │")
        print(f"  │  tSI_PSNR  = {metrics['tSI_PSNR']:>10.4f} dB             │")
        print(f"  └─────────────────────────────────────────┘")
        print(f"  data_range (p97-p3) = {metrics['data_range']:.4f}")

    # ── Compute averages across files (final challenge score) ──
    if len(all_results) > 1:
        avg = {}
        metric_keys = [k for k in all_results[0] if k not in ("file_denoised", "file_clean")]
        for k in metric_keys:
            avg[k] = float(np.mean([r[k] for r in all_results]))
        avg["file_denoised"] = "AVERAGE"
        avg["file_clean"] = "AVERAGE"
        all_results.append(avg)

        print(f"\n{'='*60}")
        print(f"AVERAGE across {len(args.denoised)} files:")
        print(f"  stSNR     = {avg['stSNR']:.4f} dB  (FINAL RANKING SCORE)")
        print(f"  stPSNR    = {avg['stPSNR']:.4f} dB")
        print(f"  stSI_PSNR = {avg['stSI_PSNR']:.4f} dB")
        print(f"{'='*60}")

    # ── Write CSV ──
    fieldnames = ["file_denoised", "file_clean",
                  "stSNR", "sSNR", "sSNR_std", "tSNR", "tSNR_std",
                  "stPSNR", "sPSNR", "sPSNR_std", "tPSNR", "tPSNR_std",
                  "stSI_PSNR", "sSI_PSNR", "sSI_PSNR_std", "tSI_PSNR", "tSI_PSNR_std",
                  "data_range"]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
