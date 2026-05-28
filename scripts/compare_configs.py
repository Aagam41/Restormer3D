#!/usr/bin/env python3
"""
Pivot a benchmark group's metrics into a clean per-(stack, config) table.

USAGE
─────

    python scripts/compare_configs.py \\
        --group-dir benchmark_results/g_modeC \\
        [--metric stSNR] \\
        [--baseline restormer3d_eval_only]

OUTPUT
──────

Console table:

    stack          eval_only     finetune     delta (FT vs EO)
    test_0         13.4521       14.8732      +1.4211
    test_1         12.9183       13.7456      +0.8273
    -----------    ---------     ---------    -----------
    MEAN           13.1852       14.3094      +1.1242

Also writes the same data as <group-dir>/compare_<metric>.csv.

If --baseline is given, the delta column is computed as (other - baseline)
for every non-baseline config. Otherwise the first config alphabetically
is treated as the baseline.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--group-dir", required=True, type=Path,
                    help="Path to <benchmark_results>/<group_id>/")
    p.add_argument("--metric", default="stSNR",
                    help="Which metric to pivot. Default: stSNR (the "
                         "CIDC25 ranking metric).")
    p.add_argument("--baseline", default=None,
                    help="Config name to treat as the baseline column. "
                         "All other columns get a delta. Default: first "
                         "config alphabetically.")
    return p.parse_args()


def _load_runs(group_dir: Path):
    """Load runs.csv → dict[run_id] -> {algo, stack_name, config_name, status}."""
    runs_path = group_dir / "runs.csv"
    if not runs_path.exists():
        sys.exit(f"runs.csv not found in {group_dir}")
    out = {}
    with open(runs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("run_id")
            if rid:
                out[rid] = row
    return out


def _load_metric_values(group_dir: Path, metric: str):
    """Load metrics.csv → dict[run_id] -> float (the metric's value)."""
    metrics_path = group_dir / "metrics.csv"
    if not metrics_path.exists():
        sys.exit(f"metrics.csv not found in {group_dir}")
    out = {}
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("metric") == metric:
                try:
                    out[row["run_id"]] = float(row["value"])
                except (KeyError, ValueError):
                    pass
    return out


def main():
    args = parse_args()
    group_dir = args.group_dir
    if not group_dir.is_dir():
        sys.exit(f"Not a directory: {group_dir}")

    runs = _load_runs(group_dir)
    metric_vals = _load_metric_values(group_dir, args.metric)

    if not metric_vals:
        sys.exit(f"No '{args.metric}' values found in {group_dir}/metrics.csv. "
                 f"Available metrics: see metrics.csv 'metric' column.")

    # Build pivot: rows = stacks, cols = config_names. Cells = metric value.
    pivot = {}      # stack -> {config_name -> value}
    config_names = set()
    successful_only = 0
    skipped = 0
    for rid, run in runs.items():
        stack = run.get("stack_name", "?")
        cfg = run.get("config_name", "?")
        status = (run.get("status") or "").lower()
        if status != "success":
            skipped += 1
            continue
        if rid not in metric_vals:
            skipped += 1
            continue
        pivot.setdefault(stack, {})[cfg] = metric_vals[rid]
        config_names.add(cfg)
        successful_only += 1

    if not pivot:
        sys.exit(f"No successful runs with metric '{args.metric}' found "
                 f"in {group_dir}.")

    config_names = sorted(config_names)
    baseline = args.baseline or config_names[0]
    if baseline not in config_names:
        sys.exit(f"--baseline '{baseline}' not in configs found: "
                 f"{config_names}")

    # Print table
    print()
    print(f"=== {args.metric} per (stack, config) — group: {group_dir.name} ===")
    print(f"    baseline column: {baseline}")
    print(f"    successful runs: {successful_only}  (skipped: {skipped})")
    print()

    # Column widths
    stack_w = max(len("stack"), max(len(s) for s in pivot.keys()))
    col_w = max(11, max(len(c) for c in config_names))
    delta_w = 14

    # Header
    header_cells = [
        f"{'stack':<{stack_w}}",
        *[f"{c:>{col_w}}" for c in config_names],
    ]
    other_cfgs = [c for c in config_names if c != baseline]
    for oc in other_cfgs:
        header_cells.append(f"{('Δ '+oc+' vs '+baseline)[:delta_w]:>{delta_w}}")
    print("  ".join(header_cells))
    print("-" * (sum(len(c) for c in header_cells) + 2 * (len(header_cells) - 1)))

    # Data rows
    stacks_sorted = sorted(pivot.keys())
    means_per_cfg = {c: [] for c in config_names}
    for stack in stacks_sorted:
        cells = [f"{stack:<{stack_w}}"]
        row_vals = pivot[stack]
        for c in config_names:
            v = row_vals.get(c)
            cells.append(
                f"{v:>{col_w}.4f}" if v is not None else f"{'—':>{col_w}}"
            )
            if v is not None:
                means_per_cfg[c].append(v)
        # delta cells
        base_v = row_vals.get(baseline)
        for oc in other_cfgs:
            other_v = row_vals.get(oc)
            if base_v is not None and other_v is not None:
                delta = other_v - base_v
                sign = "+" if delta >= 0 else ""
                cells.append(f"{sign}{delta:>{delta_w-1}.4f}")
            else:
                cells.append(f"{'—':>{delta_w}}")
        print("  ".join(cells))

    # MEAN row
    print("-" * (sum(len(c) for c in header_cells) + 2 * (len(header_cells) - 1)))
    mean_cells = [f"{'MEAN':<{stack_w}}"]
    base_mean = (statistics.mean(means_per_cfg[baseline])
                 if means_per_cfg[baseline] else None)
    for c in config_names:
        vals = means_per_cfg[c]
        if vals:
            mean_cells.append(f"{statistics.mean(vals):>{col_w}.4f}")
        else:
            mean_cells.append(f"{'—':>{col_w}}")
    for oc in other_cfgs:
        oc_mean = (statistics.mean(means_per_cfg[oc])
                   if means_per_cfg[oc] else None)
        if base_mean is not None and oc_mean is not None:
            delta = oc_mean - base_mean
            sign = "+" if delta >= 0 else ""
            mean_cells.append(f"{sign}{delta:>{delta_w-1}.4f}")
        else:
            mean_cells.append(f"{'—':>{delta_w}}")
    print("  ".join(mean_cells))

    # Write CSV
    out_csv = group_dir / f"compare_{args.metric}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["stack", *config_names,
                *[f"delta_{oc}_vs_{baseline}" for oc in other_cfgs]]
        w.writerow(cols)
        for stack in stacks_sorted:
            row_vals = pivot[stack]
            row = [stack, *[row_vals.get(c, "") for c in config_names]]
            base_v = row_vals.get(baseline)
            for oc in other_cfgs:
                ov = row_vals.get(oc)
                if base_v is not None and ov is not None:
                    row.append(ov - base_v)
                else:
                    row.append("")
            w.writerow(row)
        # MEAN row
        row = ["MEAN"]
        for c in config_names:
            vals = means_per_cfg[c]
            row.append(statistics.mean(vals) if vals else "")
        for oc in other_cfgs:
            om = (statistics.mean(means_per_cfg[oc])
                  if means_per_cfg[oc] else None)
            if base_mean is not None and om is not None:
                row.append(om - base_mean)
            else:
                row.append("")
        w.writerow(row)
    print(f"\n→ CSV: {out_csv}")


if __name__ == "__main__":
    main()
