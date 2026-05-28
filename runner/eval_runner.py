"""
Eval runner. Thin wrapper around _eval_metrics.compute_all_metrics
(verbatim implementation from the original eval.py).

We deliberately do NOT re-derive metric formulas. All algos are evaluated
by identical code paths to guarantee comparability.
"""

import numpy as np
from typing import Dict

from . import _eval_metrics as E


def evaluate_pair(denoised: np.ndarray, clean: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics on a (denoised, clean) pair.

    Returns a dict mapping metric names to values, ready to pass into
    csv_db.write_metrics(run_id, ...).
    """
    den = denoised.astype(np.float64)
    gt = clean.astype(np.float64)
    return E.compute_all_metrics(den, gt)
