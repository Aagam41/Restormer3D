"""
runner.preprocessing — single source of truth for normalization and
temporal-target strategies.

Two orthogonal strategy registries, each looked up by a string name:

    NORMALIZATION_STRATEGIES  — how to compute and apply forward/inverse
                                scaling of a stack. Used by every algo
                                via `compute_norm_params(stack, name)`
                                etc.
    TEMPORAL_TARGET_STRATEGIES — how to derive a "clean" reference from
                                 the noisy stack (warmup target,
                                 input prior, etc.).

----  How algos use this  -------------------------------------------

Every algo module declares its DEFAULT strategy names at the top:

    DEFAULT_NORMALIZATION   = "framework_default"
    DEFAULT_TEMPORAL_TARGET = "temporal_median_2d"

Inside the algo's training loop:

    norm_strategy = prep.resolve_normalization(
        config.get("normalization", DEFAULT_NORMALIZATION)
    )
    norm_params = norm_strategy.compute_params(stack)
    stack_norm  = norm_strategy.forward(stack, norm_params)
    ...
    output = norm_strategy.inverse(output, norm_params)

For the temporal target:

    tt_strategy = prep.resolve_temporal_target(
        config.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    )
    temporal_target = tt_strategy.compute(stack_norm)

----  Backwards compatibility  ---------------------------------------

The algo modules also re-export `compute_norm_params(stack)`, `normalize`,
`denormalize` as MODULE-LEVEL functions for code that imports these
names directly (older `inference.py` files, checkpoint loaders, etc.).
Those module-level functions use the algo's DEFAULT_NORMALIZATION
strategy. If you want a different strategy at training/inference time,
go through this module directly.
"""

from typing import Dict, Optional
import numpy as np


# ══════════════════════════════════════════════════════════════
# NORMALIZATION STRATEGIES
# ══════════════════════════════════════════════════════════════

class NormalizationStrategy:
    """Base class. Subclasses implement compute_params only — the
    forward/inverse implementations here cover linear shift+scale
    schemes which is all we currently use."""

    name: str = "base"

    def compute_params(self, stack: np.ndarray) -> dict:
        raise NotImplementedError

    def forward(self, stack: np.ndarray, params: dict) -> np.ndarray:
        return (stack.astype(np.float32) - params["shift"]) / params["scale"]

    def inverse(self, stack: np.ndarray, params: dict) -> np.ndarray:
        return stack.astype(np.float32) * params["scale"] + params["shift"]


class PercentileNorm(NormalizationStrategy):
    """p_lo / p_hi percentile robust scaling."""

    def __init__(self, p_lo: float, p_hi: float,
                 max_frames: int = 300, name: Optional[str] = None,
                 full_volume: bool = False):
        self.p_lo = p_lo
        self.p_hi = p_hi
        self.max_frames = max_frames
        self.full_volume = full_volume
        self.name = name or (
            f"percentile_{p_lo:g}_{p_hi:g}"
            + ("_fullvol" if full_volume else "")
        )

    def compute_params(self, stack: np.ndarray) -> dict:
        if self.full_volume:
            sampled = stack.astype(np.float64)
        else:
            n = min(self.max_frames, stack.shape[0])
            idx = np.linspace(0, stack.shape[0] - 1, n, dtype=int)
            sampled = stack[idx].astype(np.float64)
        lo = float(np.percentile(sampled, self.p_lo))
        hi = float(np.percentile(sampled, self.p_hi))
        scale = max(hi - lo, 1e-6)
        return {
            "shift": lo, "scale": scale,
            "p_lo": self.p_lo, "p_hi": self.p_hi,
            "strategy": self.name,
        }


class MinMaxNorm(NormalizationStrategy):
    """min / max scaling."""
    name = "minmax"
    def compute_params(self, stack):
        lo = float(stack.min()); hi = float(stack.max())
        return {"shift": lo, "scale": max(hi - lo, 1e-6),
                "strategy": "minmax"}


class MeanStdNorm(NormalizationStrategy):
    """Zero-mean unit-std scaling."""
    name = "meanstd"
    def compute_params(self, stack):
        mu = float(stack.mean())
        sigma = max(float(stack.std()), 1e-6)
        return {"shift": mu, "scale": sigma, "strategy": "meanstd"}


class NoopNorm(NormalizationStrategy):
    """
    Pass-through normalization for algos that handle scaling internally
    (FM2S classic). The params dict still records range info for
    bookkeeping.
    """
    name = "noop"
    def compute_params(self, stack):
        return {
            "shift": 0.0, "scale": 1.0,
            "in_min": float(stack.min()),
            "in_max": float(stack.max()),
            "strategy": "noop",
        }
    def forward(self, stack, params):
        return stack.astype(np.float32)
    def inverse(self, stack, params):
        return stack.astype(np.float32)


# Public registry
NORMALIZATION_STRATEGIES: Dict[str, NormalizationStrategy] = {
    "framework_default":  PercentileNorm(0.5, 99.5, max_frames=300,
                                          name="framework_default"),
    "p0.5_p99.5":         PercentileNorm(0.5, 99.5, max_frames=300,
                                          name="p0.5_p99.5"),
    "p1_p99":             PercentileNorm(1.0, 99.0, max_frames=300,
                                          name="p1_p99"),
    "p3_p97":             PercentileNorm(3.0, 97.0, max_frames=300,
                                          name="p3_p97"),
    # The upstream chhayansh recipe: p3-p97 on the full volume
    "chhayansh":          PercentileNorm(3.0, 97.0, max_frames=10**9,
                                          full_volume=True,
                                          name="chhayansh"),
    "p3_p97_fullvol":     PercentileNorm(3.0, 97.0, max_frames=10**9,
                                          full_volume=True,
                                          name="p3_p97_fullvol"),
    "minmax":             MinMaxNorm(),
    "meanstd":            MeanStdNorm(),
    "noop":               NoopNorm(),
}


def resolve_normalization(name) -> NormalizationStrategy:
    """
    Resolve a strategy by name. Accepts either a string name (registry
    lookup) OR an already-instantiated NormalizationStrategy (passthrough).
    Raises KeyError on unknown name.
    """
    if isinstance(name, NormalizationStrategy):
        return name
    if name not in NORMALIZATION_STRATEGIES:
        raise KeyError(
            f"Unknown normalization '{name}'. Available: "
            f"{sorted(NORMALIZATION_STRATEGIES.keys())}"
        )
    return NORMALIZATION_STRATEGIES[name]


# Optional: registration hook for users to add their own strategies
def register_normalization(strategy: NormalizationStrategy,
                            name: Optional[str] = None):
    NORMALIZATION_STRATEGIES[name or strategy.name] = strategy


# ══════════════════════════════════════════════════════════════
# TEMPORAL-TARGET STRATEGIES
# ══════════════════════════════════════════════════════════════

class TemporalTargetStrategy:
    """
    Build a clean(ish) reference from a noisy stack. Subclasses must
    set `returns` to "2d" (shape [H, W]) or "3d" (shape [F, H, W]).

    All strategies take a normalized stack and return float32.
    """
    name: str = "base"
    returns: str = "2d"

    def compute(self, stack_normalized: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class TemporalMedian2D(TemporalTargetStrategy):
    """Sub-sampled per-pixel temporal median, collapsed to a 2D image."""

    name = "temporal_median_2d"
    returns = "2d"

    def __init__(self, max_frames: int = 500,
                 name: Optional[str] = None):
        self.max_frames = max_frames
        if name:
            self.name = name

    def compute(self, stack_normalized, **kwargs):
        max_frames = kwargs.get("max_frames", self.max_frames)
        F_ = stack_normalized.shape[0]
        n = min(max_frames, F_)
        idx = np.linspace(0, F_ - 1, n, dtype=int)
        return np.median(stack_normalized[idx], axis=0).astype(np.float32)


class TemporalMean2D(TemporalTargetStrategy):
    """Sub-sampled per-pixel temporal mean — cheaper than median."""

    name = "temporal_mean_2d"
    returns = "2d"

    def __init__(self, max_frames: int = 500):
        self.max_frames = max_frames

    def compute(self, stack_normalized, **kwargs):
        max_frames = kwargs.get("max_frames", self.max_frames)
        F_ = stack_normalized.shape[0]
        n = min(max_frames, F_)
        idx = np.linspace(0, F_ - 1, n, dtype=int)
        return stack_normalized[idx].mean(axis=0).astype(np.float32)


class FullStackMedian2D(TemporalTargetStrategy):
    """Exact median over every frame. Slower but no subsampling bias."""

    name = "full_stack_median_2d"
    returns = "2d"

    def compute(self, stack_normalized, **kwargs):
        return np.median(stack_normalized, axis=0).astype(np.float32)


class PerFrameMedian3D(TemporalTargetStrategy):
    """
    Per-frame sliding-window median, returning a full 3D volume
    [F, H, W]. Each output frame is the median over `window` frames
    centered at that frame.

    This is the "per-channel simple median used at each channel"
    interpretation, applied to the time axis: each time index gets its
    OWN local median, instead of one collapsed global reference.
    """

    name = "per_frame_median_3d"
    returns = "3d"

    def __init__(self, window: int = 11):
        self.window = window

    def compute(self, stack_normalized, **kwargs):
        window = kwargs.get("window", self.window)
        F_ = stack_normalized.shape[0]
        k = window // 2
        out = np.empty_like(stack_normalized, dtype=np.float32)
        for t in range(F_):
            t0 = max(0, t - k); t1 = min(F_, t + k + 1)
            out[t] = np.median(stack_normalized[t0:t1], axis=0)
        return out


# Public registry
TEMPORAL_TARGET_STRATEGIES: Dict[str, TemporalTargetStrategy] = {
    "framework_default":    TemporalMedian2D(max_frames=500,
                                              name="framework_default"),
    "temporal_median_2d":   TemporalMedian2D(max_frames=500),
    "temporal_mean_2d":     TemporalMean2D(max_frames=500),
    "full_stack_median_2d": FullStackMedian2D(),
    "per_frame_median_3d":  PerFrameMedian3D(window=11),
}


def resolve_temporal_target(name) -> TemporalTargetStrategy:
    if isinstance(name, TemporalTargetStrategy):
        return name
    if name not in TEMPORAL_TARGET_STRATEGIES:
        raise KeyError(
            f"Unknown temporal-target strategy '{name}'. Available: "
            f"{sorted(TEMPORAL_TARGET_STRATEGIES.keys())}"
        )
    return TEMPORAL_TARGET_STRATEGIES[name]


def register_temporal_target(strategy: TemporalTargetStrategy,
                              name: Optional[str] = None):
    TEMPORAL_TARGET_STRATEGIES[name or strategy.name] = strategy


# ══════════════════════════════════════════════════════════════
# Public summary helpers (used by logs)
# ══════════════════════════════════════════════════════════════

def list_strategies() -> dict:
    return {
        "normalization": {
            k: type(v).__name__
            for k, v in NORMALIZATION_STRATEGIES.items()
        },
        "temporal_target": {
            k: f"{type(v).__name__}(returns={v.returns})"
            for k, v in TEMPORAL_TARGET_STRATEGIES.items()
        },
    }
