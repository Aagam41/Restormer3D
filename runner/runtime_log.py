"""
Runtime telemetry: GPU/CPU/memory sampling + stage timer.

GPU sampling uses NVML via `pynvml` if available, otherwise falls back to
`torch.cuda.memory_allocated()` (only memory, no utilization). Both paths
write rows to gpu_log.csv via the csv_db.

Stage timer: a tiny context manager that records start/end of named
stages and accumulates them in a dict the caller can write to timing.csv.
"""

import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import csv_db


try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# Stage timer
# ══════════════════════════════════════════════════════════════

class StageTimer:
    """
    Tracks per-stage wall-clock durations.

    Usage:
        timer = StageTimer()
        with timer.stage("training"):
            ...
        with timer.stage("inference"):
            ...
        print(timer.timings)        # {"training": 12.3, "inference": 4.5}
    """

    def __init__(self):
        self.timings = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            # If a stage is entered multiple times, accumulate.
            self.timings[name] = self.timings.get(name, 0.0) + dt


# ══════════════════════════════════════════════════════════════
# GPU sampler (background thread)
# ══════════════════════════════════════════════════════════════

def _get_static_env_info():
    info = {
        "host": socket.gethostname(),
        "pid":  os.getpid(),
        "python": "",
        "torch": "",
        "cuda_available": False,
        "gpu_name": "",
        "gpu_total_mem_mib": 0,
    }
    try:
        import sys
        info["python"] = sys.version.split()[0]
    except Exception:
        pass
    if _TORCH_AVAILABLE:
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_mem_mib"] = int(props.total_memory / 1024**2)
    return info


def _sample_gpu(handle=None):
    """Return one dict of GPU sample data."""
    sample = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_util_pct":   "",
        "gpu_mem_used_mib": "",
        "gpu_mem_total_mib": "",
        "gpu_temp_c":     "",
        "cpu_util_pct":   "",
        "ram_used_mib":   "",
    }
    if _NVML_AVAILABLE and handle is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = -1
            sample["gpu_util_pct"]      = util.gpu
            sample["gpu_mem_used_mib"]  = int(mem.used / 1024**2)
            sample["gpu_mem_total_mib"] = int(mem.total / 1024**2)
            sample["gpu_temp_c"]        = temp
        except Exception:
            pass
    elif _TORCH_AVAILABLE and torch.cuda.is_available():
        # Memory only (no utilization)
        try:
            sample["gpu_mem_used_mib"]  = int(torch.cuda.memory_allocated() / 1024**2)
            sample["gpu_mem_total_mib"] = int(
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            )
        except Exception:
            pass
    if _PSUTIL_AVAILABLE:
        try:
            sample["cpu_util_pct"] = psutil.cpu_percent(interval=None)
            sample["ram_used_mib"] = int(
                (psutil.virtual_memory().total - psutil.virtual_memory().available)
                / 1024**2
            )
        except Exception:
            pass
    return sample


class GPUSampler:
    """
    Background thread that periodically samples GPU/CPU and appends rows
    to <results_dir>/gpu_log.csv.

    Stops via the `stop()` method, which joins the thread.
    """

    def __init__(self, results_dir: Path, run_id: str,
                 interval_sec: float = 2.0):
        self.results_dir = Path(results_dir)
        self.run_id = run_id
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._handle = None

    def _loop(self):
        log_path = self.results_dir / "gpu_log.csv"
        while not self._stop.wait(self.interval):
            sample = _sample_gpu(self._handle)
            sample["run_id"] = self.run_id
            csv_db.append_row(log_path, sample)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval + 2.0)
        self._thread = None
        if _NVML_AVAILABLE and self._handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════
# Aggregate GPU stats over a run
# ══════════════════════════════════════════════════════════════

def summarize_gpu_log(results_dir: Path, run_id: str) -> dict:
    """
    Read gpu_log.csv, filter to this run, and compute peak/mean
    utilization + memory. Returns a dict ready to drop into runs.csv.
    """
    path = Path(results_dir) / "gpu_log.csv"
    if not path.exists():
        return {}
    import csv as _csv
    util_vals = []; mem_vals = []; temp_vals = []; cpu_vals = []
    with open(path, "r", newline="") as f:
        for row in _csv.DictReader(f):
            if row.get("run_id") != run_id:
                continue
            try:
                if row.get("gpu_util_pct") not in ("", None):
                    util_vals.append(float(row["gpu_util_pct"]))
            except ValueError:
                pass
            try:
                if row.get("gpu_mem_used_mib") not in ("", None):
                    mem_vals.append(float(row["gpu_mem_used_mib"]))
            except ValueError:
                pass
            try:
                if row.get("gpu_temp_c") not in ("", None):
                    temp_vals.append(float(row["gpu_temp_c"]))
            except ValueError:
                pass
            try:
                if row.get("cpu_util_pct") not in ("", None):
                    cpu_vals.append(float(row["cpu_util_pct"]))
            except ValueError:
                pass
    def _stat(v):
        if not v:
            return ("", "", "")
        return (min(v), sum(v)/len(v), max(v))
    u_min, u_mean, u_max = _stat(util_vals)
    m_min, m_mean, m_max = _stat(mem_vals)
    t_min, t_mean, t_max = _stat(temp_vals)
    c_min, c_mean, c_max = _stat(cpu_vals)
    return {
        "gpu_util_mean": u_mean,    "gpu_util_max": u_max,
        "gpu_mem_mean_mib": m_mean, "gpu_mem_max_mib": m_max,
        "gpu_temp_max_c": t_max,
        "cpu_util_mean": c_mean,    "cpu_util_max": c_max,
        "gpu_samples": len(util_vals) or len(mem_vals),
    }


# ══════════════════════════════════════════════════════════════
# Public: collect static env info into a dict
# ══════════════════════════════════════════════════════════════

def env_info() -> dict:
    return _get_static_env_info()
