"""
Generic zero-shot denoiser for AI4Life-CIDC25 Grand Challenge submission.

This wraps ANY algorithm registered in the framework's algos/ package.
You select which algorithm + config to use via TWO mechanisms (in order
of precedence):

  1. Environment variables:
       SUBMISSION_ALGO   = "dvt_unet3d" | "deepcad" | "srdtrans" | ...
       SUBMISSION_CONFIG = "dvt_unet3d_t4" | "deepcad_rt" | ... (file stem)

  2. The DEFAULT_ALGO / DEFAULT_CONFIG_NAME constants below.

The Docker image bundles the entire framework so you can switch algos
by rebuilding with different env vars baked into the Dockerfile.

For each input stack:
    1. Resolves algorithm module + config from the framework
    2. (Optional) Loads pre-trained weights from /opt/ml/model/
    3. Trains zero-shot on the noisy stack
    4. Runs sliding-window inference
    5. Writes the denoised stack with the same dtype as input
"""

from pathlib import Path
import importlib
import json
import os
import time
from glob import glob

import SimpleITK
import numpy as np


# ──────────────────────────────────────────────────────────────
# WHAT TO RUN — change these (or set env vars at build/run time)
# ──────────────────────────────────────────────────────────────
DEFAULT_ALGO = "restormer3d"            # name from algos/__init__.py REGISTRY
DEFAULT_CONFIG_NAME = "restormer3d_finetune"  # file stem in configs/

# ──────────────────────────────────────────────────────────────
# Grand Challenge paths
# ──────────────────────────────────────────────────────────────
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_DIR = Path("/opt/ml/model")        # for optional pretrained weights


# ──────────────────────────────────────────────────────────────
# Algorithm + config resolution
# ──────────────────────────────────────────────────────────────

def _algo_and_config_names():
    """Resolve which algo+config to run from env vars or defaults."""
    algo = os.environ.get("SUBMISSION_ALGO", DEFAULT_ALGO).strip()
    cfg = os.environ.get("SUBMISSION_CONFIG", DEFAULT_CONFIG_NAME).strip()
    return algo, cfg


def _load_algo_module(algo_name: str):
    """Import algos.<algo_name> as a module."""
    try:
        return importlib.import_module(f"algos.{algo_name}")
    except ImportError as e:
        # Helpful error: list available algos
        try:
            import algos
            available = sorted(algos.REGISTRY.keys())
        except Exception:
            available = ["<algos package not importable>"]
        raise RuntimeError(
            f"Could not import algos.{algo_name}: {e}\n"
            f"Available algos: {available}"
        ) from e


def _load_config(config_name: str) -> dict:
    """Load configs/<config_name>.py and return its CONFIG dict."""
    cfg_path = Path(__file__).resolve().parent / "configs" / f"{config_name}.py"
    if not cfg_path.exists():
        available = sorted(p.stem for p in (cfg_path.parent).glob("*.py")
                            if p.stem != "__init__")
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\n"
            f"Available configs: {available}"
        )
    # exec the file in an isolated namespace
    ns = {"__file__": str(cfg_path), "__name__": "__main__"}
    with open(cfg_path) as f:
        code = compile(f.read(), str(cfg_path), "exec")
    exec(code, ns)
    if "CONFIG" not in ns:
        raise RuntimeError(
            f"{cfg_path.name} must define a top-level CONFIG dict."
        )
    return dict(ns["CONFIG"])


def _resolve_pretrained_weights_path(algo_name: str):
    """Find pretrained weights for this algo in /opt/ml/model/, if any.

    Tries (in order):
        /opt/ml/model/<algo_name>.pth
        /opt/ml/model/<algo_name>_weights.pth
        /opt/ml/model/weights.pth
    Returns None if no file exists.
    """
    if not MODEL_DIR.exists():
        return None
    for candidate in (
        MODEL_DIR / f"{algo_name}.pth",
        MODEL_DIR / f"{algo_name}_weights.pth",
        MODEL_DIR / "weights.pth",
    ):
        if candidate.exists():
            return candidate
    return None


# ──────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────

def run():
    return interf0_handler()


def interf0_handler():
    import torch

    _show_torch_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    algo_name, config_name = _algo_and_config_names()
    print(f"=+= Algorithm: {algo_name}")
    print(f"=+= Config:    {config_name}")

    algo_mod = _load_algo_module(algo_name)
    cfg = _load_config(config_name)

    # Sanity: cfg should target this algo
    cfg_algo = cfg.get("algo")
    if cfg_algo and cfg_algo != algo_name:
        print(f"WARNING: config '{config_name}' targets algo "
              f"'{cfg_algo}' but we're running '{algo_name}'. "
              f"Proceeding with the user-specified algo.")
    # Strip framework-only keys that the algo doesn't expect
    for k in ("algo", "name", "description"):
        cfg.pop(k, None)

    # Optional pretrained weights — three modes:
    #
    #   MODE 1 "scratch" : no pretrained file → train from scratch
    #   MODE 2 "load"    : pretrained exists, SUBMISSION_PRETRAINED_MODE=load
    #                      → load weights, skip training entirely
    #   MODE 3 "finetune": pretrained exists (default behavior)
    #                      → load weights AS INIT, run a short fine-tune,
    #                        then infer. Requires algo to support
    #                        `init_state_dict` kwarg in train_self_supervised.
    #
    # If pretrained exists but the algo doesn't support init_state_dict,
    # we fall back to load-only mode with a printed warning.
    pretrained_path = _resolve_pretrained_weights_path(algo_name)
    requested_mode = os.environ.get("SUBMISSION_PRETRAINED_MODE",
                                     "finetune").strip().lower()
    pretrained_state = None
    mode = "scratch"
    if pretrained_path is not None:
        print(f"=+= Found pretrained weights: {pretrained_path}")
        try:
            pre_model, pre_cfg = algo_mod.load_checkpoint(
                str(pretrained_path), device=device,
            )
            pretrained_state = pre_model.state_dict()
            # Free the temporary pre_model — we only needed its state_dict
            del pre_model
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            print(f"    Failed to load pretrained: {e}. "
                   f"Falling back to scratch.")
            pretrained_state = None

        if pretrained_state is not None:
            if requested_mode == "load":
                mode = "load"
            else:
                # Check if algo supports init_state_dict (currently only dvt_unet3d)
                import inspect
                tr_sig = inspect.signature(algo_mod.train_self_supervised)
                if "init_state_dict" in tr_sig.parameters:
                    mode = "finetune"
                else:
                    print(f"    NOTE: algo '{algo_name}' doesn't support "
                          f"init_state_dict; falling back to load-only mode.")
                    mode = "load"
    else:
        print(f"=+= No pretrained weights at {MODEL_DIR}.")

    print(f"=+= MODE: {mode}")

    print("\n[1/4] Loading input stack…")
    t_total = time.time()
    input_files = load_image_file_paths(
        location=INPUT_PATH / "images/stacked-neuron-images-with-noise",
    )
    print(f"  Found {len(input_files)} input file(s).")
    if not input_files:
        raise RuntimeError(
            f"No input files found under "
            f"{INPUT_PATH / 'images/stacked-neuron-images-with-noise'}"
        )

    for input_tif in input_files:
        print(f"\n══ Processing {Path(input_tif).name} ══")
        input_tif_result = SimpleITK.ReadImage(input_tif)
        input_stack = SimpleITK.GetArrayFromImage(input_tif_result)
        print(f"  Shape: {input_stack.shape}  dtype: {input_stack.dtype}")
        print(f"  Range: [{input_stack.min()}, {input_stack.max()}]")

        if mode == "load":
            print(f"\n[2/4] Loading pretrained model (skip training)…")
            model, config = algo_mod.load_checkpoint(
                str(pretrained_path), device=device,
            )
            # Recompute per-stack normalization for this input — the model
            # was trained on a different stack.
            if hasattr(algo_mod, "compute_norm_params"):
                norm_params = algo_mod.compute_norm_params(input_stack)
                config["norm_params"] = norm_params
                print(f"  Re-computed norm params for this stack: "
                       f"shift={norm_params.get('shift', 'n/a'):.2f}, "
                       f"scale={norm_params.get('scale', 'n/a'):.2f}")
        elif mode == "finetune":
            print(f"\n[2/4] Fine-tuning ({algo_name}) from pretrained init…")
            model, config = algo_mod.train_self_supervised(
                stack=input_stack, device=device, config=dict(cfg),
                verbose=True, init_state_dict=pretrained_state,
            )
        else:
            # scratch
            print(f"\n[2/4] Training ({algo_name}) from scratch…")
            model, config = algo_mod.train_self_supervised(
                stack=input_stack, device=device, config=dict(cfg),
                verbose=True,
            )

        print(f"\n[3/4] Denoising {input_stack.shape[0]} frames…")
        denoised = algo_mod.denoise_stack(
            model=model, stack=input_stack.astype(np.float32),
            config=config, device=device, verbose=True,
        )

        print("\n[4/4] Saving output…")
        if np.issubdtype(input_stack.dtype, np.integer):
            info = np.iinfo(input_stack.dtype)
            denoised = np.clip(denoised, info.min, info.max)
            denoised = np.round(denoised).astype(input_stack.dtype)
        else:
            denoised = denoised.astype(np.float32)
        print(f"  Output shape: {denoised.shape}  dtype: {denoised.dtype}")
        print(f"  Output range: [{denoised.min()}, {denoised.max()}]")

        write_array_as_image_file(
            location=OUTPUT_PATH /
                "images/stacked-neuron-images-with-reduced-noise",
            array=denoised, name=Path(input_tif).name,
        )

        del model, denoised
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

    total_time = time.time() - t_total
    print(f"\n{'=' * 50}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 50}")
    return 0


# ──────────────────────────────────────────────────────────────
# I/O helpers (same as upstream inference.py)
# ──────────────────────────────────────────────────────────────

def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_paths(*, location):
    return (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )


def write_array_as_image_file(*, location, array, name):
    location.mkdir(parents=True, exist_ok=True)
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(image, location / f"{name}", useCompression=True)


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Torch CUDA info")
    print(f"  CUDA available: {(available := torch.cuda.is_available())}")
    if available:
        cur = torch.cuda.current_device()
        print(f"  device count : {torch.cuda.device_count()}")
        print(f"  current      : {cur}")
        print(f"  properties   : {torch.cuda.get_device_properties(cur)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
