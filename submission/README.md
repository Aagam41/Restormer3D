# Restormer3D Submission Builder

Builds a Grand Challenge-ready Docker image for Restormer3D. The image bundles
the framework (`algos/`, `configs/`, `runner/`) and runs `inference.py`, which
loads a pretrained checkpoint, fine-tunes briefly per stack, and denoises.

## Quickstart

```bash
# 1. Put your pretrained checkpoint here (filename MUST be restormer3d.pth)
cp ../pretrained_restormer3d/restormer3d_best.pth model/restormer3d.pth

# 2. Place a test stack
mkdir -p test/input/interf0/images/stacked-neuron-images-with-noise
cp your_noisy_stack.tif test/input/interf0/images/stacked-neuron-images-with-noise/

# 3. Build (defaults to restormer3d + restormer3d_finetune)
./do_build.sh

# 4. Test locally (mirrors Grand Challenge: --network none --gpus all)
time ./do_test_run.sh

# 5. Save the uploadable tarball
./do_save.sh
```

The output appears in `test/output/interf0/images/stacked-neuron-images-with-reduced-noise/`.

## Modes

Default is **finetune**: load pretrained -> 800 N2V iters on the input stack -> infer.

To switch to **eval-only** (load -> infer, no per-stack training, ~30s/stack):

```bash
./do_build.sh restormer3d restormer3d_eval_only
```

Or set the env var at run time without rebuilding:

```bash
SUBMISSION_PRETRAINED_MODE=load ./do_test_run.sh
```

## Checkpoint placement

`inference.py` looks for the pretrained weights at, in order:
```
/opt/ml/model/restormer3d.pth
/opt/ml/model/restormer3d_weights.pth
/opt/ml/model/weights.pth
```
The `model/` folder in this directory is mounted to `/opt/ml/model` at run time, so
`model/restormer3d.pth` is what you want. If no checkpoint is found, inference falls
back to training from scratch (slow) and prints `MODE: scratch`.

## What to look for in the run log

| Log line | Meaning |
|---|---|
| `MODE: finetune` | Pretrained found, fine-tune path taken (good). |
| `Initialized from pretrained state_dict (N tensors loaded)` | Weights loaded successfully. |
| `MODE: scratch` | Checkpoint NOT found - check `model/restormer3d.pth` exists. |
| `RuntimeError: ... does not match current model` | Architecture mismatch between checkpoint and finetune config. Make `configs/restormer3d_finetune.py` match what you pretrained with. |

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Image recipe. `ARG SUBMISSION_ALGO/SUBMISSION_CONFIG` default to restormer3d/restormer3d_finetune. |
| `do_build.sh [algo] [config]` | Stages framework into build context, runs docker build. |
| `do_test_run.sh [algo] [config]` | Runs the image on `test/input/` with GPU + offline. |
| `do_save.sh [algo] [config]` | `docker save \| gzip` into an uploadable `.tar.gz`. |
| `inference.py` | Generic entrypoint: loads algo + config, runs pretrain-aware train + infer. |
| `requirements.txt` | Lean inference deps (numpy, SimpleITK, tifffile). |
| `model/` | Drop `restormer3d.pth` here. |

## Notes

- **GPU required**: `do_test_run.sh` passes `--gpus all`; your local Docker needs nvidia-container-runtime.
- **Output dtype matches input** (int16 in -> int16 out, with clip+round).
- **`--network none`** mirrors Grand Challenge. If the run unexpectedly needs the network, it'll fail here first - good.
- **Image size**: bundles the whole framework + checkpoint; expect a few GB. Check your challenge's disk quota.
