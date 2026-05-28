# Restormer3D Pretraining Workflow

Pretrain on a strong GPU (e.g. RTX 5090) → load + fine-tune on T4 at submission.

## Why pretrain

Zero-shot per-stack training takes a long time on T4. With 7 test stacks and a
1-hour budget, pretraining once and fine-tuning briefly per stack keeps each
stack to ~6-7 minutes. The validation stack(s) with clean GT are used **only for
model selection** (choosing which checkpoint to save as "best"). No gradient
ever flows from val.

## Pipeline

```
TRAINING (offline, 5090)              INFERENCE (T4, submit)
pretrain_restormer3d.py reads:        submission/ container:
  train_dir/   (noisy)                  loads pretrained,
  val_noisy/ + val_clean/   --ckpt-->   fine-tunes 800 iters/stack,
writes:                                 denoises
  restormer3d_best.pth
  pretraining_log.csv
```

## Step 1 - pretraining

```bash
python scripts/pretrain_restormer3d.py \
    --train-dir   /path/to/train_noisy \
    --val-noisy-dir /path/to/val/noisy \
    --val-clean-dir /path/to/val/clean \
    --output-dir  ./pretrained_restormer3d \
    --config      configs/restormer3d_pretrain.py \
    --total-iters 50000 \
    --checkpoint-interval 500
```

What happens:
- Loads every `*.tif` from `--train-dir` as a noisy training stack.
- Loads paired noisy+clean from `--val-noisy-dir`/`--val-clean-dir` (matched by filename stem).
- Builds Restormer3D from the config architecture.
- Trains the 2-stage N2V flow with multi-stack sampling (each iter picks a random training stack).
- Every 500 iters: forwards val through `denoise_stack()`, computes stSNR vs clean.
- Saves `restormer3d_best.pth` when val stSNR improves; always saves `restormer3d_last.pth`.
- Appends a row to `pretraining_log.csv` per checkpoint.

Monitor: `tail -f ./pretrained_restormer3d/pretraining_log.csv`

CSV columns: `iter, phase, running_train_loss, val_stSNR_mean, val_stSNR_per_stack, val_stPSNR_mean, val_stSI_PSNR_mean, is_best, elapsed_sec`

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--total-iters` | 50000 | warmup + N2V iters combined |
| `--warmup-iters` | total // 25 | override warmup specifically |
| `--checkpoint-interval` | 500 | eval val + maybe save best every N iters |
| `--val-stacks-subset` | use all | only use first N val stacks (faster checkpoints) |
| `--seed` | 42 | reproducibility |
| `--force-overwrite` | off | bypass the best.pth overwrite guard |

## Step 2 - submission

```bash
cp pretrained_restormer3d/restormer3d_best.pth submission/model/restormer3d.pth
cd submission
./do_build.sh restormer3d restormer3d_finetune
./do_test_run.sh        # local test (mirrors Grand Challenge: --network none --gpus all)
./do_save.sh            # creates the uploadable .tar.gz
```

The checkpoint filename **must be `restormer3d.pth`** — `inference.py` derives it
from the algo name. At runtime `inference.py`:
1. Detects `/opt/ml/model/restormer3d.pth`
2. Loads the state_dict
3. Calls `train_self_supervised(..., init_state_dict=...)` (load pretrained, fine-tune)
4. Denoises the test stack

Expected on T4 with `restormer3d_finetune` (warmup=0, n2v=800, lr=1e-4): ~6-7 min/stack,
so ~45-50 min for 7 stacks.

## Three inference modes (`SUBMISSION_PRETRAINED_MODE` env var)

| Mode | Pretrained present? | Behavior |
|---|---|---|
| `finetune` (default) | yes | Load -> fine-tune n2v_iters from config -> infer |
| `load` | yes | Load -> skip training -> infer (fastest, ~30s/stack) |
| any | no | Train from scratch -> infer (slow) |

To use load-only at submission, build with `restormer3d_eval_only` config (which has
`warmup_iters=0, n2v_iters=0`) OR set `SUBMISSION_PRETRAINED_MODE=load`.

## Benchmarking the checkpoint (eval-only vs finetune)

```bash
python scripts/benchmark.py \
    --noisy-dir /path/to/test/noisy --clean-dir /path/to/test/clean \
    --algos restormer3d \
    --configs configs/restormer3d_eval_only.py configs/restormer3d_finetune.py \
    --pretrained pretrained_restormer3d/restormer3d_best.pth \
    --group-id g_cmp

python scripts/compare_configs.py --group-dir benchmark_results/g_cmp \
    --metric stSNR --baseline restormer3d_eval_only
```

The delta column shows how much the 800-iter fine-tune adds over pure load+infer.

## Caveats

1. **Architecture must match** across pretrain/finetune/eval_only configs (`dim`,
   `num_blocks`, `num_refinement_blocks`, `heads`, `ffn_expansion_factor`,
   `bias_free`, `normalization`, `temporal_target`). Mismatch -> loud load error.
2. **Train/val/test distribution match** is the key assumption. Pretraining only
   generalizes to test stacks from similar imaging conditions.
3. **Val chooses iteration count.** With 1-2 val stacks this signal is noisy; use more if you have them.
4. **Per-stack normalization is recomputed** at inference (p0.5-p99.5 percentiles of each stack).
5. **VRAM at pretrain** with batch_size=4, patch 32x64x64, dim=32: ~10-14 GB. Fits 5090 easily; drop batch to 2 if you OOM.
6. **Temporal-overlap fix** (50% overlap + Hann + mirror-pad time) is applied automatically at inference.

## Provenance

`pretrained_restormer3d/pretraining_config.json` records the full CLI args, config,
training-stack list, val-pair list, param count, and GPU — enough to reproduce the run.
