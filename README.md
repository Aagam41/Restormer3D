# Restormer3D — Self-Supervised Calcium Imaging Denoiser

An implementation of **Restormer3D** for the AI4Life Calcium Imaging Denoising Challenge (CIDC25).

## What's here

```
algos/restormer3d.py          The model: 3D Restormer (MDTA + GDFN), N2V training
configs/                      restormer3d_{default,pretrain,finetune,eval_only}.py
runner/                       Framework: preprocessing, IO, metrics, run_one driver
scripts/
  pretrain_restormer3d.py     Multi-stack pretraining with val-based model selection
  benchmark.py                Evaluate (algo, config) × stacks, compute stSNR/stPSNR
  compare_configs.py          Pivot benchmark results into a comparison table
  run_one.py                  Single (algo, stack) run
  eval_only.py                Metrics on an already-denoised output
submission/                   Docker submission builder (see submission/README.md)
PRETRAINING.md                Full pretrain → finetune → submit workflow
requirements.txt              Dev/benchmark deps (torch, matplotlib, scikit-image)
```

## The method

Restormer3D is a 3D adaptation of the Restormer (Zamir et al., CVPR 2022): a 4-level U-Net where each block uses Multi-Dconv Head Transposed Attention (MDTA, channel-wise attention with depthwise 3D convs) and a Gated-Dconv Feed-Forward Network (GDFN). It is trained **self-supervised** — no clean ground truth — using a two-stage scheme: a temporal-median warmup followed by 3D Noise2Void blind-spot masking. At inference it uses a sliding window with 50% temporal overlap (Hann blending + mirror-padded time axis) to avoid frame-boundary artifacts.

## Quickstart

```bash
# 1. Install dev deps
pip install -r requirements.txt

# 2. Pretrain on RTX 5090 (or any GPU; "as long as you like")
python scripts/pretrain_restormer3d.py \
    --train-dir   /path/to/train_noisy \
    --val-noisy-dir /path/to/val/noisy \
    --val-clean-dir /path/to/val/clean \
    --output-dir  ./pretrained_restormer3d \
    --total-iters 50000 --checkpoint-interval 500

# 3. Benchmark eval-only vs finetune side by side
python scripts/benchmark.py \
    --noisy-dir /path/to/test/noisy --clean-dir /path/to/test/clean \
    --algos restormer3d \
    --configs configs/restormer3d_eval_only.py configs/restormer3d_finetune.py \
    --pretrained pretrained_restormer3d/restormer3d_best.pth \
    --group-id g_cmp
python scripts/compare_configs.py --group-dir benchmark_results/g_cmp --metric stSNR

# 4. Build + test + save the submission (see submission/README.md)
cp pretrained_restormer3d/restormer3d_best.pth submission/model/restormer3d.pth
cd submission
./do_build.sh restormer3d restormer3d_finetune
./do_test_run.sh
./do_save.sh
```

## Architecture pinning (important)

The pretrain, finetune, and eval_only configs **must share identical architecture keys** (`dim`, `num_blocks`, `num_refinement_blocks`, `heads`, `ffn_expansion_factor`, `bias_free`, `normalization`, `temporal_target`). If they differ, loading the pretrained checkpoint into the finetune model fails loud with a clear error — this is intentional. The four configs shipped here all use `dim=32, num_blocks=(2,2,2,3), heads=(1,2,4,8), bias_free=True`. If you change one, change all of them.

## See also

- `PRETRAINING.md` — detailed workflow, CLI flags, timing, caveats
- `submission/README.md` — Docker build/test/save for Grand Challenge

## Citation
If you use Restormer, please consider citing:

```
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```
