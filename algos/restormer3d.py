"""
Restormer-3D for Self-Supervised Calcium Imaging Denoising
===========================================================

Adapts "Restormer: Efficient Transformer for High-Resolution Image
Restoration" (Zamir et al., CVPR 2022 — https://arxiv.org/abs/2111.09881)
to 3D volumetric denoising for AI4Life-CIDC25.

What Restormer is in the paper:
    A 4-level encoder-decoder Transformer where each block has:
      - MDTA (Multi-Dconv head Transposed Attention): attention is computed
        across the CHANNEL dimension instead of spatial. Complexity is
        O(C^2 * HW) instead of O((HW)^2), i.e. linear in spatial size.
      - GDFN (Gated Dconv Feed-Forward Network): a feed-forward layer
        with depth-wise conv + element-wise gating (GELU branch x linear
        branch), which controls what features propagate.
    Both MDTA and GDFN use 3x3 depth-wise convs to inject local context
    before the linear-complexity global mixing happens.

Why it fits calcium-imaging denoising:
    1. Linear-complexity attention -> we can run on full (D,H,W) patches
       without Swin-style window restriction.
    2. Bias-free everywhere -> avoids the "predict the local mean"
       degenerate solution that plagues N2V (Mohan et al. ICLR'19).
    3. The depth-wise convs already encode local 3D context; the
       transposed attention then mixes information *across channels*
       globally - no spatial blur from window attention.

3D adaptation (this file):
    - 2D depth-wise 3x3 conv -> 3D depth-wise 3x3x3 conv
    - PixelUnshuffle/PixelShuffle 2D -> 3D variants (with strided conv
      for downsampling so we can downsample on space and time
      independently if desired)
    - Channel-cross-covariance attention is dimension-agnostic — the
      "spatial" axis HW becomes DHW, attention weights are still C x C
    - Bias-free convolutions and LayerNorm without bias, per paper §4.4

Training:
    Same two-stage self-supervised regime as model_dvt.py:
        Stage 0 — Temporal-median warmup
        Stage 1 — 3D Noise2Void blind-spot training

API parity with model_dvt.py / model.py:
    compute_norm_params, normalize, denormalize
    train_self_supervised(stack, device, config) -> (model, cfg)
    denoise_stack(model, stack, config, device)  -> np.ndarray
    save_checkpoint, load_checkpoint
    UNet3D alias -> Restormer3D
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# NORMALIZATION (p3-p97 robust scaling — same as model_dvt.py)
# ══════════════════════════════════════════════════════════════

from runner import preprocessing as _prep

DEFAULT_NORMALIZATION = "p0.5_p99.5"
DEFAULT_TEMPORAL_TARGET = "temporal_median_2d"


def _default_norm():
    return _prep.resolve_normalization(DEFAULT_NORMALIZATION)


def compute_norm_params(stack: np.ndarray) -> dict:
    return _default_norm().compute_params(stack)


def normalize(data, params):
    return _default_norm().forward(data, params)


def denormalize(data, params):
    return _default_norm().inverse(data, params)


# ══════════════════════════════════════════════════════════════
# RESTORMER 3D BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class LayerNorm3D(nn.Module):
    """
    LayerNorm over channel dim for [B, C, D, H, W] tensors.
    BiasFree variant (no learnable bias, no learnable scale offset) per
    paper §4.4 — important for self-supervised denoising stability.
    """

    def __init__(self, num_channels: int, bias_free: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        if bias_free:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = 1e-6

    def forward(self, x):
        # x: [B, C, D, H, W]
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)
        return x


class MDTA3D(nn.Module):
    """
    Multi-Dconv head Transposed Attention — paper §3.1, Eq. (1).

    The key trick: attention is computed across CHANNELS (transposed),
    so the attention map is C x C instead of (DHW) x (DHW). This makes
    complexity linear in spatial volume, allowing us to handle full 3D
    patches without Swin-style window restriction.

    Pipeline:
        x  -> 1x1x1 conv (3x to get q,k,v projections, point-wise)
           -> 3x3x3 depth-wise conv (encodes local 3D context)
           -> reshape to [B, heads, head_dim, DHW]
           -> q_norm, k_norm = L2-normalize along DHW dim (paper trick)
           -> attn = softmax((k @ q^T) * scale)           # [heads, head_dim, head_dim]
           -> out = attn @ v                               # [heads, head_dim, DHW]
           -> reshape, 1x1x1 project, residual add
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        # learnable temperature alpha — per-head, paper Eq. 1
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1x1 expand to 3*dim for q,k,v
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)
        # 3x3x3 depth-wise conv (groups = 3*dim) to add local context
        self.qkv_dwconv = nn.Conv3d(
            dim * 3, dim * 3, kernel_size=3, padding=1,
            groups=dim * 3, bias=False,
        )
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        # qkv local-aware projection
        qkv = self.qkv_dwconv(self.qkv(x))           # [B, 3C, D, H, W]
        q, k, v = qkv.chunk(3, dim=1)                # each [B, C, D, H, W]

        # reshape to multi-head: [B, heads, head_dim, DHW]
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, D * H * W)
        k = k.reshape(B, self.num_heads, head_dim, D * H * W)
        v = v.reshape(B, self.num_heads, head_dim, D * H * W)

        # L2-normalize q and k along the spatial dim (DHW)
        # (paper does this on the flattened spatial axis to keep magnitudes
        # of the dot product controlled at high resolution)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # transposed attention: attn = (k @ q^T) -> [B, heads, head_dim, head_dim]
        # Then out = attn @ v.
        attn = (k @ q.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v                                # [B, heads, head_dim, DHW]
        out = out.reshape(B, C, D, H, W)
        out = self.project_out(out)
        return out


class GDFN3D(nn.Module):
    """
    Gated-Dconv Feed-Forward Network — paper §3.2, Eq. (2).

    Standard FFN replaced with:
        x -> 1x1x1 conv to 2*hidden  (point-wise expand into two parallel
             halves; one is the "value" path, one is the "gate" path)
        -> 3x3x3 depth-wise conv (local 3D context for both halves)
        -> split into two halves
        -> gated activation: GELU(gate_half) * value_half
        -> 1x1x1 conv back to dim
    """

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)

        # 1x1 expand to 2*hidden so we can split into gate and value
        self.project_in = nn.Conv3d(dim, hidden * 2, kernel_size=1, bias=False)
        # 3x3x3 depth-wise conv (groups = 2*hidden) for local context
        self.dwconv = nn.Conv3d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1,
            groups=hidden * 2, bias=False,
        )
        self.project_out = nn.Conv3d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)                       # [B, 2*hidden, D, H, W]
        x = self.dwconv(x)
        a, b = x.chunk(2, dim=1)                      # gate, value
        x = F.gelu(a) * b
        x = self.project_out(x)
        return x


class TransformerBlock3D(nn.Module):
    """
    One Restormer block: pre-norm MDTA + pre-norm GDFN, both with residual.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2.66,
        bias_free: bool = True,
    ):
        super().__init__()
        self.norm1 = LayerNorm3D(dim, bias_free=bias_free)
        self.attn = MDTA3D(dim, num_heads)
        self.norm2 = LayerNorm3D(dim, bias_free=bias_free)
        self.ffn = GDFN3D(dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── 3D down/up sampling — strided conv + transposed conv ──────
# The paper uses pixel-unshuffle / pixel-shuffle for 2D. For 3D there's
# no native PyTorch op, so we use strided 3x3x3 conv (down) and
# 2x2x2 transpose conv (up). Both bias-free.

class Downsample3D(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=(2, 2, 2)):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch * 2
        self.body = nn.Conv3d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False,
        )

    def forward(self, x):
        return self.body(x)


class Upsample3D(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=(2, 2, 2)):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch // 2
        self.body = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=stride, stride=stride, bias=False,
        )

    def forward(self, x):
        return self.body(x)


# ══════════════════════════════════════════════════════════════
# RESTORMER 3D — main architecture
# ══════════════════════════════════════════════════════════════

class Restormer3D(nn.Module):
    """
    4-level encoder-decoder following the Restormer paper, lifted to 3D.

    Default dims/blocks are scaled DOWN from the paper's
    dim=48 / blocks=[4,6,6,8] (designed for offline supervised training
    on 300K iters) to fit a zero-shot per-stack training budget on T4
    in <30 min. The user can scale up via `dim` and `num_blocks`.

    Input/output:  [B, 1, D, H, W] -> [B, 1, D, H, W]
    Residual learning: output = input - predicted_noise
    Spatial dims auto-padded to multiples of 8 (3 levels of /2 down).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 32,                          # paper: 48
        num_blocks: tuple = (2, 2, 2, 3),        # paper: (4, 6, 6, 8)
        num_refinement_blocks: int = 2,          # paper: 4
        heads: tuple = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.0,       # paper: 2.66
        bias_free: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # ── Initial feature extraction (1x conv) ──────────────
        self.patch_embed = nn.Conv3d(
            in_channels, dim, kernel_size=3, padding=1, bias=False,
        )

        # ── Encoder ───────────────────────────────────────────
        # level 1: dim
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample3D(dim, dim * 2)

        # level 2: 2*dim
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim * 2, num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample3D(dim * 2, dim * 4)

        # level 3: 4*dim
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim * 4, num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample3D(dim * 4, dim * 8)

        # ── Latent (deepest) ──────────────────────────────────
        self.latent = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim * 8, num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_blocks[3])
        ])

        # ── Decoder (mirror) ──────────────────────────────────
        self.up4_3 = Upsample3D(dim * 8, dim * 4)
        self.reduce_chan_level3 = nn.Conv3d(
            dim * 8, dim * 4, kernel_size=1, bias=False,
        )
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim * 4, num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample3D(dim * 4, dim * 2)
        self.reduce_chan_level2 = nn.Conv3d(
            dim * 4, dim * 2, kernel_size=1, bias=False,
        )
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim * 2, num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample3D(dim * 2, dim)
        # Per paper §3 last paragraph: at level-1 we DO NOT reduce
        # channels after concat — keeps fine textural details.
        # So decoder_level1 receives 2*dim channels.
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim * 2, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_blocks[0])
        ])

        # ── Refinement at full resolution (paper §3) ──────────
        self.refinement = nn.Sequential(*[
            TransformerBlock3D(
                dim=dim * 2, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias_free=bias_free,
            ) for _ in range(num_refinement_blocks)
        ])

        # ── Output: predict residual (noise), 3x3x3 conv ──────
        self.output = nn.Conv3d(
            dim * 2, out_channels, kernel_size=3, padding=1, bias=False,
        )

    def forward(self, x):
        """x: [B, 1, D, H, W] -> [B, 1, D, H, W]"""
        identity = x

        # Pad spatial+temporal dims to multiples of 8 (3 down-stages)
        _, _, D, H, W = x.shape
        pd = (8 - D % 8) % 8
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")
            identity = F.pad(
                identity, (0, pw, 0, ph, 0, pd), mode="reflect",
            )

        # Encoder
        f0 = self.patch_embed(x)                  # [B, dim, D, H, W]
        out_enc1 = self.encoder_level1(f0)
        in_enc2 = self.down1_2(out_enc1)
        out_enc2 = self.encoder_level2(in_enc2)
        in_enc3 = self.down2_3(out_enc2)
        out_enc3 = self.encoder_level3(in_enc3)
        in_lat = self.down3_4(out_enc3)
        latent = self.latent(in_lat)

        # Decoder with skip cat + 1x1 channel reduction
        in_dec3 = self.up4_3(latent)
        in_dec3 = _match_cat(in_dec3, out_enc3)
        in_dec3 = self.reduce_chan_level3(in_dec3)
        out_dec3 = self.decoder_level3(in_dec3)

        in_dec2 = self.up3_2(out_dec3)
        in_dec2 = _match_cat(in_dec2, out_enc2)
        in_dec2 = self.reduce_chan_level2(in_dec2)
        out_dec2 = self.decoder_level2(in_dec2)

        # Level 1: concat WITHOUT 1x1 reduction (preserve fine detail)
        in_dec1 = self.up2_1(out_dec2)
        in_dec1 = _match_cat(in_dec1, out_enc1)    # 2*dim channels
        out_dec1 = self.decoder_level1(in_dec1)

        # Full-resolution refinement
        out = self.refinement(out_dec1)

        # Predict noise residual, subtract from input
        noise = self.output(out)
        out = identity - noise

        # Strip padding
        if pd or ph or pw:
            out = out[:, :, :D, :H, :W]
        return out


def _match_cat(up, skip):
    """Pad the upsampled tensor to match skip dims, then concatenate."""
    dd = skip.shape[2] - up.shape[2]
    dh = skip.shape[3] - up.shape[3]
    dw = skip.shape[4] - up.shape[4]
    if dd or dh or dw:
        # take the smaller of the two so we never have negative-pad
        # situations from rounding
        if dd < 0 or dh < 0 or dw < 0:
            # crop the bigger one
            md = min(up.shape[2], skip.shape[2])
            mh = min(up.shape[3], skip.shape[3])
            mw = min(up.shape[4], skip.shape[4])
            up = up[:, :, :md, :mh, :mw]
            skip = skip[:, :, :md, :mh, :mw]
        else:
            up = F.pad(up, (0, dw, 0, dh, 0, dd))
    return torch.cat([up, skip], dim=1)


# Alias for drop-in compatibility with anything importing UNet3D
UNet3D = Restormer3D


# ══════════════════════════════════════════════════════════════
# 3D BLIND-SPOT MASKING (Noise2Void)
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume: torch.Tensor, mask_ratio: float = 0.015,
                radius: int = 1):
    """
    Noise2Void 3D masking.

    radius=1 by default to keep calcium transients sharp on the time
    axis (transients are 2-4 frames; sampling neighbors within radius 2
    on time often falls inside the same transient).
    """
    D, H, W = volume.shape
    n_vox = D * H * W
    n_mask = max(int(n_vox * mask_ratio), 1)

    flat_idx = torch.randperm(n_vox, device=volume.device)[:n_mask]
    mz = flat_idx // (H * W)
    my = (flat_idx % (H * W)) // W
    mx = flat_idx % W

    original = volume[mz, my, mx].clone()

    dz = torch.randint(-radius, radius + 1, (n_mask,), device=volume.device)
    dy = torch.randint(-radius, radius + 1, (n_mask,), device=volume.device)
    dx = torch.randint(-radius, radius + 1, (n_mask,), device=volume.device)
    same = (dz == 0) & (dy == 0) & (dx == 0)
    dz[same] = 1

    nz = (mz + dz).clamp(0, D - 1)
    ny = (my + dy).clamp(0, H - 1)
    nx = (mx + dx).clamp(0, W - 1)

    masked = volume.clone()
    masked[mz, my, mx] = volume[nz, ny, nx]
    return masked, (mz, my, mx), original


def _gaussian_window_3d(shape, sigma_frac=0.3, device="cpu"):
    """3D Gaussian blending window for sliding-window inference."""
    windows = []
    for s in shape:
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = max(s * sigma_frac, 1.0)
        w = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
        windows.append(w)
    w3d = (windows[0][:, None, None]
           * windows[1][None, :, None]
           * windows[2][None, None, :])
    return w3d.clamp(min=1e-6)


def _augment_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    """8-fold spatial augmentation (4 rot x 2 flip) for [D, H, W]."""
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


# ══════════════════════════════════════════════════════════════
# TRAINING (two-stage self-supervised)
# ══════════════════════════════════════════════════════════════

def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
    init_state_dict: dict = None,
):
    """
    Stage 0 — Temporal-median warmup (structural prior)
    Stage 1 — 3D Noise2Void blind-spot

    Trains in full fp32 for numerical stability.

    Args:
        stack:  [F, H, W] numpy array (raw, original values).
        device: torch device.
        config: optional overrides for any default key below.
        init_state_dict: optional pre-trained weights to load into the
            model BEFORE training starts. Used for fine-tuning from a
            pretrained checkpoint. The architecture in `config` must match
            the architecture the state_dict was saved with; loading uses
            strict matching and will fail loud on mismatch.
    """
    t0 = time.time()
    cfg = {
        # backbone
        "dim": 32,
        "num_blocks": (2, 2, 2, 3),
        "num_refinement_blocks": 2,
        "heads": (1, 2, 4, 8),
        "ffn_expansion_factor": 2.0,
        "bias_free": True,
        # patch sampling
        "patch_d": 32,
        "patch_hw": 64,
        "batch_size": 2,
        # schedule
        "warmup_iters": 200,
        "n2v_iters": 3000,
        "lr": 3e-4,
        # n2v masking
        "mask_ratio": 0.015,
        "mask_radius": 1,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" Restormer3D: dim={cfg['dim']}, blocks={cfg['num_blocks']}, "
              f"heads={cfg['heads']}, ffn={cfg['ffn_expansion_factor']}")
        print(f" Patch: {pd}x{phw}x{phw}, batch={bs}")
        print(f" Stages: warmup={cfg['warmup_iters']}, n2v={cfg['n2v_iters']}")
        print(f" Precision: fp32")

    # ── Normalize (strategy from config) ────────────────
    norm_name = cfg.get("normalization", DEFAULT_NORMALIZATION)
    norm_strategy = _prep.resolve_normalization(norm_name)
    norm_params = norm_strategy.compute_params(stack)
    cfg["norm_params"] = norm_params
    cfg["__resolved_normalization"] = norm_strategy.name
    stack_norm = norm_strategy.forward(stack, norm_params)
    if verbose:
        print(f" Norm [{norm_strategy.name}]: "
              f"shift={norm_params['shift']:.2f}, "
              f"scale={norm_params['scale']:.2f}, "
              f"range=[{stack_norm.min():.3f}, {stack_norm.max():.3f}]")

    # ── Temporal target (strategy from config) ──────────
    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    cfg["__resolved_temporal_target"] = tt_strategy.name
    if tt_strategy.returns != "2d":
        _tt_3d = tt_strategy.compute(stack_norm)
        temporal_med = np.median(_tt_3d, axis=0).astype(np.float32)
    else:
        temporal_med = tt_strategy.compute(stack_norm)
    if verbose:
        print(f" Temporal target [{tt_strategy.name}]: "
              f"[{temporal_med.min():.3f}, {temporal_med.max():.3f}]")

    stack_t = torch.from_numpy(stack_norm).float().to(device)
    tmed_t = torch.from_numpy(temporal_med).float().to(device)

    # Model
    model = Restormer3D(
        in_channels=1, out_channels=1,
        dim=cfg["dim"],
        num_blocks=cfg["num_blocks"],
        num_refinement_blocks=cfg["num_refinement_blocks"],
        heads=cfg["heads"],
        ffn_expansion_factor=cfg["ffn_expansion_factor"],
        bias_free=cfg["bias_free"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_bias = sum(1 for n, _ in model.named_parameters() if n.endswith(".bias"))
    if verbose:
        print(f" Model params: {n_params:,}  (bias params: {n_bias})")

    # Optionally initialize from pretrained weights (fine-tuning).
    if init_state_dict is not None:
        # strict=False so the caller sees missing/unexpected keys, but we
        # still raise loud on any mismatch — silently loading partial
        # weights leads to subtle quality regressions that are hard to
        # diagnose later.
        missing, unexpected = model.load_state_dict(
            init_state_dict, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Pretrained state_dict does not match current model:\n"
                f"  missing keys:    {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
                f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}\n"
                f"This usually means the architecture (dim, num_blocks, "
                f"num_refinement_blocks, heads, ffn_expansion_factor, "
                f"bias_free) differs between pretraining and fine-tuning. "
                f"They must match exactly."
            )
        if verbose:
            print(f" Initialized from pretrained state_dict "
                  f"({len(init_state_dict)} tensors loaded)")

    # Random patch helper
    def random_patch():
        d = min(pd, F_total)
        h = min(phw, H)
        w = min(phw, W)
        t0_ = np.random.randint(0, max(F_total - d, 1))
        y0 = np.random.randint(0, max(H - h, 1))
        x0 = np.random.randint(0, max(W - w, 1))
        return stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w], tmed_t[y0:y0+h, x0:x0+w]

    # ─── Stage 0: temporal-median warmup ───────────────────
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Temporal-median warmup — "
                  f"{cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1,
        )
        crit = nn.MSELoss()
        model.train()
        rl = 0.0

        for it in range(cfg["warmup_iters"]):
            patches, targets = [], []
            for _ in range(bs):
                vol, tmed_crop = random_patch()
                aug = np.random.randint(0, 8)
                vol = _augment_3d(vol, aug)
                tmed_crop = _augment_3d(
                    tmed_crop.unsqueeze(0).expand(vol.shape[0], -1, -1), aug,
                )
                patches.append(vol.unsqueeze(0))
                targets.append(tmed_crop.unsqueeze(0))

            inp = torch.stack(patches, dim=0).to(device)
            tgt = torch.stack(targets, dim=0).to(device)

            opt.zero_grad()
            pred = model(inp)
            loss = crit(pred, tgt)

            if not torch.isfinite(loss):
                if verbose:
                    print(f"   WARN non-finite loss at iter {it}, skipping")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()

            if verbose and (it + 1) % 100 == 0:
                print(f"   {it+1:>5}/{cfg['warmup_iters']} "
                      f"loss={rl/100:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # ─── Stage 1: 3D Noise2Void ────────────────────────────
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] 3D Noise2Void — {cfg['n2v_iters']} iters")
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"] * 0.5, weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6,
        )
        model.train()
        rl = 0.0

        for it in range(cfg["n2v_iters"]):
            all_orig = []
            patches = []
            for _ in range(bs):
                vol, _ = random_patch()
                aug = np.random.randint(0, 8)
                vol = _augment_3d(vol, aug)
                masked, (mz, my, mx), orig = n2v_mask_3d(
                    vol,
                    mask_ratio=cfg["mask_ratio"],
                    radius=cfg["mask_radius"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((mz, my, mx, orig))

            inp = torch.stack(patches, dim=0).to(device)

            opt.zero_grad()
            pred = model(inp)
            loss = torch.tensor(0.0, device=device)
            for b, (mz, my, mx, orig) in enumerate(all_orig):
                pred_at_mask = pred[b, 0, mz, my, mx]
                loss = loss + F.mse_loss(pred_at_mask, orig)
            loss = loss / bs

            if not torch.isfinite(loss):
                if verbose:
                    print(f"   WARN non-finite loss at iter {it}, skipping")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()

            if verbose and (it + 1) % 250 == 0:
                lr_now = sch.get_last_lr()[0]
                print(f"   {it+1:>5}/{cfg['n2v_iters']} "
                      f"loss={rl/250:.6f} lr={lr_now:.2e} "
                      f"{time.time()-t0:.1f}s")
                rl = 0.0

    elapsed = time.time() - t0
    if verbose:
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return model, cfg


# ══════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(
    model: Restormer3D,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    """
    Sliding-window inference with Gaussian spatial blending.
    """
    model.eval()
    model = model.float()

    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    pd = min(config.get("patch_d", 32), F_total)
    phw = min(config.get("patch_hw", 64), H, W)
    # multiples of 8 (3 down-stages)
    pd = max((pd // 8) * 8, 8)
    phw = max((phw // 8) * 8, 8)
    # Temporal overlap: 50% by default. The previous "stride_d = pd"
    # (no temporal overlap) produced a periodic noisy/clean pattern at
    # period = pd because frames sitting at the START or END of each
    # tile saw context only on ONE side, while center frames saw both.
    # With 50% overlap every frame is now blended from two tiles — one
    # where it's near the head, one where it's near the tail — so the
    # head/tail asymmetry averages out. Cost: ~2x more tiles.
    # Set config["temporal_overlap"] = 0 to restore the old behavior.
    temporal_overlap_frac = float(config.get("temporal_overlap", 0.5))
    if temporal_overlap_frac <= 0:
        stride_d = pd
    else:
        stride_d = max(int(pd * (1.0 - temporal_overlap_frac)), 1)
    stride_hw = max(phw // 2, 8)        # 50% spatial overlap

    if verbose:
        print(f" Sliding window: patch={pd}x{phw}x{phw}, "
              f"stride={stride_d}x{stride_hw}x{stride_hw} "
              f"(temporal overlap {100*(1 - stride_d/pd):.0f}%)")

    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    # ── Temporal window ───────────────────────────────────────
    # Hann taper when overlapping, flat when not. The Hann window
    # smoothly down-weights the edges of each tile, so when two
    # overlapping tiles both contribute to the same output frame, the
    # tile in which that frame is near the center dominates. The flat
    # window is kept for the no-overlap case where every output frame
    # comes from exactly one tile.
    if stride_d < pd:
        # Hann window with a small floor so the very-edge frames still
        # get a tiny positive weight (otherwise weight_sum could go to
        # zero at boundary cases).
        idx = torch.arange(pd, device=device, dtype=torch.float32)
        hann = 0.5 - 0.5 * torch.cos(2 * math.pi * idx / max(pd - 1, 1))
        g_t = (0.1 + 0.9 * hann).clamp(min=1e-3)
    else:
        g_t = torch.ones(pd, device=device)
    spatial_g = _gaussian_window_3d(
        (1, phw, phw), sigma_frac=0.3, device=device,
    ).squeeze(0)                           # [phw, phw]
    gauss_win = (
        g_t[:, None, None] * spatial_g[None, :, :]
    ).clamp(min=1e-6)                      # [pd, phw, phw]

    # ── Mirror-pad the time axis ──────────────────────────────
    # With 50% temporal overlap, interior frames get two overlapping
    # tiles, but the first/last `pd//2` frames would otherwise only
    # be covered by the single boundary tile (with a Hann edge weight
    # of nearly zero). Mirror-padding the stack by pd//2 frames on
    # each side gives every ORIGINAL frame at least two tiles' worth
    # of coverage. We crop back to the original length at the end.
    if stride_d < pd:
        tpad = pd // 2
    else:
        tpad = 0
    if tpad > 0:
        # Mirror reflection along the time axis. Requires tpad < F.
        if tpad >= F_total:
            tpad = max(F_total - 1, 0)
        # F.pad reflect on time: treat as [1, F, H, W] -> pad d dim
        # We pad with reflect: indices [tpad-1, tpad-2, ..., 1, 0, 0, 1, ..., F-1, F-2, ..., F-tpad]
        # which means new[0:tpad] = stack[tpad:0:-1] (excluding the seam),
        # consistent with torch reflect-padding semantics.
        stack_pad = torch.cat([
            torch.flip(stack_t[1:tpad+1], dims=[0]),
            stack_t,
            torch.flip(stack_t[-tpad-1:-1], dims=[0]),
        ], dim=0)
        F_padded = stack_pad.shape[0]
    else:
        stack_pad = stack_t
        F_padded = F_total

    output_sum_pad = torch.zeros(F_padded, H, W, device=device)
    weight_sum_pad = torch.zeros(F_padded, H, W, device=device)

    z_starts = list(range(0, max(F_padded - pd, 0) + 1, stride_d))
    if not z_starts or z_starts[-1] + pd < F_padded:
        z_starts.append(max(F_padded - pd, 0))
    y_starts = list(range(0, max(H - phw, 0) + 1, stride_hw))
    if not y_starts or y_starts[-1] + phw < H:
        y_starts.append(max(H - phw, 0))
    x_starts = list(range(0, max(W - phw, 0) + 1, stride_hw))
    if not x_starts or x_starts[-1] + phw < W:
        x_starts.append(max(W - phw, 0))
    z_starts = sorted(set(z_starts))
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    total = len(z_starts) * len(y_starts) * len(x_starts)
    if verbose:
        print(f" Patches: {len(z_starts)}x{len(y_starts)}x{len(x_starts)}"
              f" = {total}  (mirror-pad time +{tpad})")

    t0 = time.time()
    done = 0
    for z0 in z_starts:
        z1 = min(z0 + pd, F_padded); ad = z1 - z0
        for y0 in y_starts:
            y1 = min(y0 + phw, H); ah = y1 - y0
            for x0 in x_starts:
                x1 = min(x0 + phw, W); aw = x1 - x0

                patch = stack_pad[z0:z1, y0:y1, x0:x1]
                if (ad < pd) or (ah < phw) or (aw < phw):
                    patch = F.pad(
                        patch,
                        (0, phw - aw, 0, phw - ah, 0, pd - ad),
                        mode="reflect",
                    )
                inp = patch.unsqueeze(0).unsqueeze(0).float()
                pred = model(inp).squeeze(0).squeeze(0).float()
                pred = pred[:ad, :ah, :aw]
                win = gauss_win[:ad, :ah, :aw]

                output_sum_pad[z0:z1, y0:y1, x0:x1] += pred * win
                weight_sum_pad[z0:z1, y0:y1, x0:x1] += win

                done += 1
                if verbose:
                    print(f"   {done}/{total} patches "
                          f"({100*done/total:.0f}%)  "
                          f"{time.time()-t0:.1f}s", end="\r")

    if verbose:
        print(f"\n Inference: {total} patches in {time.time()-t0:.1f}s")

    # Crop back to original time length (remove the mirror-pad we added)
    if tpad > 0:
        output_sum = output_sum_pad[tpad:tpad+F_total]
        weight_sum = weight_sum_pad[tpad:tpad+F_total]
    else:
        output_sum = output_sum_pad
        weight_sum = weight_sum_pad

    output = output_sum / weight_sum.clamp(min=1e-8)
    output = output.cpu().numpy()
    output = norm_strategy.inverse(output, norm_params)

    # Replace any NaN/Inf with sane values (general numerical safety)
    output = np.nan_to_num(
        output,
        nan=float(stack.mean()),
        posinf=float(stack.max()),
        neginf=float(stack.min()),
    )

    # Clip to input range with a small headroom for genuine bright peaks
    in_lo, in_hi = float(stack.min()), float(stack.max())
    in_hi_padded = in_hi + 0.1 * max(in_hi - in_lo, 1.0)
    output = np.clip(output, in_lo, in_hi_padded)

    # Detect-and-replace any flat frames (safety net)
    per_frame_std = output.std(axis=(1, 2))
    flat_mask = per_frame_std < 1.0
    if flat_mask.any():
        n_flat = int(flat_mask.sum())
        if verbose:
            print(f" WARN: {n_flat} flat frame(s) — replacing with input")
        output[flat_mask] = stack[flat_mask].astype(np.float32)

    return output


# ══════════════════════════════════════════════════════════════
# CHECKPOINTS (drop-in compatible names)
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "arch": "Restormer3D",
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Restormer3D(
        in_channels=1, out_channels=1,
        dim=cfg.get("dim", 32),
        num_blocks=cfg.get("num_blocks", (2, 2, 2, 3)),
        num_refinement_blocks=cfg.get("num_refinement_blocks", 2),
        heads=cfg.get("heads", (1, 2, 4, 8)),
        ffn_expansion_factor=cfg.get("ffn_expansion_factor", 2.0),
        bias_free=cfg.get("bias_free", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
