#!/usr/bin/env python
"""
text-to-motion with lucidrains/mmdit  ──  flow matching (SiT-style) + optional REPA
====================================================================================

A ready-to-run text-to-motion training / sampling script. The backbone uses lucidrains/mmdit's
generalized MMDiT (dual-stream joint attention). Design highlights:

  * Two modalities: text (CLIP token-level) + motion (HumanML3D 263-dim)
  * SiT-style linear interpolant + velocity loss (rectified flow / flow matching)
  * logit-normal timestep sampling (SD3 approach, can be disabled)
  * classifier-free guidance (text dropout during training, dual-path interpolation at sampling)
  * Optional REPA: hook the motion hidden of some MMDiTBlock layer and align it to a frozen motion encoder
    (a stand-in encoder is used by default so the code path runs; for production replace it with TMR / MotionCLIP)
  * checkpoint relay: --resume auto-resumes training (works on local / Colab / Kaggle; Kaggle needs separate read/write paths only because its input is read-only)

Dependencies: already declared in mmdit-motion/pyproject.toml; uv run installs them automatically
  (core: mmdit / torch / einops; transformers is only needed with --text_encoder clip)

Examples:
  # Smoke test first (no data, no download: dummy text encoder + synthetic motion, do not pass --data_root)
  uv run mmdit-motion/main.py train --text_encoder dummy --steps 50 --ckpt_out runs/ckpt.pt

  # Training (real data)
  uv run mmdit-motion/main.py train --data_root /path/to/HumanML3D --text_encoder clip \
      --dim_motion 512 --depth 8 --batch_size 64 --steps 200000 --ckpt_out runs/ckpt.pt

  # H100 optimization (= the training command plus bf16 + compile; other params same as defaults)
  uv run mmdit-motion/main.py train --data_root /path/to/HumanML3D --text_encoder clip \
      --dim_motion 512 --depth 8 --batch_size 64 --steps 200000 \
      --amp_dtype bf16 --compile --save_every 5000 --ckpt_out runs/ckpt.pt

  # Resume (--steps must be greater than the already-trained steps. Locally/Colab, resume and ckpt_out can be the same file)
  uv run mmdit-motion/main.py train --data_root /path/to/HumanML3D \
      --resume runs/ckpt.pt --ckpt_out runs/ckpt.pt --steps 400000

  # Sampling
  uv run mmdit-motion/main.py sample --resume runs/ckpt.pt \
      --text_encoder clip --prompt "a person walks forward then sits down" \
      --steps 50 --cfg 4.0 --out sample.npy

  # View model architecture / parameter count (torchinfo)
  uv run mmdit-motion/main.py summary --depth 8 --dim_motion 512
"""

import os
import math
import shutil
import argparse
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from mmdit.mmdit_generalized_pytorch import MMDiT


# ----------------------------------------------------------------------------- #
# config
# ----------------------------------------------------------------------------- #
@dataclass
class Config:
    # data
    data_root: str = ""
    motion_dim: int = 263          # HumanML3D feature dim
    max_motion_len: int = 196      # frames
    fps: int = 20

    # text
    text_encoder: str = "clip"     # 'clip' | 'dummy'
    clip_name: str = "openai/clip-vit-base-patch32"
    dim_text: int = 512            # CLIP ViT-B/32 text hidden size
    max_text_len: int = 77

    # model (MMDiT)
    dim_motion: int = 512
    depth: int = 8
    heads: int = 8
    dim_head: int = 64
    dim_cond: int = 256            # timestep conditioning width
    qk_rmsnorm: bool = True
    num_residual_streams: int = 4

    # flow matching
    logit_normal_t: bool = True    # SD3-style timestep sampling
    p_uncond: float = 0.1          # CFG text-dropout prob

    # REPA (optional)
    repa: bool = False
    # which MMDiTBlock's motion hidden to align (0-indexed)
    repa_layer: int = 4
    repa_dim: int = 256            # target encoder embedding dim
    repa_weight: float = 0.5

    # optim
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    grad_clip: float = 1.0
    steps: int = 200_000
    warmup: int = 1000
    ema_decay: float = 0.999
    amp: bool = True
    # 'auto'|'bf16'|'fp16' (auto: bf16-capable cards like H100 use bf16, otherwise fp16)
    amp_dtype: str = "auto"
    # torch.compile (big speedup on H100; first compile is slower, hyper-conn may cause graph breaks)
    compile: bool = False
    flash_attn: bool = True        # joint attention uses flash (CUDA only)
    tf32: bool = True              # enable TF32 matmul on Ampere+/H100 (free speedup)

    # io / relay
    ckpt_out: str = "ckpt.pt"
    resume: str = ""
    save_every: int = 2000         # every N steps: update the rolling ckpt_out + save a non-overwriting step snapshot
    snapshot_dir: str = ""         # folder for step snapshots (empty = snapshots/ next to ckpt_out)
    log_every: int = 50
    seed: int = 0
    num_workers: int = 2


# ----------------------------------------------------------------------------- #
# text encoders
# ----------------------------------------------------------------------------- #
class CLIPTextTower(nn.Module):
    """Frozen CLIP text encoder → token-level embeddings + mask."""

    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel
        self.tok = CLIPTokenizer.from_pretrained(cfg.clip_name)
        self.model = CLIPTextModel.from_pretrained(cfg.clip_name).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_text_len = cfg.max_text_len
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, prompts: list[str], device):
        batch = self.tok(
            prompts, padding="max_length", truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        ).to(device)
        out = self.model(**batch).last_hidden_state          # (B, L, D)
        mask = batch["attention_mask"].bool()                # (B, L)
        return out, mask


class DummyTextTower(nn.Module):
    """Deterministic offline text 'encoder' for smoke tests (no downloads)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.dim = cfg.dim_text
        self.max_text_len = cfg.max_text_len

    @torch.no_grad()
    def forward(self, prompts: list[str], device):
        B, L, D = len(prompts), self.max_text_len, self.dim
        embs = torch.zeros(B, L, D, device=device)
        masks = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i, p in enumerate(prompts):
            words = (p.split() or ["<empty>"])[:L]
            for j, w in enumerate(words):
                g = torch.Generator(device="cpu").manual_seed(
                    hash(w) % (2**31))
                embs[i, j] = torch.randn(D, generator=g).to(device)
                masks[i, j] = True
        return embs, masks


def build_text_tower(cfg: Config):
    return CLIPTextTower(cfg) if cfg.text_encoder == "clip" else DummyTextTower(cfg)


# ----------------------------------------------------------------------------- #
# datasets
# ----------------------------------------------------------------------------- #
class HumanML3DDataset(Dataset):
    """
    Standard HumanML3D layout:
      {root}/new_joint_vecs/{name}.npy   (T, 263)
      {root}/texts/{name}.txt            lines: 'caption#tokens#start#end'
      {root}/Mean.npy  {root}/Std.npy    (263,)
      {root}/train.txt                   list of sample names
    """

    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.vec_dir = os.path.join(cfg.data_root, "new_joint_vecs")
        self.text_dir = os.path.join(cfg.data_root, "texts")
        self.mean = np.load(os.path.join(
            cfg.data_root, "Mean.npy")).astype(np.float32)
        self.std = np.load(os.path.join(
            cfg.data_root, "Std.npy")).astype(np.float32)
        with open(os.path.join(cfg.data_root, f"{split}.txt")) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        # keep only names that actually have both motion + text
        self.names = [n for n in names
                      if os.path.exists(os.path.join(self.vec_dir, n + ".npy"))
                      and os.path.exists(os.path.join(self.text_dir, n + ".txt"))]
        assert self.names, f"no samples found under {cfg.data_root}"

    def __len__(self):
        return len(self.names)

    def _load_caption(self, name):
        with open(os.path.join(self.text_dir, name + ".txt")) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        line = random.choice(lines)
        return line.split("#")[0].strip()

    def __getitem__(self, idx):
        name = self.names[idx]
        m = np.load(os.path.join(self.vec_dir, name + ".npy")
                    ).astype(np.float32)  # (T,263)
        m = (m - self.mean) / (self.std + 1e-8)
        T = m.shape[0]
        L = self.cfg.max_motion_len
        if T > L:                                  # random crop
            s = random.randint(0, T - L)
            m = m[s:s + L]
            T = L
        motion = np.zeros((L, self.cfg.motion_dim), dtype=np.float32)
        motion[:T] = m
        mask = np.zeros((L,), dtype=bool)
        mask[:T] = True
        return {
            "motion": torch.from_numpy(motion),
            "mask": torch.from_numpy(mask),
            "text": self._load_caption(name),
        }


class SyntheticMotionDataset(Dataset):
    """Random motion + canned captions, for offline smoke-testing the pipeline."""
    PROMPTS = ["a person walks forward", "someone jumps then turns around",
               "a person sits down slowly", "a man waves his right hand",
               "a person runs in a circle"]

    def __init__(self, cfg: Config, n: int = 256):
        self.cfg, self.n = cfg, n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        L = self.cfg.max_motion_len
        T = random.randint(L // 2, L)
        motion = np.zeros((L, self.cfg.motion_dim), dtype=np.float32)
        motion[:T] = np.random.randn(
            T, self.cfg.motion_dim).astype(np.float32) * 0.5
        mask = np.zeros((L,), dtype=bool)
        mask[:T] = True
        return {
            "motion": torch.from_numpy(motion),
            "mask": torch.from_numpy(mask),
            "text": random.choice(self.PROMPTS),
        }


# ----------------------------------------------------------------------------- #
# building blocks
# ----------------------------------------------------------------------------- #
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0,1] → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half,
                                        device=t.device).float() / max(half - 1, 1)
    )
    args = t[:, None].float() * 1000.0 * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, dim_cond: int):
        super().__init__()
        self.dim_cond = dim_cond
        self.net = nn.Sequential(
            nn.Linear(dim_cond, dim_cond * 4), nn.SiLU(),
            nn.Linear(dim_cond * 4, dim_cond),
        )

    def forward(self, t):
        return self.net(sinusoidal_embedding(t, self.dim_cond))


class StandInMotionEncoder(nn.Module):
    """
    Frozen stand-in target for REPA so the code path runs end-to-end.
    >>> REPLACE THIS with a pretrained TMR / MotionCLIP motion encoder. <<<
    Must return a global motion embedding (B, repa_dim).
    """

    def __init__(self, motion_dim: int, repa_dim: int):
        super().__init__()
        self.gru = nn.GRU(motion_dim, repa_dim, batch_first=True)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, motion, mask):
        lengths = mask.sum(-1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            motion, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return F.normalize(h[-1], dim=-1)           # (B, repa_dim)


# ----------------------------------------------------------------------------- #
# the model
# ----------------------------------------------------------------------------- #
class MotionMMDiT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.motion_in = nn.Linear(cfg.motion_dim, cfg.dim_motion)
        self.motion_out = nn.Linear(cfg.dim_motion, cfg.motion_dim)
        self.pos_emb = nn.Parameter(torch.randn(
            1, cfg.max_motion_len, cfg.dim_motion) * 0.02)
        self.time_mlp = TimeMLP(cfg.dim_cond)

        # learned null text for classifier-free guidance
        self.null_text = nn.Parameter(torch.randn(1, 1, cfg.dim_text) * 0.02)

        self.mmdit = MMDiT(
            depth=cfg.depth,
            dim_modalities=(cfg.dim_text, cfg.dim_motion),
            dim_cond=cfg.dim_cond,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            qk_rmsnorm=cfg.qk_rmsnorm,
            flash_attn=cfg.flash_attn,
            num_residual_streams=cfg.num_residual_streams,
        )

        # ---- REPA hook (optional) ----
        self.repa_feat = None
        if cfg.repa:
            assert 0 <= cfg.repa_layer < cfg.depth, (
                f"--repa_layer={cfg.repa_layer} is out of range, must be in [0, depth={cfg.depth})")
            self.repa_proj = nn.Sequential(
                nn.Linear(cfg.dim_motion, cfg.dim_motion), nn.SiLU(),
                nn.Linear(cfg.dim_motion, cfg.repa_dim),
            )

            def _hook(_m, _inp, out):
                # block out = [text(B,Lt,S,D), motion(B,Lm,S,Dm)]; reduce residual streams
                motion_hidden = out[1].mean(
                    dim=2)         # (B, Lm, dim_motion)
                self.repa_feat = motion_hidden
            self.mmdit.blocks[cfg.repa_layer].register_forward_hook(_hook)

    def forward(self, noised_motion, t, text_tokens, text_mask, motion_mask):
        x = self.motion_in(noised_motion) + \
            self.pos_emb[:, : noised_motion.size(1)]
        time_cond = self.time_mlp(t)
        out_text, out_motion = self.mmdit(
            modality_tokens=(text_tokens, x),
            modality_masks=(text_mask, motion_mask),
            time_cond=time_cond,
        )
        # predicted velocity (B,Lm,263)
        v = self.motion_out(out_motion)
        return v

    def repa_loss(self, target_emb, motion_mask):
        """cosine alignment of pooled projected hidden vs frozen encoder embedding."""
        if self.repa_feat is None:
            return torch.zeros((), device=target_emb.device)
        proj = self.repa_proj(self.repa_feat)              # (B, Lm, repa_dim)
        m = motion_mask.unsqueeze(-1).float()
        # masked mean → (B, repa_dim)
        pooled = (proj * m).sum(1) / m.sum(1).clamp(min=1)
        pooled = F.normalize(pooled, dim=-1)
        return (1.0 - (pooled * target_emb).sum(-1)).mean()


# ----------------------------------------------------------------------------- #
# flow matching
# ----------------------------------------------------------------------------- #
def sample_t(bsz, device, logit_normal: bool):
    if logit_normal:
        return torch.sigmoid(torch.randn(bsz, device=device))
    return torch.rand(bsz, device=device)


def fm_loss(model, batch, text_tower, cfg, device):
    motion = batch["motion"].to(device)                    # (B,L,263)
    mmask = batch["mask"].to(device)                       # (B,L)
    prompts = batch["text"]

    text_tokens, text_mask = text_tower(prompts, device)

    # classifier-free guidance: randomly drop text → learned null token
    drop = torch.rand(motion.size(0), device=device) < cfg.p_uncond
    if drop.any():
        null = model.null_text.expand(motion.size(0), -1, -1)
        text_tokens = text_tokens.clone()
        text_mask = text_mask.clone()
        # pad/truncate null to match text length is unnecessary: replace whole seq
        # build a fresh masked tensor where dropped rows use the null token only
        null_full = torch.zeros_like(text_tokens)
        null_full[:, :1] = null
        null_mask = torch.zeros_like(text_mask)
        null_mask[:, 0] = True
        text_tokens = torch.where(drop[:, None, None], null_full, text_tokens)
        text_mask = torch.where(drop[:, None], null_mask, text_mask)

    # SiT linear interpolant: x0=data, x1=noise, x_t=(1-t)x0 + t x1, v* = x1 - x0
    t = sample_t(motion.size(0), device, cfg.logit_normal_t)
    noise = torch.randn_like(motion)
    tt = t[:, None, None]
    x_t = (1 - tt) * motion + tt * noise
    v_target = noise - motion

    v_pred = model(x_t, t, text_tokens, text_mask, mmask)

    m = mmask.unsqueeze(-1).float()
    loss = (((v_pred - v_target) ** 2) * m).sum() / \
        m.sum().clamp(min=1) / motion.size(-1)
    return loss, motion, mmask


# ----------------------------------------------------------------------------- #
# sampling (Euler ODE, noise t=1 → data t=0)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def sample(model, text_tower, cfg, prompts, device, steps=50, cfg_scale=4.0, length=None):
    model.eval()
    B = len(prompts)
    L = length or cfg.max_motion_len
    mmask = torch.ones(B, L, dtype=torch.bool, device=device)

    text_tokens, text_mask = text_tower(prompts, device)
    null = model.null_text.expand(B, -1, -1)
    null_tokens = torch.zeros_like(text_tokens)
    null_tokens[:, :1] = null
    null_mask = torch.zeros_like(text_mask)
    null_mask[:, 0] = True

    # start from noise (t=1)
    x = torch.randn(B, L, cfg.motion_dim, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(B)
        dt = (ts[i] - ts[i + 1]).item()                    # positive
        v_c = model(x, t, text_tokens, text_mask, mmask)
        if cfg_scale != 1.0:
            v_u = model(x, t, null_tokens, null_mask, mmask)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        x = x - dt * v                                     # Euler step toward data
    model.train()
    # (B,L,263) normalized space
    return x


# ----------------------------------------------------------------------------- #
# EMA + checkpoint relay
# ----------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def save_ckpt(path, model, opt, ema, step, cfg):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt else None,
        "ema": ema.shadow if ema else None,
        "step": step,
        "cfg": asdict(cfg),
    }, path)


def load_ckpt(path, model, opt=None, ema=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    if opt and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if ema and ck.get("ema"):
        ema.shadow = ck["ema"]
    return ck.get("step", 0)


# ----------------------------------------------------------------------------- #
# train
# ----------------------------------------------------------------------------- #
def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * step / max(cfg.warmup, 1)
    return cfg.lr


def pick_device():
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")


def setup_backend(cfg: Config, device: str):
    """H100/Ampere-friendly settings: TF32 matmul (free speedup) + flash attention resolution (only meaningful on CUDA)."""
    if device == "cuda" and cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    cfg.flash_attn = bool(cfg.flash_attn and device == "cuda")


def resolve_amp(cfg: Config, device: str):
    """Return (enabled, dtype, use_scaler). bf16 needs no GradScaler; fp16 does."""
    if not (cfg.amp and device == "cuda"):
        return False, torch.float16, False
    if cfg.amp_dtype == "bf16":
        dtype = torch.bfloat16
    elif cfg.amp_dtype == "fp16":
        dtype = torch.float16
    else:                                  # auto: bf16-capable cards (e.g. H100) use bf16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return True, dtype, (dtype == torch.float16)


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = pick_device()
    setup_backend(cfg, device)
    print(f"[device] {device}")

    text_tower = build_text_tower(cfg).to(device)
    cfg.dim_text = text_tower.dim          # sync text width to encoder

    ds = (HumanML3DDataset(cfg, "train") if cfg.data_root
          else SyntheticMotionDataset(cfg))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True,
                    pin_memory=(device == "cuda"),
                    persistent_workers=(cfg.num_workers > 0))
    assert len(dl) > 0, (
        f"DataLoader is empty: batch_size={cfg.batch_size} exceeds the available sample count {len(ds)} "
        f"(drop_last=True drops the tail smaller than one batch). Please reduce --batch_size.")
    steps_per_epoch = max(1, len(dl))

    # always use this for EMA / saving / loading / clip / .train()
    model = MotionMMDiT(cfg).to(device)
    repa_target = StandInMotionEncoder(
        cfg.motion_dim, cfg.repa_dim).to(device) if cfg.repa else None
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    ema = EMA(model, cfg.ema_decay)

    amp_enabled, amp_dtype, use_scaler = resolve_amp(cfg, device)
    scaler = GradScaler(enabled=use_scaler)

    start = 0
    if cfg.resume and os.path.exists(cfg.resume):
        start = load_ckpt(cfg.resume, model, opt, ema, map_location=device)
        print(f"[resume] from {cfg.resume} @ step {start}")

    # used for forward; after compile it's a wrapped callable, while other ops still use model
    fwd = model
    if cfg.compile and device == "cuda":
        fwd = torch.compile(model)
        print("[compile] torch.compile on (first step is slower; drop --compile if it errors)")

    snap_dir = cfg.snapshot_dir or os.path.join(
        os.path.dirname(os.path.abspath(cfg.ckpt_out)), "snapshots")

    n_params = sum(p.numel()
                   for p in model.parameters() if p.requires_grad) / 1e6
    amp_note = ("bf16" if amp_dtype ==
                torch.bfloat16 else "fp16") if amp_enabled else "off"
    print(f"[model] {n_params:.1f}M trainable | text_dim={cfg.dim_text} | "
          f"steps/epoch={steps_per_epoch}")
    print(f"[perf]  amp={amp_note} | flash={cfg.flash_attn} | "
          f"tf32={cfg.tf32 and device == 'cuda'} | compile={cfg.compile and device == 'cuda'}")

    model.train()
    step = start
    epoch = start // steps_per_epoch
    while step < cfg.steps:
        epoch += 1
        running, n_batches = 0.0, 0
        for batch in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.split(":")[0],
                          dtype=amp_dtype, enabled=amp_enabled):
                loss, motion, mmask = fm_loss(
                    fwd, batch, text_tower, cfg, device)
                total = loss
                repa_val = torch.zeros((), device=device)
                if repa_target is not None:
                    with torch.no_grad():
                        tgt = repa_target(motion, mmask)       # (B, repa_dim)
                    repa_val = model.repa_loss(tgt, mmask)
                    total = total + cfg.repa_weight * repa_val

            # with bf16/off the scaler is enabled=False → scale/unscale/update are all no-ops, step equals opt.step
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            running += loss.item()
            n_batches += 1

            if step % cfg.log_every == 0:
                msg = f"step {step:>7} | ep {epoch} | fm {loss.item():.4f}"
                if cfg.repa:
                    msg += f" | repa {repa_val.item():.4f}"
                msg += f" | lr {opt.param_groups[0]['lr']:.2e}"
                print(msg, flush=True)

            # every N steps: update the rolling ckpt_out (latest, convenient for resume / preempt safety)
            # + save a non-overwriting step snapshot (milestone; copy bytes to save a serialization)
            if cfg.save_every and step > start and step % cfg.save_every == 0:
                save_ckpt(cfg.ckpt_out, model, opt, ema, step, cfg)
                os.makedirs(snap_dir, exist_ok=True)
                snap = os.path.join(snap_dir, f"ckpt_step{step:06d}.pt")
                shutil.copyfile(cfg.ckpt_out, snap)
                print(
                    f"[ckpt] step {step}: rolling → {cfg.ckpt_out} + snapshot → {snap}", flush=True)

            step += 1
            if step >= cfg.steps:
                break

        # ---- end of epoch: print average loss (saving is step-based, see above) ----
        avg = running / max(1, n_batches)
        print(f"[epoch {epoch:>4}] avg_fm_loss={avg:.4f} | "
              f"steps_done={step} | lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    save_ckpt(cfg.ckpt_out, model, opt, ema, step, cfg)
    os.makedirs(snap_dir, exist_ok=True)
    final_snap = os.path.join(snap_dir, f"ckpt_step{step:06d}.pt")
    shutil.copyfile(cfg.ckpt_out, final_snap)
    print(
        f"[done] final ckpt → {cfg.ckpt_out} + snapshot → {final_snap} @ step {step}")


# ----------------------------------------------------------------------------- #
# sample CLI
# ----------------------------------------------------------------------------- #
def run_sample(cfg: Config, prompt: str, steps: int, cfg_scale: float,
               out: str, use_ema: bool):
    device = pick_device()
    setup_backend(cfg, device)
    text_tower = build_text_tower(cfg).to(device)
    cfg.dim_text = text_tower.dim
    model = MotionMMDiT(cfg).to(device)

    assert cfg.resume and os.path.exists(
        cfg.resume), "need --resume <ckpt> to sample"
    ck = torch.load(cfg.resume, map_location=device)
    state = ck["ema"] if (use_ema and ck.get("ema")) else ck["model"]
    model.load_state_dict(state)
    print(
        f"[sample] loaded {'EMA' if use_ema and ck.get('ema') else 'model'} from {cfg.resume}")

    x = sample(model, text_tower, cfg, [prompt], device,
               steps=steps, cfg_scale=cfg_scale)
    arr = x[0].float().cpu().numpy()                       # (L,263) normalized
    # NOTE: de-normalize with the dataset Mean/Std before feeding a renderer:
    #   motion = arr * (Std + 1e-8) + Mean
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, arr)
    print(f"[sample] saved normalized motion {arr.shape} → {out}")


# ----------------------------------------------------------------------------- #
# summary CLI (torchinfo: inspect MMDiT architecture and parameter count, consistent with summary.py in mmdit-*-test)
# ----------------------------------------------------------------------------- #
def run_summary(cfg: Config):
    from torchinfo import summary

    device = pick_device()
    cfg.flash_attn = False                 # only inspecting architecture/params; flash doesn't affect layer structure or param count
    model = MotionMMDiT(cfg).to(device).eval()

    if cfg.resume and os.path.exists(cfg.resume):
        ck = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"[summary] loaded weights from {cfg.resume}\n")
    else:
        print("[summary] no weights loaded, showing architecture (layer structure and param count are identical)\n")

    # forward needs 5 inputs; use dummy (don't load CLIP / motion encoder, just inspect the MMDiT body)
    B, L, Lt = 1, cfg.max_motion_len, cfg.max_text_len
    dummy = (
        torch.randn(B, L, cfg.motion_dim, device=device),       # noised_motion
        torch.rand(B, device=device),                           # t
        torch.randn(B, Lt, cfg.dim_text, device=device),        # text_tokens
        torch.ones(B, Lt, dtype=torch.bool, device=device),     # text_mask
        torch.ones(B, L, dtype=torch.bool, device=device),      # motion_mask
    )
    summary(
        model,
        input_data=dummy,
        depth=4,
        col_names=("input_size", "output_size", "num_params"),
        row_settings=("var_names",),
    )


# ----------------------------------------------------------------------------- #
# argparse
# ----------------------------------------------------------------------------- #
def add_common(p):
    p.add_argument("--data_root", default="")
    p.add_argument("--text_encoder", default="clip", choices=["clip", "dummy"])
    p.add_argument("--clip_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--dim_motion", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--dim_cond", type=int, default=256)
    p.add_argument("--max_motion_len", type=int, default=196)
    p.add_argument("--num_residual_streams", type=int, default=4)
    p.add_argument("--repa", action="store_true")
    p.add_argument("--repa_layer", type=int, default=4)
    p.add_argument("--repa_weight", type=float, default=0.5)
    p.add_argument("--resume", default="")


def cfg_from_args(a) -> Config:
    c = Config()
    for k, v in vars(a).items():
        if hasattr(c, k) and v is not None:
            setattr(c, k, v)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    add_common(pt)
    pt.add_argument("--batch_size", type=int, default=64)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--steps", type=int, default=200_000)
    pt.add_argument("--save_every", type=int, default=2000)
    pt.add_argument("--snapshot_dir", default="")
    pt.add_argument("--log_every", type=int, default=50)
    pt.add_argument("--num_workers", type=int, default=2)
    pt.add_argument("--ckpt_out", default="ckpt.pt")
    pt.add_argument("--no_amp", action="store_true")
    pt.add_argument("--amp_dtype", default="auto",
                    choices=["auto", "bf16", "fp16"])
    pt.add_argument("--compile", action="store_true")
    pt.add_argument("--no_flash", action="store_true")
    pt.add_argument("--no_tf32", action="store_true")

    ps = sub.add_parser("sample")
    add_common(ps)
    ps.add_argument("--prompt", required=True)
    ps.add_argument("--steps", type=int, default=50)
    ps.add_argument("--cfg", type=float, default=4.0)
    ps.add_argument("--out", default="sample.npy")
    ps.add_argument("--use_ema", action="store_true")

    psum = sub.add_parser("summary")
    add_common(psum)

    a = ap.parse_args()
    cfg = cfg_from_args(a)

    if a.cmd == "train":
        if getattr(a, "no_amp", False):
            cfg.amp = False
        if getattr(a, "no_flash", False):
            cfg.flash_attn = False
        if getattr(a, "no_tf32", False):
            cfg.tf32 = False
        train(cfg)
    elif a.cmd == "sample":
        run_sample(cfg, a.prompt, a.steps, a.cfg, a.out, a.use_ema)
    else:  # summary
        run_summary(cfg)


if __name__ == "__main__":
    main()
