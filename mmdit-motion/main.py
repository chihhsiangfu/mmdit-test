#!/usr/bin/env python
"""
text-to-motion with lucidrains/mmdit — flow matching (SiT-style) + optional REPA.

A single-file download / train / sample / summary script. Backbone: lucidrains/mmdit's generalized
MMDiT (dual-stream joint attention). Method:

  * two modalities: text (CLIP) + motion (HumanML3D 263-dim); token-level text goes to
    joint attention, pooled text is added to the adaLN timestep modulation (SD3-style)
  * SiT linear interpolant + velocity loss (rectified flow)
  * logit-normal timestep sampling + SD3 timestep shift at sampling + classifier-free guidance
  * optional REPA alignment; --resume auto-resumes (model + optimizer + EMA + step)

Get real data (HumanML3D from a HuggingFace mirror — no AMASS pipeline):
  uv run mmdit-motion/main.py download --data_root data/HumanML3D

Quick smoke test (no data, no download):
  uv run mmdit-motion/main.py train --text_encoder dummy --steps 50 --ckpt_out runs/ckpt.pt

See README.md for the full argument reference and the download / train / sample / summary commands.
"""

import os
import math
import shutil
import argparse
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from mmdit.mmdit_generalized_pytorch import MMDiT
from ema_pytorch import EMA


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
    # ema-pytorch: delay EMA start (model is noisy early) + update cadence
    ema_update_after_step: int = 100
    ema_update_every: int = 10
    amp: bool = True
    # 'auto'|'bf16'|'fp16' (auto: bf16-capable cards like H100 use bf16, otherwise fp16)
    amp_dtype: str = "auto"
    # torch.compile (big speedup on H100; first compile is slower, hyper-conn may cause graph breaks)
    compile: bool = False
    flash_attn: bool = True        # joint attention uses flash (CUDA only)
    # enable TF32 matmul on Ampere+/H100 (free speedup)
    tf32: bool = True

    # io / relay
    ckpt_out: str = "ckpt.pt"
    resume: str = ""
    # every N steps: update the rolling ckpt_out + save a non-overwriting step snapshot
    save_every: int = 2000
    # folder for step snapshots (empty = snapshots/ next to ckpt_out)
    snapshot_dir: str = ""
    log_every: int = 50
    seed: int = 0
    num_workers: int = 2


# ----------------------------------------------------------------------------- #
# text encoders
# ----------------------------------------------------------------------------- #
class CLIPTextTower(nn.Module):
    """Frozen CLIP text encoder → token-level embeddings + mask + pooled (sentence) vector."""

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
        out = self.model(**batch)
        tokens = out.last_hidden_state                       # (B, L, D) token-level
        pooled = out.pooler_output                           # (B, D) sentence-level (EOS token)
        mask = batch["attention_mask"].bool()                # (B, L)
        return tokens, mask, pooled


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
        m = masks.unsqueeze(-1).float()
        pooled = (embs * m).sum(1) / m.sum(1).clamp(min=1)   # (B, D) masked mean
        return embs, masks, pooled


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
        # learned null pooled-text (the unconditional pooled vector for CFG)
        self.null_pooled = nn.Parameter(torch.randn(1, cfg.dim_text) * 0.02)
        # project pooled text → dim_cond, added to the timestep modulation (SD3-style)
        self.text_pool_proj = nn.Linear(cfg.dim_text, cfg.dim_cond)

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

    def forward(self, noised_motion, t, text_tokens, text_mask, motion_mask, text_pooled):
        x = self.motion_in(noised_motion) + \
            self.pos_emb[:, : noised_motion.size(1)]
        # adaLN conditioning = timestep embedding + projected pooled text (SD3-style)
        time_cond = self.time_mlp(t) + self.text_pool_proj(text_pooled)
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

    text_tokens, text_mask, text_pooled = text_tower(prompts, device)

    # classifier-free guidance: randomly drop text → learned null token + null pooled
    drop = torch.rand(motion.size(0), device=device) < cfg.p_uncond
    if drop.any():
        null = model.null_text.expand(motion.size(0), -1, -1)
        text_tokens = text_tokens.clone()
        text_mask = text_mask.clone()
        # dropped rows: replace the whole text seq with the null token (only position 0 unmasked)
        null_full = torch.zeros_like(text_tokens)
        null_full[:, :1] = null
        null_mask = torch.zeros_like(text_mask)
        null_mask[:, 0] = True
        text_tokens = torch.where(drop[:, None, None], null_full, text_tokens)
        text_mask = torch.where(drop[:, None], null_mask, text_mask)
        null_pooled = model.null_pooled.expand(motion.size(0), -1)
        text_pooled = torch.where(drop[:, None], null_pooled, text_pooled)

    # SiT linear interpolant: x0=data, x1=noise, x_t=(1-t)x0 + t x1, v* = x1 - x0
    t = sample_t(motion.size(0), device, cfg.logit_normal_t)
    noise = torch.randn_like(motion)
    tt = t[:, None, None]
    x_t = (1 - tt) * motion + tt * noise
    v_target = noise - motion

    v_pred = model(x_t, t, text_tokens, text_mask, mmask, text_pooled)

    m = mmask.unsqueeze(-1).float()
    loss = (((v_pred - v_target) ** 2) * m).sum() / \
        m.sum().clamp(min=1) / motion.size(-1)
    return loss, motion, mmask


# ----------------------------------------------------------------------------- #
# sampling (Euler ODE, noise t=1 → data t=0, SD3 timestep shift)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def sample(model, text_tower, cfg, prompts, device, steps=50, cfg_scale=4.0, length=None, shift=3.0):
    model.eval()
    B = len(prompts)
    L = length or cfg.max_motion_len
    mmask = torch.ones(B, L, dtype=torch.bool, device=device)

    text_tokens, text_mask, text_pooled = text_tower(prompts, device)
    null = model.null_text.expand(B, -1, -1)
    null_tokens = torch.zeros_like(text_tokens)
    null_tokens[:, :1] = null
    null_mask = torch.zeros_like(text_mask)
    null_mask[:, 0] = True
    null_pooled = model.null_pooled.expand(B, -1)

    # start from noise (t=1)
    x = torch.randn(B, L, cfg.motion_dim, device=device)
    # SD3 timestep shift: warp the schedule toward higher noise (shift>1; shift=1 disables)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    if shift != 1.0:
        ts = shift * ts / (1 + (shift - 1) * ts)
    for i in range(steps):
        t = ts[i].expand(B)
        dt = (ts[i] - ts[i + 1]).item()                    # positive
        v_c = model(x, t, text_tokens, text_mask, mmask, text_pooled)
        if cfg_scale != 1.0:
            v_u = model(x, t, null_tokens, null_mask, mmask, null_pooled)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        x = x - dt * v                                     # Euler step toward data
    model.train()
    # (B,L,263) normalized space
    return x


# ----------------------------------------------------------------------------- #
# EMA helpers + checkpoint relay
# (EMA itself is ema-pytorch: EMA(model, beta=...); call ema.update() each step)
# ----------------------------------------------------------------------------- #
def load_ema_into(model: nn.Module, ema_state: dict):
    """Copy EMA weights from an ema-pytorch wrapper state_dict (keys prefixed
    'ema_model.') into `model` in place."""
    prefix = "ema_model."
    weights = {k[len(prefix):]: v
               for k, v in ema_state.items() if k.startswith(prefix)}
    model.load_state_dict(weights)


def save_ckpt(path, model, opt, ema, step, cfg):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt else None,
        "ema": ema.state_dict() if ema is not None else None,
        "step": step,
        "cfg": asdict(cfg),
    }, path)


def load_ckpt(path, model, opt=None, ema=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if ema is not None and ck.get("ema") is not None:
        ema.load_state_dict(ck["ema"])
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
    """H100/Ampere backend: enable TF32 matmul + resolve flash attention (both CUDA-only)."""
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
    # auto: bf16-capable cards (e.g. H100) use bf16
    else:
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

    model = MotionMMDiT(cfg).to(device)
    repa_target = StandInMotionEncoder(
        cfg.motion_dim, cfg.repa_dim).to(device) if cfg.repa else None
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    ema = EMA(
        model,
        beta=cfg.ema_decay,
        update_after_step=cfg.ema_update_after_step,
        update_every=cfg.ema_update_every,
        include_online_model=False,   # online weights are saved separately under "model"
    )

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
        print(
            "[compile] torch.compile on (first step is slower; drop --compile if it errors)")

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
            ema.update()

            running += loss.item()
            n_batches += 1

            if step % cfg.log_every == 0:
                msg = f"step {step:>7} | ep {epoch} | fm {loss.item():.4f}"
                if cfg.repa:
                    msg += f" | repa {repa_val.item():.4f}"
                msg += f" | lr {opt.param_groups[0]['lr']:.2e}"
                print(msg, flush=True)

            # refresh the rolling ckpt, then copy it to a non-overwriting milestone snapshot
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
               out: str, use_ema: bool, shift: float):
    device = pick_device()
    setup_backend(cfg, device)
    text_tower = build_text_tower(cfg).to(device)
    cfg.dim_text = text_tower.dim
    model = MotionMMDiT(cfg).to(device)

    assert cfg.resume and os.path.exists(
        cfg.resume), "need --resume <ckpt> to sample"
    ck = torch.load(cfg.resume, map_location=device)
    if use_ema and ck.get("ema"):
        load_ema_into(model, ck["ema"])
    else:
        model.load_state_dict(ck["model"])
    print(
        f"[sample] loaded {'EMA' if use_ema and ck.get('ema') else 'model'} from {cfg.resume}")

    x = sample(model, text_tower, cfg, [prompt], device,
               steps=steps, cfg_scale=cfg_scale, shift=shift)
    arr = x[0].float().cpu().numpy()                       # (L,263) normalized
    # NOTE: de-normalize with the dataset Mean/Std before feeding a renderer:
    #   motion = arr * (Std + 1e-8) + Mean
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, arr)
    print(f"[sample] saved normalized motion {arr.shape} → {out}")


# ----------------------------------------------------------------------------- #
# summary CLI (torchinfo: inspect MMDiT architecture and parameter count)
# ----------------------------------------------------------------------------- #
def run_summary(cfg: Config):
    from torchinfo import summary

    device = pick_device()
    # only inspecting architecture/params; flash doesn't affect layer structure or param count
    cfg.flash_attn = False
    model = MotionMMDiT(cfg).to(device).eval()

    if cfg.resume and os.path.exists(cfg.resume):
        ck = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"[summary] loaded weights from {cfg.resume}\n")
    else:
        print("[summary] no weights loaded, showing architecture (layer structure and param count are identical)\n")

    # forward needs 6 inputs; use dummy (don't load CLIP / motion encoder, just inspect the MMDiT body)
    B, L, Lt = 1, cfg.max_motion_len, cfg.max_text_len
    dummy = (
        torch.randn(B, L, cfg.motion_dim, device=device),       # noised_motion
        torch.rand(B, device=device),                           # t
        torch.randn(B, Lt, cfg.dim_text, device=device),        # text_tokens
        torch.ones(B, Lt, dtype=torch.bool, device=device),     # text_mask
        torch.ones(B, L, dtype=torch.bool, device=device),      # motion_mask
        torch.randn(B, cfg.dim_text, device=device),            # text_pooled
    )
    summary(
        model,
        input_data=dummy,
        depth=4,
        col_names=("input_size", "output_size", "num_params"),
        row_settings=("var_names",),
    )


# ----------------------------------------------------------------------------- #
# download CLI (fetch HumanML3D from a HuggingFace mirror → standard data_root layout)
# ----------------------------------------------------------------------------- #
STATS_REPO = "NamYeongCho/HumanML3D"   # tiny canonical Mean.npy / Std.npy live here


def _write_mean_std(data_root):
    """Fetch the canonical 263-dim Mean/Std (tiny) from a mirror; if that fails,
    compute them from the downloaded new_joint_vecs (per-dim over all frames)."""
    try:
        from huggingface_hub import hf_hub_download
        for fn in ("Mean.npy", "Std.npy"):
            src = hf_hub_download(repo_id=STATS_REPO, filename=fn, repo_type="dataset")
            shutil.copyfile(src, os.path.join(data_root, fn))
        print(f"[download] Mean/Std: fetched canonical stats from {STATS_REPO}", flush=True)
        return
    except Exception as e:
        print(f"[download] Mean/Std: canonical fetch failed ({e}); computing from data ...", flush=True)

    vec_dir = os.path.join(data_root, "new_joint_vecs")
    files = [f for f in os.listdir(vec_dir) if f.endswith(".npy")]
    if not files:
        raise RuntimeError(f"no .npy motions under {vec_dir} to compute Mean/Std")
    dim = int(np.load(os.path.join(vec_dir, files[0])).shape[1])
    s = np.zeros(dim, dtype=np.float64)
    ss = np.zeros(dim, dtype=np.float64)
    cnt = 0
    for f in files:
        m = np.load(os.path.join(vec_dir, f)).astype(np.float64)        # (T, D)
        s += m.sum(0); ss += (m * m).sum(0); cnt += m.shape[0]
    mean = s / max(cnt, 1)
    std = np.sqrt(np.clip(ss / max(cnt, 1) - mean ** 2, 1e-12, None))
    np.save(os.path.join(data_root, "Mean.npy"), mean.astype(np.float32))
    np.save(os.path.join(data_root, "Std.npy"), std.astype(np.float32))
    print(f"[download] Mean/Std: computed from {len(files)} motions ({cnt} frames)", flush=True)


def run_download(data_root, hf_dataset, splits, max_per_split):
    from datasets import load_dataset
    vec_dir = os.path.join(data_root, "new_joint_vecs")
    txt_dir = os.path.join(data_root, "texts")
    os.makedirs(vec_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    total = 0
    for split in [s.strip() for s in splits.split(",") if s.strip()]:
        print(f"[download] streaming split '{split}' from {hf_dataset} "
              f"(first rows trigger the download/cache) ...", flush=True)
        ds = load_dataset(hf_dataset, split=split, streaming=True)
        names = []
        for i, row in enumerate(ds):
            if max_per_split and i >= max_per_split:
                break
            if not row["caption"].strip():
                continue
            name = row["meta_data"]["name"]
            motion = np.asarray(row["motion"], dtype=np.float32)        # (T, 263)
            np.save(os.path.join(vec_dir, name + ".npy"), motion)
            with open(os.path.join(txt_dir, name + ".txt"), "w") as f:
                f.write(row["caption"])         # already HumanML3D 'caption#tokens#start#end' lines
            names.append(name)
            if (i + 1) % 500 == 0:
                print(f"[download]   {split}: {i + 1} motions ...", flush=True)
        with open(os.path.join(data_root, f"{split}.txt"), "w") as f:
            f.write("\n".join(names) + "\n")
        total += len(names)
        print(f"[download] split '{split}': {len(names)} motions → {split}.txt", flush=True)

    _write_mean_std(data_root)
    print(f"[download] DONE → {data_root}  ({total} motions).  "
          f"Train with:  --data_root {data_root}", flush=True)
    # datasets' streaming HTTP layer can leave non-daemon threads that stall interpreter
    # exit; everything is written and stdout is flushed, so terminate the CLI promptly.
    os._exit(0)


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
    ps.add_argument("--shift", type=float, default=3.0)
    ps.add_argument("--out", default="sample.npy")
    ps.add_argument("--use_ema", action="store_true")

    psum = sub.add_parser("summary")
    add_common(psum)

    pdl = sub.add_parser("download")
    pdl.add_argument("--data_root", required=True,
                     help="target folder for the standard HumanML3D layout")
    pdl.add_argument("--hf_dataset", default="TeoGchx/HumanML3D",
                     help="HuggingFace datasets repo (263-dim HumanML3D parquet)")
    pdl.add_argument("--splits", default="train,val,test")
    pdl.add_argument("--max_per_split", type=int, default=0,
                     help="0 = all; set e.g. 200 for a quick subset")

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
        run_sample(cfg, a.prompt, a.steps, a.cfg, a.out, a.use_ema, a.shift)
    elif a.cmd == "download":
        run_download(a.data_root, a.hf_dataset, a.splits, a.max_per_split)
    else:  # summary
        run_summary(cfg)


if __name__ == "__main__":
    main()
