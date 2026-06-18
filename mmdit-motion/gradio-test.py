#!/usr/bin/env python
"""Gradio app: load a trained text-to-motion MMDiT, enter a prompt → generate motion → display it as a 3D skeleton animation.

Same purpose as the gradio-test.py in the other mmdit-*-test projects (load a trained model for a demo);
the difference is that what we display here is motion, so the approach is:

  prompt ─T5/CLIP─▶ MMDiT(Rectified Flow ODE) ─▶ (T,263) HumanML3D features
        ─denormalize─▶ recover_from_ric ─▶ (T,22,3) joint xyz trajectories
        ─matplotlib 3D skeleton + FuncAnimation─▶ animated GIF (the browser plays it automatically)

Usage:
    # Train and save a checkpoint with main.py first (defaults to reading runs/ckpt.pt)
    uv run mmdit-motion/gradio-test.py --ckpt runs/ckpt.pt --use_ema
    # To correctly restore the motion scale (meters), add the HumanML3D folder so Mean/Std can be loaded:
    uv run mmdit-motion/gradio-test.py --ckpt runs/ckpt.pt --data_root /path/to/HumanML3D
Then open the browser at http://127.0.0.1:7860

Notes:
  - The checkpoint stores the training-time cfg; here we rebuild the model directly from it, with no need to manually align --dim_motion/--depth…
  - Without --data_root there is no Mean/Std, so the motion still moves but the scale/proportions will be off (the UI notes this).
"""

import os
import argparse
import tempfile
from dataclasses import fields

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")   # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import gradio as gr

# main.py in the same folder (flat import, consistent with main.py itself)
from main import Config, MotionMMDiT, build_text_tower, sample, pick_device, setup_backend


# --------------------------------------------------------------
# HumanML3D 22-joint skeleton (standard t2m kinematic chain, used to draw bones)
# --------------------------------------------------------------
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],        # right leg
    [0, 1, 4, 7, 10],        # left leg
    [0, 3, 6, 9, 12, 15],    # spine → head
    [9, 14, 17, 19, 21],     # right arm
    [9, 13, 16, 18, 20],     # left arm
]
_CHAIN_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


# --------------------------------------------------------------
# 263-dim HumanML3D features → (T,22,3) global joint xyz
# Standard recover_from_ric: first restore the root rotation/translation, then convert the
# rotation-invariant local joint coordinates back to world coordinates. (A minimal rewrite of the official HumanML3D approach)
# --------------------------------------------------------------

def _qinv(q):
    """Quaternion conjugate (inverse of a unit quaternion). q: (...,4) = (w,x,y,z)."""
    mask = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    return q * mask


def _qrot(q, v):
    """Rotate vector v by quaternion q. q: (...,4), v: (...,3) → (...,3)."""
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def _recover_root_rot_pos(data):
    """Recover each frame's root rotation quaternion and world position from the 263-dim features. data: (...,T,263)."""
    rot_vel = data[..., 0]                          # root angular velocity about Y
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)     # cumulative → per-frame angle

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)       # rotation about the Y axis

    # XZ-plane linear velocity (in the root frame) → rotate back to world → accumulate into position; Y uses root height directly
    vel = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    vel[..., 1:, 0] = data[..., :-1, 1]
    vel[..., 1:, 2] = data[..., :-1, 2]
    vel = _qrot(_qinv(r_rot_quat), vel)
    r_pos = torch.cumsum(vel, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=22):
    """(...,T,263) HumanML3D features → (...,T,joints_num,3) world-coordinate joint positions."""
    r_rot_quat, r_pos = _recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]            # ric_data: the other 21 joints
    positions = positions.view(positions.shape[:-1] + (-1, 3))   # (...,T,21,3) root-local
    # rotate back to world coordinates (using the inverse of the root rotation), then add the root's XZ translation
    positions = _qrot(_qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    # put the root joint itself back as (T,1,3), concatenated with the other 21 joints to make 22
    positions = torch.cat([r_pos[..., None, :], positions], dim=-2)
    return positions


# --------------------------------------------------------------
# Mean/Std (for denormalization; stored in the HumanML3D folder)
# --------------------------------------------------------------

def load_stats(data_root):
    """Return (mean, std) np.float32 (263,) for denormalization; return None if not found."""
    if not data_root:
        return None
    mp = os.path.join(data_root, "Mean.npy")
    sp = os.path.join(data_root, "Std.npy")
    if not (os.path.exists(mp) and os.path.exists(sp)):
        return None
    return np.load(mp).astype(np.float32), np.load(sp).astype(np.float32)


# --------------------------------------------------------------
# Load the trained model (rebuild from the cfg stored in the checkpoint, no manual dimension alignment needed)
# --------------------------------------------------------------

def load_model(ckpt_path, use_ema, device):
    ck = torch.load(ckpt_path, map_location=device)
    known = {f.name for f in fields(Config)}
    saved = {k: v for k, v in ck.get("cfg", {}).items() if k in known}
    cfg = Config(**saved)
    cfg.compile = False                 # inference doesn't need torch.compile
    setup_backend(cfg, device)          # flash only takes effect on cuda (auto-off for non-cuda)

    text_tower = build_text_tower(cfg).to(device)
    cfg.dim_text = text_tower.dim       # sync with training: text width follows the encoder
    model = MotionMMDiT(cfg).to(device)

    state = ck["ema"] if (use_ema and ck.get("ema")) else ck["model"]
    model.load_state_dict(state)
    model.eval()
    tag = "EMA" if (use_ema and ck.get("ema")) else "model"
    print(f"loaded {tag} from {ckpt_path}  (step={ck.get('step', '?')}, text={cfg.text_encoder})")
    return model, text_tower, cfg


# --------------------------------------------------------------
# 3D skeleton animation → GIF
# --------------------------------------------------------------

def render_motion_gif(joints, fps=20):
    """joints: (T,22,3) (HumanML3D: Y is up, XZ is the ground) → save as an animated GIF, return the file path."""
    joints = np.asarray(joints, dtype=np.float32)
    T = joints.shape[0]
    pts = joints.reshape(-1, 3)
    mins, maxs = pts.min(0), pts.max(0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max()) / 2.0 + 0.3   # margin; same range on all three axes → correct proportions

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    def draw(f):
        ax.clear()
        ax.set_xlim(center[0] - radius, center[0] + radius)   # X
        ax.set_ylim(center[2] - radius, center[2] + radius)   # Z (depth)
        ax.set_zlim(mins[1], mins[1] + 2 * radius)            # Y (height, feet on the ground)
        ax.view_init(elev=12, azim=-60)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_axis_off()
        p = joints[f]
        # note the axis mapping: matplotlib's "up" is z, so feed (X, Z, Y) to make the vertical axis = HumanML3D's Y
        for chain, c in zip(KINEMATIC_CHAIN, _CHAIN_COLORS):
            ax.plot(p[chain, 0], p[chain, 2], p[chain, 1], "-o", color=c, ms=2.5, lw=2.0)
        ax.set_title(f"frame {f + 1}/{T}", fontsize=9)
        return []   # blit=False, the return value is ignored; an empty artist list satisfies the type

    anim = FuncAnimation(fig, draw, frames=T, interval=1000.0 / max(fps, 1))
    tmp = tempfile.NamedTemporaryFile(prefix="motion_", suffix=".gif", delete=False)
    anim.save(tmp.name, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return tmp.name


# --------------------------------------------------------------
# Generation (callback for the Generate button)
# --------------------------------------------------------------

@torch.no_grad()
def generate_motion(model, text_tower, cfg, device, stats, prompt, cfg_scale, steps, length):
    x = sample(model, text_tower, cfg, [prompt], device,
               steps=int(steps), cfg_scale=float(cfg_scale), length=int(length))
    arr = x[0].float().cpu().numpy()                      # (L, 263) normalized
    if stats is not None:                                 # restore to real scale
        mean, std = stats
        arr = arr * (std + 1e-8) + mean
    joints = recover_from_ric(torch.from_numpy(arr).float(), 22).numpy()   # (L, 22, 3)
    return render_motion_gif(joints, fps=cfg.fps)


# --------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------

def build_demo(model, text_tower, cfg, device, stats):
    note = "" if stats is not None else \
        "\n\n> ⚠️ No `--data_root` (Mean/Std) provided, so motion is shown in normalized space and the scale/proportions will be off."

    def _gen(prompt, cfg_scale, steps, length):
        return generate_motion(model, text_tower, cfg, device, stats,
                               prompt, cfg_scale, steps, length)

    max_len = int(cfg.max_motion_len)
    with gr.Blocks(title="MMDiT · text-to-motion") as demo:
        gr.Markdown(
            "# MMDiT + Rectified Flow — text-to-motion Demo\n"
            "Enter a text prompt → generate HumanML3D motion → 3D skeleton animation." + note)
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a person walks forward then sits down")
                cfg_s = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG scale")
                steps_s = gr.Slider(10, 100, value=50, step=5, label="ODE steps")
                len_s = gr.Slider(min(20, max_len), max_len,
                                  value=min(120, max_len), step=1, label="Frames (length)")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                out = gr.Image(label="3D motion", type="filepath")
        btn.click(fn=_gen, inputs=[prompt, cfg_s, steps_s, len_s], outputs=out)
    return demo


# --------------------------------------------------------------
# main
# --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Gradio demo: load a text-to-motion MMDiT, prompt → 3D skeleton animation.")
    p.add_argument("--ckpt", default="runs/ckpt.pt", help="path to the trained checkpoint")
    p.add_argument("--use_ema", action="store_true", help="load EMA-smoothed weights")
    p.add_argument("--data_root", default="",
                   help="HumanML3D folder (load Mean/Std for denormalization; omit to display in normalized space)")
    p.add_argument("--share", action="store_true", help="gradio public share link")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(
            f"Checkpoint {args.ckpt} not found. Train and save one with main.py first (or specify a path with --ckpt).")

    device = pick_device()
    print(f"[device] {device}")
    model, text_tower, cfg = load_model(args.ckpt, args.use_ema, device)
    stats = load_stats(args.data_root)
    if stats is None:
        print("(Mean/Std not loaded: pass --data_root to restore real scale; currently displaying normalized)")

    demo = build_demo(model, text_tower, cfg, device, stats)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
