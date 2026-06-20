# mmdit-motion

## Setup

```bash
git clone https://github.com/chihhsiangfu/mmdit-test
```

```bash
cd mmdit-test && uv sync
```

## Data preparation

- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)

  ```
  <data_root>/
    new_joint_vecs/<name>.npy   # (T, 263) motion features
    texts/<name>.txt            # each line: caption#tokens#start#end
    Mean.npy  Std.npy           # (263,) for normalization
    train.txt                   # list of sample names
  ```

## Arguments

`train` / `sample` / `summary` all accept the **Shares** arguments below; each command then adds the arguments in its own table (`summary` adds none). **Model-structure args** (`dim_motion`, `depth`, `heads`, `dim_head`, `dim_cond`, `max_motion_len`, `num_residual_streams`, `repa`) define the network and **must match the checkpoint** when you `--resume`, `sample`, or load weights in `summary` — the model is rebuilt from the CLI args, not restored from the checkpoint. `flag` args are off by default; pass them to turn the behavior on.

### Shares

| arg                  | type  | default                        | description                                                                                                        |
| -------------------- | ----- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| data_root            | str   | `""`                           | HumanML3D root dir; empty → synthetic random-motion data (offline smoke test)                                      |
| text_encoder         | str   | `clip`                         | `clip` (frozen CLIP) or `dummy` (offline, no download)                                                             |
| clip_name            | str   | `openai/clip-vit-base-patch32` | HF id of the CLIP text encoder (only used with `--text_encoder clip`)                                              |
| dim_motion           | int   | 512                            | motion-stream hidden width                                                                                         |
| depth                | int   | 8                              | number of MMDiT blocks                                                                                             |
| heads                | int   | 8                              | attention heads                                                                                                    |
| dim_head             | int   | 64                             | dim per attention head                                                                                             |
| dim_cond             | int   | 256                            | timestep-conditioning width                                                                                        |
| max_motion_len       | int   | 196                            | max motion frames (sequence length)                                                                                |
| num_residual_streams | int   | 4                              | MMDiT hyper-connection residual streams                                                                            |
| repa                 | flag  | off                            | enable the optional REPA alignment auxiliary loss                                                                  |
| repa_layer           | int   | 4                              | MMDiT block whose motion hidden is aligned (0-indexed, `< depth`)                                                  |
| repa_weight          | float | 0.5                            | weight of the REPA loss term                                                                                       |
| resume               | str   | `""`                           | checkpoint to load: `train` resumes model/opt/EMA/step (missing → fresh); `sample` requires it; `summary` optional |

> Not every shared arg affects every command: `data_root` → `train` only; `text_encoder` / `clip_name` → `train` and `sample` (`summary` uses no text encoder); `repa_layer` / `repa_weight` → `train` only.

### Train

| arg          | type  | default   | description                                                                                        |
| ------------ | ----- | --------- | -------------------------------------------------------------------------------------------------- |
| batch_size   | int   | 64        | training batch size (DataLoader uses `drop_last`, so it must be ≤ the sample count)                |
| lr           | float | 1e-4      | AdamW learning rate (betas 0.9/0.95); linear warmup over the first 1000 steps                      |
| steps        | int   | 200000    | total step ceiling (`while step < steps`); when resuming, set higher than the already-trained step |
| save_every   | int   | 2000      | every N steps: refresh the rolling `ckpt_out` and write a non-overwriting snapshot                 |
| snapshot_dir | str   | `""`      | folder for step snapshots (empty → `snapshots/` next to `ckpt_out`)                                |
| log_every    | int   | 50        | print a training log line every N steps                                                            |
| num_workers  | int   | 2         | DataLoader worker processes                                                                        |
| ckpt_out     | str   | `ckpt.pt` | path of the rolling (latest) checkpoint                                                            |
| no_amp       | flag  | off       | disable AMP mixed precision (AMP is active on CUDA only)                                           |
| amp_dtype    | str   | `auto`    | AMP dtype — `auto` (bf16 on bf16-capable GPUs e.g. H100, else fp16) / `bf16` / `fp16`              |
| compile      | flag  | off       | enable `torch.compile` (CUDA only; large H100 speedup, slower first step)                          |
| no_flash     | flag  | off       | disable flash attention (flash is CUDA-only regardless)                                            |
| no_tf32      | flag  | off       | disable TF32 matmul (applies only on CUDA Ampere+/H100)                                            |

### Sample

| arg     | type  | default      | description                                                |
| ------- | ----- | ------------ | ---------------------------------------------------------- |
| prompt  | str   | _(required)_ | text description to generate motion from                   |
| steps   | int   | 50           | number of Euler ODE sampling steps                         |
| cfg     | float | 4.0          | classifier-free guidance scale (`1.0` = disabled)          |
| shift   | float | 3.0          | SD3 timestep shift on the sampling schedule (`1.0` = off)  |
| out     | str   | `sample.npy` | output path for the generated **normalized** motion `.npy` |
| use_ema | flag  | off          | load EMA-averaged weights instead of the raw model         |

### Summary

_No command-specific arguments — `summary` uses only the Shares arguments above._
