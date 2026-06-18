# mmdit-motion

text-to-motion (HumanML3D) + lucidrains/mmdit (generalized MMDiT, dual-stream joint attention) + Rectified Flow. A single file `main.py` provides three subcommands: `train` / `sample` / `summary`.

## Data preparation

- **HumanML3D is not downloaded automatically; you must provide it yourself.** Follow the official process (the `EricGuo5513/HumanML3D` GitHub repo, which requires registering on AMASS first) to produce the standard directory, then point to it with `--data_root`:

  ```
  <data_root>/
    new_joint_vecs/<name>.npy   # (T, 263) motion features
    texts/<name>.txt            # each line: caption#tokens#start#end
    Mean.npy  Std.npy           # (263,) for normalization
    train.txt                   # list of sample names
  ```

  Missing files raise an immediate error (`no samples found under <data_root>`).

- **To run the whole pipeline first (no data, no download)**: use the dummy text encoder + synthetic random motion, and **do not** pass `--data_root`:

  ```bash
  uv run mmdit-motion/main.py train --text_encoder dummy --steps 50 --ckpt_out runs/ckpt.pt
  ```

- Note: `--text_encoder clip` downloads CLIP (`clip-vit-base-patch32`) automatically on first use; only HumanML3D itself needs manual preparation.

## Commands

### Training

```bash
uv run mmdit-motion/main.py train \
  --data_root /path/to/HumanML3D \
  --text_encoder clip \
  --dim_motion 512 --depth 8 \
  --batch_size 64 \
  --steps 200000 \
  --ckpt_out runs/ckpt.pt
```

### H100 optimization

bf16 + flash + tf32 + torch.compile, plus saving an archived snapshot every 5000 steps:

```bash
uv run mmdit-motion/main.py train \
  --data_root /path/to/HumanML3D \
  --text_encoder clip \
  --dim_motion 512 --depth 8 \
  --batch_size 64 \
  --steps 200000 \
  --amp_dtype bf16 \
  --compile \
  --save_every 5000 \
  --ckpt_out runs/ckpt.pt
```

> Same as the "Training" command, just with the added `--amp_dtype bf16 --compile` (H100 speedup). Of these, `--text_encoder clip / --dim_motion 512 / --depth 8 / --batch_size 64 / --steps 200000` are all **defaults**, written out only for clarity; omitting them gives the same result.

> `--save_every N`: every N steps, update the rolling `runs/ckpt.pt` (latest, convenient for resume) plus save a separate non-overwriting archived snapshot `runs/snapshots/ckpt_step000NNN.pt`. Snapshots accumulate, so for long runs set N larger (or use `--snapshot_dir` to choose a folder).

### Resume training (resume / continue training)

Supported. `--resume <ckpt>` restores the **model + optimizer + EMA + step counter** (the full training state, not just the weights), continuing from the last step toward `--steps`.

```bash
uv run mmdit-motion/main.py train \
  --data_root /path/to/HumanML3D \
  --resume runs/ckpt.pt \
  --ckpt_out runs/ckpt.pt \
  --steps 400000
```

```bash
uv run mmdit-motion/main.py train \
  --data_root /path/to/HumanML3D \
  --text_encoder clip \
  --dim_motion 512 --depth 8 \
  --batch_size 64 \
  --steps 200000 \
  --amp_dtype bf16 \
  --compile \
  --save_every 5000 \
  --ckpt_out runs/ckpt.pt \
  --resume runs/ckpt.pt \
  --steps 400000
```

Notes:

- **`--steps` is the "total step ceiling", not "how many more steps to add"**: the training loop is `while step < steps`, and step continues from the checkpoint. To resume, `--steps` must be set **greater** than the already-trained steps (e.g. if you reached 200k last time, set `--steps 400000` to keep training); keeping the same value ends immediately because `step ≥ steps`.
- **The model structure parameters must match the checkpoint** (`--dim_motion / --depth / --heads / --dim_head / --dim_cond / --num_residual_streams / --max_motion_len`), otherwise `load_state_dict` will fail with a shape mismatch — the script rebuilds the model from the CLI parameters, it does not restore the structure from the ckpt.
- If the `--resume` path **does not exist, it automatically starts from scratch** (no error), so it can be omitted for the first training run.
- **Resuming still allows H100 speedups**: `--amp_dtype bf16 --compile` (and the automatic flash / tf32) **are not structure parameters**, do not affect `load_state_dict`, and can be freely added / changed / turned off when resuming (even fp16 last time and bf16 this time). A full H100 resume:

  ```bash
  uv run mmdit-motion/main.py train \
    --data_root /path/to/HumanML3D \
    --resume runs/ckpt.pt --ckpt_out runs/ckpt.pt --steps 400000 \
    --amp_dtype bf16 --compile --save_every 5000
  ```

  (If you originally trained with non-default structure parameters, remember to **carry the same** `--dim_motion / --depth / ...` when resuming.)

### Sampling

```bash
uv run mmdit-motion/main.py sample \
  --resume runs/ckpt.pt \
  --text_encoder clip \
  --prompt "a person walks forward then sits down" \
  --steps 50 --cfg 4.0 \
  --out sample.npy
```

> Add `--use_ema` to load the EMA-smoothed weights instead.

### View model architecture / parameter count (torchinfo)

```bash
uv run mmdit-motion/main.py summary --dim_motion 512 --depth 8
```

### Launch the Gradio UI (3D motion display)

Load a trained model, enter a prompt → generate motion → **3D skeleton animation** (GIF):

```bash
uv run mmdit-motion/gradio-test.py --ckpt runs/ckpt.pt --use_ema
# To correctly restore the motion scale (meters): add the HumanML3D folder to load Mean/Std
uv run mmdit-motion/gradio-test.py --ckpt runs/ckpt.pt --data_root /path/to/HumanML3D
```

Then open the browser at http://127.0.0.1:7860.

> Flow: `prompt → MMDiT(ODE) → (T,263) features →` `recover_from_ric` `→ (T,22,3) joints → matplotlib 3D skeleton animation`. The model architecture is rebuilt directly from the `cfg` stored in the checkpoint (no manual dimension alignment). Without `--data_root` (Mean/Std) the motion still moves, but the scale/proportions will be off (the UI notes this).

## Run platforms (local / Colab / Kaggle)

The script **is not tied to any platform**; the only differences are where the checkpoint lives and whether you resume across sessions:

| Platform   | Where the ckpt lives                                                               | How to resume                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Local**  | local path `runs/ckpt.pt`                                                          | `--resume` and `--ckpt_out` can point to the **same file**; stop and rerun to auto-continue                                    |
| **Colab**  | Google Drive (so it survives across sessions) `/content/drive/MyDrive/.../ckpt.pt` | same as above, point to the same Drive file                                                                                    |
| **Kaggle** | `/kaggle/working/` (writable, but cleared when the session ends)                   | input is read-only → `--resume /kaggle/input/<prev>/ckpt.pt --ckpt_out /kaggle/working/ckpt.pt` (read one place, write another) |

> Only Kaggle needs "read and write different paths" because its `input` is read-only; locally / Colab just let `--resume` and `--ckpt_out` point to the same file.
