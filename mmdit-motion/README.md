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
