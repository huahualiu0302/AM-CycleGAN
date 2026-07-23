# AM-CycleGAN

**Accelerating Steady-State Flow Simulation in Digital Rock Using Unsupervised
Learning and an Attention Mechanism**

This repository contains the source code for **AM-CycleGAN**, an unpaired
image-to-image translation framework that couples CycleGAN with a Convolutional
Block Attention Module (CBAM) to predict pore-scale steady-state velocity
fields in digital sandstone images. The two translation domains are defined by
the driving direction of the Lattice Boltzmann Method (LBM) simulation (domain A
= x-direction driving, domain B = y-direction driving). Once trained, the
generators reproduce the orthogonal-direction velocity field at negligible
inference cost, so each slice only needs to be simulated in a single direction.

---

## 1. Repository structure

| File | Purpose |
| --- | --- |
| `models.py`  | Network definitions: `Generator`, `Discriminator`, residual blocks, and the CBAM attention module (`ChannelAttention`, `SpatialAttention`, `CBAM`). |
| `dataset.py` | Dataset class, 5:2:3 train/validation/test split, and normalization / denormalization utilities. |
| `config.py`  | Hyperparameters (image size, learning rate, number of residual blocks, cycle-loss weight, etc.). |
| `utils.py`   | `ImagePool` for stabilizing adversarial training. |
| `train.py`   | Full training entry point; saves the best generator/discriminator weights by validation cycle loss. |
| `demo.py`    | **Quick test / example.** Runs the full model on small synthetic data, with no external dataset required (see Section 5). |
| `requirements.txt` | Python dependencies. |
| `LICENSE`    | MIT license. |

---

## 2. Requirements

- Python 3.9+
- PyTorch 1.10+
- NumPy

Install the dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training but **not** required for the
quick test in Section 5, which runs on CPU.

Reference hardware used in the paper: NVIDIA GeForce RTX 4090 GPU with an
Intel Core i7-13700K CPU.

---

## 3. Data format

Training reads two NumPy arrays produced by LBM simulation:

- `Banderagray_size200_xdrive_num500.npy` — domain A (pressure gradient along x)
- `Banderagray_size200_ydrive_num500.npy` — domain B (pressure gradient along y)

Each array has shape `(N, 4, 200, 200)`, where for every 200×200 slice the four
channels are:

| Index | Content |
| --- | --- |
| 0 | Binary geometry (0 = pore, 1 = grain) |
| 1 | (reserved) |
| 2 | LBM velocity component `vx` |
| 3 | LBM velocity component `vy` |

The scalar velocity magnitude `v = sqrt(vx**2 + vy**2)` is computed internally
and used as the modeling target.

The micro-CT datasets used to generate these arrays are publicly available (see
Section 8, *Data availability*).

---

## 4. Training

Place the two `.npy` files in the repository root and run:

```bash
python train.py
```

The script builds two generators (`G_A`, `G_B`) and two discriminators
(`D_A`, `D_B`), trains them with the adversarial + cycle-consistency objective,
and saves the best weights (lowest validation cycle loss) as
`cycle_size200_GA_best.pth`, `cycle_size200_DA_best.pth`,
`cycle_size200_GB_best.pth`, and `cycle_size200_DB_best.pth`.

Key hyperparameters can be edited in `config.py`
(e.g. `num_resnet = 3`, `lrG = lrD = 0.0001`, corresponding to the optimal
configuration `b3r10^-4` reported in the paper).

---

## 5. Quick test / example (no external data needed)

To verify that the code and environment work without downloading any dataset,
run the included example:

```bash
python demo.py
```

`demo.py` fabricates a small batch of synthetic 200×200 fields, instantiates the
generators and discriminators, performs a forward pass in both translation
directions, and executes a single optimization step. On success it prints the
output tensor shapes and:

```
Quick test PASSED.
```

The test completes in well under a minute on a CPU. This confirms the model,
loss computation, and one training iteration run end-to-end before you commit to
a full training run on real data.

---

## 6. Inference on your own slices

After training, load a saved generator and apply it to a normalized velocity
field of shape `(1, 1, 200, 200)`:

```python
import torch
from dataset import invert_normalization  # denormalize back to physical units

G_A = torch.load('cycle_size200_GA_best.pth', map_location='cpu')
G_A.eval()
with torch.no_grad():
    fake_B = G_A(input_field)   # x-driven field -> predicted y-driven field
```

Use the normalization statistics returned by `dataset.get_dataloaders()` (or
saved from training) with `invert_normalization` to map predictions back to
physical velocity units.

---

## 7. Citation

If you use this code, please cite:

> Xia, Y., Liu, C., Liang, J., Wang, H., Cai, J. Accelerating Steady-State Flow
> Simulation in Digital Rock Using Unsupervised Learning and an Attention
> Mechanism. *Computers & Geosciences* (under review).

---

## 8. Data availability

The Bandera Gray, Parker, and Bentheimer micro-CT datasets are publicly
available from the Digital Rocks Portal:

1. Neumann, R. F., Barsi-Andreeta, M., Lucas-Oliveira, E., et al. High accuracy
   capillary network representation in digital rock reveals permeability scaling
   functions. *Scientific Reports*, 2021, 11(1): 11370.
2. Lucas-Oliveira, E., Barsi-Andreeta, M., Neumann, R. F., et al. Micro-computed
   tomography of sandstone rocks: Raw, filtered and segmented datasets.
   *Data in Brief*, 2022, 41: 107893.

The Berea sandstone dataset is available from Imperial College London:

3. Dong, H., Blunt, M. J. Pore-network extraction from micro-computerized-
   tomography images. *Physical Review E*, 2009, 80(3): 036307.
