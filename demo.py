"""
demo.py -- Quick test / example for AM-CycleGAN.

Purpose
-------
Verify that the AM-CycleGAN model, loss computation, and a single training
iteration run end-to-end, WITHOUT requiring any external dataset. Small
synthetic velocity-magnitude fields are generated on the fly so that a reviewer
can confirm the code works in under a minute on a CPU.

Usage
-----
    python demo.py

Expected output
---------------
The script prints the shapes of the generated tensors and, on success:

    Quick test PASSED.
"""

import itertools
import torch

from config import params
from models import Generator, Discriminator
from utils import ImagePool


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running quick test on: {device}")

    # ---- Synthetic data: two single-channel 200x200 velocity fields --------
    img_size = params["input_size"]          # 200
    real_A = torch.randn(1, 1, img_size, img_size, device=device)  # domain A
    real_B = torch.randn(1, 1, img_size, img_size, device=device)  # domain B

    # ---- Build the AM-CycleGAN models --------------------------------------
    G_A = Generator(1, params["ngf"], 1, params["num_resnet"]).to(device)
    G_B = Generator(1, params["ngf"], 1, params["num_resnet"]).to(device)
    D_A = Discriminator(1, params["ndf"], 1).to(device)
    D_B = Discriminator(1, params["ndf"], 1).to(device)

    MSE = torch.nn.MSELoss().to(device)
    L1 = torch.nn.L1Loss().to(device)

    G_opt = torch.optim.Adam(
        itertools.chain(G_A.parameters(), G_B.parameters()),
        lr=params["lrG"], betas=(params["beta1"], params["beta2"]),
    )
    D_A_opt = torch.optim.Adam(D_A.parameters(), lr=params["lrD"],
                               betas=(params["beta1"], params["beta2"]))
    D_B_opt = torch.optim.Adam(D_B.parameters(), lr=params["lrD"],
                               betas=(params["beta1"], params["beta2"]))

    fake_A_pool, fake_B_pool = ImagePool(50), ImagePool(50)

    # ---- One training iteration (A <-> B) ----------------------------------
    fake_B = G_A(real_A)                       # A -> B
    recon_A = G_B(fake_B)                      # B -> A
    fake_A = G_B(real_B)                       # B -> A
    recon_B = G_A(fake_A)                      # A -> B

    print(f"real_A  shape: {tuple(real_A.shape)}")
    print(f"fake_B  shape: {tuple(fake_B.shape)}  (G_A output)")
    print(f"recon_A shape: {tuple(recon_A.shape)} (cycle A->B->A)")
    print(f"D_B(fake_B) shape: {tuple(D_B(fake_B).shape)} (PatchGAN map)")

    # Generator objective: adversarial + cycle-consistency
    d_b_fake = D_B(fake_B)
    d_a_fake = D_A(fake_A)
    G_adv = MSE(d_b_fake, torch.ones_like(d_b_fake)) + \
            MSE(d_a_fake, torch.ones_like(d_a_fake))
    G_cycle = params["lambdaA"] * L1(recon_A, real_A) + \
              params["lambdaB"] * L1(recon_B, real_B)
    G_loss = G_adv + G_cycle
    G_opt.zero_grad(); G_loss.backward(); G_opt.step()

    # Discriminator objective (A)
    dA_real = D_A(real_A)
    dA_fake = D_A(fake_A_pool.query(fake_A).detach())
    D_A_loss = 0.5 * (MSE(dA_real, torch.ones_like(dA_real)) +
                      MSE(dA_fake, torch.zeros_like(dA_fake)))
    D_A_opt.zero_grad(); D_A_loss.backward(); D_A_opt.step()

    # Discriminator objective (B)
    dB_real = D_B(real_B)
    dB_fake = D_B(fake_B_pool.query(fake_B).detach())
    D_B_loss = 0.5 * (MSE(dB_real, torch.ones_like(dB_real)) +
                      MSE(dB_fake, torch.zeros_like(dB_fake)))
    D_B_opt.zero_grad(); D_B_loss.backward(); D_B_opt.step()

    print(f"G_loss = {G_loss.item():.4f} | "
          f"D_A_loss = {D_A_loss.item():.4f} | "
          f"D_B_loss = {D_B_loss.item():.4f}")

    # ---- Sanity checks ------------------------------------------------------
    assert fake_B.shape == real_A.shape, "Generator output shape mismatch"
    assert recon_A.shape == real_A.shape, "Cycle reconstruction shape mismatch"
    assert torch.isfinite(G_loss), "Non-finite loss"

    print("\nQuick test PASSED.")


if __name__ == "__main__":
    main()
