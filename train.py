import os
import time
import itertools
import torch
from config import params, batchsize
from models import Generator, Discriminator
from utils import ImagePool
from dataset import get_dataloaders


def evaluate_cycle_loss(G_A, G_B, loader, device, L1_Loss):

    G_A.eval()
    G_B.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for mask, mask_, ydata, ydata_ in loader:
            Ytrain = torch.as_tensor(ydata, dtype=torch.float).to(device).unsqueeze(0)
            Ytrain_ = torch.as_tensor(ydata_, dtype=torch.float).to(device).unsqueeze(0)
            recon_A = G_B(G_A(Ytrain))
            recon_B = G_A(G_B(Ytrain_))
            total += (L1_Loss(recon_A, Ytrain) + L1_Loss(recon_B, Ytrain_)).item()
            n += 1
    G_A.train()
    G_B.train()
    return total / max(n, 1)


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    loaders, statsA, statsB = get_dataloaders()
    train_loader = loaders['train']
    val_loader = loaders['val']

    # Build Model
    G_A = Generator(1, params['ngf'], 1, params['num_resnet']).to(device)
    G_B = Generator(1, params['ngf'], 1, params['num_resnet']).to(device)
    D_A = Discriminator(1, params['ndf'], 1).to(device)
    D_B = Discriminator(1, params['ndf'], 1).to(device)

    G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()),
                                   lr=params['lrG'], betas=(params['beta1'], params['beta2']))
    D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params['lrD'], betas=(params['beta1'], params['beta2']))
    D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params['lrD'], betas=(params['beta1'], params['beta2']))

    MSE_Loss = torch.nn.MSELoss().to(device)
    L1_Loss = torch.nn.L1Loss().to(device)

    num_pool = 50
    fake_A_pool = ImagePool(num_pool)
    fake_B_pool = ImagePool(num_pool)

    iterations = 800
    best_val_loss = float('inf')

    output_dir = "output_train"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(iterations):
        D_A_losses = []
        D_B_losses = []
        G_A_losses = []
        G_B_losses = []
        cycle_A_losses = []
        cycle_B_losses = []

        for i, (mask, mask_, ydata, ydata_) in enumerate(train_loader):
            start_time = time.perf_counter()

            # Ytrain: 域A（x方向驱动）真实样本；Ytrain_: 域B（y方向驱动）真实样本
            Ytrain = torch.as_tensor(ydata, dtype=torch.float).to(device)
            Ytrain_ = torch.as_tensor(ydata_, dtype=torch.float).to(device)
            Ytrain = Ytrain.unsqueeze(0)
            Ytrain_ = Ytrain_.unsqueeze(0)

            # ======================= 生成器G_A前向 (A -> B) =======================
            fake_B = G_A(Ytrain)
            D_B_fake_decision = D_B(fake_B)
            G_A_loss = MSE_Loss(D_B_fake_decision, torch.ones_like(D_B_fake_decision))

            recon_A = G_B(fake_B)
            cycle_A_loss = L1_Loss(recon_A, Ytrain) * params['lambdaA']

            # ======================= 生成器G_B前向 (B -> A) =======================
            fake_A = G_B(Ytrain_)
            D_A_fake_decision = D_A(fake_A)
            G_B_loss = MSE_Loss(D_A_fake_decision, torch.ones_like(D_A_fake_decision))

            # ======================= 反向循环检查 =======================
            recon_B = G_A(fake_A)
            cycle_B_loss = L1_Loss(recon_B, Ytrain_) * params['lambdaB']

            G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # -------------------- train discriminator D_A (域A判别器) --------------------

            D_A_real_decision = D_A(Ytrain)
            D_A_real_loss = MSE_Loss(D_A_real_decision, torch.ones_like(D_A_real_decision))

            fake_A_pooled = fake_A_pool.query(fake_A)
            D_A_fake_decision = D_A(fake_A_pooled.detach())
            D_A_fake_loss = MSE_Loss(D_A_fake_decision, torch.zeros_like(D_A_fake_decision))

            D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
            D_A_optimizer.zero_grad()
            D_A_loss.backward()
            D_A_optimizer.step()

            # -------------------- train discriminator D_B (域B判别器) --------------------

            D_B_real_decision = D_B(Ytrain_)
            D_B_real_loss = MSE_Loss(D_B_real_decision, torch.ones_like(D_B_real_decision))

            fake_B_pooled = fake_B_pool.query(fake_B)
            D_B_fake_decision = D_B(fake_B_pooled.detach())
            D_B_fake_loss = MSE_Loss(D_B_fake_decision, torch.zeros_like(D_B_fake_decision))

            D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
            D_B_optimizer.zero_grad()
            D_B_loss.backward()
            D_B_optimizer.step()

            D_A_losses.append(D_A_loss.item())
            D_B_losses.append(D_B_loss.item())
            G_A_losses.append(G_A_loss.item())
            G_B_losses.append(G_B_loss.item())
            cycle_A_losses.append(cycle_A_loss.item())
            cycle_B_losses.append(cycle_B_loss.item())

        
        val_loss = evaluate_cycle_loss(G_A, G_B, val_loader, device, L1_Loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(G_A, 'cycle_size200_GA_best.pth')
            torch.save(D_A, 'cycle_size200_DA_best.pth')
            torch.save(G_B, 'cycle_size200_GB_best.pth')
            torch.save(D_B, 'cycle_size200_DB_best.pth')


if __name__ == '__main__':
    main()
