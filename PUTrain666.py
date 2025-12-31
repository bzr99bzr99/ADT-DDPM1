import time
import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 自定义模块导入
from PUDataset import PUDataset, split_dataset_by_class_count
from Config import Config
from DfModel import GaussianDiffusion
from v111 import UNet1D

# --- Hyperparameters & Config ---
SEED = 42
EPOCHS = 3000
BATCH_SIZE = 10
LEARNING_RATE = 0.001
SEQ_LENGTH = 1024
INPUT_SIZE = 1
TRAIN_SAMPLE_NUM = 20
TEST_SAMPLE_NUM = 1
LAMBDA_DIVERSITY = 0.06  # 多样性损失权重

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("./PUmodel/v111/loss/12-24/")
CSV_FILE = './data/PU/'


def set_seed(seed: int):
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random Seed: {seed}")


def calc_diversity_loss(x_pred: torch.Tensor) -> torch.Tensor:
    """计算多样性损失 (Cosine Similarity based)"""
    b, _, _ = x_pred.shape
    x_flat = x_pred.view(b, -1)
    x_norm = F.normalize(x_flat, p=2, dim=1)
    # 计算相似度矩阵
    sim_matrix = x_norm @ x_norm.T
    # (Sum - Trace) / (N * (N-1))
    loss = (sim_matrix.sum() - sim_matrix.trace()) / (b * (b - 1))
    return loss


def get_dataloader() -> DataLoader:
    """准备数据加载器"""
    dataset = PUDataset(CSV_FILE, SEQ_LENGTH, SEQ_LENGTH)
    print(f"Total Dataset: {len(dataset)}")

    train_ds, test_ds = split_dataset_by_class_count(dataset, TRAIN_SAMPLE_NUM, TEST_SAMPLE_NUM)
    print(f"Train Size: {len(train_ds)} | Test Size: {len(test_ds)}")

    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


def train():
    # 1. Setup
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # set_seed(SEED)

    train_loader = get_dataloader()

    model = UNet1D(num_classes=5, num_diffusion_timesteps=Config.timesteps).to(DEVICE)
    diffusion = GaussianDiffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    loss_fn = torch.nn.MSELoss()

    print(f"--- Model Loaded on {DEVICE} | Start Training ---")

    best_loss = float('inf')
    start_time = time.time()

    # 2. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for batch_data, batch_labels in pbar:
            # Data prep
            x_0 = batch_data.view(-1, INPUT_SIZE, SEQ_LENGTH).to(DEVICE, non_blocking=True)
            c = batch_labels.to(DEVICE, non_blocking=True).long()
            t = torch.randint(0, Config.timesteps, (x_0.shape[0],), device=DEVICE)

            # Diffusion Forward
            noise = torch.randn_like(x_0)
            x_t = diffusion.q_sample(x_0, t, noise).to(DEVICE, non_blocking=True)

            # Prediction
            pred_noise = model(x_t, t, c)

            # Loss Calculation
            mse_loss = loss_fn(pred_noise, noise)

            # Diversity Loss
            x_0_pred = diffusion.predict_start_from_noise(x_t, t, pred_noise)
            div_loss = calc_diversity_loss(x_0_pred)

            total_loss = mse_loss + LAMBDA_DIVERSITY * div_loss

            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update Log & Checkpoint
            current_loss = total_loss.item()
            pbar.set_postfix(loss=f"{current_loss:.6f}")

            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(model.state_dict(), SAVE_DIR / 'Model_best.pth')

    # 3. Finish
    elapsed = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=int(elapsed)))

    torch.save(model.state_dict(), SAVE_DIR / 'Model_final.pth')

    print("-" * 40)
    print(f"Training Completed.")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Total Time: {time_str} ({elapsed:.2f}s)")
    print("-" * 40)


if __name__ == '__main__':
    train()