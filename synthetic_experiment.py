"""Deterministic PyTorch regression experiment for checkpoint and resume validation."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATES = (1e-2, 1e-3)
CHECKPOINT_DIR = os.path.join("model", "synthetic")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(size: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((size, 1), generator=generator) * 4.0 - 2.0
    noise = torch.randn((size, 1), generator=generator) * 0.05
    y = 1.5 * x.square() - 0.7 * x + 0.3 * torch.sin(3.0 * x) + noise
    return TensorDataset(x, y)


class RegressionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            total_loss += float(nn.functional.mse_loss(model(x), y, reduction="sum").item())
            count += len(x)
    return total_loss / count


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                    loader: DataLoader) -> None:
    model.train()
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()


def optimizer_step_count(optimizer: torch.optim.Optimizer) -> int:
    steps = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
    return max(steps, default=0)


def run_experiment(lr: float, train_set: TensorDataset,
                   val_loader: DataLoader, test_loader: DataLoader) -> float:
    set_seed(SEED)
    train_generator = torch.Generator().manual_seed(SEED + 4)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator
    )
    model = RegressionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_lr_{lr:g}.pth")
    best_val_loss = float("inf")
    sample_x, _ = next(iter(test_loader))
    best_predictions = None

    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_loader)
        val_loss = evaluate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.eval()
            with torch.no_grad():
                best_predictions = model(sample_x).detach().clone()
            torch.save({
                "epoch": epoch,
                "learning_rate": lr,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_loss),
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restored_model = RegressionNet()
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=lr)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if best_predictions is None:
        raise AssertionError("No best-model predictions were captured")
    restored_model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            restored_model(sample_x), best_predictions, rtol=0.0, atol=0.0
        )
    test_loss = evaluate(restored_model, test_loader)

    steps_before = optimizer_step_count(restored_optimizer)
    if not restored_optimizer.state:
        raise AssertionError("Optimizer state was not restored")
    restored_model.train()
    x_batch, y_batch = next(iter(train_loader))
    restored_optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(restored_model(x_batch), y_batch)
    loss.backward()
    restored_optimizer.step()
    steps_after = optimizer_step_count(restored_optimizer)
    if steps_after <= steps_before:
        raise AssertionError("Optimizer did not continue from restored state")

    print(
        f"lr={lr:g} | best_val_loss={best_val_loss:.6f} | test_mse={test_loss:.6f} | "
        f"optimizer_steps={steps_before}->{steps_after} | predictions_restored=OK"
    )
    return test_loss


def main() -> None:
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_set = make_dataset(1024, SEED + 1)
    val_set = make_dataset(256, SEED + 2)
    test_set = make_dataset(256, SEED + 3)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print("Synthetic regression experiment (CPU)")
    results = {
        lr: run_experiment(lr, train_set, val_loader, test_loader)
        for lr in LEARNING_RATES
    }
    print("\nLearning-rate comparison (test MSE):")
    for lr, loss in results.items():
        print(f"  lr={lr:g}: {loss:.6f}")


if __name__ == "__main__":
    main()
