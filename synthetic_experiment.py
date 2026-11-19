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
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            total_loss += float(
                nn.functional.mse_loss(
                    model(x), y, reduction="sum"
                ).item()
            )
            count += len(x)

    return total_loss / count


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
) -> None:
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
            steps.append(
                int(step.item()) if torch.is_tensor(step) else int(step)
            )

    return max(steps, default=0)


def train_candidate(
    lr: float,
    train_set: TensorDataset,
    val_loader: DataLoader,
) -> dict:
    set_seed(SEED)

    train_generator = torch.Generator().manual_seed(SEED + 4)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )

    model = RegressionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"best_lr_{lr:g}.pth"
    )

    probe_x = train_set.tensors[0][:8]
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_loader)
        val_loss = evaluate(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            model.eval()
            with torch.no_grad():
                probe_predictions = model(probe_x).detach().clone()

            torch.save(
                {
                    "epoch": best_epoch,
                    "learning_rate": lr,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "probe_predictions": probe_predictions,
                },
                checkpoint_path,
            )

    return {
        "learning_rate": lr,
        "val_loss": best_val_loss,
        "epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
    }


def main() -> None:
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_set = make_dataset(1024, SEED + 1)
    val_set = make_dataset(256, SEED + 2)
    test_set = make_dataset(256, SEED + 3)

    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False
    )

    print("Synthetic regression experiment (CPU)")

    results = [
        train_candidate(lr, train_set, val_loader)
        for lr in LEARNING_RATES
    ]

    print("\nValidation comparison:")
    for result in results:
        print(
            f"lr={result['learning_rate']:g} | "
            f"best_val_loss={result['val_loss']:.6f} | "
            f"best_epoch={result['epoch']}"
        )

    selected = min(results, key=lambda result: result["val_loss"])

    print(
        f"\nSelected learning rate: "
        f"{selected['learning_rate']:g}"
    )

    checkpoint = torch.load(
        selected["checkpoint_path"],
        map_location="cpu",
    )

    restored_model = RegressionNet()
    restored_model.load_state_dict(checkpoint["model_state_dict"])

    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(),
        lr=selected["learning_rate"],
    )
    restored_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    probe_x = train_set.tensors[0][:8]
    restored_model.eval()

    with torch.no_grad():
        restored_predictions = restored_model(probe_x)

    torch.testing.assert_close(
        restored_predictions,
        checkpoint["probe_predictions"],
        rtol=0.0,
        atol=0.0,
    )

    print("Checkpoint predictions restored: OK")

    test_loss = evaluate(restored_model, test_loader)
    print(f"Selected model test_mse={test_loss:.6f}")

    steps_before = optimizer_step_count(restored_optimizer)

    if not restored_optimizer.state:
        raise AssertionError("Optimizer state was not restored")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + 4),
    )

    restored_model.train()
    x_batch, y_batch = next(iter(train_loader))

    restored_optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(restored_model(x_batch), y_batch)
    loss.backward()
    restored_optimizer.step()

    steps_after = optimizer_step_count(restored_optimizer)

    if steps_after <= steps_before:
        raise AssertionError(
            "Optimizer did not continue from restored state"
        )

    print(
        f"Optimizer state restored: "
        f"{steps_before}->{steps_after}"
    )
    print("Experiment completed successfully")


if __name__ == "__main__":
    main()
