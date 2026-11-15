#!/usr/bin/env python3
"""CPU-only reproducible PyTorch experiment on synthetic regression data."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
INPUT_DIM = 5
BATCH_SIZE = 32
EPOCHS = 30


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_datasets():
    rng = np.random.default_rng(SEED)
    features = rng.normal(size=(600, INPUT_DIM)).astype(np.float32)
    true_weights = np.array([[1.5], [-2.0], [0.7], [3.0], [-0.5]], dtype=np.float32)
    noise = rng.normal(scale=0.15, size=(600, 1)).astype(np.float32)
    targets = features @ true_weights + 0.35 + noise
    permutation = np.random.default_rng(SEED + 1).permutation(len(features))
    indices = (permutation[:420], permutation[420:510], permutation[510:])
    return [
        TensorDataset(torch.from_numpy(features[i]), torch.from_numpy(targets[i]))
        for i in indices
    ]


def make_loaders():
    train_set, val_set, test_set = make_datasets()
    generator = torch.Generator().manual_seed(SEED + 2)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        generator=generator, num_workers=0,
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_epoch(model, optimizer, loader):
    model.train()
    total, count = 0.0, 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(features), targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * targets.numel()
        count += targets.numel()
    return total / count


def evaluate(model, loader):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for features, targets in loader:
            total += nn.functional.mse_loss(
                model(features), targets, reduction="sum"
            ).item()
            count += targets.numel()
    return total / count


def save_checkpoint(path, model, optimizer, epoch, best_val):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val,
        "seed": SEED,
    }, path)


def load_checkpoint(path, model, optimizer):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def run_experiment(output_dir):
    seed_everything()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders()
    model = Regressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    best_val = float("inf")
    best_path = output_dir / "best_checkpoint.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_mse = train_epoch(model, optimizer, train_loader)
        val_mse = evaluate(model, val_loader)
        if val_mse < best_val:
            best_val = val_mse
            save_checkpoint(best_path, model, optimizer, epoch, best_val)
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(f"epoch={epoch:02d} train_mse={train_mse:.8f} val_mse={val_mse:.8f}")

    best_model = Regressor()
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-2)
    best_state = load_checkpoint(best_path, best_model, best_optimizer)
    test_mse = evaluate(best_model, test_loader)

    sample_features, _ = next(iter(test_loader))
    best_model.eval()
    with torch.no_grad():
        predictions_before = best_model(sample_features).clone()
    restored_model = Regressor()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-2)
    restored_state = load_checkpoint(best_path, restored_model, restored_optimizer)
    restored_model.eval()
    with torch.no_grad():
        predictions_after = restored_model(sample_features).clone()
    predictions_identical = torch.equal(predictions_before, predictions_after)

    optimizer_state_entries = len(restored_optimizer.state)
    continued_train_mse = train_epoch(restored_model, restored_optimizer, train_loader)

    (output_dir / "training_log.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    summary = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": SEED,
        "split_sizes": {"train": 420, "validation": 90, "test": 90},
        "best_epoch": int(best_state["epoch"]),
        "best_validation_mse": float(best_state["best_val_loss"]),
        "test_mse": float(test_mse),
        "checkpoint_predictions_identical": bool(predictions_identical),
        "restored_optimizer_state_entries": optimizer_state_entries,
        "continued_training_mse": float(continued_train_mse),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("CHECKPOINT_PREDICTIONS_IDENTICAL:", predictions_identical)
    print("RESTORED_OPTIMIZER_STATE_ENTRIES:", optimizer_state_entries)
    print("TEST_MSE:", f"{test_mse:.8f}")
    print("SUMMARY:", json.dumps(summary, sort_keys=True))
    return summary


def compare_learning_rates():
    loader_a, val_a, _ = make_loaders()
    loader_b, val_b, _ = make_loaders()
    seed_everything()
    initial_state = copy.deepcopy(Regressor().state_dict())
    results = {}
    for learning_rate, name, train_loader, val_loader in (
        (1e-3, "lr_0.001", loader_a, val_a),
        (1e-1, "lr_0.1", loader_b, val_b),
    ):
        model = Regressor()
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for _ in range(20):
            train_epoch(model, optimizer, train_loader)
        results[name] = evaluate(model, val_loader)
    print("LEARNING_RATE_COMPARISON:", json.dumps(results, sort_keys=True))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-lr-comparison", action="store_true")
    args = parser.parse_args()
    run_experiment(args.output_dir)
    if not args.skip_lr_comparison:
        compare_learning_rates()


if __name__ == "__main__":
    main()
