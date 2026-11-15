#!/usr/bin/env python3
"""CPU-only reproducible PyTorch experiment on two overlapping 2D classes."""

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
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-2


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_datasets():
    """Create balanced, partially overlapping Gaussian classes and stratified splits."""
    rng = np.random.default_rng(SEED)
    covariance = np.array([[1.0, 0.35], [0.35, 1.0]])

    class_0 = rng.multivariate_normal([-0.75, -0.35], covariance, 300)
    class_1 = rng.multivariate_normal([0.75, 0.35], covariance, 300)
    features = np.concatenate([class_0, class_1]).astype(np.float32)
    labels = np.concatenate([
        np.zeros(300, dtype=np.float32),
        np.ones(300, dtype=np.float32),
    ])

    class_splits = []
    for class_id in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == class_id))
        class_splits.append((
            indices[:210],
            indices[210:255],
            indices[255:],
        ))

    split_indices = []
    for split_id in range(3):
        merged = np.concatenate([
            class_splits[0][split_id],
            class_splits[1][split_id],
        ])
        split_indices.append(rng.permutation(merged))

    return [
        TensorDataset(
            torch.from_numpy(features[indices]),
            torch.from_numpy(labels[indices]).unsqueeze(1),
        )
        for indices in split_indices
    ]


def make_loaders():
    train_set, validation_set, test_set = make_datasets()
    generator = torch.Generator().manual_seed(SEED + 1)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    return train_loader, validation_loader, test_loader


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features):
        return self.layers(features)


def train_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0

    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += ((logits >= 0) == (targets >= 0.5)).sum().item()
        count += batch_size

    return {"loss": total_loss / count, "accuracy": correct / count}


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for features, targets in loader:
            logits = model(features)
            total_loss += nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum"
            ).item()
            correct += ((logits >= 0) == (targets >= 0.5)).sum().item()
            count += targets.size(0)

    return {"loss": total_loss / count, "accuracy": correct / count}


def save_checkpoint(path, model, optimizer, epoch, best_validation):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation["loss"],
        "best_validation_accuracy": best_validation["accuracy"],
        "seed": SEED,
    }, path)


def load_checkpoint(path, model, optimizer):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def compare_learning_rates():
    """Compare rates from identical initial weights, data and batch ordering."""
    seed_everything()
    initial_model = Classifier()
    initial_state = copy.deepcopy(initial_model.state_dict())
    results = {}

    for learning_rate in (1e-3, 1e-1):
        train_loader, validation_loader, _ = make_loaders()
        model = Classifier()
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for _ in range(EPOCHS):
            train_epoch(model, optimizer, train_loader)

        metrics = evaluate(model, validation_loader)
        results[f"lr_{learning_rate:g}"] = {
            "validation_loss": metrics["loss"],
            "validation_accuracy": metrics["accuracy"],
        }

    return results


def run_experiment(output_dir: Path):
    seed_everything()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = make_loaders()

    model = Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation = {"loss": float("inf"), "accuracy": 0.0}
    best_path = output_dir / "best_checkpoint.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(model, optimizer, train_loader)
        validation_metrics = evaluate(model, validation_loader)

        improved = (
            validation_metrics["loss"] < best_validation["loss"]
        )
        if improved:
            best_validation = validation_metrics.copy()
            save_checkpoint(
                best_path, model, optimizer, epoch, best_validation
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_acc={validation_metrics['accuracy']:.4f}"
        )

    best_model = Classifier()
    best_optimizer = torch.optim.Adam(
        best_model.parameters(), lr=LEARNING_RATE
    )
    best_checkpoint = load_checkpoint(
        best_path, best_model, best_optimizer
    )

    # The held-out test set is evaluated once, after validation-based selection.
    test_metrics = evaluate(best_model, test_loader)

    sample_features, _ = next(iter(test_loader))
    best_model.eval()
    with torch.no_grad():
        predictions_before = best_model(sample_features).clone()

    restored_model = Classifier()
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(), lr=LEARNING_RATE
    )
    load_checkpoint(best_path, restored_model, restored_optimizer)
    restored_model.eval()
    with torch.no_grad():
        predictions_after = restored_model(sample_features).clone()

    predictions_identical = torch.equal(
        predictions_before, predictions_after
    )
    optimizer_state_entries = len(restored_optimizer.state)
    continued_metrics = train_epoch(
        restored_model, restored_optimizer, train_loader
    )

    learning_rate_results = compare_learning_rates()

    (output_dir / "training_log.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    summary = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": SEED,
        "dataset": "two partially overlapping Gaussian classes in 2D",
        "dataset_size": 600,
        "split_sizes": {"train": 420, "validation": 90, "test": 90},
        "architecture": "Linear(2, 16) -> ReLU -> Linear(16, 1)",
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_loss": float(best_checkpoint["best_validation_loss"]),
        "best_validation_accuracy": float(
            best_checkpoint["best_validation_accuracy"]
        ),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "checkpoint_predictions_identical": bool(predictions_identical),
        "restored_optimizer_state_entries": optimizer_state_entries,
        "continued_training_loss": float(continued_metrics["loss"]),
        "continued_training_accuracy": float(continued_metrics["accuracy"]),
        "learning_rate_comparison": learning_rate_results,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("CHECKPOINT_PREDICTIONS_IDENTICAL:", predictions_identical)
    print("TEST_METRICS:", json.dumps(test_metrics, sort_keys=True))
    print("LEARNING_RATE_COMPARISON:", json.dumps(learning_rate_results, sort_keys=True))
    print("SUMMARY:", json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    args = parser.parse_args()
    run_experiment(args.output_dir)


if __name__ == "__main__":
    main()
