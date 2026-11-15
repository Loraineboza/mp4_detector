"""Train Faster R-CNN on PASCAL VOC 2007 with reproducible checkpoints.

Run from the repository root:
    python train.py
Resume the latest checkpoint:
    python train.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as v2

from voc2007dataset import VOC2007, NUM_CLASS, DEVICE


DEFAULT_CONFIG = {
    "seed": 42,
    "batch_size": 2,
    "epochs": 30,
    "learning_rate": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.1,
    "num_workers": 2,
    "patience": 10,
    "min_delta": 1e-4,
    "max_grad_norm": 10.0,
    "dataset_root": "./VOCdevkit/VOC2007",
    "model_dir": "./model",
}

model_dir = DEFAULT_CONFIG["model_dir"]
WEIGHTS_PATH = os.path.join(model_dir, "weights_faster_rcnn_voc.pth")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int, pretrained: bool = True):
    weights = (
        torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        if pretrained else None
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None if not pretrained else None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def train_one_epoch(model, optimizer, loader, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc="train")
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += float(loss.detach().item())
        batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if batches == 0:
        raise RuntimeError("Training DataLoader is empty")
    return total_loss / batches


@torch.no_grad()
def compute_val_loss(model, loader, device):
    """Compute detector losses without autograd.

    Torchvision Faster R-CNN returns its loss dictionary in training mode and
    detections in evaluation mode. Therefore the model stays in train mode for
    this loss-based validation pass; BatchNorm layers are switched to eval so
    their running statistics are not updated.
    """
    was_training = model.training
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    total_loss = 0.0
    batches = 0
    try:
        for images, targets in tqdm(loader, desc="validation"):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images, targets)
            total_loss += float(sum(loss_dict.values()).item())
            batches += 1
    finally:
        model.train(was_training)

    if batches == 0:
        raise RuntimeError("Validation DataLoader is empty")
    return total_loss / batches


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss,
                    epochs_no_improve, config, loader_generator):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "config": config,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator_state": loader_generator.get_state(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scheduler, loader_generator, device):
    # Checkpoints are created by this project and contain Python/NumPy RNG state.
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        state = torch.load(path, map_location=device)

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_random_state_all"])
    loader_generator.set_state(state["loader_generator_state"])
    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on VOC 2007")
    parser.add_argument("--resume", action="store_true", help="resume from model/checkpoint_last.pt")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_CONFIG["model_dir"]))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive; num-workers cannot be negative")

    config = dict(DEFAULT_CONFIG)
    config.update({
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "model_dir": str(args.model_dir),
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "model_weights": "FasterRCNN_ResNet50_FPN_Weights.DEFAULT",
    })
    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.model_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    seed_everything(args.seed)
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = VOC2007(train=True, transform=transform)
    val_dataset = VOC2007(train=False, transform=transform)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=str(DEVICE).startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=str(DEVICE).startswith("cuda"),
        worker_init_fn=seed_worker,
    )
    print(f"device={DEVICE} train={len(train_dataset)} validation={len(val_dataset)}")
    print(f"torch={torch.__version__} torchvision={torchvision.__version__}")

    model = get_model(NUM_CLASS, pretrained=not args.resume).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=DEFAULT_CONFIG["learning_rate"],
        momentum=DEFAULT_CONFIG["momentum"], weight_decay=DEFAULT_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=DEFAULT_CONFIG["scheduler_step_size"],
        gamma=DEFAULT_CONFIG["scheduler_gamma"],
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 0
    last_path = args.model_dir / "checkpoint_last.pt"
    best_path = args.model_dir / "checkpoint_best.pt"
    weights_path = args.model_dir / "weights_faster_rcnn_voc.pth"

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint not found: {last_path}")
        state = load_checkpoint(
            last_path, model, optimizer, scheduler, loader_generator, DEVICE
        )
        start_epoch = int(state["epoch"])
        best_val_loss = float(state["best_val_loss"])
        epochs_no_improve = int(state["epochs_no_improve"])
        print(f"Resumed after epoch {start_epoch}; best validation loss={best_val_loss:.6f}")

    log_path = args.model_dir / "training_log.csv"
    if not args.resume or not log_path.exists():
        log_path.write_text("epoch,train_loss,val_loss,learning_rate\n", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(
            model, optimizer, train_loader, DEVICE, DEFAULT_CONFIG["max_grad_norm"]
        )
        val_loss = compute_val_loss(model, val_loader, DEVICE)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} lr={current_lr:.3e}"
        )
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{epoch + 1},{train_loss:.8f},{val_loss:.8f},{current_lr:.8e}\n")

        improved = val_loss < best_val_loss - DEFAULT_CONFIG["min_delta"]
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch + 1, best_val_loss,
                epochs_no_improve, config, loader_generator,
            )
            print(f"Best checkpoint saved: {best_path}")
        else:
            epochs_no_improve += 1

        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch + 1, best_val_loss,
            epochs_no_improve, config, loader_generator,
        )

        if epochs_no_improve >= DEFAULT_CONFIG["patience"]:
            print(f"Early stopping after {epochs_no_improve} epochs without improvement")
            break

    print(f"Training finished. Best validation loss: {best_val_loss:.6f}")
    print(f"Best model weights: {weights_path}")
    print(f"Full resumable checkpoint: {last_path}")


if __name__ == "__main__":
    main()
