import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as v2

from voc2007dataset import VOC2007, NUM_CLASS, DEVICE


SEED = 42
batch_size = 2
epochs = 30
lr = 1e-3
num_workers = 2

patience = 10
min_delta = 1e-4

model_dir = "./model"
WEIGHTS_PATH = os.path.join(model_dir, "weights_faster_rcnn_voc.pth")
CONFIG_PATH = os.path.join(model_dir, "training_config.json")


def set_seed(seed: int = SEED):
    """Fix random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_training_config(path: str, config: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="train")

    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix({"loss": f"{losses.item():.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def compute_val_loss(model, loader, device):
    # loss у faster r-cnn считается только в train-режиме, потому что нужен
    # матчинг предсказаний с ground truth. поэтому держим модель в train,
    # но замораживаем batchnorm, чтобы running stats не портились на валидации.
    was_training = model.training
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    total_loss = 0.0
    pbar = tqdm(loader, desc="val")
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        total_loss += losses.item()
        pbar.set_postfix({"val_loss": f"{losses.item():.4f}"})

    if was_training:
        model.train()
    else:
        model.eval()
    return total_loss / len(loader)


def main():
    set_seed(SEED)
    os.makedirs(model_dir, exist_ok=True)

    run_config = {
        "seed": SEED,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "num_workers": num_workers,
        "patience": patience,
        "min_delta": min_delta,
        "model_dir": model_dir,
        "device": str(DEVICE),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_training_config(CONFIG_PATH, run_config)

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_dataset = VOC2007(train=True, transform=transform)
    val_dataset = VOC2007(train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"[CONFIG] seed={SEED}, batch_size={batch_size}, epochs={epochs}, lr={lr}, device={DEVICE}")
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")

    model = get_model(NUM_CLASS).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"\nэпоха {epoch + 1} из {epochs}")

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
        val_loss = compute_val_loss(model, val_loader, DEVICE)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch + 1}/{epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            with open(WEIGHTS_PATH, "wb") as f:
                torch.save(model.state_dict(), f)
            print(f"val loss улучшился, веса сохранены в {WEIGHTS_PATH}")
        else:
            epochs_no_improve += 1
            print(
                f"улучшения нет уже {epochs_no_improve} эпох подряд "
                f"(лучший val loss {best_val_loss:.4f})"
            )
            if epochs_no_improve >= patience:
                print(
                    f"останавливаем обучение: {patience} эпох без улучшения, "
                    f"последняя эпоха {epoch + 1}"
                )
                break

    print(f"\nобучение закончено, лучший val loss {best_val_loss:.4f}")
    print(f"веса лежат тут: {WEIGHTS_PATH}")
    print(f"конфигурация сохранена в: {CONFIG_PATH}")


if __name__ == "__main__":
    main()


path: "detection_utils.py",
content: "from __future__ import annotations\n\nimport numpy as np\n\n\ndef _to_box_array(boxes):\n    arr = np.asarray(boxes, dtype=np.float64)\n    if arr.size == 0:\n        return arr.reshape(0, 4)\n    if arr.ndim == 1:\n        if arr.shape[0] != 4:\n            raise ValueError(f\"Expected a single box with 4 coordinates, got shape {arr.shape}\")\n        return arr.reshape(1, 4)\n    if arr.ndim != 2 or arr.shape[1] != 4:\n        raise ValueError(f\"Boxes must have shape (N, 4), got {arr.shape}\")\n    return arr\n\n\ndef _valid_box_mask(boxes):\n    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]\n    widths = np.maximum(0.0, x2 - x1)\n    heights = np.maximum(0.0, y2 - y1)\n    return (widths > 0.0) & (heights > 0.0)\n\n\ndef compute_iou(box_a, box_b):\n    \"\"\"IoU for a single pair of boxes in xyxy format.\"\"\"\n    return compute_iou_matrix(np.asarray([box_a], dtype=np.float64), np.asarray([box_b], dtype=np.float64))[0, 0]\n\n\ndef compute_iou_matrix(boxes_a, boxes_b):\n    \"\"\"Vectorized IoU matrix of shape (N, M).\"\"\"\n    boxes_a = _to_box_array(boxes_a)\n    boxes_b = _to_box_array(boxes_b)\n\n    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:\n        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)\n\n    mask_a = _valid_box_mask(boxes_a)\n    mask_b = _valid_box_mask(boxes_b)\n    if not np.all(mask_a) or not np.all(mask_b):\n        boxes_a = boxes_a.copy()\n        boxes_b = boxes_b.copy()\n        boxes_a[~mask_a] = 0.0\n        boxes_b[~mask_b] = 0.0\n\n    inter_x1 = np.maximum(boxes_a[:, 0][:, None], boxes_b[None, :, 0])\n    inter_y1 = np.maximum(boxes_a[:, 1][:, None], boxes_b[None, :, 1])\n    inter_x2 = np.minimum(boxes_a[:, 2][:, None], boxes_b[None, :, 2])\n    inter_y2 = np.minimum(boxes_a[:, 3][:, None], boxes_b[None, :, 3])\n\n    inter_w = np.maximum(0.0, inter_x2 - inter_x1)\n    inter_h = np.maximum(0.0, inter_y2 - inter_y1)\n    inter_area = inter_w * inter_h\n\n    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])\n    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])\n    union = area_a[:, None] + area_b[None, :] - inter_area\n\n    with np.errstate(divide='ignore', invalid='ignore'):\n        iou = np.divide(inter_area, union, out=np.zeros_like(inter_area), where=union > 0.0)\n    return iou\n\n\ndef non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.0):\n    \"\"\"Classic NMS for boxes in xyxy format. Returns kept indices.\"\"\"\n    boxes = _to_box_array(boxes)\n    scores = np.asarray(scores, dtype=np.float64)\n\n    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:\n        raise ValueError(f\"scores shape {scores.shape} does not match boxes shape {boxes.shape}\")\n\n    if boxes.shape[0] == 0:\n        return np.array([], dtype=np.int64)\n\n    valid = np.arange(boxes.shape[0])\n    if score_threshold is not None:\n        valid = valid[scores >= score_threshold]\n\n    if valid.size == 0:\n        return np.array([], dtype=np.int64)\n\n    boxes = boxes[valid]\n    scores = scores[valid]\n    order = np.argsort(scores)[::-1]\n    keep = []\n\n    while order.size > 0:\n        current = order[0]\n        keep.append(valid[current])\n        if order.size == 1:\n            break\n\n        remaining = order[1:]\n        remaining_boxes = boxes[remaining]\n        iou = compute_iou_matrix(np.asarray([boxes[current]], dtype=np.float64), remaining_boxes)[0]\n        order = remaining[iou <= iou_threshold]\n\n    return np.asarray(keep, dtype=np.int64)\n\n"},{