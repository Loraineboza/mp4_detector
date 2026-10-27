import os
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as v2

from voc2007dataset import VOC2007, NUM_CLASS, DEVICE


batch_size = 2  
epochs = 30
lr = 1e-3
num_workers = 2

patience = 10
min_delta = 1e-4

model_dir = "./model"
WEIGHTS_PATH = os.path.join(model_dir, "weights_faster_rcnn_voc.pth")


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Wei
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
    os.makedirs(model_dir, exist_ok=True)

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
            f"эпоха {epoch + 1}: train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, lr {current_lr:.2e}"
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


if __name__ == "__main__":
    main()
