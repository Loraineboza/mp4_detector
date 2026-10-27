import torch
import torch.utils.data as data
from torch import nn
from script import get_dataset, get_xml_text
from pathlib import Path
from PIL import Image
import torchvision
import torchvision.transforms.v2 as v2 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

root = Path("./VOCdevkit/VOC2007")
train_root = root / "ImageSets" / "Main" / "train.txt"
val_root = root / "ImageSets" / "Main" / "val.txt"

img_root = root / "JPEGImages"
target_root = root / "Annotations"

VOC_CLASSES = [
    "", "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASS = len(VOC_CLASSES)

class VOC2007(data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.current_root = train_root if train else val_root
        self.transform = transform 
        self.target_transform = target_transform

        with open(self.current_root, "r") as f:
            self.current_root = f.read().splitlines()
        self.n = len(self.current_root)
        self.dataset = get_dataset(self.current_root, self.n)
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n:
            raise IndexError(f"list index out of range. idx={idx} of maximum={self.n-1}")

        PIL_image = Image.open(self.dataset[idx]['image_path']).convert("RGB")
        orig_w, orig_h = PIL_image.size
        image = self.transform(PIL_image) if self.transform is not None else PIL_image

        obj = self.dataset[idx]['objects']
        total_obj = self.dataset[idx]['total_obj']

        list_classes = []
        list_bboxes = []

        for i in range(total_obj):
            list_classes.append(VOC_CLASSES.index(obj[i]['class']))
            bbox = obj[i]['bndbox']

            xmin = bbox[0] 
            ymin = bbox[1] 
            xmax = bbox[2] 
            ymax = bbox[3]  

            if xmax <= xmin: xmax = xmin + 1.0
            if ymax <= ymin: ymax = ymin + 1.0

            list_bboxes.append([xmin, ymin, xmax, ymax])
        
        if total_obj == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(list_bboxes, dtype=torch.float32)
            labels = torch.tensor(list_classes, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels
        }
        return image, target
        
    def __len__(self):
        return self.n


