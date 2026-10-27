from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch

root = Path("./VOCdevkit/VOC2007/")
train = root / "ImageSets" / "Main" / "train.txt"
val = root / "ImageSets" / "Main" / "val.txt"

img_root = root / "JPEGImages"
target_root = root / "Annotations"


def get_xml_text(element, path):
    return element.find(path).text

def get_dataset(lines, n):
    dataset: list = []
    for i in range(n):
        id = lines[i] 
        imgs = img_root / f'{id}.jpg' 
        target = target_root / f'{id}.xml'
        
        target_el = ET.parse(target).getroot()
        obj = {}
        
        objs = []
        total_obj = 0
        for j in target_el.findall("object"):
            _class = get_xml_text(j, "name")
            xmin = int(get_xml_text(j, "bndbox/xmin"))
            ymin = int(get_xml_text(j, "bndbox/ymin"))
            xmax = int(get_xml_text(j, "bndbox/xmax"))
            ymax = int(get_xml_text(j, "bndbox/ymax"))
            
            objs.append(
                {
                    "class": _class,
                    "bndbox": [xmin, ymin, xmax, ymax]
                }
            )
            total_obj += 1

        dataset.append(
            {
                "image_id": id,
                "image_path": str(imgs),
                "objects": objs,
                "total_obj": total_obj
            }
        )
    return dataset

        
def voc2007():
    if not train.exists() or not val.exists():
        print("Файл не существует")
        exit(1)

    with open(train, "r") as f1, open(val, "r") as f2:
        train_lines = f1.read().splitlines() 
        val_lines = f2.read().splitlines()


    
    train_n, val_n = len(train_lines), len(val_lines)

    train_dataset = get_dataset(train_lines, train_n)
    val_dataset = get_dataset(val_lines, val_n)

    #print(f"len(train_lines)={len(train_lines)}, len(train_dataset)={len(train_dataset)}")
    #print(f"len(val_lines)={len(val_lines)}, len(val_dataset)={len(val_dataset)}")

    return train_dataset, val_dataset

def JpegImageFile(dataset: list):
    img = Image.open(dataset[0]["image_path"])
    #print(type(img)) # JpegImageFile
    #print(img.size) # h, w
    #print(img.mode) # RGB/L/RGBA; L-1 канал; RGBA-RGB и доп. канал просрачности
    arr = np.array(img)
    #print(arr)
    #print(arr.dtype) # uint8
    #print(arr.shape) # (w, h, channels)
    #print(arr.max()) # 255
    #print(arr.min()) # 0
    
    tensor = torch.from_numpy(arr).float() / 255 #нормализация pixel/255
    tensor = tensor.permute(2, 0, 1) #перемещение осей
    print(tensor.dtype)
    print(tensor.shape) # (channels, h, w)
    print(tensor.max(), tensor.min())


if __name__ == "__main__":
    train_dataset, val_dataset = voc2007()
