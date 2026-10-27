import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import sys
from voc_classes import VOC_CLASSES
from train import WEIGHTS_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 21
SCORE_THRESHOLD = 0.5

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)


def load_model(model_path):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def detect(image_path, model):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.functional.to_tensor(img_rgb).to(DEVICE)

    predictions = model([img_tensor])[0]

    keep = predictions["scores"] > SCORE_THRESHOLD
    boxes = predictions["boxes"][keep].cpu().numpy()
    labels = predictions["labels"][keep].cpu().numpy()
    scores = predictions["scores"][keep].cpu().numpy()

    return img_bgr, boxes, labels, scores


def draw_detections(img_bgr, boxes, labels, scores):
    img_out = img_bgr.copy()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[label].tolist()
        class_name = VOC_CLASSES[label]

        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        text = f"{class_name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )

        text_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10
        cv2.rectangle(
            img_out,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            img_out, text, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return img_out


def main(image_path, model_path):
    model = load_model(model_path)
    img_bgr, boxes, labels, scores = detect(image_path, model)

    print(f"Найдено объектов: {len(boxes)}")
    for box, label, score in zip(boxes, labels, scores):
        print(f"  {VOC_CLASSES[label]}: {score:.3f} at {box.astype(int)}")

    img_result = draw_detections(img_bgr, boxes, labels, scores)

    cv2.imshow("Detection Result", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("result.jpg", img_result)


if __name__ == "__main__":
    main(
        #image_path=sys.argv[1] if len(sys.argv) > 1 else "VOCdevkit/VOC2007/JPEGImages/008872.jpg",
        image_path=sys.argv[1] if len(sys.argv) > 1 else "img/cars.png",
        model_path=sys.argv[2] if len(sys.argv) > 2 else WEIGHTS_PATH,
    )
