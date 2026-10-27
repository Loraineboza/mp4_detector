import torch
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

OUTPUT_PATH = "./result_video.mp4"
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
def detect_frame(frame_bgr, model):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.functional.to_tensor(img_rgb).to(DEVICE)

    predictions = model([img_tensor])[0]

    keep = predictions["scores"] > SCORE_THRESHOLD
    boxes = predictions["boxes"][keep].cpu().numpy()
    labels = predictions["labels"][keep].cpu().numpy()
    scores = predictions["scores"][keep].cpu().numpy()

    return boxes, labels, scores


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


def process_video(video_path, model, output_path=OUTPUT_PATH):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            boxes, labels, scores = detect_frame(frame, model)
            frame_out = draw_detections(frame, boxes, labels, scores)

            writer.write(frame_out)

            if frame_idx % 10 == 0 or frame_idx == total_frames:
                print(
                    f"Обработано кадров: {frame_idx}/{total_frames} "
                    f"(объектов в кадре: {len(boxes)})"
                )

            cv2.imshow("Video Detection", frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC или q — выход
                print("Остановлено пользователем.")
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Результат сохранён в: {output_path}")


def main(video_path, model_path):
    model = load_model(model_path)
    process_video(video_path, model)


if __name__ == "__main__":
    main(
        video_path=sys.argv[1] if len(sys.argv) > 1 else "./videos/rase.mp4",
        model_path=sys.argv[2] if len(sys.argv) > 2 else WEIGHTS_PATH,
    )
