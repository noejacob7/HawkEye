# detection/yolo_wrapper.py
from ultralytics import YOLO
import torch

class YOLOv11Wrapper:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.3, device=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, frame):
        results = self.model.predict(source=frame, imgsz=640, conf=self.conf_threshold, device=0 if self.device == "cuda" else "cpu", verbose=False)
        pred = results[0]

        boxes = []
        confidences = []
        class_ids = []

        for det in pred.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            if conf >= self.conf_threshold:
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(conf))
                class_ids.append(int(cls))

        return boxes, confidences, class_ids
