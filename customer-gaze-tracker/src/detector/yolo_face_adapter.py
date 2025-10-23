from __future__ import annotations
from typing import List, Tuple
import numpy as np

# Ultralytics YOLO
from ultralytics import YOLO

class YOLOFaceDetector:
    """
    Face detector wrapper for Ultralytics YOLO with face weights.
    Returns bboxes as (x1, y1, x2, y2, conf) in image pixel coords.
    """
    def __init__(self,
                 model_path: str = "data/weights/yolov12n-face.pt",
                 conf: float = 0.25,
                 imgsz: int = 640,
                 device: str = "cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def infer(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        # Ultralytics expects BGR np.ndarray OK
        results = self.model.predict(
            source=frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )
        out: List[Tuple[float, float, float, float, float]] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        return out
