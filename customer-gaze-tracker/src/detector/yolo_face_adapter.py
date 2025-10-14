import os
from ultralytics import YOLO

class YOLOFaceDetector:
    def __init__(self,
                 weights_path="third_party/yolo-face/weights/yolov12n-face.pt",
                 conf_thres=0.25, iou_thres=0.5, imgsz=640, device=None):
        assert os.path.isfile(weights_path), f"weights not found: {weights_path}"
        self.model = YOLO(weights_path)
        self.conf = conf_thres
        self.iou = iou_thres
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame):
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou,
                                     imgsz=self.imgsz, verbose=False, device=self.device)
        out = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), c in zip(xyxy, conf):
                out.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        return out
