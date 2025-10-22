# src/run_headpose_demo.py
import os, json, math
import cv2
from tqdm import tqdm

from detector.yolo_face_adapter import YOLOFaceDetector
from tracker.deepsort_adapter import DeepSortTracker
from pose.repnet_adapter import SixDRepNetAdapter

def draw_axes(img, center, yaw, pitch, roll, size=80):
    import numpy as np
    cx, cy = int(center[0]), int(center[1])
    yaw, pitch, roll = [math.radians(a) for a in (yaw, pitch, roll)]
    Rx = np.array([[1,0,0],
                   [0,math.cos(roll),-math.sin(roll)],
                   [0,math.sin(roll), math.cos(roll)]])
    Ry = np.array([[ math.cos(pitch),0,math.sin(pitch)],
                   [0,1,0],
                   [-math.sin(pitch),0,math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw),-math.sin(yaw),0],
                   [math.sin(yaw), math.cos(yaw),0],
                   [0,0,1]])
    R = Rz @ Ry @ Rx
    axes = np.float32([[size,0,0],[0,size,0],[0,0,size]])
    proj = axes @ R.T
    pts = proj[:,:2] + np.float32([cx,cy])
    X,Y,Z = map(tuple, pts.astype(int))
    cv2.line(img,(cx,cy),X,(0,0,255),2)   # yaw(빨)
    cv2.line(img,(cx,cy),Y,(0,255,0),2)   # pitch(초)
    cv2.line(img,(cx,cy),Z,(255,0,0),2)   # roll(파)

def main():
    os.makedirs("data/outputs", exist_ok=True)
    src = "data/samples/store_01.mp4"     # 여기에 테스트 영상 배치
    out_vid = "data/outputs/headpose_overlay.mp4"
    out_log = "data/outputs/headpose_log.jsonl"

    det = YOLOFaceDetector("weights/yolov12n-face.pt", conf_thres=0.25, imgsz=640)
    trk = DeepSortTracker()
    hp  = SixDRepNetAdapter("weights/6drepnet.pth")   # 가중치 없으면 dummy(0,0,0)로 진행

    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"Cannot open {src}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw  = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    logf = open(out_log, "w", encoding="utf-8")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for fidx in tqdm(range(total)):
        ok, frame = cap.read()
        if not ok: break

        dets = det.detect(frame)                # [[x1,y1,x2,y2,conf], ...]
        tracks = trk.update(frame, dets)        # [{"id","tlbr":[...]}...]

        for t in tracks:
            x1,y1,x2,y2 = t["tlbr"]
            pose = hp.infer(frame, (x1,y1,x2,y2))  # {"yaw","pitch","roll"} or None
            if pose is None: 
                continue

            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
            txt = f"ID {t['id']} | YPR {pose['yaw']:.1f}/{pose['pitch']:.1f}/{pose['roll']:.1f}"
            cv2.putText(frame, txt, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            draw_axes(frame, (cx,cy), pose["yaw"], pose["pitch"], pose["roll"])

            rec = {"frame": fidx, "id": t["id"], "bbox":[x1,y1,x2,y2], **pose}
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        vw.write(frame)

    cap.release(); vw.release(); logf.close()
    print("[DONE]", out_vid, out_log)

if __name__ == "__main__":
    main()
