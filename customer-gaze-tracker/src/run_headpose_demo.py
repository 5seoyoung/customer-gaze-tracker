from __future__ import annotations
import argparse, os, json
import cv2

from src.detector.yolo_face_adapter import YOLOFaceDetector
from src.pose.repnet_adapter import RepNetHeadPose
from src.tracker.deepsort_adapter import DeepSortAdapter as DeepSORTTracker

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="data/samples/store_01.mp4")
    ap.add_argument("--out", type=str, default="data/outputs/overlay.mp4")
    ap.add_argument("--face-weights", type=str, default="data/weights/yolov12n-face.pt")
    ap.add_argument("--pose-weights", type=str, default="data/weights/6DRepNet_300W_LP_AFLW2000.pth")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="cpu")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    det  = YOLOFaceDetector(model_path=args.face_weights, conf=args.conf, device=args.device)
    trk  = DeepSORTTracker()  # ✅ DeepSORTTracker로 생성
    pose = RepNetHeadPose(weights_path=args.pose_weights, device=args.device)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    per_frame_log = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = det.infer(frame)                 # [(x1,y1,x2,y2,conf), ...]
        tracks = trk.update(frame, dets)        # [{"id","tlbr":[...]}...]

        tlbrs = [tuple(t["tlbr"]) for t in tracks]
        ypr = pose.infer(frame, tlbrs) if tlbrs else []

        for t, (yaw, pitch, roll) in zip(tracks, ypr):
            box = tuple(t["tlbr"])
            RepNetHeadPose.draw_axis_on_face(frame, box, yaw, pitch, roll, color=(0, 255, 255))
            x1, y1 = int(box[0]), int(box[1])
            cv2.putText(frame, f"ID {t['id']}", (x1, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"ID {t['id']}", (x1, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)

        frame_log = {
            "frame_idx": idx,
            "tracks": [
                {"id": int(t["id"]), "tlbr": [float(v) for v in t["tlbr"]],
                 "yaw": float(ypr[i][0]), "pitch": float(ypr[i][1]), "roll": float(ypr[i][2])}
                for i, t in enumerate(tracks)
            ]
        }
        per_frame_log.append(frame_log)
        idx += 1

    writer.release()
    cap.release()

    with open("data/outputs/headpose_log.json", "w") as f:
        json.dump(per_frame_log, f)

    print("[DONE] overlay:", args.out)
    print("[DONE] log: data/outputs/headpose_log.json")

if __name__ == "__main__":
    main()
