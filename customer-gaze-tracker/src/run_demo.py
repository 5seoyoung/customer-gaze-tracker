import os, cv2, json, yaml
from tqdm import tqdm

from .detector.yolo_face_adapter import YOLOFaceDetector
from .tracker.deepsort_adapter import DeepSortTracker
from .gaze_estimator.gaze360_adapter import Gaze360Estimator
from .utils.geometry import point_in_which_roi, gaze_intersection_2d

def main(cfg_path="configs/rois_example.yaml", out_dir="data/outputs"):
    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path))
    cap = cv2.VideoCapture(cfg["video_source"])
    assert cap.isOpened(), f"Cannot open {cfg['video_source']}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det = YOLOFaceDetector()
    trk = DeepSortTracker()
    gaze = Gaze360Estimator()

    vw = cv2.VideoWriter(os.path.join(out_dir,"overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    logf = open(os.path.join(out_dir,"gaze_log.jsonl"), "w")
    counts = {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total, desc="Processing")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        dets = det(frame)                   # [(x1,y1,x2,y2,score), ...]
        tracks = trk.update(frame, dets)    # [{"id","tlbr":[...]}...]

        for t in tracks:
            x1,y1,x2,y2 = t["tlbr"]
            g = gaze(frame, (x1,y1,x2,y2))
            (sx,sy),(ex,ey) = gaze_intersection_2d((x1,y1,x2,y2), g["gaze_vec"])

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.arrowedLine(frame,(sx,sy),(ex,ey),(255,0,0),2,tipLength=0.2)
            cv2.putText(frame,f"ID {t['id']}",(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            roi = point_in_which_roi((ex,ey), cfg)
            if roi:
                key = (t["id"], roi)
                counts[key] = counts.get(key, 0) + 1
                rec = {"frame": frame_idx, "track_id": t["id"],
                       "bbox": [x1,y1,x2,y2], "gaze_end": [ex,ey], "roi": roi}
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        vw.write(frame); pbar.update(1)

    cap.release(); vw.release(); logf.close(); pbar.close()

    # 초 단위 요약
    summary = {}
    for (tid, roi), fcnt in counts.items():
        summary.setdefault(str(tid), {})[roi] = round(fcnt / fps, 2)

    with open(os.path.join(out_dir,"gaze_summary.json"),"w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] overlay:", os.path.join(out_dir,"overlay.mp4"))
    print("[DONE] per-frame log:", os.path.join(out_dir,"gaze_log.jsonl"))
    print("[DONE] summary:", os.path.join(out_dir,"gaze_summary.json"))

if __name__ == "__main__":
    main()
