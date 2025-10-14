from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self, max_age=30, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, frame, dets_xyxy_scores):
        tlbr = [[*d[:4]] for d in dets_xyxy_scores]
        conf = [d[4] for d in dets_xyxy_scores]
        tracks = self.tracker.update_tracks(tlbr, frame=frame, confidences=conf)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = [int(v) for v in t.to_tlbr()]
            out.append({"id": int(t.track_id), "tlbr": [x1,y1,x2,y2], "state":"confirmed"})
        return out
