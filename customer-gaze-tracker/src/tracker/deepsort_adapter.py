from __future__ import annotations
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortAdapter:
    """
    간단 어댑터:
    - update(frame, dets)
        dets: List[Tuple[x1,y1,x2,y2,conf]]  (얼굴 박스)
    - return:
        List[{"id": int, "tlbr": (x1,y1,x2,y2)}]
    """
    def __init__(self,
                 max_age: int = 30,
                 n_init: int = 3,
                 nn_budget: int = 100,
                 max_cosine_distance: float = 0.2,
                 # ✅ 임베더 기본 활성화 (CPU)
                 embedder: str | None = "mobilenet",
                 embedder_gpu: bool = False):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,          # "mobilenet" (PyTorch) 임베더 사용
            embedder_gpu=embedder_gpu,  # CPU로 고정
        )

    def update(self, frame, dets):
        """
        dets: list of (x1,y1,x2,y2,conf) 또는 (x1,y1,x2,y2,conf,cls)
        """
        if dets is None:
            dets = []

        ds_in = []
        for d in dets:
            if len(d) == 5:
                x1,y1,x2,y2,conf = d
                cls = 0
            elif len(d) >= 6:
                x1,y1,x2,y2,conf,cls = d[:6]
            else:
                continue
            ds_in.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)])

        tracks = self.tracker.update_tracks(ds_in, frame=frame)
        outs = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tlbr = t.to_tlbr()
            outs.append({
                "id": int(t.track_id),
                "tlbr": (float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
            })
        return outs
