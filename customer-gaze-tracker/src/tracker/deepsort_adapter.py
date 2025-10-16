# src/tracker/deepsort_adapter.py
from typing import List, Tuple, Dict, Any
import math

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as e:
    raise RuntimeError("pip install deep-sort-realtime 가 필요합니다.") from e


class DeepSortTracker:
    """
    DeepSORT 어댑터 (PyPI: deep-sort-realtime)

    Contract:
      update(frame, dets_xyxy_scores) -> List[TrackDict]
      - dets_xyxy_scores: List[(x1,y1,x2,y2,score)]
      - 반환: [{"id": int, "tlbr": [x1,y1,x2,y2], "state": "confirmed"}, ...]

    주의:
      - update_tracks()의 시그니처는 detections, frame=None 만 받습니다.
      - detections 포맷: [([x1,y1,x2,y2], conf, class_name_or_id), ...]
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: int = None,
    ) -> None:
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

    def update(self, frame, dets_xyxy_scores: List[Tuple[int, int, int, int, float]]) -> List[Dict[str, Any]]:
        # DeepSort 입력 포맷으로 변환
        detections = []
        for x1, y1, x2, y2, conf in dets_xyxy_scores:
            # class_name/id는 None 또는 문자열 사용 가능. 여기선 "face"로 태깅.
            detections.append(([int(x1), int(y1), int(x2), int(y2)], float(conf), "face"))

        # 핵심: confidences= 같은 키워드 인자 없이 호출
        tracks = self.tracker.update_tracks(detections, frame=frame)

        out = []
        for t in tracks:
            if not getattr(t, "is_confirmed", lambda: True)():
                # 상태가 confirmed가 아닐 경우 스킵 (초기화 중 등)
                continue

            # tlbr 좌표 획득 (버전별 메서드명이 다를 수 있어 안전 처리)
            if hasattr(t, "to_tlbr"):
                l, t_, r, b = t.to_tlbr()
            elif hasattr(t, "to_ltrb"):
                l, t_, r, b = t.to_ltrb()
            else:
                # tlwh만 있을 수 있음
                if hasattr(t, "to_tlwh"):
                    l, t_, w, h = t.to_tlwh()
                elif hasattr(t, "tlwh"):
                    l, t_, w, h = t.tlwh
                else:
                    # 마지막 안전망: bbox 속성에서 가져오기
                    bb = getattr(t, "bbox", None)
                    if bb is not None and len(bb) == 4:
                        l, t_, r, b = bb
                    else:
                        continue
                r, b = l + w, t_ + h

            out.append({
                "id": int(getattr(t, "track_id", -1)),
                "tlbr": [int(l), int(t_), int(r), int(b)],
                "state": "confirmed",
            })
        return out
