# src/pose/repnet_adapter.py
from __future__ import annotations
from typing import List, Tuple
import os, math
import cv2
import numpy as np
import torch

Tlbr = Tuple[float, float, float, float]

def _patch_cuda_noop_for_cpu():
    """
    CPU 환경에서 sixdrepnet이 내부적으로 .cuda()를 호출해도
    에러가 나지 않도록 .cuda()를 no-op으로 패치한다.
    (Module.cuda / Tensor.cuda 모두 커버)
    """
    if torch.cuda.is_available():
        return  # GPU 있으면 패치 불필요

    # Module.cuda -> no-op
    import types
    def _noop_module_cuda(self, device=None):
        return self
    torch.nn.Module.cuda = _noop_module_cuda  # type: ignore[attr-defined]

    # Tensor.cuda -> no-op
    def _noop_tensor_cuda(self, device=None):
        return self
    # torch.Tensor는 C 확장 클래스라 mypy가 타입 경고할 수 있음
    torch.Tensor.cuda = _noop_tensor_cuda  # type: ignore[attr-defined]


class RepNetHeadPose:
    """
    6DRepNet 래퍼 (CPU 안전)
    - weights_path: 6DRepNet 최종 체크포인트(.pth)
    - device: "cpu" | "cuda"
    """
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # ✅ CPU 강제: CUDA 가시성 차단 + .cuda() no-op 패치
        if str(self.device).startswith("cpu"):
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            _patch_cuda_noop_for_cpu()

        # ⬇️ 이제 import (패치가 적용된 상태에서 이뤄져야 함)
        from sixdrepnet import SixDRepNet

        # 기본 모델 생성 (내부가 .cuda()를 호출해도 CPU에선 no-op이므로 안전)
        self.model = SixDRepNet()
        self.model.to(self.device)
        self.model.eval()

        # 우리가 받은 체크포인트로 덮어쓰기 (선택)
        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and any(k.endswith("state_dict") for k in state.keys()):
                for k in ("state_dict", "model_state_dict", "model"):
                    if k in state and isinstance(state[k], dict):
                        state = state[k]
                        break
            try:
                self.model.load_state_dict(state, strict=False)
            except Exception:
                self.model.load_state_dict(
                    {k.replace("module.", ""): v for k, v in state.items()},
                    strict=False
                )

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray, boxes_tlbr: List[Tlbr]) -> List[Tuple[float,float,float]]:
        """
        각 박스에 대해 (yaw, pitch, roll) [deg] 반환
        (SixDRepNet.predict는 (pitch, yaw, roll) 순서를 내므로 재정렬)
        """
        if not boxes_tlbr:
            return []

        out: List[Tuple[float,float,float]] = []
        h, w = frame_bgr.shape[:2]
        for (x1, y1, x2, y2) in boxes_tlbr:
            ix1 = max(0, int(x1)); iy1 = max(0, int(y1))
            ix2 = min(w, int(x2)); iy2 = min(h, int(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                out.append((0.0, 0.0, 0.0)); continue

            crop = frame_bgr[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                out.append((0.0, 0.0, 0.0)); continue

            # sixdrepnet.SixDRepNet.predict -> (pitch, yaw, roll)
            pitch, yaw, roll = self.model.predict(crop)
            out.append((float(yaw), float(pitch), float(roll)))
        return out

    @staticmethod
    def draw_axis_on_face(frame_bgr, box_tlbr: Tlbr,
                          yaw: float, pitch: float, roll: float,
                          color=(0,255,255), length: int = 50, thickness: int = 2):
        x1, y1, x2, y2 = map(int, box_tlbr)
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        yr, pr, rr = map(math.radians, (yaw, pitch, roll))

        x_end = (int(cx + length * (math.cos(yr) * math.cos(rr))),
                 int(cy + length * (math.cos(pr) * math.sin(rr) + math.sin(pr) * math.sin(yr) * math.cos(rr))))
        y_end = (int(cx + length * (-math.cos(yr) * math.sin(rr))),
                 int(cy + length * (math.cos(pr) * math.cos(rr) - math.sin(pr) * math.sin(yr) * math.sin(rr))))
        z_end = (int(cx + length * (math.sin(yr))),
                 int(cy + length * (math.sin(pr) * math.cos(yr))))

        cv2.line(frame_bgr, (cx, cy), x_end, color, thickness)
        cv2.line(frame_bgr, (cx, cy), y_end, color, thickness)
        cv2.line(frame_bgr, (cx, cy), z_end, color, thickness)

DeepSORTTracker = DeepSortAdapter