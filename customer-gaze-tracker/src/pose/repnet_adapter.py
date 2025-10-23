from __future__ import annotations
import cv2
import numpy as np
import torch

# --- 안전한 cuda 우회 (CPU에서 에러 방지) ---
_old_cuda = torch.nn.Module.cuda
def _safe_cuda(self, device=None):
    # cuda 사용 가능 + 정수 GPU id일 때만 실제 이동, 그 외엔 no-op
    if isinstance(device, int) and device >= 0 and torch.cuda.is_available():
        return _old_cuda(self, device)
    return self
torch.nn.Module.cuda = _safe_cuda  # 전역 패치

# pip 패키지의 레그레서(편의 래퍼)
from sixdrepnet import SixDRepNet

def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return int(x1), int(y1), int(x2), int(y2)

class RepNetHeadPose:
    """
    - infer(frame, tlbr_boxes) -> [(yaw, pitch, roll), ...]
    - CPU 기본. GPU 쓰려면 device='cuda:0'로 만들고 위 패치를 제거/수정하면 됨.
    """
    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        # SixDRepNet 레그레서: weights_path를 직접 주입하는 API는 없음
        # -> 내부 자동 다운로드(or 캐시 사용)를 그대로 활용
        # (이미 data/weights에 받아둔 pth는 캐시에 복사되어 사용됨)
        self.model = SixDRepNet()  # 내부에서 cuda() 호출하지만 위에서 no-op 처리됨
        self.device = device

    @staticmethod
    def _prepare_face(img_bgr: np.ndarray):
        # SixDRepNet 레그레서의 predict는 BGR 이미지 입력을 받음
        return img_bgr

    def infer(self, frame_bgr: np.ndarray, tlbr_list):
        """
        tlbr_list: [(x1,y1,x2,y2), ...]
        return: [(yaw, pitch, roll), ...]
        """
        if not tlbr_list:
            return []
        h, w = frame_bgr.shape[:2]
        ypr_list = []
        for (x1, y1, x2, y2) in tlbr_list:
            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                ypr_list.append((0.0, 0.0, 0.0))
                continue
            face = frame_bgr[y1:y2, x1:x2].copy()
            face = self._prepare_face(face)
            pitch, yaw, roll = self.model.predict(face)  # NOTE: 레그레서 시그니처는 (pitch,yaw,roll)
            # 우리 파이프라인은 (yaw, pitch, roll)로 맞춰 반환
            ypr_list.append((float(yaw), float(pitch), float(roll)))
        return ypr_list

    @staticmethod
    def draw_axis_on_face(img, box, yaw, pitch, roll, color=(0, 255, 255)):
        # 간단한 가시화: 박스 좌상단에 yaw/pitch/roll 숫자로 표기
        x1, y1, x2, y2 = map(int, box)
        txt = f"y:{yaw:.1f} p:{pitch:.1f} r:{roll:.1f}"
        cv2.putText(img, txt, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
