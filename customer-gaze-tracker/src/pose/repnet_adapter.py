# src/pose/repnet_adapter.py
from __future__ import annotations
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T

# pip로 설치된 패키지의 올바른 경로
from sixdrepnet.model import SixDRepNet


class RepNetHeadPose:
    """
    6DRepNet 어댑터.
    - 입력: BGR frame, 얼굴 bbox 리스트(x1,y1,x2,y2)
    - 출력: 각 bbox별 (yaw, pitch, roll) [degrees]
    """
    def __init__(
        self,
        weights_path: str | None = None,   # 헤드포즈 가중치(pth)
        device: str = "cpu",
        backbone_name: str = "RepVGG-B1g2",
        backbone_file: str | None = None   # RepVGG 백본 가중치(pth). 없으면 ""로 생성자만 통과
    ):
        self.device = torch.device(device)

        # ✅ sixdrepnet은 backbone_file이 필수 인자. 없으면 빈 문자열로 통과.
        bb_file = backbone_file if backbone_file is not None else ""

        # 일부 포크는 deploy=True 옵션을 기대함
        self.model = SixDRepNet(backbone_name=backbone_name, backbone_file=bb_file, deploy=True)
        self.model.to(self.device).eval()

        # (옵션) 헤드포즈 가중치 로드
        if weights_path and os.path.isfile(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            if isinstance(ckpt, dict):
                # 다양한 키 케이스 처리
                for k in ("state_dict", "model_state_dict", "net", "model"):
                    if k in ckpt and isinstance(ckpt[k], dict):
                        ckpt = ckpt[k]
                        break
            try:
                self.model.load_state_dict(ckpt, strict=False)
            except Exception:
                # 키 불일치 대비: 교집합만 로딩
                model_keys = set(self.model.state_dict().keys())
                filt = {k: v for k, v in ckpt.items() if k in model_keys}
                self.model.load_state_dict(filt, strict=False)

        # 전처리 파이프라인
        self.tf = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def infer(self, frame: np.ndarray, bboxes_xyxy: list[tuple[float,float,float,float]]):
        """returns: list[(yaw, pitch, roll)] in degrees"""
        outs = []
        if frame is None:
            return [(0.0, 0.0, 0.0)] * len(bboxes_xyxy)

        h, w = frame.shape[:2]
        for (x1, y1, x2, y2) in bboxes_xyxy:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                outs.append((0.0, 0.0, 0.0)); continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                outs.append((0.0, 0.0, 0.0)); continue

            # BGR -> RGB
            inp = self.tf(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            pred = self.model(inp)  # 보통 (B,3) [yaw, pitch, roll], 라디안/도 버전 섞임

            ypr = pred[0].detach().cpu().numpy().astype(float)

            # 라디안 히유리스틱 → 도 단위로 변환
            if np.mean(np.abs(ypr)) < 3.14:
                ypr = np.degrees(ypr)

            outs.append(tuple(ypr.tolist()))
        return outs
