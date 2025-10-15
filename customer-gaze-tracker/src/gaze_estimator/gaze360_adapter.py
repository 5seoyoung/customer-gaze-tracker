# src/gaze_estimator/gaze360_adapter.py
import os, sys, math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Gaze360 repo 파이썬 모듈 경로 추가
TP = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "gaze360", "code")
if TP not in sys.path and os.path.isdir(TP):
    sys.path.append(TP)

# ── 레포 내부 구조가 업데이트될 수 있어 try-import로 안전하게 처리 ──
# 흔한 케이스: code/ 에 model 정의가 있고, get_arch()/GazeModel 등으로 접근
_GAZE_MODEL_FACTORY = None
try:
    # 예: model.py 에 get_arch 같은 팩토리 함수가 있는 경우
    from model import get_arch as _get_arch  # type: ignore
    _GAZE_MODEL_FACTORY = ("get_arch", _get_arch)
except Exception:
    try:
        # 다른 포크/버전: models.py에 GazeLSTM/GazeNet 등 클래스가 있을 수 있음
        from models import GazeLSTM as _GazeNet  # type: ignore
        def _get_arch(num_out=2):
            return _GazeNet()
        _GAZE_MODEL_FACTORY = ("GazeLSTM", _get_arch)
    except Exception:
        pass

def _to_unit_vector_from_yaw_pitch(yaw, pitch):
    """
    yaw: 좌우(+왼쪽), pitch: 상하(+위) [라디안]
    3D 단위 벡터로 변환 (카메라 좌표계 가정)
    """
    dx = math.cos(pitch) * math.sin(yaw)
    dy = math.sin(pitch)
    dz = math.cos(pitch) * math.cos(yaw)
    v = np.array([dx, dy, dz], dtype=np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n

def _square_crop_safe(img, x1, y1, x2, y2, pad=0.2):
    # 얼굴 bbox 중심 기준으로 정사각형 crop (+pad)
    h, w = img.shape[:2]
    bw, bh = (x2 - x1), (y2 - y1)
    cx, cy = x1 + bw/2, y1 + bh/2
    side = int(max(bw, bh) * (1.0 + pad))
    x1n, y1n = int(cx - side/2), int(cy - side/2)
    x2n, y2n = x1n + side, y1n + side
    # 경계 보정
    x1n = max(0, x1n); y1n = max(0, y1n)
    x2n = min(w, x2n); y2n = min(h, y2n)
    crop = img[y1n:y2n, x1n:x2n]
    return crop

class Gaze360Estimator:
    def __init__(self,
                 weights_path="third_party/gaze360/trained_models/model.pth",
                 device=None,
                 input_size=224):
        """
        반환: {"gaze_vec": (dx,dy,dz), "yaw_pitch_roll": (yaw,pitch,roll)}
        roll은 대부분 모델에서 직접 제공하지 않으므로 0.0으로 둠
        """
        self.device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.ready = False

        # 모델 팩토리 확인
        if _GAZE_MODEL_FACTORY is None:
            # 레포 내부 API가 달라졌을 수 있음 → 사용자에게 경로/함수 확인 유도
            print("[Gaze360] could not import model factory from repo. Check 'third_party/gaze360/code/'.")
            return

        # 모델 생성
        name, factory = _GAZE_MODEL_FACTORY
        try:
            self.model = factory()
        except TypeError:
            # 어떤 구현은 출력 차원 인자가 필요할 수 있음
            self.model = factory(num_out=2)  # yaw, pitch

        self.model.to(self.device).eval()

        # 가중치 로드
        assert os.path.isfile(weights_path), f"[Gaze360] weights not found: {weights_path}"
        ckpt = torch.load(weights_path, map_location=self.device)
        # 서로 다른 포맷 호환
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)

        self.ready = True
        torch.set_grad_enabled(False)

        # 전처리 파이프라인 (ImageNet 통상값; 레포 스펙과 다르면 맞춰 조정)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, bgr_face):
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # CHW
        ten = torch.from_numpy(img).unsqueeze(0).to(self.device)  # 1x3xHxW
        return ten

    def __call__(self, frame, face_xyxy):
        """
        face_xyxy: (x1,y1,x2,y2) in frame coords
        """
        if not self.ready:
            # 준비 안 되었으면 정면 벡터 반환(이전 fallback과 동일)
            return {"gaze_vec": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    "yaw_pitch_roll": (0.0, 0.0, 0.0)}

        x1, y1, x2, y2 = map(int, face_xyxy)
        crop = _square_crop_safe(frame, x1, y1, x2, y2, pad=0.2)
        if crop.size == 0:
            return {"gaze_vec": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    "yaw_pitch_roll": (0.0, 0.0, 0.0)}

        ten = self._preprocess(crop)

        # ── 추론 ──
        # 많은 구현에서 (yaw, pitch) 2값을 직접 회귀하거나, 벡터를 예측 후 atan2로 변환
        out = self.model(ten)  # shape: [1, 2] or [1, 3]
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.squeeze(0)

        if out.numel() >= 2:
            # (yaw, pitch) 라디안 가정
            yaw = float(out[0].item())
            pitch = float(out[1].item())
            v = _to_unit_vector_from_yaw_pitch(yaw, pitch)
            return {"gaze_vec": v, "yaw_pitch_roll": (yaw, pitch, 0.0)}

        # 혹시 3D 벡터로 직접 주는 구현이면 정규화 후 사용
        v = out.detach().float().cpu().numpy().reshape(-1)
        if v.shape[0] == 3:
            n = np.linalg.norm(v) + 1e-8
            v = v / n
            # yaw/pitch 역변환(정보용)
            yaw = math.atan2(v[0], v[2])
            pitch = math.asin(v[1])
            return {"gaze_vec": v.astype(np.float32), "yaw_pitch_roll": (yaw, pitch, 0.0)}

        # 파싱 실패 시 정면
        return {"gaze_vec": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "yaw_pitch_roll": (0.0, 0.0, 0.0)}
