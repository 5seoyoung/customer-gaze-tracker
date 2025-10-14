import numpy as np

class Gaze360Estimator:
    def __init__(self, weights_path="third_party/gaze360/trained_models/model.pth"):
        self.ready = False  # 추후 실제 모델 로드로 변경 예정

    def __call__(self, frame, face_xyxy):
        # 임시: 정면 벡터
        return {"gaze_vec": np.array([0.0, 0.0, 1.0]),
                "yaw_pitch_roll": (0.0, 0.0, 0.0)}
