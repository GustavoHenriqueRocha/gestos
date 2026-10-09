"""Motor de gestos antigo: MediaPipe Hands + regras do reconhecedor.py."""

import time

from .reconhecedor import Reconhecedor


class MotorMediapipe:
    def __init__(self, modelo, ajustes):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self._mp = mp
        self.detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(modelo)),
            running_mode=RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.5,
        ))
        self.reconhecedor = Reconhecedor(**{k: v for k, v in ajustes.items() if hasattr(Reconhecedor, k)})
        self.inicio = time.monotonic()
        self.ultimo_ts = -1
        self.pontos = self.mundo = None
        self.caixa = None
        self.confianca = 0.0

    @property
    def pose(self):
        return self.reconhecedor._pose

    def processar(self, quadro, agora):
        import cv2
        ts = max(int((agora - self.inicio) * 1000), self.ultimo_ts + 1)
        self.ultimo_ts = ts
        rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
        r = self.detector.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        self.pontos = r.hand_landmarks[0] if r.hand_landmarks else None
        self.mundo = r.hand_world_landmarks[0] if r.hand_world_landmarks else None
        eventos = self.reconhecedor.atualizar(self.pontos, agora, mundo=self.mundo)
        self.pontos = self.reconhecedor.pontos
        return eventos

    def registro(self):
        return {
            "p": [[round(q.x, 4), round(q.y, 4)] for q in self.pontos] if self.pontos else None,
            "m": [[round(q.x, 4), round(q.y, 4), round(q.z, 4)] for q in self.mundo] if self.mundo else None,
        }

    def fechar(self):
        self.detector.close()
