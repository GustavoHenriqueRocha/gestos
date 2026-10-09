"""Motor de gestos: MediaPipe acha a mão, o classificador treinado com as mãos
do usuário (poses.py) diz a pose, e poses seguradas viram eventos."""

import time

from .poses import ClassificadorPoses


class Segurador:
    """Pose estável por alguns quadros; dispara depois de `segurar` s e repete a cada `repetir` s."""

    def __init__(self, regras, quadros_estaveis=4):
        self.regras, self.quadros_estaveis = regras, quadros_estaveis
        self.pose = None
        self._candidata, self._contagem = None, 0
        self._desde = self._ultimo = 0.0
        self._disparou = False

    def __call__(self, bruta, agora):
        if bruta == self._candidata:
            self._contagem += 1
        else:
            self._candidata, self._contagem = bruta, 1
        if self._candidata != self.pose and self._contagem >= self.quadros_estaveis:
            self.pose, self._desde, self._disparou = self._candidata, agora, False
        regra = self.regras.get(self.pose)
        if not regra:
            return []
        if not self._disparou:
            if agora - self._desde >= regra.get("segurar", 0.8):
                self._disparou, self._ultimo = True, agora
                return [self.pose]
            return []
        repetir = regra.get("repetir", 0)
        if repetir and agora - self._ultimo >= repetir:
            self._ultimo = agora
            return [self.pose]
        return []


class MotorPoses:
    def __init__(self, modelo_mp, poses, confianca_min=0.8, quadros_estaveis=4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self._mp = mp
        self.detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(modelo_mp)), running_mode=RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.5,
        ))
        self.classificador = ClassificadorPoses()
        self.segurador = Segurador(poses, quadros_estaveis)
        self.confianca_min = confianca_min
        self.inicio = time.monotonic()
        self.ultimo_ts = -1
        self.pontos = self.mundo = None
        self.caixa = None
        self.bruta, self.confianca = None, 0.0

    @property
    def pose(self):
        return self.segurador.pose

    def processar(self, quadro, agora):
        import cv2
        ts = max(int((agora - self.inicio) * 1000), self.ultimo_ts + 1)
        self.ultimo_ts = ts
        r = self.detector.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)), ts)
        self.pontos = r.hand_landmarks[0] if r.hand_landmarks else None
        self.mundo = r.hand_world_landmarks[0] if r.hand_world_landmarks else None
        classe, self.confianca = self.classificador(self.pontos, self.mundo)
        self.bruta = classe if self.confianca >= self.confianca_min else None
        return self.segurador(self.bruta, agora)

    def registro(self):
        return {
            "classe": self.bruta, "conf": round(self.confianca, 3),
            "p": [[round(q.x, 4), round(q.y, 4)] for q in self.pontos] if self.pontos else None,
            "m": [[round(q.x, 4), round(q.y, 4), round(q.z, 4)] for q in self.mundo] if self.mundo else None,
        }

    def fechar(self):
        self.detector.close()
