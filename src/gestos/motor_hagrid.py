"""Motor de gestos com os modelos treinados do HaGRID (ai-forever/dynamic_gestures).

Um detector acha as mãos, um classificador diz qual das 45 poses é cada uma
(com confiança) e o controlador deles reconhece os gestos dinâmicos (swipes,
zoom, toque) a partir da sequência de poses. Por cima disso, aqui:

- poses seguradas: depois de `segurar` segundos com a mesma pose, dispara o
  evento com o nome da pose (ex.: "like"); com `repetir`, repete enquanto segura;
- swipes etc. viram eventos com nomes em português (ex.: "swipe_esquerda").
"""

from pathlib import Path

import numpy as np

from .hagrid.main_controller import MainController
from .hagrid.utils import Event, targets

MODELOS = Path(__file__).parent / "hagrid" / "models"

EVENTOS = {
    Event.SWIPE_LEFT: "swipe_esquerda", Event.SWIPE_LEFT2: "swipe_esquerda", Event.SWIPE_LEFT3: "swipe_esquerda",
    Event.SWIPE_RIGHT: "swipe_direita", Event.SWIPE_RIGHT2: "swipe_direita", Event.SWIPE_RIGHT3: "swipe_direita",
    Event.SWIPE_UP: "swipe_cima", Event.SWIPE_UP2: "swipe_cima", Event.SWIPE_UP3: "swipe_cima",
    Event.SWIPE_DOWN: "swipe_baixo", Event.SWIPE_DOWN2: "swipe_baixo", Event.SWIPE_DOWN3: "swipe_baixo",
    Event.FAST_SWIPE_UP: "swipe_rapido_cima", Event.FAST_SWIPE_DOWN: "swipe_rapido_baixo",
    Event.ZOOM_IN: "zoom_mais", Event.ZOOM_OUT: "zoom_menos",
    Event.TAP: "toque", Event.DOUBLE_TAP: "toque_duplo",
    Event.DRAG: "arrastar", Event.DRAG2: "arrastar", Event.DRAG3: "arrastar",
    Event.DROP: "soltar", Event.DROP2: "soltar", Event.DROP3: "soltar",
}


class MotorHagrid:
    def __init__(self, poses=None, confianca_min=0.6, quadros_estaveis=4):
        """`poses`: {nome_da_pose: {"segurar": s, "repetir": s ou 0}} para as poses que viram eventos."""
        self.ctl = MainController(str(MODELOS / "hand_detector.onnx"), str(MODELOS / "crops_classifier.onnx"))
        self.poses = poses or {}
        self.confianca_min = confianca_min
        self.quadros_estaveis = quadros_estaveis
        self.pose = None  # pose estável atual da mão principal
        self.caixa = None  # (x1, y1, x2, y2) da mão principal
        self.confianca = 0.0
        self.bruta = None
        self._candidata, self._contagem = None, 0
        self._desde = self._ultimo = 0.0
        self._disparou = False

    def processar(self, quadro, agora):
        """Recebe um quadro BGR (já espelhado) e devolve a lista de eventos."""
        ctl = self.ctl
        caixas, probs = ctl.detection_model(quadro)
        bruta = None
        if len(caixas):
            rotulos = ctl.classification_model(quadro, caixas)
            confs = ctl.classification_model.confiancas
            ctl.update(np.concatenate((caixas, np.expand_dims(probs, axis=1)), axis=1), rotulos)
            # mão principal = a maior caixa
            areas = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
            i = int(np.argmax(areas))
            self.caixa = tuple(int(v) for v in caixas[i])
            self.confianca = float(confs[i])
            if self.confianca >= self.confianca_min:
                bruta = targets[int(rotulos[i])]
        else:
            ctl.update(np.empty((0, 5)), None)
            self.caixa, self.confianca = None, 0.0

        eventos = []
        for trk in ctl.tracks:
            acao = trk["hands"].action
            if acao is not None:
                trk["hands"].action = None
                if acao in EVENTOS:
                    eventos.append(EVENTOS[acao])
        self.bruta = bruta
        return eventos + self._segurar(bruta, agora)

    pontos = None  # este motor não tem os pontos da mão (só a caixa)

    def registro(self):
        return {"caixa": self.caixa, "classe": self.bruta, "conf": round(self.confianca, 3)}

    def fechar(self):
        pass

    def _segurar(self, bruta, agora):
        """Pose estável por alguns quadros; dispara depois de `segurar` e repete com `repetir`."""
        if bruta == self._candidata:
            self._contagem += 1
        else:
            self._candidata, self._contagem = bruta, 1
        if self._candidata != self.pose and self._contagem >= self.quadros_estaveis:
            self.pose, self._desde, self._disparou = self._candidata, agora, False
        regra = self.poses.get(self.pose)
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
