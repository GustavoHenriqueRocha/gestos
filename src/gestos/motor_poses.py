"""Motor de gestos: MediaPipe acha a mão, o classificador treinado com as mãos
do usuário (poses.py) diz a pose, e poses seguradas viram eventos."""

import json
import subprocess
import time
from pathlib import Path

from .poses import ClassificadorPoses
from .reconhecedor import ANELAR, INDICADOR, MEDIO, MINIMO, dedo_esticado


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


class Arrasto:
    """Pose de "agarrar" (ex.: pinça): segurou `segurar` s, agarra; subir/descer a
    mão emite `<pose>_mais` / `<pose>_menos` a cada `passo` (fração da altura da
    imagem). Falhas de até `tolerancia` s no meio do movimento não soltam."""

    def __init__(self, pose, segurar=0.3, passo=0.04, tolerancia=0.3):
        self.pose, self.segurar, self.passo, self.tolerancia = pose, segurar, passo, tolerancia
        self.agarrado = False
        self._desde = self._visto = 0.0
        self._ancora = None

    def __call__(self, bruta, pontos, agora):
        if bruta == self.pose and pontos:
            # ponto de pega: meio entre as pontas do polegar e do indicador
            y = (pontos[4].y + pontos[8].y) / 2
            if not self._visto or agora - self._visto > self.tolerancia:
                self._desde = agora
            self._visto = agora
            if not self.agarrado:
                if agora - self._desde >= self.segurar:
                    self.agarrado, self._ancora = True, y
                return []
            passos = int((self._ancora - y) / self.passo)
            if passos:
                self._ancora -= passos * self.passo
                return [f"{self.pose}_mais" if passos > 0 else f"{self.pose}_menos"] * min(abs(passos), 3)
            return []
        if self.agarrado and agora - self._visto > self.tolerancia:
            self.agarrado, self._ancora = False, None
        return []


def _hypr(*args):
    try:
        return subprocess.run(["hyprctl", *args], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


class ArrastoJanela:
    """Agarrar e arrastar a janela ativa: mão abre e fecha (✋ → ✊) para agarrar;
    a janela vira flutuante e segue a mão; abrir a mão solta.

    Exigir a mão aberta logo antes evita agarrar com o punho em repouso."""

    def __init__(self, pose="fist", segurar=0.25, ganho=1.5, aberta_antes=1.0, tolerancia=0.3, avisar=None):
        self.pose, self.segurar, self.ganho = pose, segurar, ganho
        self.aberta_antes, self.tolerancia, self.avisar = aberta_antes, tolerancia, avisar
        self.endereco = None  # janela agarrada
        self._aberta = self._desde = self._visto = 0.0
        self._ultimo = None  # (x, y) da mão no último movimento
        self._resto = [0.0, 0.0]
        self._tela = None

    def __call__(self, bruta, pontos, agora):
        if pontos and sum(dedo_esticado(pontos, d) for d in (INDICADOR, MEDIO, ANELAR, MINIMO)) >= 4:
            self._aberta = agora
        fechada = bruta == self.pose and pontos
        if fechada:
            if not self._visto or agora - self._visto > self.tolerancia:
                self._desde = agora
            self._visto = agora
        if self.endereco:
            if not fechada and agora - self._visto > self.tolerancia:
                self._soltar()
            elif fechada:
                self._mover(pontos)
            return []
        if fechada and agora - self._desde >= self.segurar and self._desde - self._aberta <= self.aberta_antes:
            self._agarrar(pontos)
        return []

    def _agarrar(self, pontos):
        janela = json.loads(_hypr("activewindow", "-j") or "{}")
        if not janela.get("address"):
            return
        self.endereco = janela["address"]
        self._era_flutuante = bool(janela.get("floating"))
        if not janela.get("floating"):
            _hypr("dispatch", f'hl.dsp.window.float({{ action = "enable", window = "address:{self.endereco}" }})')
        monitor = next((m for m in json.loads(_hypr("monitors", "-j") or "[]") if m.get("focused")), None)
        if monitor:
            self._tela = (monitor["width"] / monitor["scale"], monitor["height"] / monitor["scale"])
        self._ultimo = (pontos[MEDIO[0]].x, pontos[MEDIO[0]].y)
        self._resto = [0.0, 0.0]
        if self.avisar:
            self.avisar(f"✊ agarrou: {janela.get('class', 'janela')}")

    def _mover(self, pontos):
        if not self._tela:
            return
        x, y = pontos[MEDIO[0]].x, pontos[MEDIO[0]].y
        dx = (x - self._ultimo[0]) * self._tela[0] * self.ganho + self._resto[0]
        dy = (y - self._ultimo[1]) * self._tela[1] * self.ganho + self._resto[1]
        self._ultimo = (x, y)
        ix, iy = int(dx), int(dy)
        self._resto = [dx - ix, dy - iy]
        if ix or iy:
            _hypr("dispatch", f'hl.dsp.window.move({{ x = {ix}, y = {iy}, relative = true, window = "address:{self.endereco}" }})')

    def _soltar(self):
        endereco, self.endereco = self.endereco, None
        if not self._era_flutuante:
            # volta para o mosaico (dwindle) onde foi largada: leva o cursor até o
            # centro da janela; o dwindle encaixa ao lado da janela sob o cursor
            janela = next((c for c in json.loads(_hypr("clients", "-j") or "[]") if c["address"] == endereco), None)
            if janela:
                cx, cy = (int(v) for v in _hypr("cursorpos").replace(",", " ").split())
                alvo_x = janela["at"][0] + janela["size"][0] // 2
                alvo_y = janela["at"][1] + janela["size"][1] // 2
                subprocess.run([str(Path.home() / ".local/bin/wlrctl-roda"), "pointer", "move",
                                str(alvo_x - cx), str(alvo_y - cy)], capture_output=True, timeout=2)
                time.sleep(0.03)
            _hypr("dispatch", f'hl.dsp.window.float({{ action = "disable", window = "address:{endereco}" }})')
        if self.avisar:
            self.avisar("✋ soltou a janela")


class MotorPoses:
    def __init__(self, modelo_mp, poses, confianca_min=0.8, quadros_estaveis=4, arrastos=None, arrastar_janela=None, avisar=None):
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
        self.arrastos = [Arrasto(nome, **cfg) for nome, cfg in (arrastos or {}).items()]
        self.arrasto_janela = ArrastoJanela(avisar=avisar, **arrastar_janela) if arrastar_janela is not None else None
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
        eventos = self.segurador(self.bruta, agora)
        for arrasto in self.arrastos:
            eventos += arrasto(self.bruta, self.pontos, agora)
        if self.arrasto_janela:
            self.arrasto_janela(self.bruta, self.pontos, agora)
        return eventos

    def registro(self):
        return {
            "classe": self.bruta, "conf": round(self.confianca, 3),
            "p": [[round(q.x, 4), round(q.y, 4)] for q in self.pontos] if self.pontos else None,
            "m": [[round(q.x, 4), round(q.y, 4), round(q.z, 4)] for q in self.mundo] if self.mundo else None,
        }

    def fechar(self):
        self.detector.close()
