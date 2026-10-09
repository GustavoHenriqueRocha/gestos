"""Lê a webcam, reconhece gestos da mão e executa comandos no Omarchy."""

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from pathlib import Path

from .reconhecedor import Reconhecedor

MODELO_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODELO = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gestos" / "hand_landmarker.task"
CONFIG_USUARIO = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gestos" / "gestos.toml"
CONFIG_PADRAO = Path(__file__).resolve().parents[2] / "gestos.toml"


def carregar_config(caminho):
    with open(caminho, "rb") as f:
        return tomllib.load(f)


def baixar_modelo():
    if not MODELO.exists():
        MODELO.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando modelo para {MODELO}...", flush=True)
        urllib.request.urlretrieve(MODELO_URL, MODELO)
    return MODELO


POSES = {
    "dois_dedos": "✌ dois dedos", "pinca": "🤏 pinça", "mao_aberta": "✋ mão aberta",
    "joinha": "👍 joinha", "punho": "✊ punho", None: "· sem gesto",
}


class Painel:
    """Uma notificação fixa no canto, atualizada no lugar (notify-send -r),
    mostrando a pose atual e o último gesto reconhecido."""

    def __init__(self):
        self.id = None
        self.pose = None
        self.gesto = ""
        self._fila = queue.Queue(maxsize=1)
        threading.Thread(target=self._enviar, daemon=True).start()

    def atualizar(self, pose=..., gesto=None):
        if pose is not ...:
            self.pose = pose
        if gesto:
            self.gesto = f"{gesto}  ({time.strftime('%H:%M:%S')})"
        try:
            self._fila.get_nowait()  # só a versão mais nova importa
        except queue.Empty:
            pass
        self._fila.put_nowait((POSES.get(self.pose, self.pose), self.gesto))

    def _enviar(self):
        while True:
            titulo, corpo = self._fila.get()
            cmd = ["notify-send", "-p", "-a", "Gestos", "-u", "low", "-t", "0"]
            if self.id:
                cmd += ["-r", self.id]
            try:
                saida = subprocess.run(cmd + [titulo, corpo or " "], capture_output=True, text=True, timeout=3)
                self.id = saida.stdout.strip() or self.id
            except (OSError, subprocess.TimeoutExpired):
                pass

    def fechar(self):
        if self.id:
            subprocess.run(["notify-send", "-r", self.id, "-a", "Gestos", "-t", "1", "Gestos desligado"],
                           capture_output=True, timeout=3)


def executar(gesto, gestos, painel):
    acao = gestos.get(gesto)
    rotulo = (acao or {}).get("rotulo") or gesto.replace("_", " ")
    if painel:
        painel.atualizar(gesto=rotulo + ("" if acao else "  (sem ação)"))
    if not acao:
        return
    print(f"[gesto] {gesto} → {acao['comando']}", flush=True)
    subprocess.Popen(acao["comando"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="arquivo TOML (padrão: ~/.config/gestos/gestos.toml)")
    parser.add_argument("--debug", action="store_true", help="mostra a pose detectada a cada mudança")
    parser.add_argument("--simular", action="store_true", help="reconhece mas não executa comandos")
    parser.add_argument("--gravar", type=Path, help="salva os pontos da mão de cada quadro (JSONL) para análise")
    args = parser.parse_args()

    caminho = args.config or (CONFIG_USUARIO if CONFIG_USUARIO.exists() else CONFIG_PADRAO)
    config = carregar_config(caminho)
    ajustes = config.get("ajustes", {})
    gestos = config.get("gestos", {})
    cam = config.get("camera", {})
    print(f"Config: {caminho}", flush=True)

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(baixar_modelo())),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ))

    captura = cv2.VideoCapture(cam.get("dispositivo", 0), cv2.CAP_V4L2)
    captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("largura", 640))
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("altura", 480))
    if not captura.isOpened():
        sys.exit("Não consegui abrir a câmera")

    reconhecedor = Reconhecedor(
        swipe_distancia=ajustes.get("swipe_distancia", 0.15),
        segurar=ajustes.get("segurar", 0.8),
        pinca_janela=ajustes.get("pinca_janela", 1.0),
        pinca_repetir=ajustes.get("pinca_repetir", 0.4),
        rolar_zona_morta=ajustes.get("rolar_zona_morta", 0.04),
        rolar_velocidade=ajustes.get("rolar_velocidade", 4.5),
    )
    painel = Painel() if ajustes.get("notificar", True) and not args.simular else None
    pose_painel = object()
    inicio = time.monotonic()
    ultimo_ts = -1
    pose_anterior = None
    quadros, relogio_fps = 0, time.monotonic()
    # SIGTERM (systemctl stop, kill) encerra igual ao Ctrl+C, fechando câmera e gravação
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    gravacao = open(args.gravar, "w") if args.gravar else None
    print("Gestos ativo.", flush=True)

    try:
        while True:
            ok, quadro = captura.read()
            if not ok:
                time.sleep(0.05)
                continue
            quadro = cv2.flip(quadro, 1)  # espelha: direita na imagem = direita do usuário
            rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
            ts = int((time.monotonic() - inicio) * 1000)
            if ts <= ultimo_ts:
                ts = ultimo_ts + 1
            ultimo_ts = ts
            resultado = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            pontos = resultado.hand_landmarks[0] if resultado.hand_landmarks else None

            eventos = reconhecedor.atualizar(pontos)
            if gravacao:
                pts = [[round(q.x, 4), round(q.y, 4)] for q in pontos] if pontos else None
                gravacao.write(json.dumps({"t": ts, "w": round(time.time(), 3), "pose": reconhecedor._pose, "eventos": eventos, "p": pts}) + "\n")
            for gesto in eventos:
                if args.simular:
                    print(f"[gesto] {gesto}", flush=True)
                else:
                    executar(gesto, gestos, painel)
            if painel and reconhecedor._pose != pose_painel:
                pose_painel = reconhecedor._pose
                painel.atualizar(pose=pose_painel)

            if args.debug:
                quadros += 1
                if reconhecedor._pose != pose_anterior:
                    pose_anterior = reconhecedor._pose
                    print(f"[pose] {pose_anterior}", flush=True)
                if time.monotonic() - relogio_fps >= 5:
                    print(f"[fps] {quadros / (time.monotonic() - relogio_fps):.1f}", flush=True)
                    quadros, relogio_fps = 0, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if gravacao:
            gravacao.close()
        if painel:
            painel.fechar()
        captura.release()
        detector.close()
