"""Teste guiado curto: compara, nos mesmos quadros, o detector do HaGRID com o
MediaPipe + classificador do HaGRID no recorte da mão.

Uso: .venv/bin/python comparar_motores.py  (para o serviço e religa no fim)
Grava gravacoes/comparacao-<data>.jsonl e imprime o placar.
"""

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from gestos import baixar_modelo, visual
from gestos.hagrid.onnx_models import HandClassification, HandDetection
from gestos.hagrid.utils import targets
from gestos.motor_hagrid import MODELOS

POSES = [
    ("like", "JOINHA (polegar para cima)"),
    ("dislike", "Polegar para BAIXO"),
    ("one", "Indicador para CIMA"),
    ("one_down", "Indicador para BAIXO"),
    ("peace", "Paz e amor (V)"),
    ("ok", "OK (polegar e indicador em circulo)"),
    ("palm", "Mao aberta, palma para a camera"),
    ("fist", "Punho fechado"),
]
SEGUNDOS = 5


def caixa_mediapipe(pontos, w, h, margem=0.25):
    xs = [q.x * w for q in pontos]
    ys = [q.y * h for q in pontos]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (x2 - x1) * margem, (y2 - y1) * margem
    return [int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w - 1, x2 + mx)), int(min(h - 1, y2 + my))]


def main():
    ativo = subprocess.run(["systemctl", "--user", "is-active", "--quiet", "gestos"]).returncode == 0
    if ativo:
        subprocess.run(["systemctl", "--user", "stop", "gestos"])
        time.sleep(1)
    visual.JANELA = "Comparacao de gestos"
    det = HandDetection(str(MODELOS / "hand_detector.onnx"))
    cls = HandClassification(str(MODELOS / "crops_classifier.onnx"))
    mpd = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(baixar_modelo())), running_mode=RunningMode.VIDEO, num_hands=1))
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(3, 640)
    cap.set(4, 480)
    destino = Path(__file__).parent / "gravacoes" / time.strftime("comparacao-%Y%m%d-%H%M%S.jsonl")
    saida = open(destino, "w")
    inicio = time.monotonic()
    ts_ant = -1
    try:
        etapas = [("nada", "Prepare-se...", 3)]
        for rotulo, texto in POSES:
            etapas += [("prep", "PROXIMA: " + texto, 2.5), (rotulo, "AGORA: " + texto, SEGUNDOS)]
        for rotulo, texto, duracao in etapas:
            fim = time.monotonic() + duracao
            while time.monotonic() < fim:
                ok, f = cap.read()
                if not ok:
                    continue
                f = cv2.flip(f, 1)
                h, w = f.shape[:2]
                ts = max(int((time.monotonic() - inicio) * 1000), ts_ant + 1)
                ts_ant = ts
                # A: detector + classificador do HaGRID
                caixas, _ = det(f)
                a = None
                if len(caixas):
                    areas = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
                    i = int(np.argmax(areas))
                    lab = cls(f, caixas[i:i + 1])
                    a = (targets[int(lab[0])], float(cls.confiancas[0]))
                # B: MediaPipe acha a mão, classificador do HaGRID no recorte
                r = mpd.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), ts)
                b = caixa = None
                if r.hand_landmarks:
                    caixa = caixa_mediapipe(r.hand_landmarks[0], w, h)
                    lab = cls(f, np.array([caixa]))
                    b = (targets[int(lab[0])], float(cls.confiancas[0]))
                saida.write(json.dumps({"rotulo": rotulo, "a": a, "b": b}) + "\n")
                linhas = [texto, f"HaGRID puro: {a[0] if a else '-'}", f"MediaPipe+HaGRID: {b[0] if b else '-'}"]
                destaque = "troque a pose" if rotulo in ("nada", "prep") else f"FAZENDO {fim - time.monotonic():.0f}"
                if not visual.mostrar(visual.desenhar(f, None, linhas, destaque, caixa=caixa, rotulo_caixa=b[0] if b else ""), largura=800):
                    return
    finally:
        saida.close()
        cap.release()
        visual.fechar()
        if ativo:
            subprocess.run(["systemctl", "--user", "start", "gestos"])
    placar(destino)


def placar(arquivo):
    linhas = [json.loads(l) for l in open(arquivo)]
    print(f"{'pose':10} {'HaGRID puro':>22} {'MediaPipe+HaGRID':>22}")
    for rotulo, _ in POSES:
        qs = [q for q in linhas if q["rotulo"] == rotulo]
        if not qs:
            continue
        res = []
        for k in ("a", "b"):
            vistos = [q[k] for q in qs if q[k]]
            certos = sum(1 for c, conf in vistos if c == rotulo)
            outros = Counter(c for c, _ in vistos if c != rotulo).most_common(1)
            res.append(f"vê {len(vistos) * 100 // len(qs):3d}% acerta {certos * 100 // len(qs):3d}%" + (f" ({outros[0][0]})" if outros else ""))
        print(f"{rotulo:10} {res[0]:>34}   {res[1]}")


if __name__ == "__main__":
    main()
