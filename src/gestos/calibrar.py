"""Calibração guiada: pede cada gesto várias vezes na janelinha e grava tudo rotulado.

Uso: gestos-calibrar [--repeticoes 5]

Para o serviço (a câmera só abre num processo), grava em
gravacoes/calibracao-<data>/dados.jsonl e religa o serviço no fim.
Cada linha tem os pontos 2D/3D, o rótulo do momento ("repouso", "nada" ou o
gesto pedido), a fase (aviso, prepara, agora) e o número da repetição.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from . import baixar_modelo, visual

PASTA = Path(__file__).resolve().parents[2] / "gravacoes"

# (rótulo, instrução, segundos de "AGORA" por repetição)
ROTEIRO = [
    ("macaneta_horario", "Gire o pulso como chave, sentido HORARIO, e volte", 2.5),
    ("macaneta_anti_horario", "Gire o pulso como chave, ANTI-HORARIO, e volte", 2.5),
    ("giro_horario", "Dois dedos fazendo CIRCULOS horarios (sem virar o pulso)", 3.5),
    ("giro_anti_horario", "Dois dedos fazendo CIRCULOS anti-horarios", 3.5),
    ("rolar_baixo", "Dois dedos parados, DESCA a mao um pouco e segure", 3.0),
    ("rolar_cima", "Dois dedos parados, SUBA a mao um pouco e segure", 3.0),
    ("pinca_abrir", "Pontas encostadas -> abra em L", 2.5),
    ("pinca_fechar", "Em L -> encoste as pontas", 2.5),
    ("mao_aberta", "Mao aberta, em pe, dedos afastados: segure", 2.5),
    ("joinha", "Joinha: segure", 2.5),
]
NADA_SEGUNDOS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeticoes", type=int, default=5)
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    visual.JANELA = "Calibracao de gestos"  # janela grande, fora da regra do canto
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    servico_ativo = subprocess.run(["systemctl", "--user", "is-active", "--quiet", "gestos"]).returncode == 0
    if servico_ativo:
        subprocess.run(["systemctl", "--user", "stop", "gestos"])
        time.sleep(1)

    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(baixar_modelo())),
        running_mode=RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.6, min_tracking_confidence=0.5,
    ))
    captura = cv2.VideoCapture(0, cv2.CAP_V4L2)
    captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    destino = PASTA / time.strftime("calibracao-%Y%m%d-%H%M%S")
    destino.mkdir(parents=True, exist_ok=True)
    saida = open(destino / "dados.jsonl", "w")
    inicio = time.monotonic()
    ultimo_ts = -1

    def rodar(segundos, rotulo, fase, rep, linhas, destaque=None, cor=(0, 200, 255)):
        """Lê a câmera por `segundos`, gravando e mostrando; False se a janela foi fechada."""
        nonlocal ultimo_ts
        fim = time.monotonic() + segundos
        while time.monotonic() < fim:
            ok, quadro = captura.read()
            if not ok:
                continue
            quadro = cv2.flip(quadro, 1)
            ts = max(int((time.monotonic() - inicio) * 1000), ultimo_ts + 1)
            ultimo_ts = ts
            r = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)), ts)
            p = r.hand_landmarks[0] if r.hand_landmarks else None
            m = r.hand_world_landmarks[0] if r.hand_world_landmarks else None
            saida.write(json.dumps({
                "t": ts, "rotulo": rotulo, "fase": fase, "rep": rep,
                "p": [[round(q.x, 4), round(q.y, 4)] for q in p] if p else None,
                "m": [[round(q.x, 4), round(q.y, 4), round(q.z, 4)] for q in m] if m else None,
            }) + "\n")
            falta = fim - time.monotonic()
            texto = destaque if destaque != "contagem" else f"{falta:.0f}"
            if not visual.mostrar(visual.desenhar(quadro, p, linhas, texto, cor), largura=800):
                return False
        return True

    try:
        total = len(ROTEIRO)
        if not rodar(6, "repouso", "aviso", 0, ["CALIBRACAO DE GESTOS", "Siga as instrucoes na tela.", "Feche a janela para cancelar."], "comecando..."):
            return
        if not rodar(NADA_SEGUNDOS, "nada", "agora", 0, ["Mexa a mao A TOA (sem gestos):", "digite, coce o rosto, gesticule"], "contagem", (180, 180, 180)):
            return
        for n, (rotulo, instrucao, duracao) in enumerate(ROTEIRO, 1):
            cab = [f"[{n}/{total}] {instrucao}"]
            if not rodar(4, "repouso", "aviso", 0, cab + ["Prepare-se..."], rotulo.replace("_", " ")):
                return
            for rep in range(1, args.repeticoes + 1):
                linhas = cab + [f"repeticao {rep}/{args.repeticoes}"]
                if not rodar(1.5, "repouso", "prepara", rep, linhas, "prepara", (180, 180, 180)):
                    return
                if not rodar(duracao, rotulo, "agora", rep, linhas, "AGORA!", (0, 230, 0)):
                    return
        rodar(3, "repouso", "aviso", 0, ["Pronto! Obrigado."], "fim")
        print(destino)
    except KeyboardInterrupt:
        print("cancelado")
    finally:
        saida.close()
        captura.release()
        detector.close()
        visual.fechar()
        if servico_ativo:
            subprocess.run(["systemctl", "--user", "start", "gestos"])
