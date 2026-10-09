"""Classificador de poses treinado com as mãos do próprio usuário.

O MediaPipe acha os 21 pontos da mão; daqui saem as características
(pontos 2D relativos ao pulso e em escala da mão + pontos 3D em metros) e um
classificador pequeno (scikit-learn) diz qual pose é.

  gestos-treinar gravar   grava exemplos guiados (janela com instruções)
  gestos-treinar          treina com todas as gravações e mostra o placar
"""

import argparse
import json
import os
import pickle
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parents[2] / "gravacoes"
MODELO = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gestos" / "modelo-poses.pkl"

# (classe, instrução na tela). "nada" = mão à toa; palm e fist são poses de descanso.
CLASSES = [
    ("like", "JOINHA (polegar para cima)"),
    ("dislike", "POLEGAR PARA BAIXO"),
    ("one", "INDICADOR PARA CIMA"),
    ("one_down", "INDICADOR PARA BAIXO"),
    ("point_right", "APONTANDO PARA A DIREITA"),
    ("point_left", "APONTANDO PARA A ESQUERDA"),
    ("peace", "PAZ E AMOR (V)"),
    ("ok", "OK (polegar e indicador em circulo)"),
    ("palm", "MAO ABERTA, palma para a camera"),
    ("fist", "PUNHO FECHADO"),
    ("nada", "MAO A TOA: digite, coce, gesticule, abaixe"),
]


def caracteristicas(pontos, mundo):
    """Vetor de características de uma mão (ou None)."""
    if not pontos:
        return None
    p = np.array([[q.x, q.y] for q in pontos], dtype=np.float32)
    p -= p[0]
    escala = np.linalg.norm(p, axis=1).max() or 1.0
    p /= escala
    partes = [p.ravel()]
    if mundo:
        m = np.array([[q.x, q.y, q.z] for q in mundo], dtype=np.float32)
        m -= m[0]
        partes.append((m / (np.linalg.norm(m, axis=1).max() or 1.0)).ravel())
    else:
        partes.append(np.zeros(63, dtype=np.float32))
    return np.concatenate(partes)


class ClassificadorPoses:
    def __init__(self, caminho=MODELO):
        with open(caminho, "rb") as f:
            dados = pickle.load(f)
        self.modelo, self.classes = dados["modelo"], dados["classes"]

    def __call__(self, pontos, mundo):
        """(classe, probabilidade) da mão, ou (None, 0)."""
        x = caracteristicas(pontos, mundo)
        if x is None:
            return None, 0.0
        prob = self.modelo.predict_proba(x[None])[0]
        i = int(np.argmax(prob))
        return self.classes[i], float(prob[i])


# ---------------------------------------------------------------- gravação

def gravar(segundos=12):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    from . import baixar_modelo, visual

    ativo = subprocess.run(["systemctl", "--user", "is-active", "--quiet", "gestos"]).returncode == 0
    if ativo:
        subprocess.run(["systemctl", "--user", "stop", "gestos"])
        time.sleep(1)
    visual.JANELA = "Treino de gestos"
    det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(baixar_modelo())), running_mode=RunningMode.VIDEO, num_hands=1))
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(3, 640)
    cap.set(4, 480)
    destino = PASTA / time.strftime("treino-%Y%m%d-%H%M%S")
    destino.mkdir(parents=True, exist_ok=True)
    saida = open(destino / "dados.jsonl", "w")
    inicio = time.monotonic()
    ts_ant = -1
    etapas = [("prep", "Treino: siga as instrucoes. Feche a janela p/ cancelar.", 4)]
    for classe, texto in CLASSES:
        etapas += [("prep", "PROXIMA: " + texto, 3), (classe, texto, segundos)]
    try:
        for classe, texto, duracao in etapas:
            fim = time.monotonic() + duracao
            while time.monotonic() < fim:
                ok, f = cap.read()
                if not ok:
                    continue
                f = cv2.flip(f, 1)
                ts = max(int((time.monotonic() - inicio) * 1000), ts_ant + 1)
                ts_ant = ts
                r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), ts)
                p = r.hand_landmarks[0] if r.hand_landmarks else None
                m = r.hand_world_landmarks[0] if r.hand_world_landmarks else None
                if classe != "prep":
                    saida.write(json.dumps({
                        "classe": classe, "t": ts,
                        "p": [[round(q.x, 4), round(q.y, 4)] for q in p] if p else None,
                        "m": [[round(q.x, 4), round(q.y, 4), round(q.z, 4)] for q in m] if m else None,
                    }) + "\n")
                falta = fim - time.monotonic()
                if classe == "prep":
                    linhas, destaque, cor = [texto], "troque a pose", (180, 180, 180)
                else:
                    linhas = ["AGORA: " + texto, "mexa um pouco: perto, longe, incline, gire"]
                    destaque, cor = f"FAZENDO {falta:.0f}", (0, 230, 0)
                if not visual.mostrar(visual.desenhar(f, p, linhas, destaque, cor), largura=800):
                    print("cancelado")
                    return
        print(destino)
    finally:
        saida.close()
        cap.release()
        det.close()
        visual.fechar()
        if ativo:
            subprocess.run(["systemctl", "--user", "start", "gestos"])


# ---------------------------------------------------------------- treino

def carregar_exemplos():
    from types import SimpleNamespace
    X, y, bloco = [], [], []
    for arq in sorted(PASTA.glob("treino-*/dados.jsonl")):
        linhas = [json.loads(l) for l in open(arq)]
        por_classe = Counter(l["classe"] for l in linhas)
        vistos = Counter()
        for l in linhas:
            vistos[l["classe"]] += 1
            if not l["p"]:
                continue
            p = [SimpleNamespace(x=a, y=b) for a, b in l["p"]]
            m = [SimpleNamespace(x=a, y=b, z=c) for a, b, c in l["m"]] if l["m"] else None
            X.append(caracteristicas(p, m))
            y.append(l["classe"])
            # último terço de cada classe em cada gravação vira teste (quadros vizinhos são quase iguais)
            bloco.append("teste" if vistos[l["classe"]] > por_classe[l["classe"]] * 2 / 3 else "treino")
    return np.array(X), np.array(y), np.array(bloco)


def treinar():
    from sklearn.metrics import confusion_matrix
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y, bloco = carregar_exemplos()
    if not len(X):
        raise SystemExit("Nenhuma gravação de treino. Rode: gestos-treinar gravar")
    print(f"{len(X)} exemplos: {dict(Counter(y))}")
    tr, te = bloco == "treino", bloco == "teste"
    candidatos = {
        "svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=600, random_state=0)),
    }
    melhor, nota_melhor = None, -1
    for nome, modelo in candidatos.items():
        modelo.fit(X[tr], y[tr])
        nota = (modelo.predict(X[te]) == y[te]).mean()
        print(f"{nome}: {nota:.1%} de acerto nos exemplos de teste")
        if nota > nota_melhor:
            melhor, nota_melhor = nome, nota

    modelo = candidatos[melhor]
    classes = sorted(set(y))
    pred = modelo.predict(X[te])
    cm = confusion_matrix(y[te], pred, labels=classes)
    print(f"\nescolhido: {melhor}. Acerto por pose (teste):")
    for i, c in enumerate(classes):
        total = cm[i].sum()
        erros = {classes[j]: int(cm[i, j]) for j in np.argsort(-cm[i]) if j != i and cm[i, j]}
        print(f"  {c:12} {cm[i, i] / max(total, 1):6.1%}  {dict(list(erros.items())[:3]) or ''}")

    modelo.fit(X, y)  # modelo final com todos os exemplos
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELO, "wb") as f:
        pickle.dump({"modelo": modelo, "classes": list(modelo.classes_)}, f)
    print(f"\nmodelo salvo em {MODELO}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("acao", nargs="?", default="treinar", choices=["gravar", "treinar"])
    parser.add_argument("--segundos", type=float, default=12)
    args = parser.parse_args()
    gravar(args.segundos) if args.acao == "gravar" else treinar()
