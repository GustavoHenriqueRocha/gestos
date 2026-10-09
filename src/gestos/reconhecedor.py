"""Transforma os 21 pontos da mão (MediaPipe) em gestos.

Coordenadas normalizadas (0..1), com a imagem já espelhada: x cresce para a
direita do usuário e y cresce para baixo.

Limites ajustados com gravações reais (ver analisar.py e sessao-guiada.sh).
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

# Índices dos pontos da mão no MediaPipe
PULSO = 0
POLEGAR = (1, 2, 3, 4)  # cmc, mcp, ip, ponta
INDICADOR = (5, 6, 7, 8)  # mcp, pip, dip, ponta
MEDIO = (9, 10, 11, 12)
ANELAR = (13, 14, 15, 16)
MINIMO = (17, 18, 19, 20)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def dedo_esticado(p, dedo):
    """Dedo esticado = ponta mais longe do pulso do que a articulação do meio."""
    mcp, pip, _, ponta = dedo
    return dist(p[ponta], p[PULSO]) > dist(p[pip], p[PULSO]) * 1.15


def polegar_esticado(p):
    _, mcp, ip, ponta = POLEGAR
    return dist(p[ponta], p[MINIMO[0]]) > dist(p[ip], p[MINIMO[0]]) * 1.1 and dist(p[ponta], p[INDICADOR[0]]) > dist(p[mcp], p[INDICADOR[0]])


def escala(p):
    """Tamanho da mão: pulso até a base do dedo médio."""
    return dist(p[PULSO], p[MEDIO[0]]) or 1e-6


def abertura_pinca(p):
    """Distância entre as pontas do polegar e do indicador, relativa ao tamanho da mão."""
    return dist(p[POLEGAR[3]], p[INDICADOR[3]]) / escala(p)


def zona_pinca(p):
    """'fechada' (pontas encostadas), 'aberta' ("L" largo) ou None.

    As zonas são bem separadas de propósito: a mão relaxada fica numa
    meia-pinça (abertura ~0,35) e não pode virar volume.
    """
    if any(dedo_esticado(p, d) for d in (MEDIO, ANELAR, MINIMO)):
        return None
    abertura = abertura_pinca(p)
    if abertura > 0.9 and dedo_esticado(p, INDICADOR) and polegar_esticado(p):
        return "aberta"
    # no punho fechado as pontas também se encostam, mas com o indicador encolhido
    if abertura < 0.25 and dist(p[INDICADOR[3]], p[PULSO]) > 1.15 * escala(p):
        return "fechada"
    return None


def mao_inteira_visivel(p, margem=0.02):
    return all(-margem <= q.x <= 1 + margem and -margem <= q.y <= 1 + margem for q in p)


def classificar_pose(p):
    """Pose instantânea: pinca, dois_dedos, mao_aberta, joinha, punho ou None."""
    s = escala(p)
    indicador, medio, anelar, minimo = (dedo_esticado(p, d) for d in (INDICADOR, MEDIO, ANELAR, MINIMO))
    polegar = polegar_esticado(p)

    if indicador and medio and not anelar and not minimo:
        return "dois_dedos"
    if zona_pinca(p):
        return "pinca"
    if not mao_inteira_visivel(p):
        return None  # poses paradas só com a mão inteira na imagem
    if indicador and medio and anelar and minimo and polegar:
        # dedos para cima; mão pendurada no pé da imagem não conta
        return "mao_aberta" if p[MEDIO[3]].y < p[PULSO].y else None
    if not (indicador or medio or anelar or minimo):
        if polegar and p[POLEGAR[3]].y < p[POLEGAR[2]].y < p[POLEGAR[1]].y and p[POLEGAR[3]].y < p[INDICADOR[1]].y - 0.3 * s:
            return "joinha"
        return "punho"
    return None


@dataclass
class Reconhecedor:
    """Acompanha as poses ao longo do tempo e emite eventos de gesto."""

    swipe_distancia: float = 0.15  # fração da largura da imagem
    swipe_janela: float = 0.6  # segundos para completar o movimento
    segurar: float = 0.8  # segundos parado para mao_aberta/joinha/punho
    pinca_janela: float = 1.0  # segundos para ir de uma zona da pinça à outra
    frames_estaveis: int = 3

    _pose: str | None = None
    _candidata: str | None = None
    _contagem: int = 0
    _inicio_pose: float = 0.0
    _disparou: bool = False
    _zona: tuple = (None, 0.0, 0)  # (zona, desde quando, quadros seguidos)
    _ultima_zona: tuple = (None, 0.0)  # (zona confirmada, quando saiu dela)
    # swipe: posição e pose bruta dos últimos quadros com mão
    _rastro: deque = field(default_factory=lambda: deque(maxlen=60))
    _repouso: tuple | None = None  # (t, x, y, pose) do último momento parado
    _ultimo_swipe: tuple = (0.0, None)  # (t, direção)

    def atualizar(self, pontos, agora=None):
        """Recebe os pontos (ou None se não há mão) e devolve a lista de eventos."""
        agora = time.monotonic() if agora is None else agora
        pose = classificar_pose(pontos) if pontos else None
        eventos = self._swipe(pontos, pose, agora) + self._pinca(pontos, agora)

        # só troca de pose depois de alguns frames iguais, para não piscar
        if pose == self._candidata:
            self._contagem += 1
        else:
            self._candidata, self._contagem = pose, 1
        if self._candidata != self._pose and self._contagem >= self.frames_estaveis:
            self._pose = self._candidata
            self._inicio_pose = agora
            self._disparou = False

        if self._pose in (None, "dois_dedos", "pinca") or not pontos:
            return eventos
        if not self._disparou and agora - self._inicio_pose >= self.segurar:
            self._disparou = True
            eventos.append(self._pose)
        return eventos

    def _swipe(self, p, pose, agora):
        """Swipe = movimento rápido saindo do repouso, com dois dedos.

        Durante o movimento a imagem borra e a pose some, então a pose é
        conferida no repouso de onde o movimento saiu. A volta da mão (direção
        oposta logo em seguida) é ignorada.
        """
        if not p:
            self._rastro.clear()
            self._repouso = None
            return []
        x = (p[INDICADOR[0]].x + p[MEDIO[0]].x) / 2  # base dos dedos: borra menos que as pontas
        y = (p[INDICADOR[0]].y + p[MEDIO[0]].y) / 2
        self._rastro.append((agora, x, y, pose))

        recentes = [r for r in self._rastro if agora - r[0] <= 0.2]
        if len(recentes) >= 3 and _amplitude(recentes) < 0.03:
            self._repouso = (agora, x, y, pose)
            return []
        if self._repouso is None or agora - self._repouso[0] > self.swipe_janela:
            return []

        _, x0, y0, pose0 = self._repouso
        dx, dy = x - x0, y - y0
        if max(abs(dx), abs(dy)) < self.swipe_distancia:
            return []
        movimento = [r[3] for r in self._rastro if r[0] >= self._repouso[0]]
        if pose0 != "dois_dedos" and movimento.count("dois_dedos") < 0.4 * len(movimento):
            return []

        if abs(dx) > abs(dy):
            direcao = "swipe_direita" if dx > 0 else "swipe_esquerda"
        else:
            direcao = "swipe_baixo" if dy > 0 else "swipe_cima"
        self._repouso = None  # precisa parar de novo para o próximo swipe
        t_ant, dir_ant = self._ultimo_swipe
        if dir_ant == _OPOSTO[direcao] and agora - t_ant < 1.2:
            return []  # é a mão voltando
        self._ultimo_swipe = (agora, direcao)
        return [direcao]

    def _pinca(self, p, agora):
        """Passar de pinça fechada para aberta (ou o contrário) em até `pinca_janela`."""
        zona = zona_pinca(p) if p else None
        atual, desde, n = self._zona
        n = n + 1 if zona == atual else 1
        self._zona = (zona, desde if zona == atual else agora, n)
        if zona is None or n != 3:  # zona confirmada no 3º quadro seguido
            if zona is None and atual is not None and n == 1:
                self._ultima_zona = (atual, agora)
            return []
        anterior, saiu = self._ultima_zona
        self._ultima_zona = (zona, agora)
        if anterior is None or anterior == zona or agora - saiu > self.pinca_janela:
            return []
        return ["pinca_abrir" if zona == "aberta" else "pinca_fechar"]


_OPOSTO = {
    "swipe_direita": "swipe_esquerda", "swipe_esquerda": "swipe_direita",
    "swipe_cima": "swipe_baixo", "swipe_baixo": "swipe_cima",
}


def _amplitude(rastro):
    xs = [r[1] for r in rastro]
    ys = [r[2] for r in rastro]
    return max(max(xs) - min(xs), max(ys) - min(ys))
