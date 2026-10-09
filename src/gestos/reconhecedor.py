"""Transforma os 21 pontos da mão (MediaPipe) em gestos.

Coordenadas normalizadas (0..1), com a imagem já espelhada: x cresce para a
direita do usuário e y cresce para baixo.
"""

import math
import time
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


def classificar_pose(p):
    """Pose instantânea da mão: pinca, dois_dedos, mao_aberta, joinha, punho ou None."""
    s = escala(p)
    dedos = [dedo_esticado(p, d) for d in (INDICADOR, MEDIO, ANELAR, MINIMO)]
    polegar = polegar_esticado(p)

    # pinça: pontas do polegar e indicador juntas, mas longe da palma (no punho
    # fechado elas também se encostam, só que perto do pulso)
    pontas_juntas = dist(p[POLEGAR[3]], p[INDICADOR[3]]) < 0.28 * s
    if pontas_juntas and not dedos[0] and dist(p[INDICADOR[3]], p[PULSO]) > 1.1 * s:
        return "pinca"
    if dedos == [True, True, False, False]:
        return "dois_dedos"
    if all(dedos) and polegar:
        return "mao_aberta"
    if not any(dedos):
        # polegar para cima: ponta bem acima da base e da mão fechada
        if polegar and p[POLEGAR[3]].y < p[POLEGAR[2]].y < p[POLEGAR[1]].y and p[POLEGAR[3]].y < p[INDICADOR[1]].y - 0.3 * s:
            return "joinha"
        return "punho"
    return None


@dataclass
class Reconhecedor:
    """Acompanha as poses ao longo do tempo e emite eventos de gesto."""

    swipe_distancia: float = 0.15  # fração da largura da imagem
    swipe_janela: float = 0.7  # segundos para completar o movimento
    segurar: float = 0.8  # segundos parado para mao_aberta/joinha/punho
    pinca_passo: float = 0.04  # deslocamento vertical por passo de volume
    frames_estaveis: int = 3

    _pose: str | None = None
    _candidata: str | None = None
    _contagem: int = 0
    _inicio_pose: float = 0.0
    _ancora: tuple | None = None
    _historico: list = field(default_factory=list)
    _disparou: bool = False

    def atualizar(self, pontos, agora=None):
        """Recebe os pontos (ou None se não há mão) e devolve a lista de eventos."""
        agora = time.monotonic() if agora is None else agora
        pose = classificar_pose(pontos) if pontos else None

        # só troca de pose depois de alguns frames iguais, para não piscar
        if pose == self._candidata:
            self._contagem += 1
        else:
            self._candidata, self._contagem = pose, 1
        if self._candidata != self._pose and self._contagem >= self.frames_estaveis:
            self._pose = self._candidata
            self._inicio_pose = agora
            self._ancora = None
            self._historico.clear()
            self._disparou = False

        if self._pose is None or not pontos:
            return []

        if self._pose == "dois_dedos":
            return self._swipe(pontos, agora)
        if self._pose == "pinca":
            # 0,3 s de pinça antes de mexer no volume, para ignorar transições
            return self._pinca(pontos) if agora - self._inicio_pose >= 0.3 else []
        if not self._disparou and agora - self._inicio_pose >= self.segurar:
            self._disparou = True
            return [self._pose]
        return []

    def _swipe(self, p, agora):
        x = (p[INDICADOR[3]].x + p[MEDIO[3]].x) / 2
        y = (p[INDICADOR[3]].y + p[MEDIO[3]].y) / 2
        self._historico.append((agora, x, y))
        self._historico = [h for h in self._historico if agora - h[0] <= self.swipe_janela]
        if self._disparou:
            # espera a mão parar antes de aceitar outro swipe
            if len(self._historico) > 2 and _amplitude(self._historico) < self.swipe_distancia / 4:
                self._disparou = False
                self._historico.clear()
            return []

        _, x0, y0 = self._historico[0]
        dx, dy = x - x0, y - y0
        if max(abs(dx), abs(dy)) < self.swipe_distancia:
            return []
        self._disparou = True
        if abs(dx) > abs(dy):
            return ["swipe_direita" if dx > 0 else "swipe_esquerda"]
        return ["swipe_baixo" if dy > 0 else "swipe_cima"]

    def _pinca(self, p):
        y = (p[POLEGAR[3]].y + p[INDICADOR[3]].y) / 2
        if self._ancora is None:
            self._ancora = (y,)
            return []
        passos = int((self._ancora[0] - y) / self.pinca_passo)
        if passos == 0:
            return []
        self._ancora = (self._ancora[0] - passos * self.pinca_passo,)
        evento = "pinca_cima" if passos > 0 else "pinca_baixo"
        return [evento] * abs(passos)


def _amplitude(historico):
    xs = [h[1] for h in historico]
    ys = [h[2] for h in historico]
    return max(max(xs) - min(xs), max(ys) - min(ys))
