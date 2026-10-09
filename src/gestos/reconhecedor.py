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
        # em pé (até 30° de inclinação) e dedos bem afastados: mão cruzada ou
        # relaxada tem os dedos juntos e fica deitada
        inclinacao = math.degrees(math.atan2(p[MEDIO[0]].x - p[PULSO].x, p[PULSO].y - p[MEDIO[0]].y))
        afastados = dist(p[INDICADOR[3]], p[MINIMO[3]]) / s
        return "mao_aberta" if abs(inclinacao) <= 30 and afastados >= 0.7 else None
    if not (indicador or medio or anelar or minimo):
        if polegar and p[POLEGAR[3]].y < p[POLEGAR[2]].y < p[POLEGAR[1]].y and p[POLEGAR[3]].y < p[INDICADOR[1]].y - 0.3 * s:
            return "joinha"
        return "punho"
    return None


@dataclass
class Reconhecedor:
    """Acompanha as poses ao longo do tempo e emite eventos de gesto."""

    swipe_distancia: float = 0.25  # fração da largura da imagem
    swipe_janela: float = 0.5  # segundos para completar o movimento
    swipe_volta: float = 2.0  # segundos em que a direção oposta é tratada como volta da mão
    segurar: float = 0.8  # segundos parado para mao_aberta/joinha/punho
    pinca_janela: float = 1.0  # segundos para ir de uma zona da pinça à outra
    rolar_zona_morta: float = 0.04  # quanto a mão sobe/desce antes de começar a rolar
    rolar_velocidade: float = 4.5  # passos de rolagem por quadro, por unidade além da zona morta
    pinca_repetir: float = 0.4  # segurando a pinça na zona final, repete a cada X s (0 desliga)
    frames_estaveis: int = 3

    _pose: str | None = None
    _candidata: str | None = None
    _contagem: int = 0
    _inicio_pose: float = 0.0
    _disparou: bool = False
    _zona: tuple = (None, 0.0, 0)  # (zona, desde quando, quadros seguidos)
    _ultima_zona: tuple = (None, 0.0)  # (zona confirmada, quando saiu dela)
    _repeticao: tuple = (None, 0.0)  # (evento, quando repetir) enquanto segura a pinça
    _rolar_centro: tuple | None = None  # (x, y) da mão quando a rolagem começou
    _rolar_cancelada: bool = False
    # giro: pontas dos dois dedos no último segundo e ângulo acumulado
    giro_passo: float = 60.0  # graus de giro por passo de rolagem
    _giro_pontos: deque = field(default_factory=lambda: deque(maxlen=30))
    _giro_acumulado: float = 0.0
    _giro_angulo: float | None = None
    _giro_recente: float = 0.0  # último instante em que houve giro de verdade
    _giro_deltas: deque = field(default_factory=deque)  # (t, graus) do último segundo
    # maçaneta: mão aberta girando a partir de um ângulo neutro
    macaneta_limite: float = 35.0  # graus além do neutro para disparar
    _mac_neutro: float | None = None
    _mac_armada: bool = True
    _mac_recentes: deque = field(default_factory=lambda: deque(maxlen=6))
    _mac_visto: float = 0.0
    _palmas: deque = field(default_factory=lambda: deque(maxlen=15))  # (t, ângulo da palma), ~0,5 s
    _mac_saiu: float = 0.0  # quando a mão saiu da zona neutra
    _mac_pulso: deque = field(default_factory=lambda: deque(maxlen=15))  # (x, y) do pulso, ~0,5 s
    _rolar_acumulado: float = 0.0
    # swipe: posição e pose bruta dos últimos quadros com mão
    _rastro: deque = field(default_factory=lambda: deque(maxlen=60))
    _swipe_pausa: float = 0.0
    _ultimo_swipe: tuple = (0.0, None)  # (t, direção)

    def atualizar(self, pontos, agora=None, mundo=None):
        """Recebe os pontos (ou None se não há mão) e devolve a lista de eventos.

        `mundo` são os mesmos pontos em 3D (metros, hand_world_landmarks); com
        eles a maçaneta mede o giro da palma em torno do antebraço."""
        agora = time.monotonic() if agora is None else agora
        pose = classificar_pose(pontos) if pontos else None
        if mundo:
            self._palmas.append((agora, angulo_palma(mundo)))
        eventos = self._giro(pontos, pose, agora) + self._macaneta(pontos, agora, mundo, pose)
        eventos += self._swipe(pontos, pose, agora) + self._pinca(pontos, agora)

        # só troca de pose depois de alguns frames iguais, para não piscar
        if pose == self._candidata:
            self._contagem += 1
        else:
            self._candidata, self._contagem = pose, 1
        if self._candidata != self._pose and self._contagem >= self.frames_estaveis:
            self._pose = self._candidata
            self._inicio_pose = agora
            self._disparou = False

        if self._pose == "dois_dedos" and pontos:
            return eventos + self._rolagem(pontos, agora)
        self._rolar_centro, self._rolar_cancelada = None, False
        if self._pose in (None, "pinca") or not pontos:
            return eventos
        if not self._disparou and agora - self._inicio_pose >= self.segurar:
            self._disparou = True
            eventos.append(self._pose)
        return eventos

    def _swipe(self, p, pose, agora):
        """Swipe = varredura longa, rápida e reta com dois dedos.

        Não exige repouso antes: a mão costuma entrar já em movimento, vindo de
        fora da imagem. Durante o movimento a imagem borra e a pose some, então
        basta parte dos quadros com dois dedos. A volta da mão (direção oposta
        logo em seguida) é ignorada.
        """
        if not p:
            self._rastro.clear()
            return []
        x = (p[INDICADOR[0]].x + p[MEDIO[0]].x) / 2  # base dos dedos: borra menos que as pontas
        y = (p[INDICADOR[0]].y + p[MEDIO[0]].y) / 2
        self._rastro.append((agora, x, y, pose))
        if agora < self._swipe_pausa or agora - self._giro_recente < 0.5:
            return []

        trecho = [r for r in self._rastro if agora - r[0] <= self.swipe_janela]
        _, x0, y0, _ = trecho[0]
        dx, dy = x - x0, y - y0
        if max(abs(dx), abs(dy)) < self.swipe_distancia:
            return []
        if _curva(trecho) > 45:
            return []  # caminho curvo: é giro, não swipe
        poses = [r[3] for r in trecho]
        if poses.count("dois_dedos") < 0.3 * len(poses):
            return []

        if abs(dx) > abs(dy):
            direcao = "swipe_direita" if dx > 0 else "swipe_esquerda"
        else:
            direcao = "swipe_baixo" if dy > 0 else "swipe_cima"
        self._rastro.clear()
        self._swipe_pausa = agora + 0.4  # termina a varredura sem disparar de novo
        t_ant, dir_ant = self._ultimo_swipe
        if dir_ant == _OPOSTO[direcao] and agora - t_ant < self.swipe_volta:
            self._ultimo_swipe = (agora, dir_ant)  # volta da mão: estende a espera
            return []
        self._ultimo_swipe = (agora, direcao)
        return [direcao]

    def _giro(self, p, pose, agora):
        """Dois dedos girando em círculo: horário rola para baixo, anti-horário para cima.

        O centro do círculo é a média das pontas no último segundo; cada
        `giro_passo` graus acumulados no mesmo sentido viram um passo.
        """
        if not p or pose not in ("dois_dedos", None):
            # um quadro ou outro com pose errada não interrompe o giro
            if p and self._giro_pontos and agora - self._giro_pontos[-1][0] < 0.3:
                return []
            self._giro_pontos.clear()
            self._giro_acumulado, self._giro_angulo = 0.0, None
            self._giro_deltas.clear()
            return []
        if dedo_esticado(p, ANELAR) and dedo_esticado(p, MINIMO):
            return []  # mão aberta girando é a maçaneta, não o giro dos dois dedos
        palmas = [a for t, a in self._palmas if agora - t <= 0.5]
        if len(palmas) >= 5 and max(_dif(a, palmas[0]) for a in palmas) > 40:
            self._giro_deltas.clear()
            self._giro_acumulado = 0.0
            return []  # a palma está virando: é maçaneta/chave, não círculo
        x = (p[INDICADOR[3]].x + p[MEDIO[3]].x) / 2
        y = (p[INDICADOR[3]].y + p[MEDIO[3]].y) / 2
        self._giro_pontos.append((agora, x, y))
        while self._giro_pontos and agora - self._giro_pontos[0][0] > 1.0:
            self._giro_pontos.popleft()
        if len(self._giro_pontos) < 10:
            return []
        cx = sum(g[1] for g in self._giro_pontos) / len(self._giro_pontos)
        cy = sum(g[2] for g in self._giro_pontos) / len(self._giro_pontos)
        if math.hypot(x - cx, y - cy) < 0.04:
            return []  # perto demais do centro: ângulo não confiável
        angulo = math.degrees(math.atan2(y - cy, x - cx))
        if self._giro_angulo is None:
            self._giro_angulo = angulo
            return []
        delta = (angulo - self._giro_angulo + 180) % 360 - 180
        self._giro_angulo = angulo
        if abs(delta) > 45:
            return []  # salto: vai-e-vem em linha reta passa pelo centro e o ângulo pula 180°
        self._giro_deltas.append((agora, delta))
        while agora - self._giro_deltas[0][0] > 1.0:
            self._giro_deltas.popleft()
        if self._giro_acumulado and (delta > 0) != (self._giro_acumulado > 0):
            self._giro_acumulado = 0.0  # mudou de sentido
        self._giro_acumulado += delta
        # só vale como giro com quase meia volta no mesmo sentido no último segundo
        giro_total = abs(sum(d for _, d in self._giro_deltas))
        if giro_total >= 100:
            self._giro_recente = agora  # girando: segura swipe e joystick
        if giro_total < 150:
            # ainda não é giro: não guarda saldo para não soltar vários passos de uma vez
            self._giro_acumulado = max(-self.giro_passo, min(self.giro_passo, self._giro_acumulado))
            return []
        if abs(self._giro_acumulado) < self.giro_passo:
            self._giro_recente = agora
            return []
        self._giro_recente = agora
        self._giro_acumulado -= math.copysign(self.giro_passo, self._giro_acumulado)
        # y cresce para baixo, então ângulo crescendo = sentido horário na tela
        return ["giro_horario" if delta > 0 else "giro_anti_horario"]

    def _macaneta(self, p, agora, mundo=None, pose=None):
        """Mão girando como maçaneta: além de `macaneta_limite` graus do neutro
        dispara uma vez; voltar ao neutro rearma sem disparar nada.

        Com pontos 3D mede o giro da palma em torno do antebraço (vale com os
        dedos abertos ou curvados, como segurando uma esfera). Sem 3D, usa a
        inclinação da mão na imagem.
        """
        valida = p and escala(p) >= 0.12 and pose not in ("pinca", "punho")
        if valida and mundo:
            angulo = angulo_palma(mundo)
        elif valida and sum(dedo_esticado(p, d) for d in (INDICADOR, MEDIO, ANELAR, MINIMO)) >= 3:
            angulo = math.degrees(math.atan2(p[MEDIO[0]].x - p[PULSO].x, p[PULSO].y - p[MEDIO[0]].y))
        elif p and self._mac_neutro is not None and agora - self._mac_visto < 0.4:
            return []  # no meio do giro a detecção falha às vezes: espera antes de esquecer
        else:
            if not p or agora - self._mac_visto > 0.4:
                self._mac_neutro, self._mac_armada = None, True
                self._mac_recentes.clear()
                self._mac_pulso.clear()
            return []
        self._mac_visto = agora
        self._mac_recentes.append(angulo)
        self._mac_pulso.append((p[PULSO].x, p[PULSO].y))
        if self._mac_neutro is None:
            # neutro = ângulo da mão parada (6 quadros quase iguais)
            recentes = list(self._mac_recentes)
            if len(recentes) == self._mac_recentes.maxlen and max(_dif(a, recentes[0]) for a in recentes) < 12:
                self._mac_neutro = recentes[-1]
            return []
        desvio = (angulo - self._mac_neutro + 180) % 360 - 180
        if not self._mac_armada:
            if abs(desvio) < 20:
                self._mac_armada = True
            return []
        if abs(desvio) < 15:
            self._mac_neutro += 0.05 * desvio  # neutro acompanha a mão devagar
            self._mac_saiu = agora
            return []
        if abs(desvio) < self.macaneta_limite:
            return []
        if agora - self._mac_saiu > 0.7:
            self._mac_neutro = angulo  # deriva lenta não é giro: adota a posição nova
            return []
        xs = [q[0] for q in self._mac_pulso]
        ys = [q[1] for q in self._mac_pulso]
        if max(max(xs) - min(xs), max(ys) - min(ys)) > 0.25:
            return []  # o pulso andou muito: é a mão passando (swipe), não girando
        self._mac_armada = False
        return ["macaneta_horario" if desvio > 0 else "macaneta_anti_horario"]

    def _rolagem(self, p, agora):
        """Joystick: com dois dedos, a distância vertical do ponto inicial dá a velocidade.

        Só começa com os dois dedos parados um instante (swipe é rápido) e
        desliga se a mão andar para o lado (aí é swipe, que costuma ir na diagonal).
        """
        x = (p[INDICADOR[0]].x + p[MEDIO[0]].x) / 2
        y = (p[INDICADOR[0]].y + p[MEDIO[0]].y) / 2
        palmas = [a for t, a in self._palmas if agora - t <= 0.5]
        virando = len(palmas) >= 5 and max(_dif(a, palmas[0]) for a in palmas) > 30
        if agora - self._giro_recente < 0.5 or virando:
            self._rolar_cancelada = True  # girando (dedos ou palma): o joystick fica de fora
        if self._rolar_cancelada or agora - self._inicio_pose < 0.3:
            return []
        if self._rolar_centro is None:
            self._rolar_centro, self._rolar_acumulado = (x, y), 0.0
            return []
        if abs(x - self._rolar_centro[0]) > 0.08:
            self._rolar_cancelada = True  # até baixar os dedos e levantar de novo
            return []
        desvio = y - self._rolar_centro[1]
        if abs(desvio) <= self.rolar_zona_morta:
            self._rolar_acumulado = 0.0
            return []
        self._rolar_acumulado += (abs(desvio) - self.rolar_zona_morta) * self.rolar_velocidade
        if self._rolar_acumulado < 1:
            return []
        passos = min(int(self._rolar_acumulado), 3)
        self._rolar_acumulado -= int(self._rolar_acumulado)
        return ["rolar_baixo" if desvio > 0 else "rolar_cima"] * passos

    def _pinca(self, p, agora):
        """Passar de pinça fechada para aberta (ou o contrário) em até `pinca_janela`."""
        zona = zona_pinca(p) if p else None
        atual, desde, n = self._zona
        n = n + 1 if zona == atual else 1
        self._zona = (zona, desde if zona == atual else agora, n)
        evento, quando = self._repeticao
        if evento and zona == atual and n > 3 and agora >= quando:
            self._repeticao = (evento, agora + self.pinca_repetir)
            return [evento]
        if zona != atual:
            self._repeticao = (None, 0.0)
        if zona is None or n != 3:  # zona confirmada no 3º quadro seguido
            if zona is None and atual is not None and n == 1:
                self._ultima_zona = (atual, agora)
            return []
        anterior, saiu = self._ultima_zona
        self._ultima_zona = (zona, agora)
        if anterior is None or anterior == zona or agora - saiu > self.pinca_janela:
            return []
        evento = "pinca_abrir" if zona == "aberta" else "pinca_fechar"
        if self.pinca_repetir > 0:
            # primeira repetição espera um pouco mais, para um gesto rápido dar um passo só
            self._repeticao = (evento, agora + self.pinca_repetir + 0.3)
        return [evento]


_OPOSTO = {
    "swipe_direita": "swipe_esquerda", "swipe_esquerda": "swipe_direita",
    "swipe_cima": "swipe_baixo", "swipe_baixo": "swipe_cima",
}


def angulo_palma(m):
    """Giro da palma em torno do antebraço (graus), a partir dos pontos 3D.

    Eixo = pulso → base do médio; normal da palma = (pulso→indicador) × (pulso→mínimo).
    O ângulo é medido no plano perpendicular ao eixo, com referência na direção
    da câmera; sinal ajustado para horário (visto pelo usuário) ser positivo.
    """
    def sub(a, b): return (a.x - b.x, a.y - b.y, a.z - b.z)
    def cruz(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    def esc(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    def unit(a):
        n = math.sqrt(esc(a, a)) or 1e-9
        return (a[0] / n, a[1] / n, a[2] / n)
    eixo = unit(sub(m[MEDIO[0]], m[PULSO]))
    normal = unit(cruz(sub(m[INDICADOR[0]], m[PULSO]), sub(m[MINIMO[0]], m[PULSO])))
    camera = (0.0, 0.0, -1.0)
    r1 = unit(tuple(c - esc(camera, eixo) * e for c, e in zip(camera, eixo)))
    r2 = cruz(eixo, r1)
    return -math.degrees(math.atan2(esc(normal, r2), esc(normal, r1)))


def _dif(a, b):
    return abs((a - b + 180) % 360 - 180)


def _curva(trecho):
    """Quanto a direção do movimento mudou (graus) entre a 1ª e a 2ª metade do trecho."""
    if len(trecho) < 4:
        return 0.0
    meio = len(trecho) // 2
    a, b, c = trecho[0], trecho[meio], trecho[-1]
    d1 = math.atan2(b[2] - a[2], b[1] - a[1])
    d2 = math.atan2(c[2] - b[2], c[1] - b[1])
    return abs(math.degrees((d2 - d1 + math.pi) % (2 * math.pi) - math.pi))


def _amplitude(rastro):
    xs = [r[1] for r in rastro]
    ys = [r[2] for r in rastro]
    return max(max(xs) - min(xs), max(ys) - min(ys))
