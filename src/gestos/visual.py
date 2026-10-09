"""Desenho da pré-visualização: imagem da câmera, esqueleto da mão e textos."""

import cv2

LIGACOES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
JANELA = "Gestos"


def desenhar(quadro, pontos, linhas, destaque=None, cor_destaque=(0, 200, 255), caixa=None, rotulo_caixa=""):
    """Desenha a mão (pontos ou caixa) e as linhas de texto no quadro (BGR, já espelhado)."""
    h, w = quadro.shape[:2]
    if caixa:
        x1, y1, x2, y2 = caixa
        cv2.rectangle(quadro, (x1, y1), (x2, y2), (80, 220, 80), 3)
        _texto(quadro, rotulo_caixa, (x1, max(20, y1 - 8)), 0.7, (80, 255, 80))
    if pontos:
        xy = [(int(q.x * w), int(q.y * h)) for q in pontos]
        for a, b in LIGACOES:
            cv2.line(quadro, xy[a], xy[b], (80, 220, 80), 2, cv2.LINE_AA)
        for i, c in enumerate(xy):
            cv2.circle(quadro, c, 4 if i % 4 else 6, (255, 255, 255), -1, cv2.LINE_AA)
    y = 28
    for texto in linhas:
        _texto(quadro, texto, (10, y), 0.7)
        y += 28
    if destaque:
        escala = 1.6 if len(destaque) < 18 else 1.0
        (tw, th), _ = cv2.getTextSize(destaque, cv2.FONT_HERSHEY_SIMPLEX, escala, 3)
        _texto(quadro, destaque, ((w - tw) // 2, h - 30), escala, cor_destaque, 3)
    return quadro


def _texto(img, texto, org, escala, cor=(255, 255, 255), grossura=2):
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grossura + 3, cv2.LINE_AA)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grossura, cv2.LINE_AA)


def mostrar(quadro, largura=480):
    """Mostra na janela; devolve False se o usuário fechou a janela."""
    h, w = quadro.shape[:2]
    cv2.imshow(JANELA, cv2.resize(quadro, (largura, int(h * largura / w))))
    cv2.waitKey(1)
    try:
        return cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def fechar():
    try:
        cv2.destroyWindow(JANELA)
        cv2.waitKey(1)
    except cv2.error:
        pass
