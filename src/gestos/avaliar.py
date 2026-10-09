"""Placar do reconhecedor numa calibração rotulada (gestos-calibrar).

Uso: gestos-avaliar gravacoes/calibracao-<data> [--param nome=valor ...]

Para cada repetição pedida, conta acerto se o gesto certo saiu durante o
"AGORA" (com 0,8 s de folga) e anota os outros gestos que saíram junto
(confusões). Gestos que saem no "nada" ou no repouso são falsos positivos.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from .reconhecedor import Reconhecedor

FOLGA = 0.8  # segundos depois do "AGORA" em que o gesto ainda conta


def carregar(pasta):
    quadros = []
    for linha in open(Path(pasta) / "dados.jsonl"):
        q = json.loads(linha)
        q["p"] = [SimpleNamespace(x=x, y=y) for x, y in q["p"]] if q["p"] else None
        q["m"] = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in q["m"]] if q.get("m") else None
        quadros.append(q)
    return quadros


def avaliar(quadros, **params):
    """Devolve (placar por gesto, falsos positivos, confusões)."""
    rec = Reconhecedor(**params)
    janelas = {}  # (rotulo, rep) -> [inicio, fim]
    for q in quadros:
        if q["fase"] == "agora" and q["rotulo"] != "nada":
            j = janelas.setdefault((q["rotulo"], q["rep"]), [q["t"], q["t"]])
            j[1] = q["t"]
    eventos = []
    for q in quadros:
        for e in rec.atualizar(q["p"], q["t"] / 1000, mundo=q["m"]):
            eventos.append((q["t"], e))

    placar = defaultdict(lambda: [0, 0])  # gesto -> [acertos, pedidos]
    confusoes = Counter()
    usados = set()
    for (rotulo, rep), (ini, fim) in janelas.items():
        dentro = [(t, e) for t, e in eventos if ini <= t <= fim + FOLGA * 1000]
        placar[rotulo][1] += 1
        if any(e == rotulo for _, e in dentro):
            placar[rotulo][0] += 1
        for t, e in dentro:
            usados.add((t, e))
            if e != rotulo:
                confusoes[(rotulo, e)] += 1
    falsos = Counter(e for t, e in eventos if (t, e) not in usados and e != "punho")
    return dict(placar), falsos, confusoes


def nota(placar, falsos, confusoes):
    """Um número só para comparar ajustes: acertos menos erros (erros pesam mais)."""
    acertos = sum(a for a, _ in placar.values())
    return acertos - 1.5 * sum(falsos.values()) - 0.5 * sum(confusoes.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pasta")
    parser.add_argument("--param", action="append", default=[], help="ajuste do reconhecedor, ex.: macaneta_limite=30")
    args = parser.parse_args()
    params = {}
    for p in args.param:
        k, v = p.split("=")
        params[k] = True if v == "true" else False if v == "false" else float(v)

    placar, falsos, confusoes = avaliar(carregar(args.pasta), **params)
    print(f"{'gesto':24} acertos")
    for g, (a, n) in placar.items():
        print(f"{g:24} {a}/{n}  {'█' * a}{'·' * (n - a)}")
    total_a = sum(a for a, _ in placar.values())
    total_n = sum(n for _, n in placar.values())
    print(f"\ntotal: {total_a}/{total_n} ({100 * total_a // max(total_n, 1)}%)")
    print(f"falsos positivos (nada/repouso): {sum(falsos.values())} {dict(falsos)}")
    print("confusões (pedido → saiu):", {f"{a}→{b}": n for (a, b), n in confusoes.most_common()})
    print(f"nota: {nota(placar, falsos, confusoes):.1f}")
