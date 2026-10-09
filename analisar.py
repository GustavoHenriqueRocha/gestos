"""Repassa uma gravação rotulada pelo reconhecedor e mostra acertos e erros.

Uso: .venv/bin/python analisar.py gravacoes/sessao1 [--detalhe]
"""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from gestos.reconhecedor import (INDICADOR, MEDIO, POLEGAR, PULSO, Reconhecedor, classificar_pose, dist,
                                 escala)


def carregar(pasta):
    quadros = []
    for linha in open(pasta / "pontos.jsonl"):
        q = json.loads(linha)
        if q["p"]:
            q["p"] = [SimpleNamespace(x=x, y=y) for x, y in q["p"]]
        quadros.append(q)
    rotulos = []
    for linha in open(pasta / "rotulos.txt"):
        t, nome = linha.split()
        rotulos.append((float(t), nome))
    return quadros, rotulos


def rotulo_em(w, rotulos):
    atual = None
    for t, nome in rotulos:
        if w >= t:
            atual = nome
    return atual


def main():
    pasta = Path(sys.argv[1])
    detalhe = "--detalhe" in sys.argv
    quadros, rotulos = carregar(pasta)
    rec = Reconhecedor()
    poses, eventos, medidas = {}, {}, {}
    ordem = []
    for q in quadros:
        r = rotulo_em(q["w"], rotulos)
        if r is None or r == "fim":
            continue
        if r not in ordem:
            ordem.append(r)
        p = q["p"]
        evs = rec.atualizar(p, q["t"] / 1000)
        poses.setdefault(r, Counter())[classificar_pose(p) if p else "sem_mao"] += 1
        for e in evs:
            eventos.setdefault(r, []).append(e)
        if p:
            s = escala(p)
            medidas.setdefault(r, []).append((
                dist(p[POLEGAR[3]], p[INDICADOR[3]]) / s,
                dist(p[INDICADOR[3]], p[PULSO]) / s,
                s,
            ))

    for r in ordem:
        total = sum(poses[r].values())
        dist_poses = ", ".join(f"{k}:{v * 100 // total}%" for k, v in poses[r].most_common())
        evs = Counter(eventos.get(r, []))
        print(f"\n== {r} ({total} quadros)")
        print(f"   poses:   {dist_poses}")
        print(f"   eventos: {dict(evs) or '-'}")
        if detalhe and medidas.get(r):
            m = medidas[r]
            for i, nome in enumerate(("polegar-indicador/s", "indicador-pulso/s", "escala")):
                vals = sorted(v[i] for v in m)
                print(f"   {nome:22} min {vals[0]:.2f}  med {vals[len(vals) // 2]:.2f}  max {vals[-1]:.2f}")


if __name__ == "__main__":
    main()
