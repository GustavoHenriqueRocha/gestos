#!/usr/bin/env bash
# Grava uma sessão rotulada: avisa na tela qual gesto fazer e anota o horário.
# Uso: ./sessao-guiada.sh [pasta-de-saida]
set -uo pipefail
cd "$(dirname "$0")"
saida="${1:-gravacoes/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$saida"
systemctl --user stop gestos 2>/dev/null
.venv/bin/gestos --simular --gravar "$saida/pontos.jsonl" > "$saida/log.txt" 2>&1 &
pid=$!
avisar() { notify-send -a Gestos -t 6000 -h string:x-canonical-private-synchronous:guia "$1" "$2"; }
avisar "Sessão de gestos" "Começa em 5 s. Fique de frente para a câmera."
sleep 5
passos=(
  "nada|Mão fora da câmera"
  "swipe_direita|✌ Dois dedos: arraste para a DIREITA (2x)"
  "swipe_esquerda|✌ Dois dedos: arraste para a ESQUERDA (2x)"
  "swipe_cima|✌ Dois dedos: arraste para CIMA (2x)"
  "swipe_baixo|✌ Dois dedos: arraste para BAIXO (2x)"
  "pinca_cima|🤏 Pinça e SUBA a mão devagar"
  "pinca_baixo|🤏 Pinça e DESÇA a mão devagar"
  "mao_aberta|✋ Mão aberta parada"
  "joinha|👍 Joinha parado"
  "punho|✊ Punho fechado parado"
  "nada|Mão mexendo à toa (coçar, gesticular)"
)
for passo in "${passos[@]}"; do
  rotulo="${passo%%|*}"; texto="${passo#*|}"
  echo "$(date +%s.%N) $rotulo" >> "$saida/rotulos.txt"
  avisar "Agora:" "$texto"
  sleep 7
done
echo "$(date +%s.%N) fim" >> "$saida/rotulos.txt"
kill -TERM $pid; wait $pid
avisar "Sessão de gestos" "Terminou, obrigado!"
echo "$saida"
