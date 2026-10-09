# Gestos

Controle do Omarchy (Hyprland) por gestos de mão na webcam, usando MediaPipe Hands.

| Gesto | Ação padrão |
|---|---|
| ✋🔄 mão aberta girando como maçaneta (horário / anti-horário) | Janela da direita / esquerda (como Super+setas); voltar ao neutro não dispara |
| ✌🌀 dois dedos girando em círculo (horário / anti-horário) | Rolagem contínua para baixo / cima |
| ✌ dois dedos parados, mão sobe / desce | Rolagem joystick |
| 🤏 pontas encostadas → "L" aberto / "L" → encostadas (segurar repete) | Volume +10 / −10 |
| ✋ mão aberta em pé, dedos afastados, parada (0,8 s) | Menu do Omarchy |
| 👍 joinha parado (0,8 s) | Play/Pause |
| ✌ swipe lateral, ✊ punho | (livres, configuráveis) |

Uma notificação fixa no canto mostra a pose atual e o último gesto reconhecido.
A rolagem usa `wlrctl` compilado com rolagem de rodinha (`~/.local/bin/wlrctl-roda`,
ver `wlrctl-roda.patch`).

## Uso

```bash
./instalar.sh
gestos-alternar          # liga/desliga (atalho Super+Ctrl+G)
.venv/bin/gestos --debug --simular   # testar sem executar nada
```

Os gestos e comandos ficam em `~/.config/gestos/gestos.toml`.
O modelo da mão é baixado na primeira execução para `~/.cache/gestos/`.

Obs.: usa `mediapipe==0.10.21` — a 1.1.0 morre ao criar o detector nesta máquina.

## Ajustando com gravações

`./sessao-guiada.sh` grava uma sessão rotulada (avisa na tela qual gesto fazer).
`.venv/bin/python analisar.py gravacoes/<sessão> --detalhe` repassa a gravação pelo
reconhecedor e mostra poses, eventos e medidas por gesto.
