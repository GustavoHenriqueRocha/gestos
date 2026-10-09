# Gestos

Controle do Omarchy (Hyprland) por gestos de mão na webcam, usando MediaPipe Hands.

| Gesto | Ação padrão |
|---|---|
| ✌ dois dedos → / ← | Próximo / anterior workspace |
| ✌ dois dedos ↑ | Tela cheia |
| ✌ dois dedos ↓ | Scratchpad |
| 🤏 abrir / fechar polegar e indicador | Volume + / − |
| ✋ mão aberta parada (0,8 s) | Menu do Omarchy |
| 👍 joinha parado (0,8 s) | Play/Pause |
| ✊ punho | (livre, configurável) |

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
