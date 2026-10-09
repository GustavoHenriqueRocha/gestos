# Gestos

Controle do Omarchy (Hyprland) por gestos de mão na webcam, usando MediaPipe Hands.

Motor padrão: modelos treinados do [HaGRID](https://github.com/hukenovs/hagrid)
([ai-forever/dynamic_gestures](https://github.com/ai-forever/dynamic_gestures), Apache 2.0,
código em `src/gestos/hagrid/`) — detector de mãos + classificador de 45 poses + gestos
dinâmicos. ~6 ms por quadro na CPU.

| Gesto | Ação padrão |
|---|---|
| 👍 / 👎 segurando | Volume + / − (repete enquanto segura) |
| ☝ / 👇 segurando | Rola a janela ativa para cima / baixo |
| 👋 mão de lado varrendo para a esquerda / direita | Janela da direita / esquerda (como no celular) |
| ✌ segurando | Play/Pause |
| 👌 segurando | Menu do Omarchy |

Qualquer uma das 45 poses ou dos gestos dinâmicos pode ser ligada a um comando em
`~/.config/gestos/gestos.toml`. O motor antigo (MediaPipe + regras próprias, com pinça,
maçaneta e giro) continua disponível: `gestos-mediapipe.toml`.

Uma notificação fixa no canto mostra a pose atual e o último gesto; `Super+Ctrl+Alt+G`
abre a janelinha com a câmera, a caixa da mão, a pose e a confiança.

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
