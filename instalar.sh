#!/usr/bin/env bash
# Instala dependências, serviço do systemd, comando gestos-alternar e configuração.
set -euo pipefail
cd "$(dirname "$0")"
uv sync
mkdir -p ~/.config/gestos ~/.config/systemd/user ~/.local/bin
[ -f ~/.config/gestos/gestos.toml ] || cp gestos.toml ~/.config/gestos/gestos.toml
ln -sf "$PWD/gestos.service" ~/.config/systemd/user/gestos.service
ln -sf "$PWD/gestos-alternar" ~/.local/bin/gestos-alternar
ln -sf "$PWD/gestos-rolar" ~/.local/bin/gestos-rolar
systemctl --user daemon-reload
echo "Pronto. Ligue/desligue com: gestos-alternar"
