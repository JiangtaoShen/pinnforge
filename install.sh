#!/usr/bin/env sh
# PINNForge installer — puts the `pinnforge` CLI on PATH via uv.
#
#   curl -fsSL https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.sh | sh
#
# Pin a version (any tag, branch or commit):
#   curl -fsSL …/install.sh | PINNFORGE_VERSION=v0.2.0 sh
#
# What it does:
#   1. checks the platform (Linux or macOS; on Windows use install.ps1, or this
#      script inside WSL2 to train on an NVIDIA GPU)
#   2. installs uv if it is missing (https://astral.sh/uv)
#   3. `uv tool install` pinnforge from GitHub — an isolated environment,
#      the CLI in ~/.local/bin — on Python 3.12 (never from PyPI: the name
#      `pinnforge` there belongs to an unrelated project)
#   4. makes sure ~/.local/bin is on PATH for future shells
#
# The tool is light (PyYAML, NumPy). The JAX stack blocks train on is created
# per workspace by `pinnforge init`, with CUDA if an NVIDIA GPU is visible.
# For development, clone the repository and `uv sync` instead.
set -eu

REPO="https://github.com/JiangtaoShen/pinnforge.git"
VERSION="${PINNFORGE_VERSION:-main}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$1" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$1" >&2; exit 1; }

case "$(uname -s)" in
  Linux) ;;
  Darwin) warn "macOS: PINNForge runs, but JAX has no CUDA here — blocks train on the CPU." ;;
  MINGW*|MSYS*|CYGWIN*) die "on Windows use install.ps1 (PowerShell): powershell -ExecutionPolicy ByPass -c \"irm https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.ps1 | iex\" — or run this script inside WSL2 to train on the GPU." ;;
  *) warn "untested platform: $(uname -s)" ;;
esac

command -v git  >/dev/null 2>&1 || die "git is required. Install git and retry."
command -v curl >/dev/null 2>&1 || die "curl is required to install uv. Install curl and retry."

if ! command -v uv >/dev/null 2>&1; then
  say "uv not found — installing it from astral.sh"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  export PATH
  command -v uv >/dev/null 2>&1 || die "uv installed but is not on PATH yet — open a new shell and retry."
fi

say "installing pinnforge from ${REPO}@${VERSION}"
uv tool install --force --python 3.12 "git+${REPO}@${VERSION}"
uv tool update-shell >/dev/null 2>&1 || true

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU="$(nvidia-smi -L | wc -l | tr -d ' ') NVIDIA GPU(s) found — workspaces get the CUDA build of JAX."
else
  GPU="no NVIDIA GPU visible — workspaces get the CPU build of JAX (fine for the demo)."
fi

if command -v pinnforge >/dev/null 2>&1; then
  say "installed: $(pinnforge --version 2>/dev/null || echo pinnforge)"
else
  warn "installed, but ~/.local/bin is not on PATH in this shell:"
  warn "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
say "$GPU"
cat <<'EOF'

Next:
  pinnforge init ~/pf && cd ~/pf     create a workspace (environment + demo task)
  pinnforge demo                     a full run in minutes — no API key needed
  pinnforge setup                    choose the coding agent that runs blocks
  pinnforge task new                 describe your PDE; a reference is built if you have none

Docs: https://github.com/JiangtaoShen/pinnforge
EOF
