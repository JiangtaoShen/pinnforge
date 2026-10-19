# PINNForge installer for Windows -- puts the `pinnforge` CLI on PATH via uv.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.ps1 | iex"
#
# Pin a version (any tag, branch or commit):
#   $env:PINNFORGE_VERSION = "v0.2.0"; irm .../install.ps1 | iex
#
# What it does:
#   1. installs uv if it is missing (https://astral.sh/uv)
#   2. `uv tool install` pinnforge from GitHub -- an isolated environment, the
#      CLI in %USERPROFILE%\.local\bin -- on Python 3.12 (never from PyPI: the
#      name `pinnforge` there belongs to an unrelated project)
#   3. makes sure that directory is on PATH for future terminals
#
# On Windows blocks train on the CPU: JAX publishes no CUDA build for it. To
# train on an NVIDIA GPU, install PINNForge inside WSL2 with install.sh.
# For development, clone the repository and `uv sync` instead.
#
# Run with `irm | iex`, this script lives in your own PowerShell session, so a
# failure throws rather than exits: exiting would close your window.

$ErrorActionPreference = "Stop"

$Repo = "https://github.com/JiangtaoShen/pinnforge.git"
$Version = if ($env:PINNFORGE_VERSION) { $env:PINNFORGE_VERSION } else { "main" }

function Say([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Warn([string]$Message) { Write-Host "!!  $Message" -ForegroundColor Yellow }
function Die([string]$Message) { throw "pinnforge install: $Message" }
function Have([string]$Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

if (-not (Have "git")) {
    # uv installs the frozen stack's one git dependency, and PINNForge itself, with git
    Die "git is required. Install it (winget install Git.Git), open a new terminal, and retry."
}

if (-not (Have "uv")) {
    Say "uv not found -- installing it from astral.sh"
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Have "uv")) {
        Die "uv installed but is not on PATH yet -- open a new terminal and retry."
    }
}

Say "installing pinnforge from $Repo@$Version"
uv tool install --force --python 3.12 "git+$Repo@$Version"
if ($LASTEXITCODE -ne 0) { Die "uv tool install failed (exit $LASTEXITCODE)" }
uv tool update-shell *> $null

$Gpu = "no NVIDIA GPU visible -- workspaces get the CPU build of JAX (fine for the demo)."
if (Have "nvidia-smi") {
    $cards = @(& nvidia-smi -L 2>$null)
    if ($LASTEXITCODE -eq 0 -and $cards.Count -gt 0) {
        $Gpu = "$($cards.Count) NVIDIA GPU(s) found -- JAX has no CUDA build for Windows, so " +
               "blocks train on the CPU here. To train on the GPU, install PINNForge inside " +
               "WSL2 (install.sh)."
    }
}

$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (Have "pinnforge") {
    Say "installed: $(pinnforge --version)"
} else {
    Warn "installed, but $env:USERPROFILE\.local\bin is not on PATH in this terminal -- open a new one."
}
Say $Gpu
Write-Host @"

Next:
  pinnforge init `$HOME\pf; cd `$HOME\pf   create a workspace (environment + demo task)
  pinnforge demo                          a full run in minutes -- no API key needed
  pinnforge setup                         choose the coding agent that runs blocks
  pinnforge task new                      describe your PDE; a reference is built if you have none

Claude Code on Windows runs its shell in Git Bash, which Git for Windows installs.
Docs: https://github.com/JiangtaoShen/pinnforge
"@
