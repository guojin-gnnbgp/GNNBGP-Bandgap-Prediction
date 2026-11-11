#!/usr/bin/env bash
# GNNBGP GUI v0.6 launcher: reuse verified Python, ABACUS and Linux workspace.
set -euo pipefail
STUDIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BGP_SOURCE="${BGP_SOURCE:-/mnt/d/program/gnnbgp}"
export BGP_ENGINE_HOME="${BGP_ENGINE_HOME:-$HOME/BandGap-Studio}"
export BGP_BIN="${BGP_BIN:-$BGP_ENGINE_HOME/.venv/bin/bgp}"
export BGP_UI_DATA="${BGP_UI_DATA:-$BGP_ENGINE_HOME/workspace}"
export BGP_UI_PORT="${BGP_UI_PORT:-8780}"

# The GPU binary was compiled separately; never rebuild it from the GUI launcher.
export BGP_ABACUS="${BGP_ABACUS:-$BGP_ENGINE_HOME/build/abacus-install/bin/abacus_pw_gpu}"
export BGP_DEPS_PREFIX="${BGP_DEPS_PREFIX:-$HOME/bgp-deps}"
PY="$BGP_ENGINE_HOME/.venv/bin/python"

if [[ ! -f "$BGP_SOURCE/pyproject.toml" ]]; then
  echo "ERROR: source not found: $BGP_SOURCE/pyproject.toml" >&2
  exit 2
fi
if [[ ! -x "$PY" || ! -x "$BGP_BIN" ]]; then
  echo "ERROR: existing GNNBGP Python/CLI missing: $PY / $BGP_BIN" >&2
  exit 2
fi

# ABACUS requires a case-sensitive filesystem for distinct 'input/' and 'INPUT'.
# Avoid known Windows DrvFs paths for computation (GUI source on /mnt/d is fine).
case "$BGP_UI_DATA/" in
  /mnt/[a-z]/*)
    echo "ERROR: BGP_UI_DATA=$BGP_UI_DATA points to a Windows-mounted drive." >&2
    echo "Use a Linux workspace, e.g. $HOME/BandGap-Studio/workspace" >&2
    exit 2 ;;
esac
mkdir -p "$BGP_UI_DATA"
if [[ -d "$BGP_DEPS_PREFIX/lib" ]]; then
  # Preserve CUDA, MPI, UCX and other manually configured runtime library paths.
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$BGP_DEPS_PREFIX/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$BGP_DEPS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
else
  echo "WARNING: ABACUS dependency libraries not found: $BGP_DEPS_PREFIX/lib" >&2
fi

echo "Source      : $BGP_SOURCE"
echo "CLI         : $BGP_BIN"
echo "Workspace   : $BGP_UI_DATA"
if [[ -x "$BGP_ABACUS" ]]; then
  echo "ABACUS GPU  : $BGP_ABACUS"
else
  echo "WARNING: ABACUS binary not executable: $BGP_ABACUS" >&2
  echo "         GNN can still run; GPU/CIDER jobs require a valid ABACUS path." >&2
fi
echo "Browser     : http://127.0.0.1:$BGP_UI_PORT"
echo "Ctrl+C to stop the local GUI service."
exec "$PY" "$STUDIO_DIR/studio/server.py"
