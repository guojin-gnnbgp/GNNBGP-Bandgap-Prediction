#!/usr/bin/env bash
# Read-only diagnostics for NEW D:\program\gnnbgp and its active WSL venv.
set -euo pipefail
export BGP_SOURCE="${BGP_SOURCE:-/mnt/d/program/gnnbgp}"
ENV_DIR="${BGP_ENGINE_HOME:-$HOME/BandGap-Studio}/.venv"
if [[ ! -x "$ENV_DIR/bin/python" || ! -x "$ENV_DIR/bin/bgp" ]]; then
  echo "Missing installed virtual environment at $ENV_DIR" >&2
  exit 2
fi
"$ENV_DIR/bin/python" --version
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/bgp" --version
"$ENV_DIR/bin/bgp" backends
"$ENV_DIR/bin/bgp" assets verify
echo
echo "Source: $BGP_SOURCE"
echo "Source pyproject exists:" 
test -f "$BGP_SOURCE/pyproject.toml" && echo "yes"
