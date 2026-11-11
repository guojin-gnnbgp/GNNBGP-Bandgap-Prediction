#!/usr/bin/env bash
# Optional setup for fresh machines. NEVER runs when launching the GUI.
# Existing tested users should use start_wsl.sh instead of reinstalling torch/MACE.
set -euo pipefail
SOURCE="${BGP_SOURCE:-/mnt/d/program/gnnbgp}"
ENVROOT="${BGP_ENGINE_HOME:-$HOME/BandGap-Studio}"
if [[ ! -f "$SOURCE/requirements.txt" || ! -f "$SOURCE/pyproject.toml" ]]; then
  echo "New source missing: $SOURCE" >&2
  exit 2
fi
if [[ -e "$ENVROOT/.venv" ]]; then
  echo "Existing .venv detected at $ENVROOT/.venv. No changes made." >&2
  echo "Run scripts/check_environment.sh, or manually back up the old .venv first."
  exit 2
fi
UV="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV" ]]; then
  echo "uv missing. Install from official uv documentation first." >&2
  exit 2
fi
mkdir -p "$ENVROOT"
"$UV" python install 3.11
"$UV" venv --python 3.11 --seed "$ENVROOT/.venv"
PY="$ENVROOT/.venv/bin/python"
echo "Installing the customer's TESTED GNN pins from requirements.txt."
"$PY" -m pip install -r "$SOURCE/requirements.txt"
echo "Adding the engine package's BASE dependencies (including platformdirs)."
# Do NOT install [gnn] extra: its torch/MACE versions conflict with requirements.txt.
"$PY" -m pip install -e "$SOURCE"
"$PY" -m pip check
"$ENVROOT/.venv/bin/bgp" backends
"$ENVROOT/.venv/bin/bgp" assets verify
echo "Environment ready. Run scripts/start_wsl.sh"
