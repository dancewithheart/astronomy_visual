#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -e ".[pyvista,analysis]"
python gaia_orion_pyvista.py --mode screenshot --quality preview
