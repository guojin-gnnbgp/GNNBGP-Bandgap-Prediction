#!/usr/bin/env python3
"""Drive a runner call in a subprocess so tests can deliver a real SIGINT."""

import sys
from pathlib import Path

from bandgap_predictor.abacus.runner import run_abacus

try:
    run_abacus(
        Path(sys.argv[1]),
        Path(sys.argv[3]),
        progress=lambda _line: None,
        mpirun=Path(sys.argv[2]),
    )
except KeyboardInterrupt:
    raise SystemExit(130) from None
