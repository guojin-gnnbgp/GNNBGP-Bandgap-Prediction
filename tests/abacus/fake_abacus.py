#!/usr/bin/env python3
"""Small executable used by local-runner contract tests."""

import json
import os
import time
from pathlib import Path

Path("fake-env.json").write_text(
    json.dumps({"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")})
)
Path("fake-pid.txt").write_text(str(os.getpid()))
print("ABACUS fake startup", flush=True)
print("ITER 1 etot=-1.0", flush=True)
if os.environ.get("BGP_FAKE_MODE") == "hang":
    while True:
        time.sleep(1)
print("#SCF IS CONVERGED#", flush=True)
