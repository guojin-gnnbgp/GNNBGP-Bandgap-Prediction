#!/bin/sh
# Mock CLI for HTTP/queue tests only; NEVER scientific output validation.
case "$1" in
  --version) echo "bgp, version 0.1.0"; exit 0;;
  backends)
    echo "gnn-hse06: available"
    echo "abacus-hse06: unavailable (ABACUS not configured)"
    echo "abacus-pbe0: unavailable (ABACUS not configured)"
    echo "cider-pbe0: unavailable (ABACUS not configured)"
    exit 0;;
  assets) echo "Verified bundled assets"; exit 0;;
  doctor) echo "ABACUS missing" >&2; exit 4;;
  predict)
    shift
    work=""
    backend="gnn-hse06"
    abacus=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--work-root" ]; then
        work="$2"; shift 2
      elif [ "$1" = "--backend" ]; then
        backend="$2"; shift 2
      elif [ "$1" = "--abacus" ]; then
        abacus="$2"; shift 2
      else
        shift
      fi
    done
    test -n "$work" || exit 2
    mkdir -p "$work/mock-run"
    echo "MOCK TEST BACKEND: $backend"
    echo "MOCK ABACUS PATH: $abacus"
    echo "Loading verified legacy/provisional GNN model (MOCK TEST ONLY)"
    cat > "$work/mock-run/result.json" <<RESULT
{"schema_version":"1.0","run_id":"testing-0001","status":"success","backend":"$backend","method":"SIMULATED TEST ONLY","prediction":{"gap_ev":0.2487309,"classification":"insulator","gap_type":"direct","gap_type_probability":0.91},"structure":{"formula":"In4Sb4"},"converged":null,"settings":{"device":"cpu"},"provenance":{"model_sha256":"test-only"},"occupation_summary":{},"diagnostics":{},"warnings":["SIMULATED TEST ONLY"],"elapsed_seconds":0.12,"started_at":"2026-01-01","finished_at":"2026-01-01","artifact_path":"MOCK ONLY"}
RESULT
    echo "Band gap: 0.2487309 eV"
    exit 0;;
esac
exit 2
