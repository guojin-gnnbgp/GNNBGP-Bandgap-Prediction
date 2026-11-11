# Result and artifact contract

Every request allocates a unique directory below `--work-root`. The directory
is retained on success, scientific failure, process failure, and interruption.

Typical contents include:

- `input/`: a byte-exact copy of the submitted structure;
- `request.json`: the versioned request;
- `INPUT`, `STRU`, `site-map.json`, staged models, and pseudopotentials for an
  ABACUS calculation;
- `process.json` and `stdout.log`;
- `OUT.ABACUS/bgp_electronic_state.json`, the backend-neutral final ABACUS
  eigenvalue, occupation, k-weight, Fermi-level, and convergence evidence;
- `result.json`, written atomically as the terminal public record.

The concise terminal output is a view of `result.json`, not a separate result.
The JSON includes:

- schema version, run ID, status, backend, and scientific method;
- non-negative gap and electronic classification;
- convergence and normalized occupation summary;
- original-file and semantic structure hashes;
- the complete locked settings and preset version;
- executable, model, pseudopotential, feature, and source-revision provenance;
- start/finish times, elapsed time, diagnostics, warnings, and artifact path.

For ABACUS, a result is successful only when the process exits cleanly, the
final SCF marker says converged, and the electronic-state artifact is complete
and finite. A converged state with materially fractional occupations is
classified `partial_occupancy` and returns `0.0 eV`. Non-convergence or an
invalid/missing artifact never returns a successful numerical gap.

The ABACUS occupation artifact stores raw k-weighted `wg`. For `nspin=1`, the
parser normalizes an occupation by its k-point weight; the separate
`spin_capacity=2` field describes electron capacity and is not another divisor.

## Required-system baselines

The release audits used the final package-built executable with SHA-256
`9ad2d03bbb4f6fa8a56384a654f744fd23bd563872555b105b44f7efe544fb2e`.
Every entry below is a converged, retained calculation with exactly 0.001 eV
Gaussian smearing. The direct hybrid entries use locked preset v1.1; CIDER
uses calibrated preset v1.2 with `cider_xmix=0.15`. Exact hashes and tolerances
are in `tests/hardware/baselines.json`.

| System | Backend | Reported gap (eV) | Classification |
| --- | --- | ---: | --- |
| InSb | `abacus-hse06` | 0.2442765 | `insulator` |
| InSb | `abacus-pbe0` | 0.8610374 | `insulator` |
| InSb | `cider-pbe0` | 0.2475220 | `insulator` |
| Si | `abacus-hse06` | 1.1698225 | `insulator` |
| Si | `abacus-pbe0` | 1.7745132 | `insulator` |
| Si | `cider-pbe0` | 1.3119157 | `insulator` |

The superseded preset v1.0 used 0.015 Ry (about 0.204 eV) Gaussian smearing and
classified both InSb calculations as `partial_occupancy` with public zero gaps.
Those retained run directories remain immutable historical evidence. With the
requested 0.001 eV width, none of the three current InSb occupation sets is
materially fractional, so the public results retain their positive eigenvalue
gaps and are classified as insulators. CIDER preset v1.1 with
`cider_xmix=0.25` produced the superseded `0.6588544 eV` InSb and `1.7531899 eV`
Si results; their retained directories remain immutable historical evidence.
No-SOC remains an important limitation for InSb.

GaAs and MgO are covered by the GNN regression suite. Their release baselines
for all three ABACUS methods remain explicitly `not_baselined`; no numerical
values are inferred.

Interrupting `bgp` terminates the local process group, retains its logs, writes
an `interrupted` result when possible, and returns exit code 130. Preflight,
execution, scientific-convergence, and parsing failures use stable nonzero exit
codes and print the retained artifact path.
