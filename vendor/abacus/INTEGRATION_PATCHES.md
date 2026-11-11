# Native-CIDER and GPU-EXX integration record

The integration branch `integration/cider-gpu-exx` started at the native-CIDER
commit `7828d7cc28c8702ced99e88fd3cad51b58cf7fee`. The following GPU-EXX changes
were applied in order, preserving native-CIDER hooks:

| Imported source | Integrated commit | Functionality |
| --- | --- | --- |
| `1ace479115b59d5566b9a2bc4969ab54f41da17b` | `6d3df0de6776d378b5879ec74b1ca3b6219ee22a` | symmetry-compatible FFT grids |
| `20b5024a9442488af7107e8c1b236bfb1029ad81` | `2fcae9c66fc2101103eab01c061b85f3a2e05aea` | symmetry-remap cutoff |
| `e96a73f7e6745326f2aefaa45294b502da7d33a1` | folded into reconciliation | safe GPU real-space-symmetry default |
| `36789e304e2c1a66b73ef79aaec1f1f50392377e` | `8aaac6cf068a16457009a00f32789581ee8c90e1` | reduced-k energy and remap boundary |
| `7e24ed1be8fcd954ee07eb56849b28d82567ae16` | `ec09a6e0ea44ab3a61d3ff46c75097feecbeb22f` | reduced-k exact exchange |
| `d67df55bbedf97f8b378c95101d2ffc3685b793a` | `cd8181d182dc03e7e0606c78a374582afe182641` | automatic EXX tiling |
| `73a8bd8a8186f2b8214a8e08bcc0b39e37a51a31` | `6437808ad655d53f5046ed4e48867d701988bd36` | hardened EXX execution context |

Commit `53f9a990587c240cf4a41f66713795139defab2f` manually reconciles the two
lines. It retains the native-CIDER potential and evaluator boundaries while
restoring the selected-diagonal and energy-density interfaces, symmetry-grid
operation helpers, reduced-mesh tile-policy helpers, the training-EXX
constructor's execution context, and the EXX stress path. It also repairs the
`cal_exx_energy_op_qtile` interface shared by the imported GPU work.

Commit `681fd5c0b2d82745d96008f450a927a1ec723eb4` adds the package integration
contract. The opt-in `out_bgp_result` input is restricted to PW and `nspin=1`.
Rank zero atomically emits `OUT.<suffix>/bgp_electronic_state.json` from the
final SCF state, including convergence, k weights, eigenvalues in eV, and raw
ABACUS weighted occupations. This output is independent of training-data
generation. The same commit completes compile-time interface reconciliation
found while linking the combined `abacus_pw_gpu` target.

Commit `79389b635` guards construction of exact-exchange Coulomb options when
EXX is disabled. This is required for the native-CIDER semilocal/PBE path in
the same combined executable; enabled EXX still fails closed when its required
Coulomb parameters are absent. Dedicated unit tests cover both branches.

Commit `222a4924e` records the actual ABACUS `wg` convention in the result
writer tests: for `nspin=1`, a fully occupied band carries the k-point weight;
the `spin_capacity` field describes electron capacity but is not another
normalization divisor.

Commit `11c945349` corrects the EXX option boundary for functional-specific
Coulomb families. HSE06 legitimately supplies only screened `Erfc` parameters,
while PBE0 supplies only unscreened `Fock` parameters. The shared PW operator
now accepts either family while still rejecting enabled EXX with no Coulomb
parameters. Focused tests cover the HSE-shaped map and the fail-closed case.

Commit `bbfa7d67c` makes EXX FFT-grid translation commensurability use the same
`1e-6` coordinate tolerance as ABACUS symmetry analysis. This prevents valid
quarter-cell translations carrying symmetry-analysis roundoff from forcing a
full-k fallback. The Si 72x72x72 regression uses the exact runtime translation
values and retains the existing rejection coverage for genuinely incompatible
translations.

The integrated source was configured with MPI, CUDA, LibXC, native CIDER,
PW-only mode, and tests. Focused writer/input, native-CIDER evaluator, NLDF,
EXX tiling, reduced-k, host symmetry-remap, and CUDA symmetry-kernel tests were
run before export. Corpus-dependent CIDER oracle tests require a separate
read-only corpus and are not part of this source snapshot.
