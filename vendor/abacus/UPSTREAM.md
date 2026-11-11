# Vendored ABACUS source provenance

This directory is a clean `git archive` of the workspace-local integration
commit `bbfa7d67c0476ae4a586719bd4a9713d81df8536`. It contains source only: the
Python package does not ship an ABACUS executable and does not compile one at
package installation or import time.

The source lineage is:

- official ABACUS tag `v3.11.0-beta.3`, resolving to
  `d33c34f3fa880349a6577ef1700c6ab1bfe8356a`;
- the native-CIDER source line through
  `7828d7cc28c8702ced99e88fd3cad51b58cf7fee`;
- the combined GPU-EXX/native-CIDER reconciliation through
  `53f9a990587c240cf4a41f66713795139defab2f`;
- the BGP electronic-state integration at
  `681fd5c0b2d82745d96008f450a927a1ec723eb4`;
- the semilocal-path runtime correction at `79389b635`.
- the BGP occupation-convention test at `222a4924e`.
- the functional-specific EXX potential-family correction at `11c945349`.
- the symmetry-tolerant EXX FFT-grid compatibility correction at `bbfa7d67c`.

The official ABACUS repository is
<https://github.com/deepmodeling/abacus-develop>. The workspace-local source
clone was created from the project's existing CIDER-enabled ABACUS repository;
no source repository outside this project was modified.

The exact source export used from the project root was:

```bash
git -C .integration/abacus archive --format=tar \
  --output=/tmp/bgp-abacus-integration.tar \
  bbfa7d67c0476ae4a586719bd4a9713d81df8536
mkdir -p vendor/abacus
tar -xf /tmp/bgp-abacus-integration.tar -C vendor/abacus
```

ABACUS's license is retained as `LICENSE`. Some native-CIDER files carry
additional source-level notices; those notices remain intact in their files.
See `INTEGRATION_PATCHES.md` for the integration history and manual
reconciliation boundaries.
