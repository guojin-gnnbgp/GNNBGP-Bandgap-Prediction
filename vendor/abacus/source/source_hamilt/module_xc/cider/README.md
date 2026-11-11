# Native CIDER module boundary

This directory owns ABACUS-native CIDER inference: the safe legacy-YAML
reader, immutable typed model, normalizers and transforms, mapped spline
evaluator, semilocal and version-J NLDF plans, reverse derivatives, and the
role-separated top-level evaluator. Its model/evaluator interfaces contain no
Python or third-party YAML parser types; yaml-cpp is private to
`model_loader.cpp`.

`cider_native` is an ABACUS-owned static target. It links LibXC and the
ABACUS-selected FFTW3 or MKL backend and has no Python, NumPy, CiderPress, or
`libciderbridge` dependency. yaml-cpp must normally be discoverable through
`yaml-cpp_DIR` or `CMAKE_PREFIX_PATH`; network access is disabled by default.
`CIDER_FETCH_YAML_CPP=ON` explicitly enables the immutable pinned fallback,
and `FETCHCONTENT_SOURCE_DIR_YAML-CPP` can select an offline source tree.

The migrated NLDF C kernels retain provenance and license information in
`NLDF_PROVENANCE.md` and their source headers. Scientific synchronization must
preserve that record and be gated by the frozen oracle tests.

Durable contracts live in the
[architecture and migration record](../../../../docs/developers_guide/cider_native_architecture.md),
[legacy YAML compatibility profile](../../../../docs/developers_guide/cider_legacy_yaml_compatibility.md),
[model-authoring and troubleshooting guide](../../../../docs/developers_guide/cider_model_authoring_and_troubleshooting.md),
[validation record](../../../../docs/developers_guide/cider_native_validation.md),
and [build guide](../../../../BUILD_CIDER.md).
