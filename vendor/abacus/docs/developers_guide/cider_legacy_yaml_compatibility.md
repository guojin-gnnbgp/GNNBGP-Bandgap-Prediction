# CIDER legacy YAML compatibility profile

## Status and scope

This document is the normative input contract for the ABACUS-native CIDER model
loader. It describes a closed, safe subset of the Python-object YAML that
CiderPress currently emits.  It is not a general PyYAML deserializer and it does
not authorize importing a Python name or constructing an object from a tag.
Every tag, mapping key, scalar type, sequence shape, and cross-field relation
must be checked as data before the native model is constructed.

For runtime data flow and migration see
[`cider_native_architecture.md`](cider_native_architecture.md); for concise
model-author guidance and diagnostics see
[`cider_model_authoring_and_troubleshooting.md`](cider_model_authoring_and_troubleshooting.md);
for builds see [the CIDER build guide](cider_build.md); and for acceptance
evidence see [`cider_native_validation.md`](cider_native_validation.md).

The accepted model family is deliberately narrow:

- one `MappedXC` root containing one `MappedDFTKernel` in `SEP` mode;
- one `SplineSetEvaluator` in that kernel;
- semilocal GGA (`np`) or MGGA (`npa`) descriptors;
- either no NLDF descriptors or version-J NLDF descriptors using the observed
  `se`/`one` family;
- identity (`null`) and constant feature normalizers;
- `UMap`, `VMap`, and `ZMap` feature transforms;
- the LDA multiplicative exchange baseline and the Chachiyo or PBE additive
  GGA exchange baseline; and
- the exact NumPy reconstruction encodings described below.

`MappedXC2`, multiple kernels, multiple evaluators, RBF, neural, and linear
evaluators, NLOF, SDMX, hybrid-feature settings, non-version-J NLDF, `POL` and
`NPOL` modes, and arbitrary Python objects are outside this profile. A future
schema may add them explicitly; their existence in CiderPress does not make
them valid for native execution. A dual ABACUS build may still pass such a
model to the external CiderPress bridge with `cider_backend external`; that
does not broaden this native profile.

## Normative corpus

The following files in the parent CiderPress repository were audited as
read-only inputs.  The hashes freeze the bytes used to define this profile.

| Artifact | SHA-256 |
| --- | --- |
| `functionals/CIDER23X_SL_GGA.yaml` | `2e518727b836cd806c4f27c5566e9401b8c15db77182e34380728efdea24f0ce` |
| `functionals/CIDER23X_NL_GGA.yaml` | `b4e29d9530eaa7c94a3c61cc5e942a0cf7cfd0dd3c4a265dc35395123ae26c86` |
| `functionals/CIDER23X_SL_MGGA.yaml` | `fdd60d58ae0f981dcdf64ec24437454bdc334e65b162e379e3630bb99dc82bba` |
| `functionals/CIDER23X_NL_MGGA_DTR.yaml` | `93312ddde97a8b8e88a9f875df221a44f65a1a1b7c955a55d9214bd449e363ea` |
| `abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml` | `0bdadb33599a4f5d26baa743b1a5849a1fb63e0ce6b0fdf7ad9479feb2e28734` |

The structural differences that compatibility tests must retain are:

| Artifact | NumPy spelling | Raw / transformed features | Spline terms (maximum dimension) | Coefficient shapes | Additive baseline | Density policy |
| --- | --- | ---: | ---: | --- | --- | --- |
| SL GGA | `numpy.core` | 2 / 1 | 1 (1-D) | `(52)` | Chachiyo | default `valence` |
| NL GGA | `numpy.core` | 5 / 4 | 7 (3-D) | `(27)`, `(27,82)`, `(27,76)`, `(27,69)`, `(27,82,76)`, `(27,82,69)`, `(27,76,69)` | Chachiyo | inferred legacy `valence_pseudo_core` |
| SL MGGA | `numpy.core` | 3 / 2 | 1 (2-D) | `(52,44)` | Chachiyo | default `valence` |
| NL MGGA DTR | `numpy.core` | 6 / 5 | 11 (3-D) | `(28)`, `(28,24)`, `(28,72)`, `(28,72)`, `(28,69)`, `(28,24,72)`, `(28,24,72)`, `(28,24,69)`, `(28,72,72)`, `(28,72,69)`, `(28,72,69)` | PBE | inferred legacy `valence_pseudo_core` |
| accepted trained NL MGGA | `numpy._core` | 6 / 5 | 11 (3-D) | `(19)`, `(19,32)`, three `(19,19)`, three `(19,32,19)`, three `(19,19,19)` | PBE | explicit `valence` |

The trained artifact stores `scale` as an ordinary YAML sequence and its
`ind_sets` as YAML sequences of NumPy `int64` scalars.  Stock artifacts store
`scale` as a NumPy array and use tuples containing either plain YAML integers or
NumPy integer scalars.  Both forms are required compatibility cases.

## YAML processing boundary

Under `cider_backend native`, the ABACUS `cider_model` input accepts only paths
ending in `.yaml` or `.yml` (case-insensitive). Input conversion inspects typed
native metadata before potential allocation. That metadata, rather than the
artifact filename or a text scan, determines whether MGGA kinetic-energy-density
storage is required and which feature-density policy the model expects. In MPI
builds rank zero performs this inspection and broadcasts either the metadata
packet or the same complete failure message to every rank.

Under `cider_backend external`, the installed CiderPress bridge owns model
compatibility and native whitelist errors are not applied. `cider_backend auto`
chooses external when that capability is compiled and native otherwise; an
explicit request never falls back.

Joblib/pickle suffixes are rejected explicitly.  Other suffixes fail with
`ARTIFACT_UNSUPPORTED`; malformed or structurally unsupported YAML retains the
native loader error.  The input-level diagnostic always includes the selected
model path, native error code, YAML node path, and reason.  An explicit
`cider_feature_density` must agree with the model metadata; `auto` resolves to
the model policy and the deprecated `total` spelling canonicalizes to
`valence_pseudo_core`.

The loader must parse YAML nodes without invoking a Python-object constructor.
It must perform a complete tag-and-key validation pass before decoding a binary
payload or allocating model arrays.  Anchors and aliases are allowed because
the corpus reuses `numpy.ndarray` and `numpy.dtype` nodes, but recursive aliases,
merge keys, duplicate mapping keys, and alias expansion that exceeds the
loader's normal document limits are errors.

The only ordinary YAML tags allowed are mapping, sequence, string, integer,
floating point, Boolean, null, and binary.  Plain integers and floating-point
values are accepted only where a field below says “integer” or “number”;
Booleans are never numbers.  All numbers used by the evaluator must be finite.
YAML timestamps, sets, ordered maps, pairs, and application-specific tags are
not accepted.

Unknown mapping keys are errors, including on otherwise whitelisted object
tags.  Required keys may appear in any order.  A structural sequence may use
an ordinary YAML sequence or `!!python/tuple` only where this document permits
both.  Strings are case-sensitive unless a rule explicitly says otherwise.

## Tag and object whitelist

The table uses the short PyYAML spelling.  A YAML parser normally exposes the
corresponding full URI, for example
`tag:yaml.org,2002:python/object:ciderpress.dft.xc_evaluator.MappedXC`.

| Tag | Required keys | Optional keys and defaults | Additional constraints |
| --- | --- | --- | --- |
| `!!python/object:ciderpress.dft.xc_evaluator.MappedXC` | `kernels`, `settings`, `libxc_baseline` | Root metadata listed below | Must be the root. `kernels` has exactly one element. `libxc_baseline` must be null in this profile. |
| `!!python/object:ciderpress.dft.xc_evaluator.MappedDFTKernel` | `_add_basefunc`, `_mul_basefunc`, `feature_list`, `fevals`, `mode` | `output_scale` defaults to `1.0` | `fevals` has exactly one element, `mode` is exactly `SEP`, and `output_scale` is finite. |
| `!!python/object:ciderpress.dft.xc_evaluator.SplineSetEvaluator` | `coeff_sets`, `const`, `ind_sets`, `nterms`, `scale`, `spline_grids` | none | `nterms` is a positive integer. `const` is finite. The spline invariants below apply. |
| `!!python/object:ciderpress.dft.transform_data.FeatureList` | `feat_list` | none | `feat_list` is a nonempty sequence containing only U/V/Z maps. |
| `!!python/object:ciderpress.dft.transform_data.UMap` | `i`, `gamma`, `_bounds` | none | `i` is a valid raw-feature index, `gamma > 0`, and `_bounds` is a finite increasing pair. |
| `!!python/object:ciderpress.dft.transform_data.VMap` | `i`, `gamma`, `scale`, `center`, `_bounds` | none | `i` is valid, `gamma > 0`, `scale > 0`, and `_bounds` is a finite increasing pair. |
| `!!python/object:ciderpress.dft.transform_data.ZMap` | `i`, `gamma`, `scale`, `center`, `n` | `_bounds` defaults to `(-center, scale - center)` | `i` is valid, `gamma > 0`, `scale > 0`, and the legacy `n` value must be integer `1`. An explicit `_bounds` is a finite increasing pair. |
| `!!python/object:ciderpress.dft.settings.FeatureSettings` | `sl_settings`, `nldf_settings`, `nlof_settings`, `sdmx_settings`, `hyb_settings`, `normalizers` | none | NLOF, SDMX, and hybrid settings must be empty. |
| `!!python/object:ciderpress.dft.settings.EmptySettings` | none | none | The mapping must be empty. |
| `!!python/object:ciderpress.dft.settings.SemilocalSettings` | `level`, `mode` | none | Only `(GGA, np)` and `(MGGA, npa)` are accepted. |
| `!!python/object:ciderpress.dft.settings.NLDFSettingsVJ` | `theta_params`, `rho_mult`, `feat_specs`, `feat_params`, `l0_feat_specs`, `l1_feat_specs`, `l1_feat_dots` | `_sl_level` defaults to `MGGA` | The restricted version-J rules below apply. |
| `!!python/object:ciderpress.dft.feat_normalizer.FeatNormalizerList` | `_normalizers`, `cutoff`, `slmode` | none | `cutoff` is finite and positive. `slmode` equals `sl_settings.mode`. |
| `!!python/object:ciderpress.dft.feat_normalizer.ConstantNormalizer` | `const` | none | `const` is a finite plain number or supported NumPy `float64` scalar. |

`ZMap._bounds` is genuinely optional: it is absent in both stock MGGA files and
present in the accepted trained MGGA file.  The stated default is the current
`ZMap.bounds` compatibility behavior.  `_sl_level` is likewise present in the
stock NL GGA model and absent in both audited NL MGGA models; the current
`NLDFSettings.sl_level` fallback is `MGGA`.  No other missing object attribute
receives an implicit default.

### Named-function tags

Only these scalar tags are accepted, and their scalar payload must be empty:

- `!!python/name:ciderpress.dft.baselines.lda_x`, only at
  `$.kernels[0]._mul_basefunc`;
- `!!python/name:ciderpress.dft.baselines.gga_x_chachiyo`, only at
  `$.kernels[0]._add_basefunc`; and
- `!!python/name:ciderpress.dft.baselines.gga_x_pbe`, only at
  `$.kernels[0]._add_basefunc`.

The tag identifies an internal native enum; it never triggers name lookup.
Null or another named function at either baseline position is not accepted.

## Settings and feature-count rules

Let `N_sl` be 2 for GGA and 3 for MGGA.  An empty NLDF setting contributes
`N_nldf = 0`.  A version-J setting contributes one raw feature for every entry
of `feat_specs`.  The raw feature count is

`N_raw = N_sl + N_nldf`.

The version-J subset is constrained as follows:

- the effective `_sl_level` is `GGA` or `MGGA` and equals
  `sl_settings.level`;
- `rho_mult` is exactly `one`;
- `feat_specs` is a nonempty sequence containing only `se`;
- `feat_params` has the same length as `feat_specs`;
- `theta_params` is exactly one three-number sequence
  `[a0, grad_mul, tau_mul]`;
- every element of `feat_params` is exactly one three-number sequence with the
  same layout;
- in each such sequence, `a0 > 0`, `grad_mul >= 0`, and `tau_mul >= 0`; the
  third component is retained but ignored by the GGA exponent path; and
- the compatibility fields `l0_feat_specs`, `l1_feat_specs`, and
  `l1_feat_dots` are present and empty.

An NLDF tag from another version, `rho_mult: expnt`, another feature spec, or a
nonempty legacy l=0/l=1 field is an unsupported descriptor family, not a
request for a best-effort approximation.

`normalizers._normalizers` must have exactly `N_raw` entries.  Each entry is
either null (the identity normalizer) or a whitelisted `ConstantNormalizer`.
The number of entries in `feature_list.feat_list` defines `N_transformed`.
Every map index `i` must satisfy `0 <= i < N_raw`, and every spline descriptor
index must satisfy `0 <= index < N_transformed`.  The parser must not require
every raw feature to be referenced, but no missing feature may be fabricated.

These equalities are checked before evaluation:

- settings-derived `N_raw` equals the normalizer count;
- `nterms` equals the lengths of `scale`, `ind_sets`, `spline_grids`, and
  `coeff_sets`;
- each term's descriptor-index count equals its grid count and coefficient
  rank; and
- each term's coefficient extent in dimension `d` equals
  `spline_grids[term][d][2] + 2`.

## Spline payload

`scale` may be a one-dimensional supported `float64` NumPy array or an ordinary
sequence of finite numbers.  Its length is `nterms`.  Each `ind_sets` entry is
a list or tuple of one, two, or three distinct descriptor indices.  An index
may be a plain YAML integer or a supported NumPy `int64` scalar.  Dimension zero
and dimensions four or greater are rejected by the first-release profile.

Each `spline_grids` term is a list or tuple of triples `(lower, upper, order)`.
`lower` and `upper` are finite numbers with `lower < upper`; `order` is a
positive integer.  For every referenced transformed feature, the grid interval
must equal that feature map's effective bounds.  Each coefficient set is a
C-order `float64` ndarray of rank one through three.  Its rank and extents obey
the equalities above.  `const`, every scale, grid endpoint, transform parameter,
normalizer constant, and coefficient must be finite.

The exact corpus uses grid orders from 17 through 80 and coefficient extents
from 19 through 82.  Those observations are fixtures, not extra hard-coded
limits: structural validity and checked size arithmetic are authoritative.
Zero extents, multiplication overflow, a decoded byte count different from the
shape byte count, or an allocation larger than the loader's configured input
budget is a malformed-array error.

## NumPy reconstruction whitelist

The native reader recognizes the following tags only in the stated roles:

- `!!python/name:numpy.ndarray`, as the first reconstruct argument;
- `!!python/object/apply:numpy.dtype`, inside an ndarray or scalar encoding;
- `!!python/object/apply:numpy.core.multiarray._reconstruct` and
  `!!python/object/apply:numpy._core.multiarray._reconstruct`, as exact aliases;
- `!!python/object/apply:numpy.core.multiarray.scalar` and
  `!!python/object/apply:numpy._core.multiarray.scalar`, as exact aliases; and
- `!!python/tuple`, for the tuple positions described here and elsewhere in
  this profile.

The `numpy.core` spelling is emitted by the four stock NumPy-1-era files.  The
`numpy._core` spelling is emitted by the accepted NumPy-2-era trained file.
After recognizing the alias, both spellings have identical validation and
decode behavior.  A mixed document is valid if every node is independently
valid.

### Dtype object

The dtype node must have exactly this reconstruction shape:

- mapping keys `args` and `state` only;
- `args` is `[typecode, false, true]`, where `typecode` is `f8` or `i8`; and
- `state` is the tuple `(3, '<', null, null, null, -1, -1, 0)`.

Only little-endian IEEE-754 binary64 (`f8`) and little-endian signed two's
complement 64-bit integer (`i8`) are supported.  Big-endian (`>`), native-order
(`=`), byte-order-independent (`|`), object, structured, subarray, Unicode,
byte-string, and every other dtype are rejected.  The decoder interprets the
little-endian bytes explicitly, independent of host endianness.

### Ndarray object

The reconstruct node must have mapping keys `args` and `state` only:

- `args` is `(!!python/name:numpy.ndarray '', (0,), !!binary "Yg==")`;
- `state` is `(1, shape, dtype, false, data)`;
- `shape` is a tuple of positive integers;
- `dtype` is the supported `f8` dtype above;
- the fourth state item is exactly `false`, selecting C order; and
- `data` is YAML binary whose decoded length is exactly
  `8 * product(shape)` bytes.

Fortran order is not accepted.  Arrays are allowed only as spline coefficient
sets or `scale`; coefficient and scale shape rules further constrain them.
The placeholder binary values in `args` do not carry model data and must match
the one-byte value shown above.  The state version and every tuple arity are
fixed; trailing or missing elements are malformed.

### NumPy scalar object

The scalar node is exactly the two-element sequence `(dtype, data)`.  `data`
must decode to eight bytes.  `i8` is allowed only for spline descriptor indices,
and `f8` is allowed only where this profile permits a numeric scalar (currently
constant normalizers).  Arrays cannot be substituted for scalars or vice versa.

## Root metadata and defaults

All unlisted root keys are rejected.  The following keys are optional in
addition to the three required `MappedXC` keys.

| Key | Accepted value | Default / interpretation |
| --- | --- | --- |
| `feature_density_policy` | `valence`, `valence_pseudo_core`, deprecated alias `total`, or the missing-policy sentinels null, empty string, and `auto` | Resolution rules below. `total` normalizes to `valence_pseudo_core`; a sentinel is treated exactly like an absent key. |
| `feature_density_policy_source` | string | With an explicit policy, missing or empty resolves to `model`. It has no numerical effect. |
| `ml_exchange_scale` | null or a finite nonnegative number | Missing resolves to `1.0`, matching the current bridge. Null or zero means use the requested runtime exchange mix directly; otherwise runtime scale is requested mix divided by this value. |
| `ml_exchange_label_convention` | `full_exact_exchange` | Training provenance only; the four label keys form one atomic group. |
| `ml_exchange_label_hybrid_scaled` | `false` | Training provenance only. |
| `ml_exchange_label_scale` | finite number equal to `1.0` | Training provenance only. |
| `ml_exchange_label_source_scales` | nonempty sequence of finite positive numbers | Training provenance only. |

The four `ml_exchange_label_*` keys must either all be absent or all be present
and valid.  When absent, the native metadata representation records “legacy
unspecified”; it must not infer a label convention from the model filename.
These label fields do not rescale evaluation.  Kernel `output_scale` and the
separate runtime `ml_exchange_scale` are the only model-side scaling controls in
this profile.  The accepted trained artifact has the complete label group,
`output_scale: 1.0`, and no `ml_exchange_scale`; stock artifacts have none of
these optional fields.

The following runtime metadata is derived rather than read from additional YAML
keys: `MappedXC` has exchange-energy convention `energy_density`; MGGA or an
MGGA-level NLDF setting requires kinetic-energy density; and the required null
`libxc_baseline` resolves to the current PBE exchange/PBE correlation outer
baseline for this profile.  A YAML key attempting to override any of these
derived values is an unknown root key.  The named LDA and GGA baseline tags in
the mapped kernel are separate: they define how the learned spline residual is
assembled inside `MappedDFTKernel`.

### Feature-density policy resolution

Policy resolution occurs only after descriptor validation determines whether
the model has NLDF features:

1. If `feature_density_policy` is explicit and is not a missing-policy
   sentinel, normalize `total` as above and use it.
   `feature_density_policy_source` defaults to `model`.
2. If the policy is absent and the model has no NLDF features, use `valence`
   with source `default_valence`.
3. If the policy is absent, the model has NLDF features, and the input file's
   basename contains `CIDER23X` case-insensitively, use
   `valence_pseudo_core` with source `inferred_legacy_cider23x`.  This is the
   compatibility rule used by the two stock NLDF fixtures.
4. Every other NLDF model without an explicit policy is rejected with
   `DENSITY_POLICY_REQUIRED`.

A `feature_density_policy_source` without a policy is rejected rather than used
to bypass these rules.  The basename rule is intentionally narrow and
filename-dependent because it preserves the current bridge behavior; copying a
stock object graph to an unrelated basename requires adding explicit metadata.

## Deterministic validation failures

Every `LegacyYamlError` produced by the native model loader uses the form

`CIDER legacy YAML error [CODE] at PATH: DETAIL`

where `PATH` is `$` plus mapping keys and zero-based sequence indexes, or
`<file>` before a YAML root exists.  Tests should assert the code and path; the
detail may add values and expected constraints.  Validation stops at the first
failure in this order: file format, YAML syntax/duplicate keys, tag class,
per-node schema, NumPy payload, cross-field invariants, then density metadata.
Input conversion wraps that information as
`CIDER model inspection failed: model='...' code=CODE yaml_path=PATH reason=DETAIL`
so the selected file and the same structured fields reach every MPI rank.

| Code | Required use |
| --- | --- |
| `JOBLIB_UNSUPPORTED` | The selected input path has a case-insensitive `.joblib`, `.pkl`, or `.pickle` suffix. Joblib/pickle input is never passed to the YAML parser. A non-YAML binary given a YAML suffix produces `YAML_SYNTAX` because joblib has no single reliable magic signature. |
| `ARTIFACT_UNSUPPORTED` | Input conversion selected a suffix other than `.yaml`/`.yml` and the joblib/pickle suffixes above. |
| `YAML_SYNTAX` | Invalid YAML, invalid UTF-8, duplicate mapping keys, recursive/disallowed alias use, or a document with zero or multiple roots. |
| `UNKNOWN_TAG` | The default for a nonstandard tag that is neither whitelisted nor inside one of the recognized family prefixes enumerated below. The diagnostic includes the complete tag text. |
| `UNSUPPORTED_ROOT` | A recognized CIDER root family other than `MappedXC`, including `MappedXC2`, or a whitelisted object tag in the wrong root position. |
| `UNSUPPORTED_EVALUATOR` | A recognized XC evaluator tag other than `SplineSetEvaluator`, more than one evaluator, or more than one mapped kernel. This includes RBF, neural, and global-linear evaluators. |
| `UNSUPPORTED_DESCRIPTOR` | NLOF, SDMX, nonempty hybrid settings, non-version-J NLDF, unsupported semilocal/NLDF modes or values, nonempty l=0/l=1 version-J fields, or a feature-map tag other than U/V/Z. |
| `UNSUPPORTED_NORMALIZER` | A normalizer entry other than null or `ConstantNormalizer`. |
| `UNSUPPORTED_NUMPY_TAG` | Any non-whitelisted Python tag targeting the `numpy.*` namespace, including `python/module`, `python/name`, `python/object`, `python/object/new`, and `python/object/apply` forms, another reconstruction or scalar callable, a non-aliased module spelling, or a NumPy class tag other than the exact `numpy.ndarray` role. Structurally malformed nodes using a whitelisted NumPy tag use `MALFORMED_ARRAY` instead. |
| `UNSUPPORTED_NAMED_FUNCTION` | A `python/name:ciderpress.dft.baselines.*` tag other than the three whitelisted functions, or a whitelisted baseline-function tag used at any path other than its allowed `_mul_basefunc`/`_add_basefunc` position. |
| `SCHEMA` | Missing/extra mapping keys, wrong ordinary scalar/container type, invalid enum or numeric range not covered by a more specific code, or invalid cardinality. |
| `MALFORMED_ARRAY` | Invalid NumPy reconstruction/dtype/scalar shape, dtype, endian, order, state version, base64, byte count, dimension, coefficient shape, nonfinite payload, checked-size overflow, or input-budget violation. |
| `FEATURE_COUNT` | A mismatch among settings-derived raw features, normalizers, transform indices, transformed features, spline index sets, grids, terms, scales, and coefficient arrays. The path points to the first inconsistent field. |
| `DENSITY_POLICY_REQUIRED` | An NLDF model has no explicit policy and is not eligible for the legacy CIDER23X basename rule. |
| `DENSITY_POLICY_INVALID` | A non-sentinel policy is outside the accepted values, a source appears without an effective policy, or policy metadata has the wrong type. |
| `MODEL_LOAD_FAILED` | Input conversion caught an unexpected native model-load exception that did not already carry a structured loader code. Retain the model path and complete reason when reporting it. |

Tag classification is exhaustive and ordered.  An exact whitelisted tag in an
allowed role proceeds to node validation.  Otherwise:

- any `python/object:ciderpress.dft.xc_evaluator.*` or
  `python/object:ciderpress.dft.xc_evaluator2.*` tag at `$` other than the
  allowed `MappedXC`, and `MappedXC` itself at a non-root path, uses
  `UNSUPPORTED_ROOT`; any other disallowed tag or placement with either prefix
  below the root uses `UNSUPPORTED_EVALUATOR`;
- `python/object:ciderpress.dft.settings.*` and
  `python/object:ciderpress.dft.transform_data.*` use
  `UNSUPPORTED_DESCRIPTOR`;
- `python/object:ciderpress.dft.feat_normalizer.*` uses
  `UNSUPPORTED_NORMALIZER`;
- any Python tag whose target begins `numpy.`—including `python/module`,
  `python/name`, `python/object`, `python/object/new`, and
  `python/object/apply`—uses `UNSUPPORTED_NUMPY_TAG`;
- `python/name:ciderpress.dft.baselines.*` uses
  `UNSUPPORTED_NAMED_FUNCTION`; and
- every other nonstandard tag uses `UNKNOWN_TAG`.

No nonstandard tag may be silently treated as a generic mapping, sequence, or
scalar.  This distinction makes failures actionable without broadening the
whitelist.  No error may fall back to Python import,
joblib loading, tag construction, evaluator substitution, descriptor omission,
array reshaping, byte-order guessing, or density-policy guessing beyond the one
documented legacy rule.

## Required compatibility tests

Tasks implementing the loader and typed model should derive positive and
negative tests directly from this contract.  At minimum they must cover:

- successful metadata and structural loading of all five hashed artifacts;
- equality of the `numpy.core` and `numpy._core` reconstruction paths;
- array-backed and sequence-backed spline scales;
- plain and NumPy-scalar descriptor indices;
- accepted one-, two-, and three-dimensional terms plus rejected zero- and
  four-dimensional terms;
- non-whitelisted NumPy reconstruction/class tags, unknown baseline named
  functions, and whitelisted baseline functions in the wrong position;
- absent and explicit `ZMap._bounds`, absent and explicit `_sl_level`, and
  absent and explicit `output_scale`;
- explicit `valence`, semilocal default `valence`, stock CIDER23X NLDF
  inference, renamed NLDF rejection, and the deprecated `total` alias;
- every deterministic error code above, including path reporting;
- byte-count, endian, Fortran-order, rank, extent, finite-value, and overflow
  failures for arrays; and
- every feature-count equality and descriptor-index bound.

Passing PyYAML loading is not evidence that a file satisfies this profile.
Conversely, a file satisfying this profile must be loadable without Python,
NumPy, joblib, or the CiderPress package at ABACUS runtime.

## Compatibility evolution

This legacy profile remains supported even if a cleaner exporter is added.
Future CiderPress tooling should emit a separate declarative format with an
explicit `schema_version`, evaluator and descriptor enums, density and energy
conventions, training-scale provenance, and explicit array dtype, order,
shape, and checksum fields. It should not contain language-specific object
constructors or depend on inferred runtime defaults. Such a format is a future
recommendation, not an accepted input today; it requires its own fixtures and
schema-version tests before ABACUS support is claimed.

## Python-reference anchors

The corpus inventory above is backed by the current Python definitions, not by
tag names alone.  The relevant reference points in the parent CiderPress tree
are `ciderpress/dft/model_utils.py` for YAML/joblib dispatch,
`ciderpress/dft/xc_evaluator.py` for `MappedXC`, kernel output scaling, and
spline evaluation, `ciderpress/dft/settings.py` for semilocal and version-J
feature counts and the legacy `_sl_level` fallback,
`ciderpress/dft/transform_data.py` for U/V/Z transforms and bounds, and
`ciderpress/dft/feat_normalizer.py` for identity/constant normalization.
`ciderpress/lib/bridge/cider_bridge_py.py` is authoritative for exchange runtime
scaling, energy convention, density-policy normalization, and the legacy
CIDER23X filename inference.  These files are implementation evidence; they do
not expand the whitelist beyond this document.
