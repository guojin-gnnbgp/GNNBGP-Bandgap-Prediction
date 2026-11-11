#!/usr/bin/env python3
"""Generate compact v1 native-CIDER oracle fixtures with parent CiderPress.

The generated payloads are raw, C-order, explicitly little-endian arrays.  No
pickle-capable serialization is used.  This script is a development-time tool;
the native ABACUS runtime must not import or invoke it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "abacus-cider-python-reference-oracle-v1"
GRID_SHAPE = (2, 2, 2)
CELL_LENGTH = 4.0
XMIX = 0.25
NALPHA = 8
ENCUT = 20.0
DEFAULT_TOLERANCE = {"atol": 1.0e-12, "rtol": 1.0e-11}
FFT_TOLERANCE = {"atol": 2.0e-11, "rtol": 2.0e-10}
LIBXC_TOLERANCE = {"atol": 2.0e-12, "rtol": 2.0e-11}

CASES = (
    {
        "id": "sl_gga_spin1_valence",
        "model": "functionals/CIDER23X_SL_GGA.yaml",
        "nspin": 1,
        "feature_density_policy": "valence",
        "coverage": ["gga", "non_nldf", "spin_unpolarized", "valence"],
    },
    {
        "id": "sl_mgga_spin2_valence",
        "model": "functionals/CIDER23X_SL_MGGA.yaml",
        "nspin": 2,
        "feature_density_policy": "valence",
        "coverage": ["mgga", "non_nldf", "spin_polarized", "valence"],
    },
    {
        "id": "nl_gga_spin2_valence_pseudo_core",
        "model": "functionals/CIDER23X_NL_GGA.yaml",
        "nspin": 2,
        "feature_density_policy": "valence_pseudo_core",
        "coverage": [
            "gga",
            "nldf",
            "spin_polarized",
            "valence_pseudo_core",
        ],
    },
    {
        "id": "nl_mgga_spin1_valence_pseudo_core",
        "model": "functionals/CIDER23X_NL_MGGA_DTR.yaml",
        "nspin": 1,
        "feature_density_policy": "valence_pseudo_core",
        "coverage": [
            "mgga",
            "nldf",
            "spin_unpolarized",
            "valence_pseudo_core",
        ],
    },
    {
        "id": "trained_nl_mgga_spin2_valence",
        "model": "abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml",
        "nspin": 2,
        "feature_density_policy": "valence",
        "coverage": [
            "mgga",
            "nldf",
            "spin_polarized",
            "valence",
            "numpy_2_core_spelling",
            "explicit_density_metadata",
            "trained_model",
        ],
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(parent: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(parent), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _class_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _callable_name(obj: Any) -> str | None:
    if obj is None:
        return None
    return f"{obj.__module__}.{obj.__name__}"


def _canonical_array(array: Any) -> tuple[np.ndarray, str, str]:
    arr = np.asarray(array)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype.kind == "f":
        arr = np.ascontiguousarray(arr, dtype="<f8")
        return arr, "<f8", "f64le"
    if arr.dtype.kind in "iu":
        if arr.dtype.itemsize <= 4:
            arr = np.ascontiguousarray(arr, dtype="<i4")
            return arr, "<i4", "i32le"
        arr = np.ascontiguousarray(arr, dtype="<i8")
        return arr, "<i8", "i64le"
    raise TypeError(f"unsupported oracle array dtype {arr.dtype}")


class FixtureWriter:
    def __init__(self, root: Path):
        self.root = root
        self.arrays: dict[str, dict[str, Any]] = {}
        self.payload_bytes = 0
        self.blob = bytearray()

    def reset(self) -> None:
        data_dir = self.root / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        blob = self.root / "arrays.bin"
        if blob.exists():
            blob.unlink()
        manifest = self.root / "manifest.json"
        if manifest.exists():
            manifest.unlink()

    def finish(self) -> None:
        (self.root / "arrays.bin").write_bytes(self.blob)

    def add(
        self,
        case: dict[str, Any],
        name: str,
        array: Any,
        stage: str,
        description: str,
        tolerance: dict[str, float] | None = None,
    ) -> None:
        arr, dtype, _ = _canonical_array(array)
        if dtype == "<f8" and not np.isfinite(arr).all():
            raise ValueError(f"non-finite values in {case['id']}.{name}")
        array_id = f"{case['id']}.{name}"
        data = arr.tobytes(order="C")
        offset = len(self.blob)
        self.blob.extend(data)
        spec = {
            "path": "arrays.bin",
            "offset": offset,
            "dtype": dtype,
            "shape": list(arr.shape),
            "bytes": len(data),
            "sha256": _bytes_sha256(data),
            "stage": stage,
            "description": description,
            "tolerance": tolerance or DEFAULT_TOLERANCE,
        }
        self.arrays[array_id] = spec
        self.payload_bytes += len(data)
        case["array_ids"].append(array_id)
        case["stages"].add(stage)


def _input_tuples(nspin: int, is_mgga: bool) -> dict[str, np.ndarray | None]:
    grid = np.arange(np.prod(GRID_SHAPE), dtype=np.float64)
    phase = 2.0 * np.pi * grid / grid.size
    if nspin == 1:
        rho_valence = (0.082 + 0.0045 * grid + 0.003 * np.sin(phase))[None, :]
        sigma_valence = (0.0038 + 0.00022 * grid + 0.0003 * np.cos(phase))[None, :]
        tau_valence = (
            (0.115 + 0.007 * grid + 0.003 * np.cos(phase))[None, :]
            if is_mgga
            else None
        )
        rho_core = (0.006 + 0.0012 * (1.0 + np.cos(phase)))[None, :]
        sigma_core = (0.00035 + 0.00004 * (1.0 + np.sin(phase)))[None, :]
        tau_core = (
            (0.012 + 0.001 * (1.0 + np.sin(phase)))[None, :]
            if is_mgga
            else None
        )
    else:
        rho_valence = np.stack(
            [
                0.052 + 0.0031 * grid + 0.002 * np.sin(phase),
                0.034 + 0.0023 * grid + 0.0015 * np.cos(phase),
            ]
        )
        sigma_valence = np.stack(
            [
                0.0028 + 0.00018 * grid + 0.0002 * np.cos(phase),
                0.00055 + 0.00003 * grid,
                0.0021 + 0.00014 * grid + 0.00015 * np.sin(phase),
            ]
        )
        tau_valence = (
            np.stack(
                [
                    0.078 + 0.005 * grid + 0.002 * np.cos(phase),
                    0.056 + 0.004 * grid + 0.0015 * np.sin(phase),
                ]
            )
            if is_mgga
            else None
        )
        core_total = 0.007 + 0.001 * (1.0 + np.cos(phase))
        rho_core = np.stack([0.5 * core_total, 0.5 * core_total])
        sigma_core = np.stack(
            [
                0.00024 + 0.00003 * (1.0 + np.sin(phase)),
                0.00005 + 0.000005 * grid,
                0.00021 + 0.000025 * (1.0 + np.cos(phase)),
            ]
        )
        tau_core = (
            np.stack(
                [
                    0.007 + 0.0006 * (1.0 + np.sin(phase)),
                    0.006 + 0.0005 * (1.0 + np.cos(phase)),
                ]
            )
            if is_mgga
            else None
        )
    return {
        "rho_valence_sg": rho_valence,
        "sigma_valence_xg": sigma_valence,
        "tau_valence_sg": tau_valence,
        "rho_total_sg": rho_valence + rho_core,
        "sigma_total_xg": sigma_valence + sigma_core,
        "tau_total_sg": (
            None if tau_valence is None or tau_core is None else tau_valence + tau_core
        ),
    }


def _sample_coefficients(feval: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    samples = np.empty((feval.nterms, 9), dtype=np.float64)
    metadata = []
    for term, coeff in enumerate(feval.coeff_sets):
        canonical = np.ascontiguousarray(coeff, dtype="<f8")
        flat = canonical.ravel(order="C")
        indexes = np.linspace(0, flat.size - 1, 9, dtype=np.int64)
        samples[term] = flat[indexes]
        metadata.append(
            {
                "term": term,
                "shape": list(canonical.shape),
                "dtype": "<f8",
                "byte_count": int(canonical.nbytes),
                "sha256": _bytes_sha256(canonical.tobytes(order="C")),
                "sample_flat_indices": indexes.tolist(),
            }
        )
    return samples, metadata


def _decoded_model_arrays(
    writer: FixtureWriter, case: dict[str, Any], feval: Any
) -> list[dict[str, Any]]:
    nterms = int(feval.nterms)
    dims = np.asarray([len(indices) for indices in feval.ind_sets], dtype=np.int32)
    max_dim = int(dims.max())
    ind_sets = np.full((nterms, max_dim), -1, dtype=np.int32)
    grids = np.zeros((nterms, max_dim, 3), dtype=np.float64)
    for term, indices in enumerate(feval.ind_sets):
        ind_sets[term, : len(indices)] = np.asarray(indices, dtype=np.int32)
        grids[term, : len(indices)] = np.asarray(feval.spline_grids[term], dtype=np.float64)
    coeff_samples, coeff_metadata = _sample_coefficients(feval)
    writer.add(case, "model_scale", feval.scale, "decoded_model_arrays", "Decoded spline term scales")
    writer.add(case, "model_ind_set_dims", dims, "decoded_model_arrays", "Active descriptor count per spline term")
    writer.add(case, "model_ind_sets", ind_sets, "decoded_model_arrays", "Decoded spline descriptor indices; -1 pads inactive columns")
    writer.add(case, "model_spline_grids", grids, "decoded_model_arrays", "Decoded spline lower/upper/order triples; zero pads inactive columns")
    writer.add(case, "model_coefficient_samples", coeff_samples, "decoded_model_arrays", "Nine deterministic flat samples from every decoded coefficient array")
    return coeff_metadata


def _feature_map_metadata(kernel: Any) -> list[dict[str, Any]]:
    result = []
    for feature in kernel.feature_list.feat_list:
        entry = {"class": _class_name(feature)}
        for name in ("i", "gamma", "scale", "center", "n", "_bounds"):
            if hasattr(feature, name):
                entry[name] = _json_value(getattr(feature, name))
        entry["effective_bounds"] = _json_value(feature.bounds)
        result.append(entry)
    return result


def _normalizer_metadata(settings: Any) -> list[dict[str, Any] | None]:
    result = []
    for normalizer in settings.normalizers._normalizers:
        if normalizer is None:
            result.append(None)
            continue
        entry = {"class": _class_name(normalizer)}
        for name, value in vars(normalizer).items():
            if isinstance(value, (str, int, float, bool, np.generic)) or value is None:
                entry[name] = _json_value(value)
        result.append(entry)
    return result


def _integrated_results(result: dict[str, np.ndarray | None], inputs: dict[str, Any]) -> np.ndarray:
    weight = CELL_LENGTH**3 / np.prod(GRID_SHAPE)

    def integral(name: str) -> float:
        return float(np.sum(result[name]) * weight)

    def contraction(prefix: str, density_role: str) -> float:
        rho = inputs[f"rho_{density_role}_sg"]
        sigma = inputs[f"sigma_{density_role}_xg"]
        tau = inputs[f"tau_{density_role}_sg"]
        value = np.sum(result[f"vrho_{prefix}_sg"] * rho)
        value += np.sum(result[f"vsigma_{prefix}_xg"] * sigma)
        if tau is not None and result[f"vtau_{prefix}_sg"] is not None:
            value += np.sum(result[f"vtau_{prefix}_sg"] * tau)
        return float(value * weight)

    return np.asarray(
        [
            weight,
            integral("exc_baseline_g"),
            integral("exc_feature_g"),
            integral("exc_total_g"),
            contraction("baseline", "baseline"),
            contraction("feature", "feature"),
        ],
        dtype=np.float64,
    )


def _generate_case(parent: Path, writer: FixtureWriter, definition: dict[str, Any]) -> dict[str, Any]:
    from ciderpress.lib.bridge import cider_bridge_py as bridge

    model_path = parent / definition["model"]
    info = bridge.create(
        str(model_path),
        GRID_SHAPE,
        (np.eye(3) * CELL_LENGTH).ravel(),
        definition["nspin"],
        xmix=XMIX,
        Nalpha=NALPHA,
        encut=ENCUT,
    )
    model = bridge._models[info["id"]]
    mlfunc = model["mlfunc"]
    kernel = mlfunc.kernels[0]
    feval = kernel.fevals[0]
    case: dict[str, Any] = {
        "id": definition["id"],
        "coverage": list(definition["coverage"]),
        "stages": {"loader_metadata"},
        "array_ids": [],
        "source_model": {
            "path": definition["model"],
            "sha256": _sha256(model_path),
            "bytes": model_path.stat().st_size,
        },
        "grid": {
            "shape": list(GRID_SHAPE),
            "cell_cv": (np.eye(3) * CELL_LENGTH).tolist(),
            "volume": CELL_LENGTH**3,
            "grid_weight": CELL_LENGTH**3 / np.prod(GRID_SHAPE),
            "layout": "spin-major C order",
        },
        "loader_metadata": {
            key: _json_value(value) for key, value in info.items() if key != "id"
        },
        "model_structure": {
            "root_class": _class_name(mlfunc),
            "kernel_class": _class_name(kernel),
            "kernel_mode": kernel.mode,
            "kernel_output_scale": float(getattr(kernel, "output_scale", 1.0)),
            "multiplicative_baseline": _callable_name(kernel._mul_basefunc),
            "additive_baseline": _callable_name(kernel._add_basefunc),
            "evaluator_class": _class_name(feval),
            "spline_nterms": int(feval.nterms),
            "spline_const": float(feval.const),
            "semilocal_level": mlfunc.settings.sl_settings.level,
            "semilocal_mode": mlfunc.settings.sl_settings.mode,
            "feature_maps": _feature_map_metadata(kernel),
            "normalizers": _normalizer_metadata(mlfunc.settings),
        },
        "requested_feature_density_policy": definition["feature_density_policy"],
        "integrated_result_labels": [
            "grid_weight",
            "exc_baseline_volume_integral",
            "exc_feature_volume_integral",
            "exc_total_volume_integral",
            "baseline_potential_density_contraction",
            "feature_potential_density_contraction",
        ],
    }
    coeff_metadata = _decoded_model_arrays(writer, case, feval)
    case["model_structure"]["coefficient_arrays"] = coeff_metadata

    inputs = _input_tuples(definition["nspin"], bool(info["is_mgga"]))
    inputs["rho_baseline_sg"] = inputs["rho_total_sg"]
    inputs["sigma_baseline_xg"] = inputs["sigma_total_xg"]
    inputs["tau_baseline_sg"] = inputs["tau_total_sg"]
    feature_suffix = (
        "valence" if definition["feature_density_policy"] == "valence" else "total"
    )
    inputs["rho_feature_sg"] = inputs[f"rho_{feature_suffix}_sg"]
    inputs["sigma_feature_xg"] = inputs[f"sigma_{feature_suffix}_xg"]
    inputs["tau_feature_sg"] = inputs[f"tau_{feature_suffix}_sg"]
    for name in (
        "rho_baseline_sg",
        "sigma_baseline_xg",
        "tau_baseline_sg",
        "rho_feature_sg",
        "sigma_feature_xg",
        "tau_feature_sg",
    ):
        if inputs[name] is not None:
            writer.add(case, name, inputs[name], "dual_density_role_outputs", f"Deterministic ABACUS-layout {name} input")

    capture: dict[str, np.ndarray | None] = {}
    original_dump = bridge._maybe_dump_feature_intermediates

    def capture_dump(_model: Any, rho: Any, sigma: Any, tau: Any, **values: Any) -> None:
        capture["rho_feature_sg"] = np.asarray(rho).copy()
        capture["sigma_feature_xg"] = np.asarray(sigma).copy()
        capture["tau_feature_sg"] = None if tau is None else np.asarray(tau).copy()
        for name, value in values.items():
            capture[name] = None if value is None else np.asarray(value).copy()

    bridge._maybe_dump_feature_intermediates = capture_dump
    try:
        result = bridge._evaluate_dual_density_core(
            model,
            inputs["rho_baseline_sg"],
            inputs["sigma_baseline_xg"],
            inputs["tau_baseline_sg"],
            inputs["rho_feature_sg"],
            inputs["sigma_feature_xg"],
            inputs["tau_feature_sg"],
        )
    finally:
        bridge._maybe_dump_feature_intermediates = original_dump

    x0t = capture["X0T"]
    x0tn = capture["X0TN"]
    assert x0t is not None and x0tn is not None
    nfeat_sl = int(info["nfeat_sl"])
    writer.add(case, "semilocal_raw_features", x0t[:, :nfeat_sl], "semilocal_raw_features", "SemilocalPlan2 raw np/npa features")
    writer.add(case, "raw_features", x0t, "semilocal_raw_features", "Combined semilocal and optional NLDF raw features", FFT_TOLERANCE)
    writer.add(case, "normalized_features", x0tn, "normalized_features", "Feature-normalizer forward result", FFT_TOLERANCE)

    x1 = kernel.get_descriptors(x0tn, force_polarize=True)
    spline_forward, spline_deriv = feval(x1)
    spline_reverse_raw = kernel.apply_descriptor_grad(x0tn, spline_deriv, force_polarize=True)
    mapped_exc, mapped_deriv_norm = mlfunc(x0tn, rhocut=bridge.RHO_TOL)
    mapped_deriv_raw = mlfunc.settings.normalizers.get_derivative_wrt_unnormed_features(
        x0t, mapped_deriv_norm
    )
    writer.add(case, "transformed_descriptors", x1, "transformed_descriptors", "UMap/VMap/ZMap transformed descriptors", FFT_TOLERANCE)
    writer.add(case, "spline_forward", spline_forward, "spline_forward", "SplineSetEvaluator residual before baselines")
    writer.add(case, "spline_derivatives", spline_deriv, "spline_derivatives", "Spline residual derivatives with respect to transformed descriptors")
    writer.add(case, "spline_reverse_raw", spline_reverse_raw, "reverse_feature_derivatives", "Spline derivative reversed through feature transforms")
    writer.add(case, "mapped_exc_unscaled", mapped_exc, "spline_forward", "MappedXC output before bridge runtime exchange scaling", FFT_TOLERANCE)
    writer.add(case, "mapped_derivative_normalized_unscaled", mapped_deriv_norm, "reverse_feature_derivatives", "MappedXC derivative with respect to normalized features", FFT_TOLERANCE)
    writer.add(case, "mapped_derivative_raw_unscaled", mapped_deriv_raw, "reverse_feature_derivatives", "MappedXC derivative reversed through feature normalizers", FFT_TOLERANCE)

    if info["has_nldf"]:
        for capture_name, output_name, stage, description in (
            ("arg_sg", "nldf_interpolation_arguments", "nldf_interpolation_arguments", "Version-J source interpolation arguments"),
            ("fun_sg", "nldf_convolution_function", "nldf_convolution_features", "Version-J real-space function sent to convolution"),
            ("feat_sig", "nldf_convolution_features", "nldf_convolution_features", "Version-J convolved NLDF features"),
        ):
            value = capture[capture_name]
            assert value is not None
            writer.add(case, output_name, value, stage, description, FFT_TOLERANCE)

        rho_eval, sigma_eval, tau_eval = bridge._regularized_feature_density_tuple(
            inputs["rho_feature_sg"], inputs["sigma_feature_xg"], inputs["tau_feature_sg"]
        )
        nldf_tuple = bridge._same_spin_rho_tuple(rho_eval, sigma_eval, tau_eval)
        _, darg = model["plan"].get_interpolation_arguments(nldf_tuple, i=-1)
        _, dfun = model["plan"].get_function_to_convolve(nldf_tuple)
        writer.add(case, "nldf_interpolation_argument_derivatives", np.stack(darg), "reverse_feature_derivatives", "Interpolation-argument derivatives with respect to rho/same-spin sigma/optional tau", FFT_TOLERANCE)
        writer.add(case, "nldf_convolution_function_derivatives", np.stack(dfun), "reverse_feature_derivatives", "Convolution-function derivatives with respect to rho/same-spin sigma/optional tau", FFT_TOLERANCE)
        plan = model["plan"]
        case["model_structure"]["nldf_plan"] = {
            "class": _class_name(plan),
            "alpha0": float(plan.alpha0),
            "lambd": float(plan.lambd),
            "nalpha": int(plan.nalpha),
            "alphas": np.asarray(plan.alphas).tolist(),
            "num_vj": int(plan.num_vj),
        }

    for prefix in ("baseline", "feature", "total"):
        for quantity in ("exc", "vrho", "vsigma", "vtau"):
            key = f"{quantity}_{prefix}_{'g' if quantity == 'exc' else ('xg' if quantity == 'vsigma' else 'sg')}"
            value = result.get(key)
            if value is not None:
                tolerance = LIBXC_TOLERANCE if prefix == "baseline" else FFT_TOLERANCE
                writer.add(case, key, value, "dual_density_role_outputs", f"Role-separated bridge {key} output", tolerance)
    writer.add(case, "integrated_results", _integrated_results(result, inputs), "integrated_results", "Volume-integrated ABACUS bridge energy and potential contractions", FFT_TOLERANCE)

    case["stages"] = sorted(case["stages"])
    bridge.destroy(info["id"])
    return case


def _provenance(parent: Path, generator: Path) -> dict[str, Any]:
    source_relpaths = (
        "ciderpress/lib/bridge/cider_bridge_py.py",
        "ciderpress/lib/bridge/nldf_fft_standalone.py",
        "ciderpress/dft/plans.py",
        "ciderpress/dft/xc_evaluator.py",
        "ciderpress/dft/feat_normalizer.py",
        "ciderpress/dft/transform_data.py",
        "ciderpress/dft/baselines.py",
    )
    try:
        commit = _git(parent, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "ciderpress_git_commit": commit,
        "ciderpress_source_sha256": {
            relpath: _sha256(parent / relpath) for relpath in source_relpaths
        },
        "generator_sha256": _sha256(generator),
        "platform": platform.platform(),
    }


def build(parent: Path, output: Path) -> dict[str, Any]:
    parent = parent.resolve()
    output = output.resolve()
    sys.path.insert(0, str(parent))
    writer = FixtureWriter(output)
    writer.reset()
    cases = [_generate_case(parent, writer, definition) for definition in CASES]
    writer.finish()
    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "byte_order": "little",
        "array_order": "C",
        "serialization": "raw binary payloads described by JSON; never pickle",
        "default_tolerances": DEFAULT_TOLERANCE,
        "generation_parameters": {
            "grid_shape": list(GRID_SHAPE),
            "cell_length": CELL_LENGTH,
            "xmix": XMIX,
            "nalpha": NALPHA,
            "encut": ENCUT,
        },
        "generation_command": "LD_LIBRARY_PATH=/home/bonan/appdir/CiderPress/ciderpress/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/bonan/appdir/CiderPress/.venv/bin/python3 source/source_hamilt/module_xc/cider/test/fixtures/v1/generate_oracle.py --parent /home/bonan/appdir/CiderPress",
        "validation_command": "python3 source/source_hamilt/module_xc/cider/test/fixtures/v1/validate_oracle.py",
        "reproducibility_command": "LD_LIBRARY_PATH=/home/bonan/appdir/CiderPress/ciderpress/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/bonan/appdir/CiderPress/.venv/bin/python3 source/source_hamilt/module_xc/cider/test/fixtures/v1/generate_oracle.py --parent /home/bonan/appdir/CiderPress --check",
        "provenance": _provenance(parent, Path(__file__).resolve()),
        "cases": cases,
        "arrays": dict(sorted(writer.arrays.items())),
        "payload_bytes": writer.payload_bytes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _generated_files(root: Path) -> dict[str, str]:
    paths = [root / "manifest.json", root / "arrays.bin"]
    return {str(path.relative_to(root)): _sha256(path) for path in paths if path.is_file()}


def check(parent: Path, output: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="cider-oracle-v1-") as tmp:
        regenerated = Path(tmp)
        build(parent, regenerated)
        expected = _generated_files(output)
        actual = _generated_files(regenerated)
        if expected != actual:
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            changed = sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            )
            print(
                f"FAIL: generated oracle differs missing={missing} extra={extra} changed={changed}",
                file=sys.stderr,
            )
            return False
    print(f"PASS: deterministic regeneration matches {len(expected)} generated files")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True, help="Read-only parent CiderPress checkout")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check", action="store_true", help="Regenerate in a temporary directory and compare checksums")
    args = parser.parse_args()
    if args.check:
        return 0 if check(args.parent, args.output) else 1
    manifest = build(args.parent, args.output)
    print(
        f"generated v1 CIDER oracle: cases={len(manifest['cases'])} "
        f"arrays={len(manifest['arrays'])} payload_bytes={manifest['payload_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
