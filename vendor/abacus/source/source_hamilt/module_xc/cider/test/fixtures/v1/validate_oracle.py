#!/usr/bin/env python3
"""Validate the version-1 native CIDER Python-reference oracle fixtures.

This validator intentionally uses only the Python standard library.  It checks
the manifest and raw little-endian array payloads without importing CiderPress,
NumPy, or a pickle-capable container reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from pathlib import PurePosixPath


SCHEMA = "abacus-cider-python-reference-oracle-v1"
REQUIRED_CASE_COVERAGE = {
    "gga",
    "mgga",
    "nldf",
    "non_nldf",
    "spin_unpolarized",
    "spin_polarized",
    "valence",
    "valence_pseudo_core",
}
REQUIRED_STAGES = {
    "loader_metadata",
    "decoded_model_arrays",
    "semilocal_raw_features",
    "nldf_interpolation_arguments",
    "nldf_convolution_features",
    "normalized_features",
    "transformed_descriptors",
    "spline_forward",
    "spline_derivatives",
    "reverse_feature_derivatives",
    "dual_density_role_outputs",
    "integrated_results",
}
DTYPES = {"<f8": (8, "d"), "<i8": (8, "q"), "<i4": (4, "i")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _product(shape: list[int]) -> int:
    value = 1
    for extent in shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ValueError(f"invalid shape extent {extent!r}")
        value *= extent
    return value


def _check_finite_payload(data: bytes, dtype: str, array_id: str) -> None:
    if dtype != "<f8":
        return
    for (value,) in struct.iter_unpack("<d", data):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in {array_id}")


def validate(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing required manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != 1:
        raise ValueError("manifest schema/version is not v1")
    if not manifest.get("generation_command") or not manifest.get("validation_command"):
        raise ValueError("manifest must record generation and validation commands")
    provenance = manifest.get("provenance", {})
    for key in ("python", "numpy", "ciderpress_git_commit", "generator_sha256"):
        if not provenance.get(key):
            raise ValueError(f"missing provenance field {key}")
    if _sha256(root / "generate_oracle.py") != provenance["generator_sha256"]:
        raise ValueError("generator checksum does not match provenance")
    tolerances = manifest.get("default_tolerances", {})
    for key in ("atol", "rtol"):
        value = tolerances.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid default tolerance {key}")

    cases = manifest.get("cases")
    arrays = manifest.get("arrays")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a nonempty list")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("manifest arrays must be a nonempty mapping")

    coverage: set[str] = set()
    stages: set[str] = set()
    case_ids: set[str] = set()
    referenced_arrays: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError(f"invalid or duplicate case id {case_id!r}")
        case_ids.add(case_id)
        coverage.update(case.get("coverage", []))
        stages.update(case.get("stages", []))
        model = case.get("source_model", {})
        if len(model.get("sha256", "")) != 64 or not model.get("path"):
            raise ValueError(f"case {case_id} lacks source-model provenance")
        if not case.get("array_ids"):
            raise ValueError(f"case {case_id} has no arrays")
        for array_id in case["array_ids"]:
            if array_id not in arrays:
                raise ValueError(f"case {case_id} references unknown array {array_id}")
            if array_id in referenced_arrays:
                raise ValueError(f"array {array_id} is referenced by more than one case")
            referenced_arrays.add(array_id)

    missing_coverage = REQUIRED_CASE_COVERAGE - coverage
    missing_stages = REQUIRED_STAGES - stages
    if missing_coverage:
        raise ValueError(f"missing case coverage: {sorted(missing_coverage)}")
    if missing_stages:
        raise ValueError(f"missing oracle stages: {sorted(missing_stages)}")
    orphan_arrays = set(arrays) - referenced_arrays
    if orphan_arrays:
        raise ValueError(f"unreferenced arrays: {sorted(orphan_arrays)}")

    payload_paths: set[str] = set()
    payload_ranges: dict[str, list[tuple[int, int, str]]] = {}
    total_bytes = 0
    for array_id, spec in arrays.items():
        relpath = spec.get("path")
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        if not isinstance(relpath, str):
            raise ValueError(f"invalid payload path for {array_id}")
        path_parts = PurePosixPath(relpath)
        if path_parts.is_absolute() or ".." in path_parts.parts or path_parts.suffix in {
            ".pkl",
            ".pickle",
            ".joblib",
        }:
            raise ValueError(f"unsafe payload path for {array_id}: {relpath}")
        payload_paths.add(relpath)
        if dtype not in DTYPES or not isinstance(shape, list) or not shape:
            raise ValueError(f"invalid dtype/shape for {array_id}")
        payload = root / relpath
        if not payload.is_file():
            raise ValueError(f"missing payload for {array_id}: {payload}")
        itemsize, _ = DTYPES[dtype]
        expected_bytes = _product(shape) * itemsize
        array_bytes = spec.get("bytes")
        offset = spec.get("offset")
        if array_bytes != expected_bytes:
            raise ValueError(
                f"byte-count mismatch for {array_id}: expected {expected_bytes}, "
                f"manifest {array_bytes}"
            )
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"invalid payload offset for {array_id}")
        with payload.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(array_bytes)
        if len(data) != array_bytes:
            raise ValueError(f"payload segment extends past end of file for {array_id}")
        if hashlib.sha256(data).hexdigest() != spec.get("sha256"):
            raise ValueError(f"checksum mismatch for {array_id}")
        payload_ranges.setdefault(relpath, []).append(
            (offset, offset + array_bytes, array_id)
        )
        tolerance = spec.get("tolerance", tolerances)
        if not isinstance(tolerance, dict) or "atol" not in tolerance or "rtol" not in tolerance:
            raise ValueError(f"missing tolerance for {array_id}")
        for key in ("atol", "rtol"):
            value = tolerance[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid {key} tolerance for {array_id}")
        _check_finite_payload(data, dtype, array_id)
        total_bytes += array_bytes

    for relpath, ranges in payload_ranges.items():
        cursor = 0
        for start, end, array_id in sorted(ranges):
            if start != cursor:
                raise ValueError(
                    f"payload ranges are overlapping or non-contiguous before {array_id}"
                )
            cursor = end
        if (root / relpath).stat().st_size != cursor:
            raise ValueError(f"unreferenced trailing bytes in {relpath}")

    expected_files = {"manifest.json", "README.md", "generate_oracle.py", "validate_oracle.py"}
    expected_files.update(payload_paths)
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "__pycache__"
    }
    unexpected = actual_files - expected_files
    missing = expected_files - actual_files
    if unexpected or missing:
        raise ValueError(
            f"fixture file inventory mismatch: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    if total_bytes != manifest.get("payload_bytes"):
        raise ValueError("manifest payload_bytes does not match array payloads")
    return {
        "cases": len(cases),
        "arrays": len(arrays),
        "payload_bytes": total_bytes,
        "coverage": sorted(coverage),
        "stages": sorted(stages),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        summary = validate(args.root.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: v1 CIDER oracle "
        f"cases={summary['cases']} arrays={summary['arrays']} "
        f"payload_bytes={summary['payload_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
