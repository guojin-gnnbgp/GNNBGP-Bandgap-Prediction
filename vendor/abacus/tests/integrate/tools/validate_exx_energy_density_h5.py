#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


ARRAY_KEY = "exx_energy_density_g"


def fail(message):
    print(f"validate_exx_energy_density_h5: {message}", file=sys.stderr)
    return 1


def load_record(handle):
    if "record_json" not in handle:
        raise KeyError("missing /record_json")
    raw = handle["record_json"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def main():
    parser = argparse.ArgumentParser(
        description="Validate optional conventional-gauge EXX energy density in an ABACUS training HDF5 file."
    )
    parser.add_argument("h5_file", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--expect-present", action="store_true")
    mode.add_argument("--expect-absent", action="store_true")
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol-ha", type=float, default=1.0e-8)
    args = parser.parse_args()

    if not args.h5_file.is_file():
        return fail(f"file not found: {args.h5_file}")

    try:
        with h5py.File(args.h5_file, "r") as handle:
            record = load_record(handle)
            arrays = record.get("arrays", {})
            exx_label = record.get("exx_label", {})
            metadata = exx_label.get("energy_density")
            dataset_path = f"arrays/{ARRAY_KEY}"
            dataset_present = dataset_path in handle
            mapping_present = arrays.get(ARRAY_KEY) == ARRAY_KEY

            if args.expect_absent:
                if dataset_present or mapping_present or metadata is not None:
                    return fail("EXX energy-density field is present when absence was expected")
                print(json.dumps({"file": str(args.h5_file), "present": False}, sort_keys=True))
                return 0

            if not dataset_present:
                return fail(f"missing /{dataset_path}")
            if not mapping_present:
                return fail(f'missing arrays["{ARRAY_KEY}"] mapping')
            if not isinstance(metadata, dict) or not metadata.get("present"):
                return fail("missing present exx_label.energy_density metadata")

            field = np.asarray(handle[dataset_path])
            nxyz = int(record["nxyz"])
            if field.shape != (nxyz,):
                return fail(f"field shape {field.shape}, expected {(nxyz,)}")
            if field.dtype != np.dtype(np.float64):
                return fail(f"field dtype {field.dtype}, expected float64")
            if not np.isfinite(field).all():
                return fail("field contains non-finite values")
            if float(np.sum(field)) >= 0.0:
                return fail("integrated field is not negative")

            required_metadata = {
                "array": ARRAY_KEY,
                "unit": "Ha Bohr^-3",
                "gauge": "conventional_fock_exchange_hole",
                "spin": "summed",
                "grid": "charge_density_fft_real_space",
                "full_range": True,
                "hybrid_scaled": False,
                "base_state": "pbe_scf_density_and_pw_wavefunctions",
            }
            for key, expected in required_metadata.items():
                if metadata.get(key) != expected:
                    return fail(f"metadata {key}={metadata.get(key)!r}, expected {expected!r}")
            max_imaginary = float(metadata["max_imaginary_abs_Ha_bohr3"])
            if not np.isfinite(max_imaginary) or max_imaginary < 0.0:
                return fail("max imaginary residual is invalid")
            if metadata.get("k_policy") not in {"pair-map", "full-active"}:
                return fail(f"unsupported k_policy metadata: {metadata.get('k_policy')!r}")

            dvol = float(record["dV_bohr3"])
            integrated = float(np.sum(field, dtype=np.float64) * dvol)
            scalar = float(exx_label["energy_Ha"])
            abs_residual = abs(integrated - scalar)
            rel_residual = abs_residual / max(abs(scalar), np.finfo(float).tiny)
            tolerance = max(args.atol_ha, args.rtol * abs(scalar))
            if abs_residual > tolerance:
                return fail(
                    f"integral mismatch: integrated={integrated:.16e} Ha "
                    f"scalar={scalar:.16e} Ha residual={abs_residual:.3e} Ha tolerance={tolerance:.3e} Ha"
                )
            for key, actual in (
                ("integrated_energy_Ha", integrated),
                ("scalar_energy_Ha", scalar),
                ("integration_abs_residual_Ha", abs_residual),
                ("integration_rel_residual", rel_residual),
            ):
                if not np.isclose(float(metadata[key]), actual, rtol=1.0e-10, atol=1.0e-12):
                    return fail(f"metadata {key} does not match independently computed value")

            summary = {
                "file": str(args.h5_file),
                "present": True,
                "shape": list(field.shape),
                "dtype": str(field.dtype),
                "scalar_energy_Ha": scalar,
                "integrated_energy_Ha": integrated,
                "integration_abs_residual_Ha": abs_residual,
                "integration_rel_residual": rel_residual,
                "max_imaginary_abs_Ha_bohr3": max_imaginary,
                "k_policy": metadata["k_policy"],
            }
            print(json.dumps(summary, sort_keys=True))
            return 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
