#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def fail(message):
    print(f"validate_band_edge_h5: {message}", file=sys.stderr)
    return 1


def load_record(handle):
    raw = handle["record_json"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def main():
    parser = argparse.ArgumentParser(description="Validate native dynamic ABACUS band-edge labels.")
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("--expect-nedge", type=int)
    parser.add_argument("--expect-band-only", action="store_true")
    args = parser.parse_args()

    if not args.h5_file.is_file():
        return fail(f"file not found: {args.h5_file}")

    try:
        with h5py.File(args.h5_file, "r") as handle:
            record = load_record(handle)
            label = record["band_edge_label"]
            if label.get("present") is not True:
                return fail("band_edge_label.present is not true")
            if label.get("edge_selection_policy") != "degenerate_frontier_average":
                return fail("unexpected edge-selection policy")
            if float(label.get("degenerate_tolerance_Ha", 0.0)) <= 0.0:
                return fail("missing positive degeneracy tolerance")
            if float(label.get("occupation_threshold", -1.0)) < 0.0:
                return fail("missing non-negative occupation threshold")

            arrays = {name: np.asarray(handle[f"arrays/{name}"]) for name in (
                "band_edge_index_tb",
                "band_edge_role_t",
                "band_edge_eigenvalue_Ha_t",
                "band_edge_occupation_t",
                "band_edge_kpt_direct_tv",
                "band_edge_kpt_cart_tv",
                "band_edge_exx_shift_Ha_t",
                "band_edge_orbital_rho_tsg",
                "band_edge_orbital_sigma_txg",
                "band_edge_orbital_tau_tsg",
            )}
            nedge = arrays["band_edge_index_tb"].shape[0]
            if int(label.get("nedge", -1)) != nedge:
                return fail("band_edge_label.nedge does not match band-edge arrays")
            nspin = int(record["nspin"])
            nsigma = int(record["nsigma"])
            nxyz = int(record["nxyz"])
            expected_shapes = {
                "band_edge_index_tb": (nedge, 3),
                "band_edge_role_t": (nedge,),
                "band_edge_eigenvalue_Ha_t": (nedge,),
                "band_edge_occupation_t": (nedge,),
                "band_edge_kpt_direct_tv": (nedge, 3),
                "band_edge_kpt_cart_tv": (nedge, 3),
                "band_edge_exx_shift_Ha_t": (nedge,),
                "band_edge_orbital_rho_tsg": (nedge, nspin, nxyz),
                "band_edge_orbital_sigma_txg": (nedge, nsigma, nxyz),
                "band_edge_orbital_tau_tsg": (nedge, nspin, nxyz),
            }
            for name, expected in expected_shapes.items():
                if arrays[name].shape != expected:
                    return fail(f"{name} shape {arrays[name].shape}, expected {expected}")
                if not np.isfinite(arrays[name]).all():
                    return fail(f"{name} contains non-finite values")
            if args.expect_nedge is not None and nedge != args.expect_nedge:
                return fail(f"nedge={nedge}, expected {args.expect_nedge}")

            roles = np.rint(arrays["band_edge_role_t"]).astype(int)
            vbm = np.flatnonzero(roles == 0)
            cbm = np.flatnonzero(roles == 1)
            if not len(vbm) or not len(cbm) or np.any((roles != 0) & (roles != 1)):
                return fail("role array must contain at least one VBM and one CBM")
            if label.get("vbm_count") != len(vbm) or label.get("cbm_count") != len(cbm):
                return fail("role-count metadata does not match role array")
            shifts = arrays["band_edge_exx_shift_Ha_t"]
            expected_gap = float(np.mean(shifts[cbm]) - np.mean(shifts[vbm]))
            if not np.isclose(float(label["gap_ref_shift_Ha"]), expected_gap, rtol=0.0, atol=1.0e-14):
                return fail("gap_ref_shift_Ha is not the CBM-minus-VBM mean shift")

            if args.expect_band_only:
                if record.get("label_mode") != "pbe_base_state_frontier_exx_band_edge_label":
                    return fail("unexpected band-only label_mode")
                if record.get("exx_energy_evaluated") is not False:
                    return fail("band-only record claims scalar EXX evaluation")
                if record.get("exx_label", {}).get("present") is not False:
                    return fail("band-only record has exx_label.present=true")

            print(json.dumps({
                "file": str(args.h5_file),
                "nedge": nedge,
                "vbm_count": len(vbm),
                "cbm_count": len(cbm),
                "gap_ref_shift_Ha": expected_gap,
            }, sort_keys=True))
            return 0
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
